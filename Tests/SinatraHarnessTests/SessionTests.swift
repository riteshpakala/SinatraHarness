import Foundation
import Testing
@testable import SinatraHarness

@Suite("Session: grounding loop, band and store")
struct SessionTests {
    let owner = OwnerID("User@Example.com")
    let emphasis: [String: Float] = ["garden#0": 2, "bread#0": 0.3, "bike#0": 0.3]

    /// Plan a turn, "decode" `answer`, measure it with the lexical estimate, record it.
    @discardableResult
    func turn(
        _ session: SinatraSession, tokenizer: SimpleTokenizer, answer: String, at: Date,
        emphasis: [String: Float]? = nil, measure: Bool = true
    ) async throws -> (InjectionPlan, GroundingMeasurement) {
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: at))
        let raw = LexicalGroundingEstimate.raw(output: tokenizer.encode(answer), plan: plan, emphasis: emphasis ?? self.emphasis)
        let measurement = measure ? await session.groundingMeasurement(plan: plan, raw: raw) : .skipped("no bare prompt")
        await session.generationDidFinish(
            owner: owner, turnId: plan.turnId, assistantText: answer,
            startedAt: at.addingTimeInterval(1), finishedAt: at.addingTimeInterval(5), measurement: measurement)
        return (plan, measurement)
    }

    @Test func aMeasuredTurnIsLabelledAtOnce() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        let (plan, measurement) = try await turn(session, tokenizer: tokenizer, answer: breadText, at: t0)
        #expect(plan.bias == nil)  // cold start: nothing learned, identity
        #expect(measurement.measured)
        #expect(measurement.summary.driftShare > 0.5)  // it answered from bread; the context pointed at the garden
        let garden = try #require(measurement.attribution.first { $0.partitionId == "garden#0" })
        #expect(garden.intent == 1 && garden.uptake == 0)
        let summary = await session.summary(owner: owner)
        #expect(summary.observations == 1 && summary.measuredTurns == 1 && summary.labelledEvents == 3)
        #expect(summary.meanDrift != nil)
    }

    @Test func anUnmeasuredTurnTeachesNothing() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        try await turn(session, tokenizer: tokenizer, answer: breadText, at: Date(), measure: false)
        let summary = await session.summary(owner: owner)
        #expect(summary.observations == 1 && summary.measuredTurns == 0 && summary.labelledEvents == 0)
        #expect(await session.groundingReport(owner: owner).unmeasured["no bare prompt"] == 1)
    }

    @Test func theContextsFavouriteEarnsAPositiveSteer() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        var clock = Date(timeIntervalSince1970: 1_758_000_000)
        // The model keeps answering from the bread note while the context points at the garden.
        for _ in 0..<6 {
            try await turn(session, tokenizer: tokenizer, answer: breadText, at: clock)
            clock = clock.addingTimeInterval(120)
        }
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: clock), dryRun: true)
        let weights = plan.perPartitionWeights
        #expect(weights["garden#0"]! > 0.3)
        #expect(weights["bread#0"]! < 0)
        #expect(weights["garden#0"]! > weights["bike#0"]!)
        #expect(plan.bias != nil)
        #expect(plan.mask?.partitionIds == ["garden#0", "bread#0", "bike#0"])
        // The garden's content tokens carry the positive bias.
        let gardenToken = tokenizer.encode("tomatoes")[0]
        #expect((plan.bias?.value(of: gardenToken) ?? 0) > 0)
    }

    @Test func aGroundedTurnKeepsItsSteer() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        var clock = Date(timeIntervalSince1970: 1_758_000_000)
        // The output follows the context's own push: nothing to correct.
        for _ in 0..<4 {
            let (_, measurement) = try await turn(session, tokenizer: tokenizer, answer: gardenText, at: clock)
            #expect(measurement.summary.driftShare == 0)
            clock = clock.addingTimeInterval(120)
        }
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: clock), dryRun: true)
        #expect(plan.perPartitionWeights.values.allSatisfy { abs($0) < 1e-6 })
        #expect(plan.bias == nil)
    }

    @Test func theThirtyDayBandIsEnforced() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        try await turn(session, tokenizer: tokenizer, answer: breadText, at: t0)
        _ = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: t0.addingTimeInterval(31 * 86_400)))
        let summary = await session.summary(owner: owner)
        #expect(summary.labelledEvents == 0)
        #expect(summary.observations == 1)
    }

    @Test func dryRunsRecordNothing() async throws {
        let session = makeSession(store: temporaryStore())
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()), dryRun: true)
        #expect(plan.dryRun && plan.hasContext)
        #expect(await session.summary(owner: owner).observations == 0)
    }

    @Test func unfinishedTurnsAreForgotten() async throws {
        let session = makeSession(store: temporaryStore())
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()))
        await session.generationAbandoned(owner: owner, turnId: plan.turnId)
        #expect(await session.summary(owner: owner).observations == 0)

        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        _ = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: t0))
        _ = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: t0.addingTimeInterval(2 * 3600)))
        #expect(await session.summary(owner: owner).observations == 1)  // the first never finished
    }

    @Test func ledgersPersistAndCanBeForgotten() async throws {
        let store = temporaryStore()
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: store, tokenizer: tokenizer)
        try await turn(session, tokenizer: tokenizer, answer: breadText, at: Date())
        await session.flush()
        let reopened = makeSession(store: store)
        let summary = await reopened.summary(owner: owner)
        #expect(summary.observations == 1 && summary.measuredTurns == 1)
        #expect(await reopened.knownOwners() == [owner])
        #expect(FeatureStore.ownerKey(owner).hasPrefix("userexamplecom-"))
        try await reopened.forget(owner: owner)
        #expect(await makeSession(store: store).summary(owner: owner).observations == 0)
    }

    @Test func aSchemaOneLedgerStartsFresh() async throws {
        let store = temporaryStore()
        let directory = store.appendingPathComponent("owners").appendingPathComponent(FeatureStore.ownerKey(owner))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"schema":1,"owner":"x","turns":[{"id":"not-a-v2-turn"}]}"#.utf8)
            .write(to: directory.appendingPathComponent("ledger.json"))
        let summary = await makeSession(store: store).summary(owner: owner)
        #expect(summary.observations == 0)
    }

    @Test func observeLabelsARecordedTurn() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()))
        let raw = LexicalGroundingEstimate.raw(output: tokenizer.encode(breadText), plan: plan, emphasis: emphasis)
        let measurement = await session.groundingMeasurement(plan: plan, raw: raw)
        let receipt = try await session.observe(owner: owner, turn: TurnObservation(turnId: plan.turnId, assistantText: breadText, measurement: measurement))
        #expect(receipt.targets["garden#0"]! > 0 && receipt.targets["bread#0"]! < 0)
        await #expect(throws: SinatraError.self) {
            try await session.observe(owner: owner, turn: TurnObservation(turnId: UUID(), measurement: measurement))
        }
    }
}
