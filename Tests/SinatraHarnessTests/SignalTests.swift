import Foundation
import Testing
@testable import SinatraHarness

@Suite("Grounding signals")
struct GroundingSignalTests {
    let configuration = SinatraConfiguration()

    @Test func classesFollowTheThreshold() {
        #expect(Grounding.classify(influence: 0.5, isContent: true, threshold: 0.5) == .grounded)
        #expect(Grounding.classify(influence: 0.49, isContent: true, threshold: 0.5) == .unsupported)
        #expect(Grounding.classify(influence: -0.5, isContent: true, threshold: 0.5) == .contradicted)
        #expect(Grounding.classify(influence: 3, isContent: false, threshold: 0.5) == .function)
    }

    @Test func sharesSplitATokenByTermFrequency() {
        let shares = Grounding.shares([[1: 3, 2: 1], [1: 1]])
        let one = Dictionary(uniqueKeysWithValues: (shares[1] ?? []).map { ($0.partition, $0.share) })
        #expect(one[0] == 0.75 && one[1] == 0.25)
        #expect(shares[2]?.count == 1 && shares[2]?.first?.share == 1)
        #expect(shares[9] == nil)
    }

    @Test func verbatimRunsAreFound() {
        let covered = Grounding.copiedPositions(output: [9, 1, 2, 3, 4, 5, 6, 9], source: [0, 1, 2, 3, 4, 5, 6, 7], n: 6)
        #expect(covered == Set(1...6))
        #expect(Grounding.copiedPositions(output: [1, 2, 3], source: [1, 2, 3], n: 6).isEmpty)
    }

    /// Two partitions: A holds tokens 10 and 11, B holds 20. The context pushes hardest
    /// toward A's token 10 at every step.
    func plan(weights: [Float] = [0, 0]) -> [PartitionTerms] {
        [
            PartitionTerms(id: "A", documentId: "a", counts: [10: 2, 11: 1], sequence: [10, 11], relevancyShare: 0.5, relevancyWeight: 1, appliedWeight: weights[0]),
            PartitionTerms(id: "B", documentId: "b", counts: [20: 1], sequence: [20], relevancyShare: 0.5, relevancyWeight: 1, appliedWeight: weights[1]),
        ]
    }

    @Test func aMeasurementOnHandMadeNumbers() {
        // Steps: 10 (from A, ι 2), 20 (from B, ι 0.2), 30 (nowhere, ι 0.1), 1 (a function token).
        // The context lifts every content step by 2 nats on average, toward A's token 10.
        let raw = GroundingRaw(
            tokens: [10, 20, 30, 1],
            logpCtx: [-1, -2.8, -2.9, -0.5], logpBare: [-3, -3, -3, -0.5],
            entropyCtx: [1, 1, 1, 1], entropyBare: [2, 2, 2, 2], contextKL: [2, 2, 2, 0],
            tune: [10, 10, 10, 10], tuneNats: [3, 3, 3, 0])
        let m = Grounding.measurement(
            raw: raw, partitions: plan(), isContent: { $0 >= 10 }, decode: { "t\($0)" }, detail: .summary,
            configuration: configuration)
        let s = m.summary
        #expect(m.measured)
        #expect(s.contentTokens == 3)
        #expect(abs(s.grounding - 1.0 / 3) < 1e-6)
        #expect(abs(s.unsupportedShare - 2.0 / 3) < 1e-6)
        // drift = KL − ι: 0, 1.8, 1.9 over content tokens
        #expect(abs(s.drift - (0 + 1.8 + 1.9) / 3) < 1e-5)
        #expect(abs(s.driftShare - 2.0 / 3) < 1e-6)
        #expect(s.firstDriftStep == 1)
        #expect(abs(s.contextDependence - (2 + 0.2 + 0.1 + 0)) < 1e-5)
        #expect(abs(s.meanContextKL - 1.5) < 1e-6)
        // The unsupported tokens were unsure: risk = 1 − p_ctx.
        let risk = ((1 - exp(Float(-2.8))) + (1 - exp(Float(-2.9)))) / 3
        #expect(abs(s.hallucinationRisk - risk) < 1e-5)
        // Attribution: A gets ι⁺ of token 10 (2), B of token 20 (0.2); token 30's 0.1 is unattributed.
        let a = m.attribution.first { $0.partitionId == "A" }!
        let b = m.attribution.first { $0.partitionId == "B" }!
        #expect(abs(a.nats - 2) < 1e-4 && abs(b.nats - 0.2) < 1e-4)
        #expect(abs(s.unattributed - 0.1) < 1e-5)
        #expect(abs(a.uptake - 2 / 2.3) < 1e-3 && abs(b.uptake - 0.2 / 2.3) < 1e-3)
        // Every missed push pointed at A's token 10.
        #expect(a.intent == 1 && b.intent == 0)
        #expect(abs(a.missed - 3.7) < 1e-4)
        #expect(abs(a.coverage - 0.5) < 1e-6 && b.coverage == 1)
        #expect(m.steps.count == 4 && m.steps[1].kind == .unsupported && m.steps[3].kind == .function)
        #expect(m.steps[1].tuneText == "t10")
    }

    @Test func noContextMeansNothingMoved() {
        let raw = GroundingRaw(
            tokens: [10, 20], logpCtx: [-2, -2], logpBare: [-2, -2], entropyCtx: [1, 1], entropyBare: [1, 1],
            contextKL: [0, 0], tune: [10, 10], tuneNats: [0, 0])
        let s = Grounding.measurement(
            raw: raw, partitions: plan(), isContent: { _ in true }, decode: { _ in nil }, detail: .summary,
            configuration: configuration).summary
        #expect(s.grounding == 0 && s.drift == 0 && s.driftShare == 0 && s.contextDependence == 0 && s.meanContextKL == 0)
    }

    @Test func theSteerTargetFollowsWhatTheOutputMissed() {
        // A grounded turn keeps its steer.
        #expect(Grounding.steerTarget(applied: 0.3, relevancyWeight: 1, intent: 1, uptake: 0, driftShare: 0, parrot: 0, configuration: configuration) == 0.3)
        // A drifting turn leans toward the under-used partition the context pushed for...
        let missed = Grounding.steerTarget(applied: 0, relevancyWeight: 1, intent: 1, uptake: 0, driftShare: 0.8, parrot: 0, configuration: configuration)
        #expect(abs(missed - 0.8) < 1e-6)
        // ...scaled by relevancy (a stale partition counts for less)...
        let stale = Grounding.steerTarget(applied: 0, relevancyWeight: 0.4, intent: 1, uptake: 0, driftShare: 0.8, parrot: 0, configuration: configuration)
        #expect(abs(stale - 0.32) < 1e-6)
        // ...and away from the one it over-used.
        #expect(Grounding.steerTarget(applied: 0, relevancyWeight: 1, intent: 0, uptake: 0.9, driftShare: 0.8, parrot: 0, configuration: configuration) < 0)
        // Copying damps only a push the steer made itself.
        #expect(Grounding.steerTarget(applied: 0, relevancyWeight: 1, intent: 0, uptake: 0, driftShare: 0, parrot: 1, configuration: configuration) == 0)
        #expect(abs(Grounding.steerTarget(applied: 0.5, relevancyWeight: 1, intent: 0, uptake: 0, driftShare: 0, parrot: 0.5, configuration: configuration) - 0.25) < 1e-6)
        #expect(Grounding.steerTarget(applied: 0.9, relevancyWeight: 1, intent: 1, uptake: 0, driftShare: 1, parrot: 0, configuration: configuration) == 1)
    }

    @Test func relevancyAndWeights() {
        #expect(Grounding.relevancy(bandWeight: 1, score01: 1) == 1)
        #expect(Grounding.relevancy(bandWeight: 0.4, score01: 0) == 0.2)
        #expect(Grounding.normalised([1, 3]) == [0.25, 0.75])
        #expect(Grounding.normalised([0, 0]) == [0.5, 0.5])
        #expect(Grounding.sampleWeight(ageDays: 3, configuration: configuration) == 1)
        #expect(Grounding.sampleWeight(ageDays: 10, configuration: configuration) == 0.7)
        #expect(Grounding.sampleWeight(ageDays: 31, configuration: configuration) == 0)
        #expect(Grounding.wordCount("Hello, world — it's 2026!") == 5)
    }

    @Test func theLexicalEstimatePointsAtTheEmphasisedPartition() async throws {
        let tokenizer = SimpleTokenizer()
        let session = makeSession(store: temporaryStore(), tokenizer: tokenizer)
        let plan = try await session.prepareTurn(TurnInput(owner: "o", retrieved: partitions()), dryRun: true)
        let raw = LexicalGroundingEstimate.raw(
            output: tokenizer.encode(breadText), plan: plan, emphasis: ["garden#0": 2, "bread#0": 0.3, "bike#0": 0.3])
        #expect(raw.tuneNats.allSatisfy { $0 == 2 })
        #expect(raw.tokens.count == raw.logpCtx.count)
        #expect(zip(raw.logpCtx, raw.logpBare).contains { $0 - $1 > 0.29 })
    }
}

@Suite("Time features and relevancy bands")
struct TimeFeatureTests {
    let configuration = SinatraConfiguration()

    @Test func bandEdges() {
        #expect(TimeFeatures.band(ageDays: 7, configuration: configuration) == .fresh)
        #expect(TimeFeatures.band(ageDays: 7.01, configuration: configuration) == .mid)
        #expect(TimeFeatures.band(ageDays: 30, configuration: configuration) == .mid)
        #expect(TimeFeatures.band(ageDays: 30.01, configuration: configuration) == .stale)
        #expect(TimeFeatures.bandWeight(.stale, configuration: configuration) == 0.4)
    }

    @Test func logNormalisation() {
        #expect(TimeFeatures.lnNorm(days: 0) == 0)
        #expect(abs(TimeFeatures.lnNorm(days: 365) - 1) < 1e-6)
        #expect(TimeFeatures.lnNorm(days: 10_000) == 1)
        #expect(TimeFeatures.lnNorm(seconds: 1800, horizon: 1800) == 1)
    }

    @Test func recencyIsTheNewestKnownDate() {
        let now = Date(timeIntervalSince1970: 1_758_000_000)
        let old = now.addingTimeInterval(-60 * 86_400)
        let recent = now.addingTimeInterval(-2 * 86_400)
        let edited = Partition(id: "p", documentId: "d", text: "x", createdAt: old, modifiedAt: recent)
        #expect(TimeFeatures.recencyReference(edited, firstSeen: now, now: now) == recent)
        let unknown = Partition(id: "p", documentId: "d", text: "x")
        #expect(TimeFeatures.recencyReference(unknown, firstSeen: old, now: now) == old)
        #expect(TimeFeatures.recencyReference(unknown, firstSeen: nil, now: now) == now)
    }

    @Test func periodicsAreUnitCircles() {
        let values = TimeFeatures.periodics(Date(timeIntervalSince1970: 1_758_000_000), timeZone: TimeZone(identifier: "UTC")!)
        #expect(values.count == 4)
        #expect(abs(values[0] * values[0] + values[1] * values[1] - 1) < 1e-5)
        #expect(abs(values[2] * values[2] + values[3] * values[3] - 1) < 1e-5)
    }
}

@Suite("Content-token filter")
struct TokenFilterTests {
    @Test func controlAndStopwordsAreNotContent() {
        let filter = TokenFilter(tokenizer: StubTokenizer())
        for id in [0, 1, 2, 5, 6, 7] { #expect(filter.isControl(id)) }
        #expect(filter.isContent(9))
        #expect(filter.isContent(12))
        #expect(!filter.isContent(8))   // "the"
        #expect(!filter.isContent(10))  // ","
        #expect(!filter.isContent(11))  // "a"
        #expect(!filter.isContent(6))
    }

    @Test func hotTokensNeedAGuaranteedShare() {
        var hot = HotTokens(capacity: 4)
        for _ in 0..<30 { hot.observe([1, 2]) }
        for i in 0..<30 { hot.observe([100 + i]) }
        #expect(hot.isHot(1, ratio: 0.5, minimumPartitions: 20))
        #expect(!hot.isHot(129, ratio: 0.5, minimumPartitions: 20))
        #expect(hot.tracked <= 4)
        #expect(!HotTokens(capacity: 4).isHot(1, ratio: 0.5, minimumPartitions: 20))
    }

    @Test func clipKeepsHeadAndTail() {
        #expect(TokenBatch.clip(Array(0..<10), maxTokens: 4) == [0, 1, 8, 9])
        #expect(TokenBatch.clip([1, 2], maxTokens: 4) == [1, 2])
        let padded = TokenBatch(rows: [[1, 2, 3], [4]]).padded()
        #expect(padded.columns == 3 && padded.ids == [1, 2, 3, 4, 0, 0] && padded.mask == [1, 1, 1, 1, 0, 0])
    }
}

@Suite("Sparse bias and impact mask")
struct SparseBiasTests {
    @Test func sparseBiasBasics() {
        let bias = SparseBias(vocabularySize: 10, entries: [5: 1.0, 2: -0.5, 9: 0.25, 12: 3])
        #expect(bias.indices == [2, 5, 9])
        #expect(bias.value(of: 5) == 1 && bias.value(of: 3) == 0)
        #expect(bias.denseVector()[2] == -0.5)
        #expect(bias.maxAbs == 1 && bias.l1 == 1.75)
        #expect(bias.top(1).first?.id == 5)
    }

    @Test func clampingKeepsAttributionConsistent() throws {
        let (bias, mask) = try #require(BiasAssembly.assemble(
            perPartition: [[3: 1.5], [3: 1.0, 4: -0.2]], partitionIds: ["a", "b"], vocabularySize: 10, cap: 2))
        #expect(bias.value(of: 3) == 2)
        let shares = mask.attribution(of: 3)
        #expect(abs((shares["a"] ?? 0) - 1.2) < 1e-5 && abs((shares["b"] ?? 0) - 0.8) < 1e-5)
        #expect(abs(mask.attribution(of: 4)["b"]! - (-0.2)) < 1e-6)
    }

    @Test func lexicalBiasFollowsWeightsAndBands() throws {
        var configuration = SinatraConfiguration()
        configuration.alpha = 1
        let input = LexicalBiasBuilder.Input(
            partitionIds: ["a"], terms: [[1: 4, 2: 1]], weights: [1], bandWeights: [1])
        let (bias, _) = try #require(LexicalBiasBuilder.build(input, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }))
        #expect(abs(bias.value(of: 1) - 1) < 1e-6 && abs(bias.value(of: 2) - 0.5) < 1e-6)

        let negative = LexicalBiasBuilder.Input(partitionIds: ["a"], terms: [[1: 4]], weights: [-0.5], bandWeights: [0.4])
        let (pushed, _) = try #require(LexicalBiasBuilder.build(negative, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }))
        #expect(abs(pushed.value(of: 1) - (-0.2)) < 1e-6)

        let idle = LexicalBiasBuilder.Input(partitionIds: ["a"], terms: [[1: 4]], weights: [0], bandWeights: [1])
        #expect(LexicalBiasBuilder.build(idle, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }) == nil)

        let excluded = LexicalBiasBuilder.build(input, vocabularySize: 10, configuration: configuration, isExcluded: { $0 == 1 })
        #expect(excluded?.0.value(of: 1) == 0)
    }

    @Test func countSketchIsDeterministicAndUnitNorm() {
        let projection = CountSketchProjection(inputDimension: 128, outputDimension: 16)
        let x = (0..<128).map { Float(sin(Double($0))) }
        let y = projection.project(x)
        #expect(y == CountSketchProjection(inputDimension: 128, outputDimension: 16).project(x))
        #expect(abs(y.reduce(0) { $0 + $1 * $1 } - 1) < 1e-3)
        #expect(projection.project([Float](repeating: 0, count: 128)).allSatisfy { $0 == 0 })
    }
}
