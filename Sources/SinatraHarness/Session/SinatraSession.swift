//
//  SinatraSession.swift
//  SinatraHarness
//
//  WHAT: The side model's orchestration actor, one per loaded LLM. Per turn:
//          prepareTurn          — sweep the 30-day band, encode ONLY the retrieved
//                                 partitions, weigh them, build the injection
//          groundingMeasurement — read the scorer's numbers against the turn's partitions
//          generationDidFinish  — record the assistant side and label the turn from its
//                                 grounding measurement
//          train                — fit the weight model on the band; IMBHS on its cadence
//  PIN:  Owns its FeatureStore outright. MLX work (encode, predict, train) happens inside
//        these methods; the harness serialises them against generation with its gate.
//

import Foundation

public actor SinatraSession {
    public let configuration: SinatraConfiguration
    public let modelKey: String
    public nonisolated let storeDirectory: URL

    let tokenizer: any SinatraTokenizing
    let encoder: (any ContextEncoding)?
    let vocabularySize: Int
    let weightModelFactory: WeightModelFactory
    let log: SinatraLog?
    let store: FeatureStore
    let filter: TokenFilter
    let schema: ModelSchema
    var projection: CountSketchProjection?
    var models: [OwnerID: any WeightModeling] = [:]
    var modelOrder: [OwnerID] = []

    public init(
        configuration: SinatraConfiguration = .init(),
        storeDirectory: URL,
        modelKey: String,
        tokenizer: any SinatraTokenizing,
        encoder: (any ContextEncoding)?,
        vocabularySize: Int,
        weightModelFactory: @escaping WeightModelFactory = PriorWeightModel.factory,
        log: SinatraLog? = nil
    ) {
        self.configuration = configuration
        self.modelKey = modelKey
        self.storeDirectory = storeDirectory
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.vocabularySize = vocabularySize
        self.weightModelFactory = weightModelFactory
        self.log = log
        self.store = FeatureStore(root: storeDirectory, configuration: configuration, log: log)
        self.filter = TokenFilter(tokenizer: tokenizer, useStopwords: configuration.useStopwords)
        self.schema = ModelSchema.current(configuration)
    }

    // MARK: Weight models

    func weightModel(for owner: OwnerID) -> any WeightModeling {
        if let model = models[owner] { return model }
        let model = weightModelFactory(schema)
        let url = store.modelURL(owner, modelKey: modelKey)
        if FileManager.default.fileExists(atPath: url.path) {
            do {
                _ = try model.load(from: url)
            } catch {
                log?.log(.warning, "weight model for \(owner) did not load (\(error)); starting untrained")
                model.reset()
            }
        }
        models[owner] = model
        modelOrder.append(owner)
        if modelOrder.count > 32 {
            let evicted = modelOrder.removeFirst()
            models[evicted] = nil
        }
        return model
    }

    func contextProjection(inputDimension: Int) -> CountSketchProjection {
        if let projection, projection.inputDimension == inputDimension { return projection }
        let made = CountSketchProjection(inputDimension: inputDimension, outputDimension: configuration.contextDim)
        projection = made
        return made
    }

    // MARK: Status and housekeeping

    public func summary(owner: OwnerID) -> OwnerSummary {
        // The ledger is swept to the band on every turn; report it as swept.
        let ledger = store.ledger(owner)
        let measured = ledger.turns.compactMap(\.grounding).filter { $0.contentTokens > 0 }
        func mean(_ values: [Float]) -> Float? { values.isEmpty ? nil : values.reduce(0, +) / Float(values.count) }
        return OwnerSummary(
            owner: owner.rawValue,
            observations: ledger.turns.count,
            measuredTurns: ledger.measuredTurns,
            labelledEvents: ledger.labelledEvents,
            meanGrounding: mean(measured.map(\.grounding)),
            meanDrift: mean(measured.map(\.drift)),
            meanHallucinationRisk: mean(measured.map(\.hallucinationRisk)),
            trainedAt: ledger.training.lastTrainedAt,
            trainingCycles: ledger.training.cycles,
            reliability: ledger.training.reliability,
            holdoutMAE: ledger.training.holdoutMAE,
            baselineMAE: ledger.training.baselineMAE,
            periods: ledger.periods,
            lastBiasMagnitude: ledger.lastBiasMagnitude,
            lastTraceId: ledger.lastTraceId,
            store: store.ownerDirectory(owner).path)
    }

    public func trace(owner: OwnerID, turnId: UUID) -> InjectionTrace? {
        store.loadTrace(owner: owner, turnId: turnId)
    }

    public func latestTrace(owner: OwnerID) -> InjectionTrace? {
        guard let id = store.ledger(owner).lastTraceId else { return nil }
        return store.loadTrace(owner: owner, turnId: id)
    }

    public func groundingReport(owner: OwnerID) -> GroundingReport {
        GroundingAnalysis.report(owner: owner, ledger: store.ledger(owner))
    }

    public func forget(owner: OwnerID) throws {
        models[owner] = nil
        modelOrder.removeAll { $0 == owner }
        try store.forget(owner)
    }

    /// Write the owner's ledger if it changed. Cheap when clean.
    public func persist(owner: OwnerID) {
        guard store.isDirty(owner) else { return }
        do {
            try store.save(owner)
        } catch {
            log?.log(.error, "saving ledger for \(owner) failed: \(error)")
        }
    }

    public func flush() {
        do {
            try store.saveDirty()
        } catch {
            log?.log(.error, "flushing ledgers failed: \(error)")
        }
    }

    public func knownOwners() -> [OwnerID] { store.knownOwners() }

    /// Read-only view for tools (`inspect`).
    public func ledgerJSON(owner: OwnerID) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try encoder.encode(store.ledger(owner))
    }
}
