//
//  OwnerLedger.swift
//  SinatraHarness
//
//  WHAT: Everything the side model remembers about one owner: turns and their grounding,
//        the partitions each turn retrieved (events) and what the measurement taught each,
//        the grounding series the indicators run over, first-seen times (the recency
//        fallback), learned stopwords, IMBHS state and training state.
//  PIN:  Bounded by the 30-day relevancy band plus hard caps. Schema 2 replaced the
//        reply-labelled ledger of schema 1; a schema-1 file is discarded on load.
//

import Foundation

struct OwnerLedger: Codable {
    static let schema = 2

    var schema: Int = OwnerLedger.schema
    var owner: String
    var createdAt: Date
    var updatedAt: Date
    var turns: [TurnRecord] = []
    var events: [RetrievalEvent] = []
    var series: [SeriesPoint] = []
    var firstSeenPartitions: [String: Date] = [:]
    var firstSeenDocuments: [String: Date] = [:]
    var hotTokens: HotTokens
    var periods: IndicatorPeriods = .default
    var harmony: HarmonyMemory
    var training = TrainingState()
    /// When the owner's previous turn happened (the query-gap feature).
    var lastTurnAt: Date?
    var lastBiasMagnitude: Float?
    var lastTraceId: UUID?

    init(owner: OwnerID, now: Date, configuration: SinatraConfiguration) {
        self.owner = owner.rawValue
        self.createdAt = now
        self.updatedAt = now
        self.hotTokens = HotTokens(capacity: configuration.hotTokenCapacity)
        self.harmony = HarmonyMemory()
    }

    // MARK: Queries

    var measuredTurns: Int { turns.lazy.filter { $0.grounding != nil }.count }
    var labelledEvents: Int { events.lazy.filter { $0.label != nil }.count }

    func turnIndex(_ id: UUID) -> Int? { turns.firstIndex { $0.id == id } }

    func eventIndices(turn id: UUID) -> [Int] {
        events.indices.filter { events[$0].turnId == id }
    }

    /// The most recent measured turn.
    var lastMeasured: TurnRecord? {
        turns.last(where: { $0.grounding != nil })
    }
}

struct TurnRecord: Codable {
    var id: UUID
    /// When the turn was prepared (≈ the user's message arrived).
    var at: Date
    var conversationId: String?
    var modelKey: String
    var mode: BiasMode
    var partitionIds: [String]
    var assistantStartedAt: Date?
    var assistantFinishedAt: Date?
    var assistantWords: Int?
    var assistantChars: Int?
    /// Mean |w_p| applied this turn: how hard it was steered.
    var meanSteer: Float = 0
    var measuredAt: Date?
    var grounding: GroundingSummary?
    /// Why the turn went unmeasured, when it did.
    var unmeasuredReason: String?
    var injection: InjectionAggregates?
    var trace: TraceAggregates?

    var isComplete: Bool { assistantFinishedAt != nil }
    var isMeasured: Bool { grounding != nil }
}

struct InjectionAggregates: Codable, Equatable {
    var biasMaxAbs: Float
    var biasNonZero: Int
    var biasL1: Float
    var gate: Float
    var weightedPartitions: Int
}

struct RetrievalEvent: Codable {
    var turnId: UUID
    /// As-of time for features: the turn's prepare time.
    var at: Date
    var partitionId: String
    var documentId: String
    var rank: Int
    var staticFeatures: [Float]
    /// Count-sketch of the pooled context embedding.
    var context: [Float]?
    var contextModelKey: String?
    /// rel_p normalised within the turn, and over the turn's largest rel.
    var relevancy: Float
    var relevancyWeight: Float
    var priorWeight: Float
    var netWeight: Float?
    var appliedWeight: Float
    var bandWeight: Float
    var label: EventLabel?
}

struct TrainingState: Codable, Equatable {
    var cycles = 0
    var lastTrainedAt: Date?
    var labelsSinceTrain = 0
    var reliability: Float = 0
    var holdoutMAE: Float?
    var baselineMAE: Float?
    var lastReport: TrainingReport?
}
