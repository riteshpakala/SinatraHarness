//
//  SinatraTypes.swift
//  SinatraHarness
//
//  WHAT: The value types that cross the harness boundary: who a turn belongs to,
//        the retrieved context, and what came back.
//  PIN:  Only `Partition.text` / `tokenIds` ever reach the side model's encoder, and the
//        user's message never enters SinatraHarness at all. A turn is judged by what its
//        retrieved context did to the model's own distributions (Signals/Grounding.swift),
//        never by how the user reacted to the answer.
//

import Foundation

/// Who the personalization belongs to. Sewn passes its lowercased auth user id.
public struct OwnerID: Hashable, Sendable, Codable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public var description: String { rawValue }

    public init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// One retrieved chunk of context.
public struct Partition: Sendable, Codable, Equatable, Identifiable {
    public var id: String
    public var documentId: String
    public var text: String
    /// Pre-tokenised content without special tokens. The session tokenises `text` when nil.
    public var tokenIds: [Int]?
    /// Retrieval score. A distance by default (lower = closer), as Thread's PQ search
    /// returns it; see `SinatraConfiguration.scoreIsDistance`.
    public var score: Float
    /// Document creation time, when the caller knows it.
    public var createdAt: Date?
    /// Last modification time, when the caller knows it. The newest of creation,
    /// modification and index time places the partition in its relevancy band.
    public var modifiedAt: Date?
    /// Index time, when the caller knows it. With none of the three known, the first time
    /// this owner retrieved the document stands in.
    public var indexedAt: Date?

    public init(
        id: String, documentId: String, text: String, tokenIds: [Int]? = nil,
        score: Float = 0, createdAt: Date? = nil, modifiedAt: Date? = nil, indexedAt: Date? = nil
    ) {
        self.id = id
        self.documentId = documentId
        self.text = text
        self.tokenIds = tokenIds
        self.score = score
        self.createdAt = createdAt
        self.modifiedAt = modifiedAt
        self.indexedAt = indexedAt
    }
}

/// Everything the side model needs to steer one generation.
public struct TurnInput: Sendable {
    public var owner: OwnerID
    public var retrieved: [Partition]
    public var conversationId: String?
    /// When the turn happens: the as-of time for every feature.
    public var now: Date
    /// Debug override of the per-partition weights (CLI `--force-weights`). Aligned with
    /// `retrieved` after de-duplication; missing entries keep the computed weight.
    public var weightOverride: [Float]?

    public init(
        owner: OwnerID, retrieved: [Partition], conversationId: String? = nil,
        now: Date = Date(), weightOverride: [Float]? = nil
    ) {
        self.owner = owner
        self.retrieved = retrieved
        self.conversationId = conversationId
        self.now = now
        self.weightOverride = weightOverride
    }
}

/// How the injection is built.
public enum BiasMode: String, Sendable, Codable, CaseIterable {
    /// No injection. Turns are still recorded and labelled, so learning continues.
    case off
    /// Sparse token bias over the retrieved context's content tokens (v1 default).
    case lexical
    /// Steering vector through the model's own output head, sparsified (experimental).
    case dense
}

/// How much of the injection's effect on the logits is recorded.
public enum TraceLevel: String, Sendable, Codable, CaseIterable {
    /// `.summary` when an injection is active, otherwise `.off`.
    case automatic
    case off
    /// Per-step entropy, divergence, gain, ranks and the counterfactual token.
    case summary
    /// `.summary` plus top-k before/after and the per-step movement over the impact mask.
    case full

    func resolved(injecting: Bool) -> TraceLevel {
        switch self {
        case .automatic: return injecting ? .summary : .off
        default: return self
        }
    }
}

/// What a measured turn taught one retrieved partition.
public struct EventLabel: Sendable, Codable, Equatable {
    /// y_p ∈ [−1, 1]: what the steer on this partition should have been
    /// (`Grounding.steerTarget`). The weight model's training target.
    public var target: Float
    /// uptake_p ∈ [0, 1]: this partition's share of the context's influence on the output.
    public var uptake: Float
    /// intent_p ∈ [0, 1]: this partition's share of the push the output did not take.
    public var intent: Float
    /// A_p: nats of the output's context influence attributed to this partition.
    public var attributionNats: Float
    /// Share of the partition's content tokens that appear in the output.
    public var coverage: Float
    /// Share of the output copied verbatim from this partition.
    public var parrot: Float
    /// rel_p normalised within the turn.
    public var relevancy: Float
    public var turnGrounding: Float
    public var turnDrift: Float
    public var turnDriftShare: Float
    public var labelledAt: Date
}

/// A turn measured by the caller (replay, tests): its grounding labels the turn's events.
public struct TurnObservation: Sendable {
    public var turnId: UUID
    public var assistantText: String?
    public var assistantStartedAt: Date?
    public var assistantFinishedAt: Date?
    public var measurement: GroundingMeasurement

    public init(
        turnId: UUID, assistantText: String? = nil, assistantStartedAt: Date? = nil,
        assistantFinishedAt: Date? = nil, measurement: GroundingMeasurement
    ) {
        self.turnId = turnId
        self.assistantText = assistantText
        self.assistantStartedAt = assistantStartedAt
        self.assistantFinishedAt = assistantFinishedAt
        self.measurement = measurement
    }
}

/// What labelling one turn produced.
public struct ObservationReceipt: Sendable, Codable, Equatable {
    public var turnId: UUID
    public var summary: GroundingSummary
    /// partition id → y_p
    public var targets: [String: Float]
}

/// A token and its bias, for diagnostics.
public struct BiasedToken: Sendable, Codable, Equatable {
    public var id: Int
    public var text: String?
    public var bias: Float
}

/// Why one partition got the weight it did.
public struct PartitionDiagnostic: Sendable, Codable, Equatable {
    public var id: String
    public var documentId: String
    public var rank: Int
    public var docAgeDays: Double
    public var band: String
    public var bandWeight: Float
    /// rel_p = band weight × (0.5 + 0.5 × score01): how relevant recency and retrieval
    /// rank make this partition, before normalising within the turn.
    public var relevancy: Float
    public var priorWeight: Float
    /// Mean uptake of this partition over its measured turns in the band (0 if none).
    public var priorUptake: Float
    public var netWeight: Float?
    public var appliedWeight: Float
    public var contentTokens: Int
    public var labelledBefore: Int
}

/// Everything `prepareTurn` decided, for the caller and for traces.
public struct TurnDiagnostics: Sendable, Codable, Equatable {
    public var turnId: UUID
    public var mode: BiasMode
    /// Fewer labelled events than the training minimum: weights come from priors only.
    public var coldStart: Bool
    public var labelledEvents: Int
    public var measuredTurns: Int
    public var observedTurns: Int
    public var partitions: [PartitionDiagnostic]
    public var weightedPartitions: Int
    public var biasNonZero: Int
    public var biasMaxAbs: Float
    public var biasL1: Float
    /// g: reliability of the learned weights, blended against the priors.
    public var gate: Float
    /// The weight model's uptake forecast, averaged over the partitions, when it ran.
    public var forecastUptake: Float?
    public var periods: IndicatorPeriods
    public var topBiased: [BiasedToken]
    public var encodeMillis: Double
    public var buildMillis: Double
    public var dryRun: Bool
    /// The weight model's input per partition, in `FeatureVector.names` order.
    public var features: [[Float]]
}

/// The side model's decision for one generation. Plain values: safe to hand across actors.
public struct InjectionPlan: Sendable, Codable {
    public let turnId: UUID
    public let owner: OwnerID
    public let mode: BiasMode
    /// nil ⇒ identity (no injection this turn).
    public let bias: SparseBias?
    public let mask: ImpactMask?
    /// partition id → applied weight w_p
    public let perPartitionWeights: [String: Float]
    public let gate: Float
    public let diagnostics: TurnDiagnostics
    public let dryRun: Bool
    /// The partitions as the grounding measurement reads them.
    let terms: [PartitionTerms]

    public var injects: Bool { bias != nil && mode != .off }
    /// Retrieved context there is to measure the output against.
    public var hasContext: Bool { !terms.isEmpty }
}

/// One retrieved partition as the grounding measurement reads it.
struct PartitionTerms: Sendable, Codable, Equatable {
    var id: String
    var documentId: String
    /// Content-token counts, the owner's hot tokens removed: what attribution is made of.
    var counts: [Int: Int]
    /// The partition's token ids in order, for spotting verbatim copies.
    var sequence: [Int]
    /// rel_p normalised within the turn.
    var relevancyShare: Float
    /// rel_p over the turn's largest rel: how much a correction on this partition counts.
    var relevancyWeight: Float
    /// w_p, the steer applied this turn.
    var appliedWeight: Float
}

/// Where an owner's personalization stands.
public struct OwnerSummary: Sendable, Codable, Equatable {
    public var owner: String
    /// Turns observed in the relevancy band.
    public var observations: Int
    /// Of those, the turns whose grounding was measured (and whose partitions are labelled).
    public var measuredTurns: Int
    public var labelledEvents: Int
    /// Means over the measured turns in the band; nil before the first.
    public var meanGrounding: Float?
    public var meanDrift: Float?
    public var meanHallucinationRisk: Float?
    public var trainedAt: Date?
    public var trainingCycles: Int
    public var reliability: Float
    public var holdoutMAE: Float?
    public var baselineMAE: Float?
    public var periods: IndicatorPeriods
    public var lastBiasMagnitude: Float?
    public var lastTraceId: UUID?
    public var store: String
}

/// What one training cycle did.
public struct TrainingReport: Sendable, Codable, Equatable {
    public var owner: String
    public var skipped: String?
    public var rows: Int
    public var trainRows: Int
    public var holdoutRows: Int
    public var steps: Int
    public var initialLoss: Float?
    public var finalLoss: Float?
    public var holdoutMAE: Float?
    public var baselineMAE: Float?
    public var reliability: Float
    public var elapsed: TimeInterval
    public var stoppedBy: String?
    public var cycle: Int
    public var harmonyRan: Bool
    public var periodsChanged: Bool
    public var periods: IndicatorPeriods

    static func skipped(_ reason: String, owner: OwnerID, rows: Int, reliability: Float, cycle: Int, periods: IndicatorPeriods) -> TrainingReport {
        TrainingReport(
            owner: owner.rawValue, skipped: reason, rows: rows, trainRows: 0, holdoutRows: 0,
            steps: 0, initialLoss: nil, finalLoss: nil, holdoutMAE: nil, baselineMAE: nil,
            reliability: reliability, elapsed: 0, stoppedBy: nil, cycle: cycle,
            harmonyRan: false, periodsChanged: false, periods: periods)
    }
}

public enum SinatraError: Error, CustomStringConvertible, Sendable {
    case notLoaded
    case embeddingNotFound(String)
    case schemaMismatch(String)
    case unknownTurn(UUID)
    case invalidArgument(String)
    case measurement(String)

    public var description: String {
        switch self {
        case .notLoaded: return "No model is loaded."
        case .embeddingNotFound(let detail): return "No input embedding table found: \(detail)"
        case .schemaMismatch(let detail): return "Saved weight model does not match this schema: \(detail)"
        case .unknownTurn(let id): return "Unknown turn \(id)"
        case .invalidArgument(let detail): return detail
        case .measurement(let detail): return "Grounding measurement: \(detail)"
        }
    }
}
