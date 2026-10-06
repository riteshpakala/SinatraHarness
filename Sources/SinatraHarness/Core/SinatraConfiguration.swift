//
//  SinatraConfiguration.swift
//  SinatraHarness
//
//  WHAT: Every tunable in one place. Defaults are the v2 (grounding) spec.
//

import Foundation

public struct SinatraConfiguration: Sendable {

    // MARK: Relevancy bands

    /// Events, turns and series points older than this are dropped. Training sees only the band.
    public var retentionDays: Double = 30
    /// Documents younger than this (by creation, modification, index or first-seen time) are "fresh".
    public var freshBandDays: Double = 7
    /// Injection multiplier per band: context older than the retention window is damped, never inverted.
    public var bandWeights = BandWeights(fresh: 1.0, mid: 0.7, stale: 0.4)
    /// Sample-weight decay by event age inside the band.
    public var midBandSampleDecay: Float = 0.7

    // MARK: Caps

    public var maxEventsPerOwner = 4000
    public var maxTurnsPerOwner = 1000
    public var maxSeriesPoints = 200
    public var maxPartitionsPerTurn = 16
    /// Tokens per partition the encoder and the injection read (head and tail kept).
    public var maxTokensPerPartition = 256
    /// Tokens per partition kept for spotting verbatim copies in the output.
    public var maxPartitionSequence = 2048
    public var maxFirstSeenEntries = 20_000

    // MARK: Grounding measurement

    /// Which turns are re-scored with and without their context after decoding.
    public var groundingMeasurement: GroundingPolicy = .always
    /// Wall-clock cap on one measurement. It runs under the harness gate, after the reply.
    public var groundingBudget: TimeInterval = 6
    /// |ι| at or above this (nats) makes a content token grounded (ι > 0) or contradicted (ι < 0).
    public var groundingThreshold: Float = 0.5
    /// A content token whose drift exceeds this (nats) counts as drifting.
    public var driftThreshold: Float = 1.0
    /// The "tune" (the context's hardest push) is searched over the union of both sides' top-k.
    public var contextTopK = 64
    /// A verbatim run of at least this many tokens from one partition counts as parroting.
    public var parrotNGram = 6
    /// Positions scored per forward pass while measuring.
    public var groundingChunk = 64
    /// `.full` measurements keep this many of the context's strongest pushes per step.
    public var groundingAlternatives = 8

    // MARK: Steer targets

    /// κ: how far a drifting turn moves a partition's steer toward what the output missed.
    public var steerGain: Float = 1.0
    /// λ: how far verbatim copying from a partition pushes its steer down.
    public var parrotPenalty: Float = 1.0

    // MARK: Turn bookkeeping

    /// A turn whose generation never finished is dropped after this long.
    public var abandonedAfter: TimeInterval = 3600
    /// Saturation of the gap-since-the-last-turn feature.
    public var queryGapHorizon: TimeInterval = 30 * 60

    // MARK: Weight model and training

    public var minimumLabelledEvents = 20
    public var trainEveryLabelledEvents = 4
    public var trainingBudget: TimeInterval = 0.4
    public var maxTrainingSteps = 150
    public var learningRate: Float = 2e-3
    public var weightDecay: Float = 1e-2
    public var holdoutFraction: Double = 0.2
    /// Weight of the uptake head's loss beside the steer head's.
    public var uptakeLossWeight: Float = 0.25
    /// Reliability ramps from 0 at `lower` labelled events to 1 at `upper`.
    public var reliabilityRamp: ClosedRange<Double> = 20...100
    /// Holdout skill over the mean predictor that earns full reliability.
    public var reliabilitySkillForFull: Float = 0.3
    /// Indicators stay neutral until the grounding series has this many points.
    public var minimumSeriesForIndicators = 20

    // MARK: Injection

    public var biasMode: BiasMode = .lexical
    public var alpha: Float = 1.0
    /// Hard cap on |bias| in nats.
    public var cap: Float = 2.0
    public var alphaDense: Float = 0.5
    public var denseTopK = 4096
    /// |w_p| below this contributes nothing.
    public var minimumWeight: Float = 1e-3
    /// Use ledger priors while the weight model is unreliable (cold start).
    public var coldStartPrior = true
    /// Multiply each partition's injection term by its band weight (applied once, in the builder).
    public var bandPrior = true
    /// Width of the count-sketch projection of pooled context embeddings.
    public var contextDim = 64
    /// Treat `Partition.score` as a distance (lower = better).
    public var scoreIsDistance = true

    // MARK: Vocabulary

    public var hotTokenCapacity = 512
    /// A token seen in at least this share of the owner's partitions is too common to bias.
    public var hotTokenExclusionRatio: Double = 0.5
    public var hotTokenMinimumPartitions = 20
    public var useStopwords = true

    // MARK: Trace

    public var traceLevel: TraceLevel = .automatic
    public var traceTopK = 8
    /// Full traces kept per owner.
    public var traceHistory = 20

    // MARK: Time

    public var timeZone: TimeZone = .current

    public init() {}

    public struct BandWeights: Sendable, Codable, Equatable {
        public var fresh: Float
        public var mid: Float
        public var stale: Float
        public init(fresh: Float, mid: Float, stale: Float) {
            self.fresh = fresh
            self.mid = mid
            self.stale = stale
        }
    }
}

/// Which turns get a grounding measurement.
public enum GroundingPolicy: String, Sendable, Codable, CaseIterable {
    /// Every recorded turn with retrieved context, and every traced one.
    case always
    /// Only turns whose decode is traced.
    case traced
    case off

    func measures(record: Bool, traced: Bool) -> Bool {
        switch self {
        case .always: return record || traced
        case .traced: return traced
        case .off: return false
        }
    }
}
