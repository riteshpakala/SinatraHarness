//
//  GroundingTrace.swift
//  SinatraHarness
//
//  WHAT: The record of what the retrieved context did to the output, the second layer of
//        a turn's trace. The injection layer (InjectionTrace) says what Sinatra did to the
//        logits during decoding. This one is measured after decoding, by scoring the
//        output twice with the model itself: once under the prompt with its context and
//        once under the same prompt without it.
//
//          ι_t   = log p_ctx(y_t) − log p_bare(y_t)   nats the context added to what was said
//          KL_t  = KL(p_ctx ‖ p_bare) = E_{v∼p_ctx}[log p_ctx(v) − log p_bare(v)]
//                                                      the lift the context gave this step, on average
//          drift = max(0, KL_t − ι_t)                  how much less the output took than that
//          tune  = argmax_v p_ctx(v) · (log p_ctx(v) − log p_bare(v))
//                                                      the token the context moved the model toward
//
//        Aggregated per turn into grounding, drift, hallucination risk and per-partition
//        attribution: the citation value of each retrieved partition in the output.
//  PIN:  Plain Codable values; floats are sanitised before they get here. Grounded and
//        unsupported are thresholds on the context's own push, not truth checks.
//

import Foundation

/// How one output token relates to the retrieved context.
public enum GroundingClass: String, Codable, Sendable, CaseIterable {
    /// ι ≥ threshold: the context pushed the model toward this token.
    case grounded
    /// ι ≤ −threshold: the context pushed against it; the model said it anyway.
    case contradicted
    /// A content token the context neither pushed for nor against: the model's own.
    case unsupported
    /// Not a content token (punctuation, stopwords, control tokens): measured, not counted.
    case function
}

public struct GroundingStep: Codable, Sendable, Equatable {
    public var index: Int
    public var token: Int
    public var text: String?
    /// log p(y_t) under the prompt with its context, and under the bare prompt.
    public var logpCtx: Float
    public var logpBare: Float
    /// ι_t = logpCtx − logpBare.
    public var influence: Float
    public var entropyCtx: Float
    public var entropyBare: Float
    /// KL(p_ctx ‖ p_bare), nats.
    public var contextKL: Float
    /// The token the context moved the model toward at this step (largest share of the KL,
    /// searched over the union of both distributions' top-k), and its log-ratio in nats.
    public var tune: Int
    public var tuneText: String?
    public var tuneNats: Float
    /// max(0, contextKL − influence): how much less lift the output took than the context
    /// gave this step on average.
    public var drift: Float
    public var kind: GroundingClass
    /// An unsupported token the model itself was unsure of: 1 − p_ctx(y_t); else 0.
    public var risk: Float
    /// `.full` only: the sampled token's rank without the context.
    public var rankBare: Int?
    /// `.full` only: the tokens the context moved the model toward most at this step, by
    /// share of the KL (logprob = their log-ratio).
    public var pushes: [TokenLogprob]?
}

/// One retrieved partition's part in the output.
public struct PartitionAttribution: Codable, Sendable, Equatable {
    public var partitionId: String
    public var documentId: String
    /// A_p = Σ_t max(0, ι_t) · share_p(y_t) over content tokens: its citation value, in nats.
    public var nats: Float
    /// A_p over everything the context contributed: its share of the context's influence.
    public var uptake: Float
    /// Drift at steps where the context moved the model toward this partition's tokens:
    /// the push the output did not take, in nats.
    public var missed: Float
    /// `missed` as a share of the turn's attributable missed push.
    public var intent: Float
    /// Share of the partition's content tokens that appear in the output.
    public var coverage: Float
    /// Share of the output copied verbatim (runs of `parrotNGram` tokens) from it.
    public var parrot: Float
    /// rel_p normalised within the turn.
    public var relevancy: Float
}

public struct GroundingSummary: Codable, Sendable, Equatable {
    public var steps: Int
    public var contentTokens: Int
    /// Share of content tokens the context pushed for.
    public var grounding: Float
    public var unsupportedShare: Float
    public var contradictedShare: Float
    /// Mean drift over content tokens, nats.
    public var drift: Float
    /// Share of content tokens drifting more than the threshold.
    public var driftShare: Float
    public var firstDriftStep: Int?
    /// Σ ι_t over every step: log p_ctx(y) − log p_bare(y), nats.
    public var contextDependence: Float
    public var meanContextKL: Float
    public var meanEntropyCtx: Float
    public var meanEntropyBare: Float
    /// Mean risk over content tokens.
    public var hallucinationRisk: Float
    /// Share of the output copied verbatim from any partition.
    public var parrotShare: Float
    /// Positive influence on tokens found in no partition: the context shaped them indirectly.
    public var unattributed: Float

    public static let empty = GroundingSummary(
        steps: 0, contentTokens: 0, grounding: 0, unsupportedShare: 0, contradictedShare: 0,
        drift: 0, driftShare: 0, firstDriftStep: nil, contextDependence: 0, meanContextKL: 0,
        meanEntropyCtx: 0, meanEntropyBare: 0, hallucinationRisk: 0, parrotShare: 0, unattributed: 0)
}

public struct GroundingMeasurement: Codable, Sendable {
    /// false: the turn could not be measured (`skippedReason` says why).
    public var measured: Bool
    public var skippedReason: String?
    /// true: the context side reused the generation's own KV cache (exact, no prefill).
    public var cacheReused: Bool
    public var promptTokens: Int
    public var bareTokens: Int
    /// Leading tokens both prompts share, whose KV cache the bare side reused.
    public var sharedPrefixTokens: Int
    /// Prefilling what the bare prompt does not share, then scoring both sides.
    public var prefillMillis: Double
    public var scoreMillis: Double
    public var detail: TraceLevel
    public var summary: GroundingSummary
    public var attribution: [PartitionAttribution]
    public var steps: [GroundingStep]

    public static func skipped(_ reason: String) -> GroundingMeasurement {
        GroundingMeasurement(
            measured: false, skippedReason: reason, cacheReused: false, promptTokens: 0, bareTokens: 0,
            sharedPrefixTokens: 0, prefillMillis: 0, scoreMillis: 0, detail: .off, summary: .empty,
            attribution: [], steps: [])
    }
}

/// The scorer's per-step output, evaluated and read back: what `Grounding.measurement`
/// turns into a `GroundingMeasurement`.
public struct GroundingRaw: Sendable, Equatable {
    public var tokens: [Int]
    public var logpCtx: [Float]
    public var logpBare: [Float]
    public var entropyCtx: [Float]
    public var entropyBare: [Float]
    public var contextKL: [Float]
    public var tune: [Int]
    public var tuneNats: [Float]
    /// `.full` only.
    public var rankBare: [Int]?
    public var pushes: [[(id: Int, nats: Float)]]?
    public var cacheReused: Bool
    public var promptTokens: Int
    public var bareTokens: Int
    public var sharedPrefixTokens: Int
    public var prefillMillis: Double
    public var scoreMillis: Double

    public init(
        tokens: [Int], logpCtx: [Float], logpBare: [Float], entropyCtx: [Float], entropyBare: [Float],
        contextKL: [Float], tune: [Int], tuneNats: [Float], rankBare: [Int]? = nil,
        pushes: [[(id: Int, nats: Float)]]? = nil, cacheReused: Bool = false, promptTokens: Int = 0,
        bareTokens: Int = 0, sharedPrefixTokens: Int = 0, prefillMillis: Double = 0, scoreMillis: Double = 0
    ) {
        self.tokens = tokens
        self.logpCtx = logpCtx
        self.logpBare = logpBare
        self.entropyCtx = entropyCtx
        self.entropyBare = entropyBare
        self.contextKL = contextKL
        self.tune = tune
        self.tuneNats = tuneNats
        self.rankBare = rankBare
        self.pushes = pushes
        self.cacheReused = cacheReused
        self.promptTokens = promptTokens
        self.bareTokens = bareTokens
        self.sharedPrefixTokens = sharedPrefixTokens
        self.prefillMillis = prefillMillis
        self.scoreMillis = scoreMillis
    }

    public static func == (a: GroundingRaw, b: GroundingRaw) -> Bool {
        a.tokens == b.tokens && a.logpCtx == b.logpCtx && a.logpBare == b.logpBare
            && a.entropyCtx == b.entropyCtx && a.entropyBare == b.entropyBare && a.contextKL == b.contextKL
            && a.tune == b.tune && a.tuneNats == b.tuneNats && a.rankBare == b.rankBare
    }
}
