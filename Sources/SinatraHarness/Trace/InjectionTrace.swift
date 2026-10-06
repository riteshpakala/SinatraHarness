//
//  InjectionTrace.swift
//  SinatraHarness
//
//  WHAT: A turn's trace. Its first layer records what the injection did to the logits,
//        step by step; its second (`grounding`, Trace/GroundingTrace.swift) what the
//        retrieved context did to the output, measured after decoding.
//
//        Because the injection is additive and constant within a turn (z' = z + b before
//        the penalty processor), its effect on every step is fully attributable. Each step
//        keeps both distributions' entropy, their divergence, the sampled token's log-prob
//        and rank before and after, how much probability mass moved into the impact mask,
//        and the COUNTERFACTUAL token: what an identically seeded sampler would have drawn
//        from the un-injected logits. Decoding with and without Sinatra can therefore be
//        compared inside a single run.
//
//  PIN:  Plain Codable values. Non-finite floats are sanitised before they get here, and
//        the store's JSON coders map any that slip through to strings.
//

import Foundation

public struct TokenLogprob: Codable, Sendable, Equatable {
    public var id: Int
    public var text: String?
    public var logprob: Float
}

public struct StepTrace: Codable, Sendable, Equatable {
    public var index: Int
    /// The token actually sampled (from the injected distribution).
    public var sampled: Int
    /// What the same sampler, same seed, would have drawn without the injection.
    public var counterfactual: Int
    public var sampledText: String?
    public var counterfactualText: String?
    /// Entropy (nats) of the un-injected and injected distributions.
    public var entropyPre: Float
    public var entropyPost: Float
    /// KL(injected ‖ un-injected), nats.
    public var kl: Float
    /// Jensen–Shannon divergence, nats.
    public var js: Float
    /// Log-prob of the sampled token without / with the injection.
    public var logprobPre: Float
    public var logprobPost: Float
    /// Rank of the sampled token (0 = most likely) without / with the injection.
    public var rankPre: Int
    public var rankPost: Int
    public var argmaxPre: Int
    public var argmaxPost: Int
    /// Probability mass the injection moved INTO the impact mask at this step.
    public var massIntoMask: Float
    /// The sampled token is one Sinatra biased.
    public var inMask: Bool
    /// `.full` only: top-k before and after.
    public var topPre: [TokenLogprob]?
    public var topPost: [TokenLogprob]?
    /// `.full` only: Δ probability (p' − p) for every mask token, aligned with
    /// `ImpactMask.tokenIds`.
    public var movement: [Float]?

    /// Nats of personalization in the sampled token.
    public var gain: Float { logprobPost - logprobPre }
    public var entropyShift: Float { entropyPost - entropyPre }
    public var diverged: Bool { sampled != counterfactual }
    public var flippedArgmax: Bool { argmaxPre != argmaxPost }
}

public struct TraceSummary: Codable, Sendable, Equatable {
    public var steps: Int
    public var meanEntropyPre: Float
    public var meanEntropyPost: Float
    public var meanEntropyShift: Float
    public var totalKL: Float
    public var meanKL: Float
    public var meanJS: Float
    /// Σ gain over sampled tokens: the personalization in the output, in nats.
    public var totalGain: Float
    public var meanMassIntoMask: Float
    /// Share of steps whose counterfactual token differs from the sampled one.
    public var divergenceRate: Float
    public var firstDivergenceStep: Int?
    public var flippedArgmaxSteps: Int
    public var sampledInMaskShare: Float
    /// partition id → Σ over steps of its bias on the sampled token.
    public var partitionAttribution: [String: Float]

    public static let empty = TraceSummary(
        steps: 0, meanEntropyPre: 0, meanEntropyPost: 0, meanEntropyShift: 0, totalKL: 0,
        meanKL: 0, meanJS: 0, totalGain: 0, meanMassIntoMask: 0, divergenceRate: 0,
        firstDivergenceStep: nil, flippedArgmaxSteps: 0, sampledInMaskShare: 0,
        partitionAttribution: [:])

    public static func compute(steps: [StepTrace], mask: ImpactMask?) -> TraceSummary {
        guard !steps.isEmpty else { return .empty }
        let n = Float(steps.count)
        var attribution: [String: Float] = [:]
        if let mask {
            for step in steps where step.inMask {
                for (id, value) in mask.attribution(of: step.sampled) {
                    attribution[id, default: 0] += value
                }
            }
        }
        let totalKL = steps.reduce(0) { $0 + $1.kl }
        return TraceSummary(
            steps: steps.count,
            meanEntropyPre: steps.reduce(0) { $0 + $1.entropyPre } / n,
            meanEntropyPost: steps.reduce(0) { $0 + $1.entropyPost } / n,
            meanEntropyShift: steps.reduce(0) { $0 + $1.entropyShift } / n,
            totalKL: totalKL,
            meanKL: totalKL / n,
            meanJS: steps.reduce(0) { $0 + $1.js } / n,
            totalGain: steps.reduce(0) { $0 + $1.gain },
            meanMassIntoMask: steps.reduce(0) { $0 + $1.massIntoMask } / n,
            divergenceRate: Float(steps.filter(\.diverged).count) / n,
            firstDivergenceStep: steps.first(where: \.diverged)?.index,
            flippedArgmaxSteps: steps.filter(\.flippedArgmax).count,
            sampledInMaskShare: Float(steps.filter(\.inMask).count) / n,
            partitionAttribution: attribution)
    }
}

public struct InjectionTrace: Codable, Sendable {
    public var traceId: UUID
    public var owner: String?
    public var level: TraceLevel
    public var mode: BiasMode
    public var seed: UInt64?
    public var temperature: Float
    public var topP: Float
    public var createdAt: Date
    public var promptTokens: Int
    public var generatedTokens: Int
    public var stopReason: String?
    public var mask: ImpactMask?
    /// The injection layer, step by step (empty when only grounding was measured).
    public var steps: [StepTrace]
    public var summary: TraceSummary
    /// The context layer: the output re-scored with and without its retrieved context.
    public var grounding: GroundingMeasurement?
}

/// What a ledger turn keeps of its trace, so analysis runs without the full files.
struct TraceAggregates: Codable, Equatable {
    var traceId: UUID
    var level: TraceLevel
    var steps: Int
    var meanEntropyPre: Float
    var meanEntropyPost: Float
    var entropyShift: Float
    var kl: Float
    var gain: Float
    var divergenceRate: Float
    var massIntoMask: Float

    init(_ trace: InjectionTrace) {
        traceId = trace.traceId
        level = trace.level
        steps = trace.summary.steps
        meanEntropyPre = trace.summary.meanEntropyPre
        meanEntropyPost = trace.summary.meanEntropyPost
        entropyShift = trace.summary.meanEntropyShift
        kl = trace.summary.totalKL
        gain = trace.summary.totalGain
        divergenceRate = trace.summary.divergenceRate
        massIntoMask = trace.summary.meanMassIntoMask
    }
}

extension Float {
    /// Finite or the fallback: traces must round-trip through JSON.
    func finite(_ fallback: Float = 0) -> Float { isFinite ? self : fallback }
}
