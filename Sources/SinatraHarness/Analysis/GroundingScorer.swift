//
//  GroundingScorer.swift
//  SinatraHarness
//
//  WHAT: The grounding measurement's MLX half. After a turn is decoded, its output y is
//        scored twice by the same model, teacher-forced, one chunk of positions at a time:
//
//          context side  the generation's own KV cache, trimmed back to all but the last
//                        prompt token, then fed [last prompt token] + y (minus its last
//                        token). Exact: the same distributions the decode saw, before any
//                        injection or penalty, with no prefill.
//          bare side     a copy of that cache trimmed to the prefix both prompts share,
//                        then prefilled with the rest of the bare prompt (the same chat
//                        without its retrieved context), then fed the same y.
//
//        Per step: log p(y_t) on both sides, both entropies, KL(p_ctx ‖ p_bare), and the
//        "tune" — the token contributing most to that KL (p_ctx · log-ratio), searched over
//        the union of both sides' top-k: where the context actually moved the model, not
//        merely a token it lifted from near zero. Signals/Grounding.swift turns these into
//        classes, drift, attribution and the turn's summary.
//  PIN:  Runs under the harness gate, after the reply has streamed; stops between chunks
//        when a generation is waiting or the budget is spent. Nothing is ever held at
//        [T, V]: each chunk's logits are reduced and evaluated before the next.
//        Frigate has no batched generation, which is why the counterfactual (no context)
//        is scored after the fact rather than decoded alongside.
//

import Foundation
import MLX
import MLXLMCommon

public enum GroundingScorer {

    public struct Options: Sendable {
        public var topK: Int
        public var chunk: Int
        public var prefillStep: Int
        public var detail: TraceLevel
        public var pushes: Int
        public var budget: TimeInterval

        public init(
            topK: Int = 64, chunk: Int = 64, prefillStep: Int = 512, detail: TraceLevel = .summary,
            pushes: Int = 8, budget: TimeInterval = 6
        ) {
            self.topK = topK
            self.chunk = chunk
            self.prefillStep = prefillStep
            self.detail = detail
            self.pushes = pushes
            self.budget = budget
        }

        public init(configuration: SinatraConfiguration, detail: TraceLevel) {
            self.init(
                topK: configuration.contextTopK, chunk: configuration.groundingChunk,
                detail: detail, pushes: configuration.groundingAlternatives,
                budget: configuration.groundingBudget)
        }
    }

    /// Why a measurement stopped short.
    public enum Stop: Error, CustomStringConvertible, Equatable {
        /// A generation is waiting for the gate.
        case waiting
        case overBudget(TimeInterval)
        case nothingToScore(String)

        public var description: String {
            switch self {
            case .waiting: return "a generation was waiting"
            case .overBudget(let seconds): return String(format: "over the %.1f s budget", seconds)
            case .nothingToScore(let reason): return reason
            }
        }
    }

    /// Score `sampled` with and without the retrieved context.
    ///
    /// `decodeCache` is the generation's own cache, holding `contextTokens` and some of
    /// `sampled`. Without it, or when a layer cannot be trimmed, both prompts are
    /// prefilled fresh (`cacheReused` false).
    public static func score(
        model: any LanguageModel, decodeCache: [KVCache]?, contextTokens: [Int], bareTokens: [Int],
        sampled: [Int], options: Options, shouldAbort: () -> Bool = { false }
    ) throws -> GroundingRaw {
        let started = Date()
        let pc = contextTokens.count
        let pb = bareTokens.count
        guard pc > 0, pb > 0 else { throw Stop.nothingToScore("an empty prompt") }
        guard !sampled.isEmpty else { throw Stop.nothingToScore("no generated tokens") }

        var shared = 0
        let sharable = min(pc - 1, pb - 1)
        while shared < sharable, contextTokens[shared] == bareTokens[shared] { shared += 1 }

        func check() throws {
            if shouldAbort() { throw Stop.waiting }
            let elapsed = Date().timeIntervalSince(started)
            if elapsed > options.budget { throw Stop.overBudget(options.budget) }
        }

        // The context side: the decode's own cache, trimmed back to the prompt.
        var reused = false
        if let decodeCache, !decodeCache.isEmpty,
            decodeCache.allSatisfy({ $0.isTrimmable && $0.offset >= pc - 1 })
        {
            for layer in decodeCache { layer.trim(layer.offset - (pc - 1)) }
            reused = decodeCache.allSatisfy { $0.offset == pc - 1 }
        }
        let ctxCache: [KVCache]
        var bareCache: [KVCache]
        var bareStart: Int
        if reused, let decodeCache {
            ctxCache = decodeCache
            (bareCache, bareStart) = fork(ctxCache, at: shared, model: model)
        } else {
            ctxCache = model.newCache(parameters: nil)
            try prefill(model, cache: ctxCache, tokens: contextTokens[0..<shared], step: options.prefillStep, check: check)
            (bareCache, bareStart) = fork(ctxCache, at: shared, model: model)
            try prefill(model, cache: ctxCache, tokens: contextTokens[shared..<(pc - 1)], step: options.prefillStep, check: check)
        }
        // The bare side: whatever of the bare prompt the context prompt does not share.
        try prefill(model, cache: bareCache, tokens: bareTokens[bareStart..<(pb - 1)], step: options.prefillStep, check: check)
        let prefillMillis = Date().timeIntervalSince(started) * 1000

        let feedCtx = [contextTokens[pc - 1]] + sampled.dropLast()
        let feedBare = [bareTokens[pb - 1]] + sampled.dropLast()
        var raw = GroundingRaw(
            tokens: sampled, logpCtx: [], logpBare: [], entropyCtx: [], entropyBare: [], contextKL: [],
            tune: [], tuneNats: [], rankBare: options.detail == .full ? [] : nil,
            pushes: options.detail == .full ? [] : nil, cacheReused: reused, promptTokens: pc,
            bareTokens: pb, sharedPrefixTokens: bareStart)
        let scoreStarted = Date()
        var offset = 0
        while offset < sampled.count {
            try check()
            let end = min(offset + max(1, options.chunk), sampled.count)
            let length = end - offset
            let zc = model(MLXArray(feedCtx[offset..<end].map { Int32(truncatingIfNeeded: $0) }, [1, length]), cache: ctxCache)
            let zb = model(MLXArray(feedBare[offset..<end].map { Int32(truncatingIfNeeded: $0) }, [1, length]), cache: bareCache)
            let targets = MLXArray(sampled[offset..<end].map { Int32(truncatingIfNeeded: $0) }, [length, 1])
            let math = chunk(context: zc[0].asType(.float32), bare: zb[0].asType(.float32), targets: targets, options: options)
            eval(math.arrays)
            math.append(to: &raw, rows: length)
            offset = end
        }
        raw.prefillMillis = prefillMillis
        raw.scoreMillis = Date().timeIntervalSince(scoreStarted) * 1000
        return raw
    }

    /// log p(continuation_t | prompt, continuation_<t) for every t, from a fresh cache.
    /// Leave-one-out attribution scores the same output under several prompts with it.
    public static func tokenLogprobs(
        model: any LanguageModel, prompt: [Int], continuation: [Int], chunk: Int = 64, prefillStep: Int = 512
    ) throws -> [Float] {
        guard !prompt.isEmpty, !continuation.isEmpty else { return [] }
        let cache = model.newCache(parameters: nil)
        try prefill(model, cache: cache, tokens: prompt[0..<(prompt.count - 1)], step: prefillStep, check: {})
        let feed = [prompt[prompt.count - 1]] + continuation.dropLast()
        var out: [Float] = []
        out.reserveCapacity(continuation.count)
        var offset = 0
        while offset < continuation.count {
            let end = min(offset + max(1, chunk), continuation.count)
            let length = end - offset
            let z = model(MLXArray(feed[offset..<end].map { Int32(truncatingIfNeeded: $0) }, [1, length]), cache: cache)[0]
                .asType(.float32)
            let targets = MLXArray(continuation[offset..<end].map { Int32(truncatingIfNeeded: $0) }, [length, 1])
            let logp = takeAlong(z - logSumExp(z, axis: -1, keepDims: true), targets, axis: -1).squeezed(axis: -1)
            eval(logp)
            out += logp.asArray(Float.self)
            offset = end
        }
        return out
    }

    // MARK: - Pieces

    /// A copy of `cache` holding its first `length` tokens, and where prefilling continues.
    /// A layer that will not trim leaves the copy empty: the caller prefills from 0.
    static func fork(_ cache: [KVCache], at length: Int, model: any LanguageModel) -> ([KVCache], Int) {
        guard length > 0 else { return (model.newCache(parameters: nil), 0) }
        let copy = cache.map { $0.copy() }
        guard copy.allSatisfy({ $0.isTrimmable && $0.offset >= length }) else {
            return (model.newCache(parameters: nil), 0)
        }
        for layer in copy { layer.trim(layer.offset - length) }
        guard copy.allSatisfy({ $0.offset == length }) else { return (model.newCache(parameters: nil), 0) }
        return (copy, length)
    }

    /// Push `tokens` through the model into `cache`, evaluating only the cache: the output
    /// head never runs for prompt positions.
    static func prefill(
        _ model: any LanguageModel, cache: [KVCache], tokens: ArraySlice<Int>, step: Int, check: () throws -> Void
    ) throws {
        var start = tokens.startIndex
        while start < tokens.endIndex {
            try check()
            let end = min(start + max(1, step), tokens.endIndex)
            let input = MLXArray(tokens[start..<end].map { Int32(truncatingIfNeeded: $0) }, [1, end - start])
            _ = model(input, cache: cache)
            eval(cache.flatMap { $0.innerState() })
            start = end
        }
    }

    struct ChunkMath {
        var logpCtx: MLXArray
        var logpBare: MLXArray
        var entropyCtx: MLXArray
        var entropyBare: MLXArray
        var contextKL: MLXArray
        var tune: MLXArray
        var tuneNats: MLXArray
        var rankBare: MLXArray?
        var pushIds: MLXArray?
        var pushNats: MLXArray?

        var arrays: [MLXArray] {
            [logpCtx, logpBare, entropyCtx, entropyBare, contextKL, tune, tuneNats]
                + [rankBare, pushIds, pushNats].compactMap { $0 }
        }

        func append(to raw: inout GroundingRaw, rows: Int) {
            raw.logpCtx += logpCtx.asArray(Float.self)
            raw.logpBare += logpBare.asArray(Float.self)
            raw.entropyCtx += entropyCtx.asArray(Float.self)
            raw.entropyBare += entropyBare.asArray(Float.self)
            raw.contextKL += contextKL.asArray(Float.self)
            raw.tune += tune.asArray(Int32.self).map(Int.init)
            raw.tuneNats += tuneNats.asArray(Float.self)
            if let rankBare {
                raw.rankBare = (raw.rankBare ?? []) + rankBare.asArray(Float.self).map { Int($0) }
            }
            if let pushIds, let pushNats {
                let width = pushIds.dim(-1)
                let ids = pushIds.asArray(Int32.self)
                let nats = pushNats.asArray(Float.self)
                var all = raw.pushes ?? []
                for row in 0..<rows {
                    var seen = Set<Int>()
                    var pushes: [(id: Int, nats: Float)] = []
                    for j in 0..<width {
                        let id = Int(ids[row * width + j])
                        if seen.insert(id).inserted { pushes.append((id, nats[row * width + j])) }
                    }
                    all.append(pushes.sorted { $0.nats > $1.nats })
                }
                raw.pushes = all
            }
        }
    }

    /// Both sides' distributions for one chunk of positions ([L, V] each), reduced.
    static func chunk(context zc: MLXArray, bare zb: MLXArray, targets: MLXArray, options: Options) -> ChunkMath {
        let lpC = zc - logSumExp(zc, axis: -1, keepDims: true)
        let lpB = zb - logSumExp(zb, axis: -1, keepDims: true)
        let pC = exp(lpC)
        let pB = exp(lpB)
        let zero = MLXArray(Float(0))
        let entropyCtx = -sum(MLX.where(pC .> 0, pC * lpC, zero), axis: -1)
        let entropyBare = -sum(MLX.where(pB .> 0, pB * lpB, zero), axis: -1)
        let kl = sum(MLX.where(pC .> 0, pC * (lpC - lpB), zero), axis: -1)
        let logpCtx = takeAlong(lpC, targets, axis: -1).squeezed(axis: -1)
        let logpBare = takeAlong(lpB, targets, axis: -1).squeezed(axis: -1)

        let k = max(1, min(options.topK, zc.dim(-1)))
        let topC = argPartition(-lpC, kth: k - 1, axis: -1)[0..., ..<k]
        let topB = argPartition(-lpB, kth: k - 1, axis: -1)[0..., ..<k]
        let union = concatenated([topC, topB], axis: -1)
        let lpCUnion = takeAlong(lpC, union, axis: -1)
        let push = lpCUnion - takeAlong(lpB, union, axis: -1)
        // Each token's share of KL(p_ctx ‖ p_bare): large only where the context both moved
        // the token and made it likely.
        let contribution = exp(lpCUnion) * push
        let best = argMax(contribution, axis: -1, keepDims: true)
        var math = ChunkMath(
            logpCtx: logpCtx, logpBare: logpBare, entropyCtx: entropyCtx, entropyBare: entropyBare,
            contextKL: kl, tune: takeAlong(union, best, axis: -1).squeezed(axis: -1).asType(.int32),
            tuneNats: takeAlong(push, best, axis: -1).squeezed(axis: -1))
        if options.detail == .full {
            math.rankBare = sum((zb .> takeAlong(zb, targets, axis: -1)).asType(.float32), axis: -1)
            let count = max(1, min(options.pushes, push.dim(-1)))
            let order = argPartition(-contribution, kth: count - 1, axis: -1)[0..., ..<count]
            math.pushIds = takeAlong(union, order, axis: -1).asType(.int32)
            math.pushNats = takeAlong(push, order, axis: -1)
        }
        return math
    }
}
