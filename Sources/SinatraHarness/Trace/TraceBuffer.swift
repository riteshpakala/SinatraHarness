//
//  TraceBuffer.swift
//  SinatraHarness
//
//  WHAT: Collects per-step trace arrays during generation and evaluates them once at the
//        end. The tracing processor stashes z and z'; the tracing sampler draws the real
//        token from z' and the counterfactual from z with an identically seeded shadow
//        sampler, then records the step.
//  PIN:  Each step's arrays are `asyncEval`'d as they are made: nothing blocks the token
//        loop, and the [1, V] logits they read are freed as soon as they are reduced.
//

import Foundation
import MLX
import MLXLMCommon

final class TraceBuffer: @unchecked Sendable {
    let level: TraceLevel
    let mask: MLXArray?
    let topK: Int
    private let lock = NSLock()
    private var pendingBase: MLXArray?
    private var pendingInjected: MLXArray?
    private var steps: [TraceMath.Step] = []

    init(level: TraceLevel, mask: MLXArray?, topK: Int) {
        self.level = level
        self.mask = mask
        self.topK = topK
    }

    func stash(base: MLXArray, injected: MLXArray) {
        lock.lock()
        pendingBase = base
        pendingInjected = injected
        lock.unlock()
    }

    func record(sampled: MLXArray, shadow: any LogitSampler) {
        lock.lock()
        let base = pendingBase
        let injected = pendingInjected
        pendingBase = nil
        pendingInjected = nil
        lock.unlock()
        guard let base, let injected else { return }
        let counterfactual = shadow.sample(logits: base)
        let step = TraceMath.step(
            base: base, injected: injected, sampled: sampled, counterfactual: counterfactual,
            mask: mask, topK: topK, full: level == .full)
        asyncEval(step.arrays)
        lock.lock()
        steps.append(step)
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return steps.count
    }

    struct RawStep {
        var entropyPre, entropyPost, kl, js, logprobPre, logprobPost, massIntoMask: Float
        var rankPre, rankPost, argmaxPre, argmaxPost, sampled, counterfactual: Int
        var topPre: [(Int, Float)]?
        var topPost: [(Int, Float)]?
        var movement: [Float]?
    }

    /// Evaluate everything recorded (first `limit` steps) and read it back.
    func drain(limit: Int?) -> [RawStep] {
        lock.lock()
        let all = steps
        steps = []
        lock.unlock()
        let used = limit.map { Array(all.prefix(max(0, $0))) } ?? all
        guard !used.isEmpty else { return [] }
        eval(used.flatMap(\.arrays))

        func floats(_ key: KeyPath<TraceMath.Step, MLXArray>) -> [Float] {
            concatenated(used.map { $0[keyPath: key].reshaped([1]).asType(.float32) }, axis: 0).asArray(Float.self)
        }
        func ints(_ key: KeyPath<TraceMath.Step, MLXArray>) -> [Int] {
            concatenated(used.map { $0[keyPath: key].reshaped([1]).asType(.int32) }, axis: 0)
                .asArray(Int32.self).map(Int.init)
        }
        let entropyPre = floats(\.entropyPre)
        let entropyPost = floats(\.entropyPost)
        let kl = floats(\.kl)
        let js = floats(\.js)
        let logprobPre = floats(\.logprobPre)
        let logprobPost = floats(\.logprobPost)
        let mass = floats(\.massIntoMask)
        let rankPre = floats(\.rankPre).map { Int($0) }
        let rankPost = floats(\.rankPost).map { Int($0) }
        let argmaxPre = ints(\.argmaxPre)
        let argmaxPost = ints(\.argmaxPost)
        let sampled = ints(\.sampled)
        let counterfactual = ints(\.counterfactual)

        return used.indices.map { i in
            var raw = RawStep(
                entropyPre: entropyPre[i].finite(), entropyPost: entropyPost[i].finite(),
                kl: kl[i].finite(), js: js[i].finite(),
                logprobPre: logprobPre[i].finite(-1e4), logprobPost: logprobPost[i].finite(-1e4),
                massIntoMask: mass[i].finite(),
                rankPre: rankPre[i], rankPost: rankPost[i], argmaxPre: argmaxPre[i], argmaxPost: argmaxPost[i],
                sampled: sampled[i], counterfactual: counterfactual[i])
            let step = used[i]
            if let ids = step.topPreIds, let values = step.topPreValues {
                raw.topPre = Self.sortedPairs(ids, values)
            }
            if let ids = step.topPostIds, let values = step.topPostValues {
                raw.topPost = Self.sortedPairs(ids, values)
            }
            if let movement = step.movement {
                raw.movement = movement.asType(.float32).asArray(Float.self).map { $0.finite() }
            }
            return raw
        }
    }

    private static func sortedPairs(_ ids: MLXArray, _ values: MLXArray) -> [(Int, Float)] {
        let idList = ids.asType(.int32).asArray(Int32.self).map(Int.init)
        let valueList = values.asType(.float32).asArray(Float.self)
        return zip(idList, valueList).map { ($0, $1.finite(-1e4)) }.sorted { $0.1 > $1.1 }
    }
}

/// Wraps the injection and the penalty processors so both z (no injection) and z' reach
/// the trace. Penalties are applied to each side separately — the counterfactual sees the
/// same history the real decode does.
struct TracingLogitProcessor: LogitProcessor {
    var injection: InjectionProcessor?
    var penalty: (any LogitProcessor)?
    let buffer: TraceBuffer

    mutating func prompt(_ prompt: MLXArray) {
        penalty?.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        let base = penalty?.process(logits: logits) ?? logits
        let injected: MLXArray
        if let injection {
            let raised = injection.process(logits: logits)
            injected = penalty?.process(logits: raised) ?? raised
        } else {
            injected = base
        }
        buffer.stash(base: base, injected: injected)
        return injected
    }

    mutating func didSample(token: MLXArray) {
        penalty?.didSample(token: token)
    }
}

/// Samples the real token, then asks an identically seeded shadow sampler what it would
/// have drawn from the un-injected logits. Both advance their random state once per step,
/// so their draws differ only where the distributions do.
struct TracingSampler: LogitSampler {
    let real: any LogitSampler
    let shadow: any LogitSampler
    let buffer: TraceBuffer

    func sample(logits: MLXArray) -> MLXArray {
        let token = real.sample(logits: logits)
        buffer.record(sampled: token, shadow: shadow)
        return token
    }
}
