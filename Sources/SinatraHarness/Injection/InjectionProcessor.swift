//
//  InjectionProcessor.swift
//  SinatraHarness
//
//  WHAT: The transformer injection layer: adds the turn's [1, V] bias to the logits after
//        the forward pass and before the sampler — i.e. right before decoding.
//  PIN:  One broadcast add per token. `prompt(_:)` is deliberately a no-op: the injection
//        never inspects the prompt, only what the side model derived from retrieved context.
//        A logits width that disagrees with the bias passes through untouched.
//

import Foundation
import MLX
import MLXLMCommon

struct InjectionProcessor: LogitProcessor {
    let bias: MLXArray
    let vocabularySize: Int
    private let casts = CastCache()

    init(bias: SparseBias) {
        self.vocabularySize = bias.vocabularySize
        self.bias = MLXArray(bias.denseVector(), [1, bias.vocabularySize])
        eval(self.bias)
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard logits.dim(-1) == vocabularySize else { return logits }
        return logits + casts.bias(bias, as: logits.dtype)
    }

    mutating func didSample(token: MLXArray) {}

    /// The bias cast once per logits dtype rather than every step.
    final class CastCache: @unchecked Sendable {
        private let lock = NSLock()
        private var cached: [DType: MLXArray] = [:]

        func bias(_ bias: MLXArray, as dtype: DType) -> MLXArray {
            if dtype == bias.dtype { return bias }
            lock.lock()
            defer { lock.unlock() }
            if let hit = cached[dtype] { return hit }
            let cast = bias.asType(dtype)
            eval(cast)
            cached[dtype] = cast
            return cast
        }
    }
}

/// Runs processors in order: injection first, then the built-in penalties, so a boosted
/// token that was just emitted is still repetition-damped.
struct CompositeLogitProcessor: LogitProcessor {
    var processors: [any LogitProcessor]

    init(_ processors: [any LogitProcessor]) { self.processors = processors }

    mutating func prompt(_ prompt: MLXArray) {
        for i in processors.indices { processors[i].prompt(prompt) }
    }

    func process(logits: MLXArray) -> MLXArray {
        processors.reduce(logits) { $1.process(logits: $0) }
    }

    mutating func didSample(token: MLXArray) {
        for i in processors.indices { processors[i].didSample(token: token) }
    }
}

/// Keeps every sampled token as a lazy array; read once, after the decode, for the
/// grounding measurement. A class so the iterator's copy and the run share one log.
final class TokenRecorder: LogitProcessor, @unchecked Sendable {
    private let lock = NSLock()
    private var tokens: [MLXArray] = []

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    func didSample(token: MLXArray) {
        lock.lock()
        tokens.append(token)
        lock.unlock()
    }

    /// The first `limit` tokens (all when nil), evaluated in one go.
    func drain(limit: Int?) -> [Int] {
        lock.lock()
        let all = tokens
        tokens = []
        lock.unlock()
        let used = limit.map { Array(all.prefix(max(0, $0))) } ?? all
        guard !used.isEmpty else { return [] }
        let joined = concatenated(used.map { $0.reshaped([1]).asType(.int32) }, axis: 0)
        eval(joined)
        return joined.asArray(Int32.self).map(Int.init)
    }
}
