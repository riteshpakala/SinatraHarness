//
//  ContextEncoding.swift
//  SinatraHarness
//
//  WHAT: The encoder box in the diagram: retrieved context in, one pooled vector per
//        partition out. `LanguageModelContextEncoder` uses the loaded LLM's own input
//        embedding table; `HashingContextEncoder` is a model-free stand-in.
//  PIN:  Vectors cross this boundary as plain [Float] so the session, tests and replay
//        never depend on a GPU. Only the prompt-free partition tokens are ever passed in.
//

import Foundation

public protocol ContextEncoding: AnyObject {
    /// Stable description, recorded with every stored context vector.
    var identifier: String { get }
    var hiddenSize: Int { get }
    /// Rows of the model's output head, when there is one.
    var vocabularySize: Int? { get }
    /// Mean-pooled input embeddings, one row per partition ([P][H]).
    func encode(_ batch: TokenBatch) throws -> [[Float]]
    /// Project vectors through the model's output head ([N][H] → [N][V]); nil when unsupported.
    func outputLogits(_ vectors: [[Float]]) throws -> [[Float]]?
}

/// Deterministic pseudo-embeddings from hashed token ids. No model, no GPU.
public final class HashingContextEncoder: ContextEncoding {
    public let identifier: String
    public let hiddenSize: Int
    public let vocabularySize: Int? = nil
    private let seed: UInt64

    public init(hiddenSize: Int = 256, seed: UInt64 = 0x5349_4E41_5452_4131) {
        self.hiddenSize = hiddenSize
        self.seed = seed
        self.identifier = "hashing-v1/\(hiddenSize)"
    }

    public func encode(_ batch: TokenBatch) throws -> [[Float]] {
        batch.rows.map { row in
            var sum = [Float](repeating: 0, count: hiddenSize)
            guard !row.isEmpty else { return sum }
            for id in row {
                var rng = SplitMix64(seed: seed ^ (UInt64(bitPattern: Int64(id)) &* 0x9E37_79B9_7F4A_7C15))
                for d in 0..<hiddenSize {
                    sum[d] += Float(Int64(bitPattern: rng.next() >> 11) - (1 << 52)) / Float(1 << 52)
                }
            }
            let n = Float(row.count)
            return sum.map { $0 / n }
        }
    }

    public func outputLogits(_ vectors: [[Float]]) throws -> [[Float]]? { nil }
}

/// Count sketch H → C: every input dimension lands in one bucket with a ±1 sign. No
/// parameters, nothing to persist, identical for any model with the same H.
public struct CountSketchProjection: Sendable, Equatable {
    public static let version = "count-sketch-v1"
    public let inputDimension: Int
    public let outputDimension: Int
    private let buckets: [Int]
    private let signs: [Float]

    public init(inputDimension: Int, outputDimension: Int = 64, seed: UInt64 = 0x434F_554E_5453_4B31) {
        self.inputDimension = inputDimension
        self.outputDimension = max(1, outputDimension)
        var buckets = [Int](repeating: 0, count: inputDimension)
        var signs = [Float](repeating: 1, count: inputDimension)
        for i in 0..<inputDimension {
            let h = FNV1a.hash(UInt64(i), seed: seed)
            buckets[i] = Int(h % UInt64(self.outputDimension))
            signs[i] = (h >> 40) & 1 == 0 ? 1 : -1
        }
        self.buckets = buckets
        self.signs = signs
    }

    /// L2-normalise, sketch, L2-normalise again (a zero vector stays zero).
    public func project(_ x: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: outputDimension)
        let n = min(x.count, inputDimension)
        guard n > 0 else { return out }
        var norm: Float = 0
        for i in 0..<n { norm += x[i] * x[i] }
        norm = norm.squareRoot()
        guard norm > 0, norm.isFinite else { return out }
        for i in 0..<n { out[buckets[i]] += signs[i] * x[i] / norm }
        var outNorm: Float = 0
        for v in out { outNorm += v * v }
        outNorm = outNorm.squareRoot()
        guard outNorm > 0 else { return out }
        return out.map { ($0 / outNorm).rounded4 }
    }
}
