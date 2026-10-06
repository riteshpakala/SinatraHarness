//
//  SparseBias.swift
//  SinatraHarness
//
//  WHAT: The injection as data: a sparse additive bias over the LLM's vocabulary, and the
//        impact mask that says which partition put how much of it on each token.
//  PIN:  Plain values, Sendable and Codable, so a plan can cross actors and land in a
//        trace file. The [1, V] MLXArray is built from `denseVector()` on the generation
//        side only.
//

import Foundation

public struct SparseBias: Sendable, Codable, Equatable {
    public let vocabularySize: Int
    /// Ascending token ids.
    public let indices: [Int32]
    public let values: [Float]

    public init(vocabularySize: Int, entries: [Int: Float]) {
        let sorted = entries.filter { $0.key >= 0 && $0.key < vocabularySize }.sorted { $0.key < $1.key }
        self.vocabularySize = vocabularySize
        self.indices = sorted.map { Int32($0.key) }
        self.values = sorted.map(\.value)
    }

    public var nonZero: Int { indices.count }
    public var maxAbs: Float { values.map(abs).max() ?? 0 }
    public var l1: Float { values.reduce(0) { $0 + abs($1) } }
    public var isEmpty: Bool { indices.isEmpty }

    public func denseVector() -> [Float] {
        var dense = [Float](repeating: 0, count: vocabularySize)
        for (i, id) in indices.enumerated() { dense[Int(id)] = values[i] }
        return dense
    }

    public func position(of token: Int) -> Int? {
        var lo = 0
        var hi = indices.count - 1
        let target = Int32(truncatingIfNeeded: token)
        while lo <= hi {
            let mid = (lo + hi) / 2
            if indices[mid] == target { return mid }
            if indices[mid] < target { lo = mid + 1 } else { hi = mid - 1 }
        }
        return nil
    }

    public func value(of token: Int) -> Float {
        position(of: token).map { values[$0] } ?? 0
    }

    /// The k entries with the largest |bias|.
    public func top(_ k: Int) -> [(id: Int, bias: Float)] {
        values.indices.sorted { abs(values[$0]) > abs(values[$1]) }
            .prefix(k)
            .map { (Int(indices[$0]), values[$0]) }
    }
}

/// Where the injection acts and who put it there.
public struct ImpactMask: Sendable, Codable, Equatable {
    /// The bias support, ascending — identical to `SparseBias.indices`.
    public let tokenIds: [Int32]
    public let bias: [Float]
    public let partitionIds: [String]
    /// [partition][mask position]: each column sums to that token's applied bias.
    public let contributions: [[Float]]
    /// Decoded text of each mask token, for reading traces without the tokenizer.
    public var tokenTexts: [String]?

    public func position(of token: Int) -> Int? {
        var lo = 0
        var hi = tokenIds.count - 1
        let target = Int32(truncatingIfNeeded: token)
        while lo <= hi {
            let mid = (lo + hi) / 2
            if tokenIds[mid] == target { return mid }
            if tokenIds[mid] < target { lo = mid + 1 } else { hi = mid - 1 }
        }
        return nil
    }

    /// partition id → its share of the bias on `token`.
    public func attribution(of token: Int) -> [String: Float] {
        guard let i = position(of: token) else { return [:] }
        var out: [String: Float] = [:]
        for (p, id) in partitionIds.enumerated() where contributions[p][i] != 0 {
            out[id] = contributions[p][i]
        }
        return out
    }
}

enum BiasAssembly {
    /// Sum per-partition contributions, clamp to ±cap (scaling each partition's share with
    /// the clamp so attribution still sums to the applied bias), drop negligible entries.
    static func assemble(
        perPartition: [[Int: Float]], partitionIds: [String], vocabularySize: Int, cap: Float,
        minimumMagnitude: Float = 1e-4
    ) -> (SparseBias, ImpactMask)? {
        var total: [Int: Float] = [:]
        for contributions in perPartition {
            for (token, value) in contributions { total[token, default: 0] += value }
        }
        var applied: [Int: Float] = [:]
        var scale: [Int: Float] = [:]
        for (token, value) in total {
            let clamped = value.clamped(-cap, cap)
            guard abs(clamped) >= minimumMagnitude, token >= 0, token < vocabularySize else { continue }
            applied[token] = clamped
            scale[token] = value != 0 ? clamped / value : 0
        }
        guard !applied.isEmpty else { return nil }
        let bias = SparseBias(vocabularySize: vocabularySize, entries: applied)
        let contributions: [[Float]] = perPartition.map { part in
            bias.indices.map { id in
                let token = Int(id)
                return (part[token] ?? 0) * (scale[token] ?? 0)
            }
        }
        let mask = ImpactMask(
            tokenIds: bias.indices, bias: bias.values, partitionIds: partitionIds,
            contributions: contributions, tokenTexts: nil)
        return (bias, mask)
    }
}
