//
//  Hashing.swift
//  SinatraHarness
//
//  WHAT: Deterministic, platform-neutral hashing and a seedable RNG. No CryptoKit:
//        these files must build anywhere Frigate's MLX targets build.
//

import Foundation

enum FNV1a {
    static func hash(_ string: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    static func hash(_ value: UInt64, seed: UInt64) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325 ^ seed
        var v = value
        for _ in 0..<8 {
            hash ^= v & 0xff
            hash = hash &* 0x0000_0100_0000_01b3
            v >>= 8
        }
        return hash
    }

    static func hex(_ string: String) -> String {
        String(format: "%016llx", hash(string))
    }
}

/// SplitMix64: tiny, fast, seedable. Used where determinism matters (tests, projections).
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

extension Float {
    /// Four decimals: enough for features, a third of the JSON.
    var rounded4: Float { (self * 10_000).rounded() / 10_000 }
}

extension Comparable {
    func clamped(_ lower: Self, _ upper: Self) -> Self { Swift.min(Swift.max(self, lower), upper) }
}
