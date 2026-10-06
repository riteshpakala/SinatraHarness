//
//  HotTokens.swift
//  SinatraHarness
//
//  WHAT: The owner's own stopwords, learned: content tokens that appear in a large share
//        of everything retrieved for them carry no signal about any one partition.
//  PIN:  Space-Saving top-k with per-entry error bounds. A token counts as hot only on its
//        guaranteed count (count − error), so a freshly inserted rare token can never be
//        mistaken for a frequent one.
//

import Foundation

public struct HotTokens: Codable, Sendable, Equatable {
    public private(set) var capacity: Int
    public private(set) var partitionsObserved: Int = 0
    private var ids: [Int] = []
    private var counts: [Int] = []
    private var errors: [Int] = []

    public init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    /// Record one partition's unique content tokens.
    public mutating func observe(_ tokens: Set<Int>) {
        partitionsObserved += 1
        var index: [Int: Int] = [:]
        index.reserveCapacity(ids.count)
        for (i, id) in ids.enumerated() { index[id] = i }
        for token in tokens {
            if let i = index[token] {
                counts[i] += 1
            } else if ids.count < capacity {
                index[token] = ids.count
                ids.append(token)
                counts.append(1)
                errors.append(0)
            } else if let victim = counts.indices.min(by: { counts[$0] < counts[$1] }) {
                let floor = counts[victim]
                index[ids[victim]] = nil
                ids[victim] = token
                counts[victim] = floor + 1
                errors[victim] = floor
                index[token] = victim
            }
        }
    }

    public func guaranteedCount(_ token: Int) -> Int {
        guard let i = ids.firstIndex(of: token) else { return 0 }
        return counts[i] - errors[i]
    }

    public func isHot(_ token: Int, ratio: Double, minimumPartitions: Int) -> Bool {
        guard partitionsObserved >= minimumPartitions else { return false }
        return Double(guaranteedCount(token)) >= ratio * Double(partitionsObserved)
    }

    public var tracked: Int { ids.count }
}
