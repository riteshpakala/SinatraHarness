//
//  TokenBatch.swift
//  SinatraHarness
//
//  WHAT: The retrieved partitions as padded token-id rows, ready to become one MLX array.
//

import Foundation

public struct TokenBatch: Sendable, Equatable {
    public let rows: [[Int]]

    public init(rows: [[Int]]) { self.rows = rows }

    public var count: Int { rows.count }
    public var lengths: [Int] { rows.map(\.count) }
    public var maxLength: Int { rows.map(\.count).max() ?? 0 }

    /// Row-major [P, T] ids (padded with `pad`) and a matching 0/1 mask.
    public func padded(pad: Int32 = 0) -> (ids: [Int32], mask: [Float], rows: Int, columns: Int) {
        let columns = max(1, maxLength)
        var ids = [Int32](repeating: pad, count: count * columns)
        var mask = [Float](repeating: 0, count: count * columns)
        for (r, row) in rows.enumerated() {
            for (c, id) in row.enumerated() {
                ids[r * columns + c] = Int32(truncatingIfNeeded: id)
                mask[r * columns + c] = 1
            }
        }
        return (ids, mask, count, columns)
    }

    /// Keep the head and tail of a long partition: openings and conclusions carry the most.
    public static func clip(_ ids: [Int], maxTokens: Int) -> [Int] {
        guard maxTokens > 0, ids.count > maxTokens else { return ids }
        let head = maxTokens / 2
        let tail = maxTokens - head
        return Array(ids.prefix(head)) + Array(ids.suffix(tail))
    }
}
