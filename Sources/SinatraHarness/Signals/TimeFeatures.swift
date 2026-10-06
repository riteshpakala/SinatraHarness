//
//  TimeFeatures.swift
//  SinatraHarness
//
//  WHAT: Time-derived feature primitives: log-normalised ages, the 30-day relevancy
//        bands, and time-of-day / day-of-week periodics.
//

import Foundation

public enum RelevancyBand: String, Sendable, Codable, CaseIterable {
    /// ≤ `freshBandDays` (7 d)
    case fresh
    /// 7 – 30 d
    case mid
    /// > `retentionDays` (30 d)
    case stale
}

enum TimeFeatures {

    static func days(_ interval: TimeInterval) -> Double {
        max(0, interval) / 86_400
    }

    /// ln(1 + days) / ln(1 + horizon), clipped to [0, 1].
    static func lnNorm(days: Double, horizon: Double = 365) -> Float {
        Float(log1p(max(0, days)) / log1p(horizon)).clamped(0, 1)
    }

    /// ln(1 + seconds) / ln(1 + horizon), clipped to [0, 1].
    static func lnNorm(seconds: Double, horizon: Double) -> Float {
        Float(log1p(max(0, seconds)) / log1p(horizon)).clamped(0, 1)
    }

    static func band(ageDays: Double, configuration: SinatraConfiguration) -> RelevancyBand {
        if ageDays <= configuration.freshBandDays { return .fresh }
        if ageDays <= configuration.retentionDays { return .mid }
        return .stale
    }

    static func bandWeight(_ band: RelevancyBand, configuration: SinatraConfiguration) -> Float {
        switch band {
        case .fresh: return configuration.bandWeights.fresh
        case .mid: return configuration.bandWeights.mid
        case .stale: return configuration.bandWeights.stale
        }
    }

    /// The partition's recency reference: the newest of its creation, modification and
    /// index times; with none known, when this owner first retrieved the document.
    static func recencyReference(_ partition: Partition, firstSeen: Date?, now: Date) -> Date {
        let known = [partition.createdAt, partition.modifiedAt, partition.indexedAt].compactMap { $0 }
        return known.max() ?? firstSeen ?? now
    }

    /// (hourSin, hourCos, dowSin, dowCos) at `date` in `timeZone`.
    static func periodics(_ date: Date, timeZone: TimeZone) -> [Float] {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let parts = calendar.dateComponents([.hour, .minute, .weekday], from: date)
        let hour = Double(parts.hour ?? 0) + Double(parts.minute ?? 0) / 60
        let weekday = Double((parts.weekday ?? 1) - 1)  // 0 = Sunday
        let h = 2 * Double.pi * hour / 24
        let d = 2 * Double.pi * weekday / 7
        return [Float(sin(h)), Float(cos(h)), Float(sin(d)), Float(cos(d))]
    }
}
