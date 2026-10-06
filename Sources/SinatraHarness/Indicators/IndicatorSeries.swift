//
//  IndicatorSeries.swift
//  SinatraHarness
//
//  WHAT: The owner's grounding series and the 11 indicator features computed from it
//        as of any moment, under any `IndicatorPeriods`.
//  PIN:  As-of only: a training row for a turn sees the points labelled at or before
//        that turn, never later ones. The MACD history the old collector kept as state
//        is recomputed from prefixes, so candidate periods can be evaluated exactly.
//

import Foundation

/// One measured turn in the owner's series.
public struct SeriesPoint: Codable, Sendable, Equatable {
    /// When the turn was measured.
    public var at: Date
    /// The turn's grounding: share of its content tokens the context pushed for, in [0, 1].
    public var value: Double
    /// Content tokens in the output.
    public var volume: Int

    public init(at: Date, value: Double, volume: Int) {
        self.at = at
        self.value = value
        self.volume = volume
    }
}

public enum IndicatorSeries {

    public static let names = [
        "ema", "sma", "macd", "macdSignal", "macdPrevSignal", "avgVolChange",
        "vwa", "stochK", "stochD", "momentum", "velocity",
    ]

    /// What each indicator reports before there is enough history.
    public static let neutral: [Float] = [0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0]

    /// The 11 indicator features over `points` (chronological), neutral below `minimumCount`.
    public static func features(
        points: [SeriesPoint], periods: IndicatorPeriods, minimumCount: Int
    ) -> [Float] {
        guard points.count >= max(1, minimumCount) else { return neutral }
        let indicators = TechnicalIndicators(points: points)
        let macdHistory = macdHistory(points: points, periods: periods)

        let values: [Double] = [
            indicators.emaWA(period: periods.emaPeriod),
            indicators.smaWA(period: periods.smaPeriod),
            indicators.macD(fastPeriod: periods.macdFast, slowPeriod: periods.macdSlow),
            TechnicalIndicators.macDSignal(macdHistory: macdHistory, signalPeriod: periods.macdSignalPeriod),
            TechnicalIndicators.macDPreviousSignal(macdHistory: macdHistory),
            indicators.avgVolChange(period: periods.avgVolPeriod).clamped(-1, 1),
            indicators.volumeWeightedAverage(period: periods.vwaPeriod),
            indicators.stochasticK(period: periods.stochKPeriod),
            indicators.stochasticD(period: periods.stochKPeriod, signalPeriod: periods.stochDSignal),
            indicators.momentum(period: periods.momentumPeriod),
            indicators.velocity(period: periods.velocityPeriod) * 0.5,
        ]
        return values.map { Float($0.isFinite ? $0 : 0) }
    }

    /// MACD evaluated on the most recent prefixes long enough for the slow EMA, oldest first.
    static func macdHistory(points: [SeriesPoint], periods: IndicatorPeriods) -> [Double] {
        let depth = max(periods.macdSignalPeriod, 10) + 1
        let n = points.count
        let firstEnd = max(periods.macdSlow, n - depth + 1)
        guard firstEnd <= n else { return [] }
        var history: [Double] = []
        history.reserveCapacity(n - firstEnd + 1)
        for end in firstEnd...n {
            let prefix = TechnicalIndicators(points: points[0..<end])
            history.append(prefix.macD(fastPeriod: periods.macdFast, slowPeriod: periods.macdSlow))
        }
        return history
    }

    /// Points at or before `date`, assuming `points` is chronological.
    static func asOf(_ points: [SeriesPoint], _ date: Date) -> [SeriesPoint] {
        guard let last = points.last, last.at > date else { return points }
        var end = points.count
        while end > 0 && points[end - 1].at > date { end -= 1 }
        return Array(points[0..<end])
    }
}
