//
//  TechnicalIndicators.swift
//  SinatraHarness
//
//  WHAT: The stock-style indicators, computed over an owner's grounding series.
//        "Price" is a measured turn's grounding in [0, 1]; "volume" is its content-token count;
//        the timestamps are when each turn was measured.
//  PIN:  Formulas ported verbatim from Sewn's Sinatra (Sources/Sinatra/ML/
//        TechnicalIndicators.swift), where the price was an LLM-judged sentiment weight.
//        Insufficient history returns the same neutral values (0.5 for levels, 0 for
//        differences).
//

import Foundation

struct TechnicalIndicators {
    private let values: [Double]
    private let timestamps: [Date]
    private let volumes: [Int]

    init(values: [Double], timestamps: [Date], volumes: [Int]) {
        self.values = values
        self.timestamps = timestamps
        self.volumes = volumes
    }

    init(points: some Collection<SeriesPoint>) {
        self.values = points.map(\.value)
        self.timestamps = points.map(\.at)
        self.volumes = points.map(\.volume)
    }

    // === EMA Weighted Average ===
    func emaWA(period: Int = 10, alpha: Double = 0.3) -> Double {
        guard period > 0, values.count >= period else { return 0.5 }
        let window = values.suffix(period)
        var ema = window.first!
        for value in window.dropFirst() {
            ema = alpha * value + (1 - alpha) * ema
        }
        return ema
    }

    // === SMA Weighted Average ===
    func smaWA(period: Int = 20) -> Double {
        guard period > 0, values.count >= period else { return 0.5 }
        return values.suffix(period).reduce(0, +) / Double(period)
    }

    // === MACD ===
    func macD(fastPeriod: Int = 5, slowPeriod: Int = 15) -> Double {
        emaWA(period: fastPeriod, alpha: 0.4) - emaWA(period: slowPeriod, alpha: 0.2)
    }

    // === MACD Signal Line (EMA of MACD values) ===
    static func macDSignal(macdHistory: [Double], signalPeriod: Int = 9) -> Double {
        guard signalPeriod > 0, macdHistory.count >= signalPeriod else { return 0.0 }
        let window = macdHistory.suffix(signalPeriod)
        var ema = window.first!
        for value in window.dropFirst() {
            ema = 0.3 * value + 0.7 * ema
        }
        return ema
    }

    // === MACD Previous Signal ===
    /// The signal line without the latest MACD value, window fixed at 9 (as ported).
    static func macDPreviousSignal(macdHistory: [Double]) -> Double {
        guard macdHistory.count >= 10 else { return 0.0 }
        return macDSignal(macdHistory: Array(macdHistory.dropLast()), signalPeriod: 9)
    }

    // === Average Volume Change ===
    /// Positive = replies arriving faster than the historical cadence, negative = slower.
    func avgVolChange(period: Int = 10, lifetimeAvgInterval: Double? = nil) -> Double {
        guard period > 0, timestamps.count >= period + 1 else { return 0.0 }

        let recentTimestamps = timestamps.suffix(period + 1)
        let recentIntervals = zip(recentTimestamps.dropLast(), recentTimestamps.dropFirst())
            .map { $1.timeIntervalSince($0) }
        guard !recentIntervals.isEmpty else { return 0.0 }
        let recentAvgInterval = recentIntervals.reduce(0, +) / Double(recentIntervals.count)

        let historicalAvgInterval: Double
        if let lifetime = lifetimeAvgInterval, lifetime > 0 {
            historicalAvgInterval = lifetime
        } else {
            let allIntervals = zip(timestamps.dropLast(), timestamps.dropFirst())
                .map { $1.timeIntervalSince($0) }
            guard !allIntervals.isEmpty else { return 0.0 }
            historicalAvgInterval = allIntervals.reduce(0, +) / Double(allIntervals.count)
        }
        guard historicalAvgInterval > 0 else { return 0.0 }
        return (historicalAvgInterval - recentAvgInterval) / historicalAvgInterval
    }

    // === Stochastic %K ===
    /// Where the current value sits in its recent high-low range: 1 = peak, 0 = trough.
    func stochasticK(period: Int = 14) -> Double {
        guard period > 0, values.count >= period else { return 0.5 }
        let window = Array(values.suffix(period))
        guard let minVal = window.min(), let maxVal = window.max(), maxVal > minVal else { return 0.5 }
        return (window.last! - minVal) / (maxVal - minVal)
    }

    // === Stochastic %D ===
    /// Mean of the most recent `signalPeriod` %K values.
    func stochasticD(period: Int = 14, signalPeriod: Int = 3) -> Double {
        let required = period + signalPeriod - 1
        guard period > 0, signalPeriod > 0, values.count >= required else { return 0.5 }
        let history = Array(values.suffix(required))
        var kValues: [Double] = []
        for i in 0..<signalPeriod {
            let windowEnd = required - i
            let windowStart = windowEnd - period
            let window = Array(history[windowStart..<windowEnd])
            guard let minVal = window.min(), let maxVal = window.max() else { continue }
            let current = history[windowEnd - 1]
            kValues.append(maxVal > minVal ? (current - minVal) / (maxVal - minVal) : 0.5)
        }
        guard !kValues.isEmpty else { return 0.5 }
        return kValues.reduce(0, +) / Double(kValues.count)
    }

    // === Momentum (raw rate of change) ===
    /// Point-to-point change over `period` turns. Range [-1, 1].
    func momentum(period: Int = 10) -> Double {
        guard period > 0, values.count >= period + 1 else { return 0.0 }
        let n = values.count
        return values[n - 1] - values[n - 1 - period]
    }

    // === Velocity (momentum acceleration) ===
    /// Change of momentum. Range [-2, 2].
    func velocity(period: Int = 10) -> Double {
        guard period > 0, values.count >= period + 2 else { return 0.0 }
        let n = values.count
        let current = values[n - 1] - values[n - 1 - period]
        let previous = values[n - 2] - values[n - 2 - period]
        return current - previous
    }

    // === Volume Weighted Average ===
    /// Grounding weighted by content-token count.
    func volumeWeightedAverage(period: Int = 15) -> Double {
        guard period > 0, values.count >= period else { return 0.5 }
        let recentValues = values.suffix(period)
        let recentVolumes = volumes.suffix(period)
        let weightedSum = zip(recentValues, recentVolumes).reduce(0.0) { $0 + ($1.0 * Double($1.1)) }
        let totalVolume = recentVolumes.reduce(0, +)
        return totalVolume > 0 ? weightedSum / Double(totalVolume) : 0.5
    }
}
