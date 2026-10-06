//
//  IndicatorPeriods.swift
//  SinatraHarness
//
//  WHAT: The 11 lookback windows of the technical indicators — IMBHS's decision vector.
//  PIN:  Ported verbatim from Sewn's Sinatra (Sources/Sinatra/ML/IndicatorPeriods.swift):
//        same defaults, bounds and joint feasibility rules. Windows count labelled turns,
//        and the 30-day retention bounds how many there can be.
//

import Foundation

public struct IndicatorPeriods: Codable, Equatable, Hashable, Sendable {
    // Level indicators
    public var emaPeriod: Int         // default 10
    public var smaPeriod: Int         // default 20

    // EMA-momentum indicators
    public var macdFast: Int          // default 5
    public var macdSlow: Int          // default 15
    public var macdSignalPeriod: Int  // default 9

    // Oscillator indicators
    public var stochKPeriod: Int      // default 14
    public var stochDSignal: Int      // default 3

    // Raw differential indicators
    public var momentumPeriod: Int    // default 10
    public var velocityPeriod: Int    // default 10

    // Volume indicators
    public var avgVolPeriod: Int      // default 10
    public var vwaPeriod: Int         // default 15

    public static let `default` = IndicatorPeriods(
        emaPeriod: 10, smaPeriod: 20,
        macdFast: 5, macdSlow: 15, macdSignalPeriod: 9,
        stochKPeriod: 14, stochDSignal: 3,
        momentumPeriod: 10, velocityPeriod: 10,
        avgVolPeriod: 10, vwaPeriod: 15
    )

    /// Per-parameter (min, max), in `asArray` order. Derived from the 20-point history
    /// minimum and the joint constraints.
    public static let bounds: [(min: Int, max: Int)] = [
        (5, 20),   // emaPeriod
        (10, 20),  // smaPeriod
        (3, 10),   // macdFast
        (10, 18),  // macdSlow      (must > macdFast)
        (3, 15),   // macdSignalPeriod
        (5, 16),   // stochKPeriod  (joint: stochKPeriod + stochDSignal - 1 ≤ 20)
        (2, 5),    // stochDSignal
        (3, 18),   // momentumPeriod (period + 2 ≤ 20)
        (3, 18),   // velocityPeriod (period + 2 ≤ 20)
        (5, 15),   // avgVolPeriod  (period + 1 ≤ 20)
        (5, 20),   // vwaPeriod
    ]

    public var asArray: [Int] {
        [emaPeriod, smaPeriod, macdFast, macdSlow, macdSignalPeriod,
         stochKPeriod, stochDSignal, momentumPeriod, velocityPeriod,
         avgVolPeriod, vwaPeriod]
    }

    public init(fromArray array: [Int]) {
        precondition(array.count == Self.bounds.count, "IndicatorPeriods needs \(Self.bounds.count) values")
        emaPeriod        = array[0]
        smaPeriod        = array[1]
        macdFast         = array[2]
        macdSlow         = array[3]
        macdSignalPeriod = array[4]
        stochKPeriod     = array[5]
        stochDSignal     = array[6]
        momentumPeriod   = array[7]
        velocityPeriod   = array[8]
        avgVolPeriod     = array[9]
        vwaPeriod        = array[10]
    }

    public init(
        emaPeriod: Int, smaPeriod: Int,
        macdFast: Int, macdSlow: Int, macdSignalPeriod: Int,
        stochKPeriod: Int, stochDSignal: Int,
        momentumPeriod: Int, velocityPeriod: Int,
        avgVolPeriod: Int, vwaPeriod: Int
    ) {
        self.emaPeriod = emaPeriod
        self.smaPeriod = smaPeriod
        self.macdFast = macdFast
        self.macdSlow = macdSlow
        self.macdSignalPeriod = macdSignalPeriod
        self.stochKPeriod = stochKPeriod
        self.stochDSignal = stochDSignal
        self.momentumPeriod = momentumPeriod
        self.velocityPeriod = velocityPeriod
        self.avgVolPeriod = avgVolPeriod
        self.vwaPeriod = vwaPeriod
    }

    /// "ema10/sma20 macd5·15·9 stoch14·3 mom10/vel10 vol10/vwa15"
    public var logDescription: String {
        "ema\(emaPeriod)/sma\(smaPeriod) macd\(macdFast)·\(macdSlow)·\(macdSignalPeriod) stoch\(stochKPeriod)·\(stochDSignal) mom\(momentumPeriod)/vel\(velocityPeriod) vol\(avgVolPeriod)/vwa\(vwaPeriod)"
    }

    /// A copy with every bound and joint constraint enforced:
    /// `macdFast < macdSlow` and `stochKPeriod + stochDSignal - 1 ≤ 20`.
    public func feasible() -> IndicatorPeriods {
        var p = self
        p.emaPeriod        = p.emaPeriod.clamped(5, 20)
        p.smaPeriod        = p.smaPeriod.clamped(10, 20)
        p.macdFast         = p.macdFast.clamped(3, 10)
        p.macdSlow         = p.macdSlow.clamped(10, 18)
        p.macdSignalPeriod = p.macdSignalPeriod.clamped(3, 15)
        p.stochKPeriod     = p.stochKPeriod.clamped(5, 16)
        p.stochDSignal     = p.stochDSignal.clamped(2, 5)
        p.momentumPeriod   = p.momentumPeriod.clamped(3, 18)
        p.velocityPeriod   = p.velocityPeriod.clamped(3, 18)
        p.avgVolPeriod     = p.avgVolPeriod.clamped(5, 15)
        p.vwaPeriod        = p.vwaPeriod.clamped(5, 20)

        if p.macdFast >= p.macdSlow {
            p.macdFast = Swift.max(3, p.macdSlow - 1)
        }
        if p.stochKPeriod + p.stochDSignal - 1 > 20 {
            p.stochKPeriod = Swift.max(5, 20 - p.stochDSignal + 1)
        }
        return p
    }
}
