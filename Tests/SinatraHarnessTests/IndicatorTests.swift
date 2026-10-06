import Foundation
import Testing
@testable import SinatraHarness

@Suite("Technical indicators")
struct IndicatorTests {
    let values = [0.2, 0.4, 0.6, 0.8, 1.0]
    let base = Date(timeIntervalSince1970: 1_000_000)

    var indicators: TechnicalIndicators {
        TechnicalIndicators(
            values: values,
            timestamps: [0, 10, 20, 30, 60].map { base.addingTimeInterval($0) },
            volumes: [1, 1, 1, 2, 2])
    }

    @Test func emaAndSma() {
        #expect(abs(indicators.emaWA(period: 5, alpha: 0.3) - 0.64538) < 1e-9)
        #expect(abs(indicators.smaWA(period: 5) - 0.6) < 1e-12)
    }

    @Test func insufficientHistoryIsNeutral() {
        #expect(indicators.emaWA(period: 10) == 0.5)
        #expect(indicators.smaWA(period: 10) == 0.5)
        #expect(indicators.stochasticK(period: 10) == 0.5)
        #expect(indicators.momentum(period: 10) == 0)
        #expect(indicators.velocity(period: 10) == 0)
        #expect(TechnicalIndicators.macDSignal(macdHistory: [0.1], signalPeriod: 9) == 0)
    }

    @Test func momentumVelocityStochastic() {
        #expect(abs(indicators.momentum(period: 2) - 0.4) < 1e-12)
        #expect(abs(indicators.velocity(period: 2)) < 1e-12)
        #expect(indicators.stochasticK(period: 5) == 1)
    }

    @Test func volumeWeightedAverage() {
        #expect(abs(indicators.volumeWeightedAverage(period: 3) - 0.84) < 1e-12)
    }

    @Test func averageVolumeChangeUsesIntervals() {
        // recent intervals [10, 30] → 20 s; all intervals → 15 s; (15 − 20) / 15
        #expect(abs(indicators.avgVolChange(period: 2) - (-1.0 / 3.0)) < 1e-9)
    }

    @Test func seriesFeaturesAreNeutralBelowTheMinimum() {
        let points = values.enumerated().map { SeriesPoint(at: base.addingTimeInterval(Double($0.offset)), value: $0.element, volume: 5) }
        #expect(IndicatorSeries.features(points: points, periods: .default, minimumCount: 20) == IndicatorSeries.neutral)
        let features = IndicatorSeries.features(points: points, periods: .default, minimumCount: 1)
        #expect(features.count == IndicatorSeries.names.count)
        #expect(features.allSatisfy { $0.isFinite })
    }

    @Test func asOfExcludesLaterPoints() {
        let points = (0..<5).map { SeriesPoint(at: base.addingTimeInterval(Double($0) * 10), value: 0.5, volume: 1) }
        #expect(IndicatorSeries.asOf(points, base.addingTimeInterval(25)).count == 3)
        #expect(IndicatorSeries.asOf(points, base.addingTimeInterval(100)).count == 5)
    }
}

@Suite("IMBHS harmony memory")
struct HarmonyMemoryTests {
    @Test func feasibilityInvariants() {
        let infeasible = IndicatorPeriods(
            emaPeriod: 99, smaPeriod: 1, macdFast: 10, macdSlow: 10, macdSignalPeriod: 0,
            stochKPeriod: 16, stochDSignal: 5, momentumPeriod: 0, velocityPeriod: 40,
            avgVolPeriod: 1, vwaPeriod: 30).feasible()
        #expect(infeasible.macdFast < infeasible.macdSlow)
        #expect(infeasible.stochKPeriod + infeasible.stochDSignal - 1 <= 20)
        for (value, bound) in zip(infeasible.asArray, IndicatorPeriods.bounds) {
            #expect(value >= bound.min && value <= bound.max)
        }
    }

    @Test func seededImprovisationStaysInBounds() {
        var rng = SplitMix64(seed: 42)
        let memory = HarmonyMemory(using: &rng)
        for _ in 0..<200 {
            let candidate = memory.improvise(using: &rng)
            #expect(candidate == candidate.feasible())
        }
        var a = SplitMix64(seed: 7)
        var b = SplitMix64(seed: 7)
        #expect(HarmonyMemory(using: &a) == HarmonyMemory(using: &b))
    }

    @Test func scheduleAndCadence() {
        var memory = HarmonyMemory()
        #expect(abs(memory.currentPAR - 0.1) < 1e-12)
        #expect(memory.currentBW == 3)
        #expect(!memory.shouldRun)
        for _ in 0..<20 { memory.incrementGeneration() }
        #expect(memory.shouldRun)
        memory.incrementGeneration()
        #expect(!memory.shouldRun)
        for _ in 0..<179 { memory.incrementGeneration() }
        #expect(abs(memory.currentPAR - 0.5) < 1e-12)
        #expect(memory.currentBW == 1)
    }

    @Test func updateNeedsOnePercent() {
        var rng = SplitMix64(seed: 1)
        var memory = HarmonyMemory(using: &rng)
        memory.recordFitness(.default, fitness: 1.0)
        let near = IndicatorPeriods.default.feasible()
        var slightly = near
        slightly.emaPeriod = 11
        // Replaces the worst (unevaluated) slot but is <1% better than the active: no switch.
        #expect(memory.update(candidate: slightly, candidateFitness: 0.995) == false)
        var better = near
        better.emaPeriod = 12
        #expect(memory.update(candidate: better, candidateFitness: 0.9) == true)
        #expect(memory.activePeriods == better)
    }

    @Test func codableKeepsInfinity() throws {
        let memory = HarmonyMemory()
        let data = try JSONEncoder().encode(memory)
        let decoded = try JSONDecoder().decode(HarmonyMemory.self, from: data)
        #expect(decoded.fitness.allSatisfy { $0 == .infinity })
        #expect(decoded == memory)
    }
}
