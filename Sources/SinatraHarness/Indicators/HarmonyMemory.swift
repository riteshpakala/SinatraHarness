//
//  HarmonyMemory.swift
//  SinatraHarness
//
//  WHAT: IMBHS (Improved Music-Based Harmony Search) over the 11 indicator windows.
//  PIN:  Ported from Sewn's Sinatra (Sources/Sinatra/ML/HarmonyMemory.swift) with two
//        changes: randomness is injectable (deterministic tests), and `recordFitness`
//        lets the caller score the active windows before the first candidate, so the 1%
//        improvement threshold is meaningful from the start instead of the first candidate
//        always replacing the unevaluated default.
//
//  Terminology (IMBHS → Sinatra):
//    Harmony vector  → IndicatorPeriods (11 integer dimensions)
//    Harmony memory  → H = 10 candidate window configurations
//    Generation gn   → one successful training cycle of the weight model
//    Fitness f(h)    → MAE of the current weight model with features rebuilt under h
//    PAR(gn)         → linear 0.1 → 0.5 over NI = 200 cycles
//    BW(gn)          → exponential 3 → 1 over NI = 200 cycles
//

import Foundation

public struct HarmonyMemory: Codable, Sendable, Equatable {

    public static let memorySize = 10
    public static let hmcr: Double = 0.9
    public static let parMin: Double = 0.1
    public static let parMax: Double = 0.5
    public static let bwMax = 3
    public static let bwMin = 1
    public static let ni = 200
    public static let cadence = 5
    public static let warmup = 20
    public static let fitnessThreshold: Double = 0.01

    public private(set) var harmonies: [IndicatorPeriods]
    /// MAE, lower is better; `.infinity` = unevaluated.
    public private(set) var fitness: [Double]
    public private(set) var generation: Int
    public private(set) var activePeriods: IndicatorPeriods

    public init() {
        var rng = SystemRandomNumberGenerator()
        self.init(using: &rng)
    }

    public init<G: RandomNumberGenerator>(using rng: inout G) {
        var h: [IndicatorPeriods] = [.default]
        for _ in 1..<Self.memorySize {
            h.append(Self.randomHarmony(using: &rng))
        }
        harmonies = h
        fitness = Array(repeating: .infinity, count: Self.memorySize)
        generation = 0
        activePeriods = .default
    }

    // MARK: Codable (JSON has no infinity)

    enum CodingKeys: String, CodingKey { case harmonies, fitness, generation, activePeriods }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        harmonies = try c.decode([IndicatorPeriods].self, forKey: .harmonies)
        fitness = try c.decode([Double].self, forKey: .fitness)
            .map { $0 >= Double.greatestFiniteMagnitude ? .infinity : $0 }
        generation = try c.decode(Int.self, forKey: .generation)
        activePeriods = try c.decodeIfPresent(IndicatorPeriods.self, forKey: .activePeriods) ?? .default
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(harmonies, forKey: .harmonies)
        try c.encode(fitness.map { $0.isFinite ? $0 : Double.greatestFiniteMagnitude }, forKey: .fitness)
        try c.encode(generation, forKey: .generation)
        try c.encode(activePeriods, forKey: .activePeriods)
    }

    // MARK: Schedule

    public var currentPAR: Double {
        let gn = Swift.min(generation, Self.ni)
        return Self.parMin + (Self.parMax - Self.parMin) / Double(Self.ni) * Double(gn)
    }

    public var currentBW: Int {
        let gn = Swift.min(generation, Self.ni)
        let c = log(Double(Self.bwMax) / Double(Self.bwMin)) / Double(Self.ni)
        let bw = Double(Self.bwMax) * exp(-c * Double(gn))
        return Swift.max(Self.bwMin, Int(bw.rounded()))
    }

    public var shouldRun: Bool {
        generation >= Self.warmup && generation % Self.cadence == 0
    }

    public mutating func incrementGeneration() {
        generation += 1
    }

    // MARK: Improvisation

    /// With probability HMCR take a value from a random stored harmony (then with
    /// probability PAR nudge it by ±1…BW); otherwise draw uniformly within bounds.
    public func improvise<G: RandomNumberGenerator>(using rng: inout G) -> IndicatorPeriods {
        let par = currentPAR
        let bw = currentBW
        let bounds = IndicatorPeriods.bounds
        var raw = [Int](repeating: 0, count: bounds.count)

        for d in 0..<bounds.count {
            let lo = bounds[d].min
            let hi = bounds[d].max
            if Double.random(in: 0..<1, using: &rng) < Self.hmcr {
                let source = harmonies[Int.random(in: 0..<harmonies.count, using: &rng)].asArray
                var value = source[d]
                if Double.random(in: 0..<1, using: &rng) < par {
                    let sign = Bool.random(using: &rng) ? 1 : -1
                    let step = Int.random(in: 1...bw, using: &rng) * sign
                    value = Swift.min(Swift.max(value + step, lo), hi)
                }
                raw[d] = value
            } else {
                raw[d] = Int.random(in: lo...hi, using: &rng)
            }
        }
        return IndicatorPeriods(fromArray: raw).feasible()
    }

    // MARK: Update

    /// Score a stored harmony (typically the active one) without replacing anything.
    public mutating func recordFitness(_ periods: IndicatorPeriods, fitness value: Double) {
        guard let index = harmonies.firstIndex(of: periods) else { return }
        fitness[index] = value
    }

    /// Replace the worst harmony with `candidate` when strictly better. If the best
    /// harmony changes and improves on the active one by at least 1%, it becomes active.
    /// Returns true when `activePeriods` changed.
    @discardableResult
    public mutating func update(candidate: IndicatorPeriods, candidateFitness: Double) -> Bool {
        guard let worst = fitness.indices.max(by: { fitness[$0] < fitness[$1] }) else { return false }
        guard candidateFitness < fitness[worst] else { return false }

        harmonies[worst] = candidate
        fitness[worst] = candidateFitness

        guard let best = fitness.indices.min(by: { fitness[$0] < fitness[$1] }) else { return false }
        let newBest = harmonies[best]
        let bestFitness = fitness[best]
        guard newBest != activePeriods else { return false }

        let activeFitness = zip(harmonies, fitness).first(where: { $0.0 == activePeriods })?.1
        if let af = activeFitness, af.isFinite, af > 0 {
            let improvement = (af - bestFitness) / af
            guard improvement >= Self.fitnessThreshold else { return false }
        }
        activePeriods = newBest
        return true
    }

    static func randomHarmony<G: RandomNumberGenerator>(using rng: inout G) -> IndicatorPeriods {
        let raw = IndicatorPeriods.bounds.map { Int.random(in: $0.min...$0.max, using: &rng) }
        return IndicatorPeriods(fromArray: raw).feasible()
    }
}
