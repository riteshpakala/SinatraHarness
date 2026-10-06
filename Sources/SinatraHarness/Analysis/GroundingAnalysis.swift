//
//  GroundingAnalysis.swift
//  SinatraHarness
//
//  WHAT: Grounding over time, for one owner. For every measured turn it sets what the
//        injection did (bias size, gain, divergence) against what the context did to the
//        output (grounding, drift, hallucination risk), and answers:
//          - does steering reduce drift?        r(biasL1, drift), steered vs unsteered bins
//          - is the steer getting more precise? first vs second half of the band
//          - which documents does the output actually cite?   attribution per document
//          - does sharper context mean better grounding?     r(contextKL, grounding), r(entropy, grounding)
//

import Foundation

public struct GroundingReport: Codable, Sendable {
    public struct Row: Codable, Sendable, Equatable {
        public var turnId: UUID
        public var at: Date
        public var mode: BiasMode
        public var steered: Bool
        public var biasL1: Float
        public var meanSteer: Float
        public var grounding: Float
        public var drift: Float
        public var driftShare: Float
        public var contextDependence: Float
        public var meanContextKL: Float
        public var hallucinationRisk: Float
        public var parrotShare: Float
        public var entropyCtx: Float
        public var entropyBare: Float
        public var contentTokens: Int
        /// The injection layer, when the turn was traced.
        public var injectionGain: Float?
        public var injectionKL: Float?
        public var divergenceRate: Float?
    }

    public struct Correlation: Codable, Sendable, Equatable {
        public var x: String
        public var y: String
        /// Pearson r; nil when undefined (n < 3 or no variance).
        public var pearson: Double?
        public var n: Int
    }

    public struct Bin: Codable, Sendable, Equatable {
        public var label: String
        public var count: Int
        public var meanGrounding: Double?
        public var meanDrift: Double?
        public var meanRisk: Double?
    }

    /// How much of the output each document accounts for, across the band.
    public struct Citation: Codable, Sendable, Equatable {
        public var documentId: String
        public var turns: Int
        public var nats: Double
        public var meanUptake: Double
        public var meanCoverage: Double
        public var meanParrot: Double
    }

    public var owner: String
    public var rows: [Row]
    public var correlations: [Correlation]
    public var bins: [Bin]
    public var citations: [Citation]
    /// The turns whose output strayed furthest from where their context pointed.
    public var mostDrifted: [Row]
    /// Why unmeasured turns went unmeasured, and how often.
    public var unmeasured: [String: Int]
}

enum GroundingAnalysis {

    static func report(owner: OwnerID, ledger: OwnerLedger) -> GroundingReport {
        let rows: [GroundingReport.Row] = ledger.turns.compactMap { turn in
            guard let g = turn.grounding, g.contentTokens > 0 else { return nil }
            let biasL1 = turn.injection?.biasL1 ?? 0
            return GroundingReport.Row(
                turnId: turn.id, at: turn.at, mode: turn.mode, steered: biasL1 > 0, biasL1: biasL1,
                meanSteer: turn.meanSteer, grounding: g.grounding, drift: g.drift, driftShare: g.driftShare,
                contextDependence: g.contextDependence, meanContextKL: g.meanContextKL,
                hallucinationRisk: g.hallucinationRisk, parrotShare: g.parrotShare,
                entropyCtx: g.meanEntropyCtx, entropyBare: g.meanEntropyBare, contentTokens: g.contentTokens,
                injectionGain: turn.trace?.gain, injectionKL: turn.trace?.kl, divergenceRate: turn.trace?.divergenceRate)
        }

        func correlate(_ x: String, _ fx: (GroundingReport.Row) -> Float?, _ y: String, _ fy: (GroundingReport.Row) -> Float) -> GroundingReport.Correlation {
            let pairs = rows.compactMap { row in fx(row).map { (Double($0), Double(fy(row))) } }
            return GroundingReport.Correlation(x: x, y: y, pearson: pearson(pairs.map(\.0), pairs.map(\.1)), n: pairs.count)
        }
        let correlations = [
            correlate("biasL1", { $0.biasL1 }, "drift", { $0.drift }),
            correlate("biasL1", { $0.biasL1 }, "grounding", { $0.grounding }),
            correlate("injectionGain", { $0.injectionGain }, "drift", { $0.drift }),
            correlate("divergenceRate", { $0.divergenceRate }, "grounding", { $0.grounding }),
            correlate("meanContextKL", { $0.meanContextKL }, "grounding", { $0.grounding }),
            correlate("entropyCtx", { $0.entropyCtx }, "grounding", { $0.grounding }),
            correlate("entropyCtx", { $0.entropyCtx }, "hallucinationRisk", { $0.hallucinationRisk }),
        ]

        func bin(_ label: String, _ members: [GroundingReport.Row]) -> GroundingReport.Bin {
            GroundingReport.Bin(
                label: label, count: members.count,
                meanGrounding: mean(members.map { Double($0.grounding) }),
                meanDrift: mean(members.map { Double($0.drift) }),
                meanRisk: mean(members.map { Double($0.hallucinationRisk) }))
        }
        let ordered = rows.sorted { $0.at < $1.at }
        let half = ordered.count / 2
        let bins = [
            bin("steered", rows.filter(\.steered)),
            bin("unsteered", rows.filter { !$0.steered }),
            bin("first half of the band", Array(ordered.prefix(half))),
            bin("second half of the band", Array(ordered.suffix(ordered.count - half))),
        ]

        struct Sums { var turns = Set<UUID>(); var nats = 0.0; var uptake = 0.0; var coverage = 0.0; var parrot = 0.0; var n = 0 }
        var byDocument: [String: Sums] = [:]
        for event in ledger.events {
            guard let label = event.label else { continue }
            var sums = byDocument[event.documentId] ?? Sums()
            sums.turns.insert(event.turnId)
            sums.nats += Double(label.attributionNats)
            sums.uptake += Double(label.uptake)
            sums.coverage += Double(label.coverage)
            sums.parrot += Double(label.parrot)
            sums.n += 1
            byDocument[event.documentId] = sums
        }
        let citations = byDocument.map { id, s in
            GroundingReport.Citation(
                documentId: id, turns: s.turns.count, nats: s.nats,
                meanUptake: s.uptake / Double(max(1, s.n)), meanCoverage: s.coverage / Double(max(1, s.n)),
                meanParrot: s.parrot / Double(max(1, s.n)))
        }.sorted { $0.nats > $1.nats }

        var unmeasured: [String: Int] = [:]
        for turn in ledger.turns where turn.grounding == nil && turn.isComplete {
            unmeasured[turn.unmeasuredReason ?? "not measured", default: 0] += 1
        }

        return GroundingReport(
            owner: owner.rawValue, rows: rows, correlations: correlations, bins: bins,
            citations: Array(citations.prefix(20)),
            mostDrifted: Array(rows.sorted { $0.drift > $1.drift }.prefix(5)),
            unmeasured: unmeasured)
    }

    static func mean(_ values: [Double]) -> Double? {
        values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
    }

    static func pearson(_ x: [Double], _ y: [Double]) -> Double? {
        let n = min(x.count, y.count)
        guard n >= 3 else { return nil }
        let mx = x.prefix(n).reduce(0, +) / Double(n)
        let my = y.prefix(n).reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for i in 0..<n {
            let dx = x[i] - mx
            let dy = y[i] - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        }
        guard sxx > 0, syy > 0 else { return nil }
        return sxy / (sxx * syy).squareRoot()
    }
}
