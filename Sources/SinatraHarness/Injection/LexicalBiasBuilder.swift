//
//  LexicalBiasBuilder.swift
//  SinatraHarness
//
//  WHAT: v1 injection. Each partition's content tokens get m_p[t] = sqrt(tf)·idf,
//        max-normalised; the bias is α · Σ_p band_p · w_p · m_p, clamped to ±cap nats.
//        A positive w_p pulls decoding toward that partition's vocabulary, a negative one
//        pushes it away. This is the only place the band weight multiplies a steer.
//

import Foundation

enum LexicalBiasBuilder {

    struct Input {
        var partitionIds: [String]
        /// Content-token term frequencies per partition.
        var terms: [[Int: Int]]
        var weights: [Float]
        var bandWeights: [Float]
    }

    static func build(
        _ input: Input, vocabularySize: Int, configuration: SinatraConfiguration,
        isExcluded: (Int) -> Bool
    ) -> (SparseBias, ImpactMask)? {
        let sets = input.terms.map { Set($0.keys) }
        let idf = Grounding.idf(partitions: sets)
        var perPartition: [[Int: Float]] = []
        for (p, terms) in input.terms.enumerated() {
            let weight = input.weights[p]
            guard abs(weight) >= configuration.minimumWeight, !terms.isEmpty else {
                perPartition.append([:])
                continue
            }
            var mass: [Int: Float] = [:]
            var peak: Float = 0
            for (token, tf) in terms where token < vocabularySize && !isExcluded(token) {
                let m = sqrt(Float(tf)) * (idf[token] ?? 1)
                mass[token] = m
                peak = max(peak, m)
            }
            guard peak > 0 else {
                perPartition.append([:])
                continue
            }
            let band = configuration.bandPrior ? input.bandWeights[p] : 1
            let scale = configuration.alpha * band * weight / peak
            perPartition.append(mass.mapValues { $0 * scale })
        }
        return BiasAssembly.assemble(
            perPartition: perPartition, partitionIds: input.partitionIds,
            vocabularySize: vocabularySize, cap: configuration.cap)
    }
}

/// v2 injection (experimental). Given each partition's pooled embedding projected through
/// the model's own output head, combine them by weight, z-score across the vocabulary and
/// keep the strongest content tokens.
enum DenseBiasCombiner {

    static func build(
        headLogits: [[Float]], partitionIds: [String], weights: [Float], bandWeights: [Float],
        vocabularySize: Int, configuration: SinatraConfiguration, isContent: (Int) -> Bool
    ) -> (SparseBias, ImpactMask)? {
        let p = headLogits.count
        guard p > 0, p == weights.count else { return nil }
        let v = min(vocabularySize, headLogits.map(\.count).min() ?? 0)
        guard v > 0 else { return nil }

        var coefficients = [Float](repeating: 0, count: p)
        var means = [Float](repeating: 0, count: p)
        for i in 0..<p {
            let band = configuration.bandPrior ? bandWeights[i] : 1
            coefficients[i] = abs(weights[i]) >= configuration.minimumWeight ? weights[i] * band : 0
            means[i] = headLogits[i].prefix(v).reduce(0, +) / Float(v)
        }
        guard coefficients.contains(where: { $0 != 0 }) else { return nil }

        var combined = [Float](repeating: 0, count: v)
        for i in 0..<p where coefficients[i] != 0 {
            let c = coefficients[i]
            let m = means[i]
            let row = headLogits[i]
            for t in 0..<v { combined[t] += c * (row[t] - m) }
        }
        let mean = combined.reduce(0, +) / Float(v)
        let variance = combined.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(v)
        let sigma = sqrt(max(variance, 1e-12))

        var chosen: [Int] = []
        chosen.reserveCapacity(configuration.denseTopK)
        for t in (0..<v).sorted(by: { abs(combined[$0]) > abs(combined[$1]) }) {
            if chosen.count >= configuration.denseTopK { break }
            if isContent(t) { chosen.append(t) }
        }
        var perPartition = [[Int: Float]](repeating: [:], count: p)
        for t in chosen {
            for i in 0..<p where coefficients[i] != 0 {
                perPartition[i][t] = configuration.alphaDense * coefficients[i] * (headLogits[i][t] - means[i]) / sigma
            }
        }
        return BiasAssembly.assemble(
            perPartition: perPartition, partitionIds: partitionIds,
            vocabularySize: vocabularySize, cap: configuration.cap)
    }
}
