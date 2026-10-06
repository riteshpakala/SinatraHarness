//
//  Grounding.swift
//  SinatraHarness
//
//  WHAT: The feedback channel, v2. A turn is judged by what its retrieved context did to
//        the model's own distributions, not by how the user reacted. From the scorer's
//        per-step numbers (Analysis/GroundingScorer.swift) this file derives, in plain Swift:
//
//          class_t   grounded / contradicted / unsupported / function, from ι_t
//          drift_t   max(0, KL_t − ι_t): the context lifted this step by KL_t on average
//                    (KL = E_{p_ctx}[log-ratio]) and the output took less than that
//          A_p       Σ ι⁺_t · share_p(y_t): each partition's citation value in the output
//          missed_p  Σ drift_t · share_p(tune_t): the push toward p the output did not take
//          parrot_p  share of the output copied verbatim from p
//
//        and the steer target the weight model learns:
//
//          y_p = clip(w_p + κ · driftShare · r_p · (intent_p − uptake_p) − λ · parrot_p · max(0, w_p), −1, 1)
//
//        A grounded turn keeps its steer. A drifting turn leans toward the partitions the
//        context pushed for and the output under-used, scaled by how relevant recency and
//        rank make them (r_p). Where the steer itself pushed toward a partition the output
//        then copied verbatim, the push is damped; a model quoting on its own is left alone.
//  PIN:  share_p(v) = tf_p(v) / Σ_q tf_q(v): the IDF factor is the same for every partition
//        holding v, so it cancels. Tokens are matched by their text, trimmed and lowercased,
//        so " Decoding" in the output counts for "decoding" in a partition. Tokens in no
//        partition go to `unattributed`.
//

import Foundation

enum Grounding {

    // MARK: Per step

    static func classify(influence: Float, isContent: Bool, threshold: Float) -> GroundingClass {
        guard isContent else { return .function }
        if influence >= threshold { return .grounded }
        if influence <= -threshold { return .contradicted }
        return .unsupported
    }

    // MARK: Attribution

    /// Smoothed IDF over the turn's partitions: ln((1 + P) / (1 + df)) + 1.
    static func idf(partitions: [Set<Int>]) -> [Int: Float] {
        var df: [Int: Int] = [:]
        for set in partitions {
            for token in set { df[token, default: 0] += 1 }
        }
        let p = Float(partitions.count)
        return df.mapValues { log((1 + p) / (1 + Float($0))) + 1 }
    }

    /// For every token in any partition: which partitions hold it, and each one's share.
    static func shares(_ partitions: [[Int: Int]]) -> [Int: [(partition: Int, share: Float)]] {
        var totals: [Int: Int] = [:]
        for counts in partitions {
            for (token, tf) in counts { totals[token, default: 0] += tf }
        }
        var out: [Int: [(partition: Int, share: Float)]] = [:]
        for (p, counts) in partitions.enumerated() {
            for (token, tf) in counts {
                guard let total = totals[token], total > 0 else { continue }
                out[token, default: []].append((p, Float(tf) / Float(total)))
            }
        }
        return out
    }

    /// Output positions covered by a verbatim run of at least `n` tokens found in `source`.
    static func copiedPositions(output: [Int], source: [Int], n: Int) -> Set<Int> {
        guard n > 0, output.count >= n, source.count >= n else { return [] }
        var grams = Set<ArraySlice<Int>>()
        grams.reserveCapacity(source.count - n + 1)
        for i in 0...(source.count - n) { grams.insert(source[i..<(i + n)]) }
        var covered = Set<Int>()
        for i in 0...(output.count - n) where grams.contains(output[i..<(i + n)]) {
            for j in i..<(i + n) { covered.insert(j) }
        }
        return covered
    }

    // MARK: Relevancy

    /// rel_p: the band weight scaled by where retrieval ranked the partition.
    static func relevancy(bandWeight: Float, score01: Float) -> Float {
        bandWeight * (0.5 + 0.5 * score01.clamped(0, 1))
    }

    /// Shares summing to 1 (uniform when everything is zero).
    static func normalised(_ values: [Float]) -> [Float] {
        let total = values.reduce(0, +)
        guard total > 0 else { return values.isEmpty ? [] : Array(repeating: 1 / Float(values.count), count: values.count) }
        return values.map { $0 / total }
    }

    // MARK: Steer target

    /// y_p: what the steer on a partition should have been, given how the turn went.
    static func steerTarget(
        applied: Float, relevancyWeight: Float, intent: Float, uptake: Float, driftShare: Float,
        parrot: Float, configuration: SinatraConfiguration
    ) -> Float {
        let correction = configuration.steerGain * driftShare * relevancyWeight * (intent - uptake)
        let damping = configuration.parrotPenalty * parrot * max(0, applied)
        return (applied + correction - damping).clamped(-1, 1)
    }

    /// Training weight of a labelled event: 1 inside the fresh band, decayed to the retention edge.
    static func sampleWeight(ageDays: Double, configuration: SinatraConfiguration) -> Float {
        if ageDays > configuration.retentionDays { return 0 }
        return ageDays <= configuration.freshBandDays ? 1 : configuration.midBandSampleDecay
    }

    static func wordCount(_ text: String) -> Int {
        var count = 0
        var inWord = false
        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                if !inWord { count += 1; inWord = true }
            } else {
                inWord = false
            }
        }
        return count
    }

    // MARK: The measurement

    /// Everything the turn's measurement says, from the scorer's per-step numbers.
    static func measurement(
        raw: GroundingRaw, partitions: [PartitionTerms], isContent: (Int) -> Bool,
        decode: (Int) -> String?, detail: TraceLevel, configuration: SinatraConfiguration
    ) -> GroundingMeasurement {
        let n = raw.tokens.count
        // Match tokens by their normalised text: case and leading-space variants are one word.
        var keys: [Int: String] = [:]
        func key(_ id: Int) -> String {
            if let known = keys[id] { return known }
            let text = decode(id)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
            let made = text.isEmpty ? "#\(id)" : text
            keys[id] = made
            return made
        }
        var keyIds: [String: Int] = [:]
        func keyId(_ id: Int) -> Int {
            let k = key(id)
            if let known = keyIds[k] { return known }
            let made = keyIds.count
            keyIds[k] = made
            return made
        }
        let keyedCounts: [[Int: Int]] = partitions.map { partition in
            var counts: [Int: Int] = [:]
            for (token, tf) in partition.counts { counts[keyId(token), default: 0] += tf }
            return counts
        }
        let shareIndex = shares(keyedCounts)
        var attribution = [Float](repeating: 0, count: partitions.count)
        var missed = [Float](repeating: 0, count: partitions.count)
        var unattributed: Float = 0
        var outputKeys = Set<Int>()

        var content = 0, grounded = 0, unsupported = 0, contradicted = 0, drifting = 0
        var driftSum: Float = 0, riskSum: Float = 0
        var dependence: Float = 0, klSum: Float = 0, hCtx: Float = 0, hBare: Float = 0
        var firstDrift: Int?
        var steps: [GroundingStep] = []
        steps.reserveCapacity(n)

        for t in 0..<n {
            let token = raw.tokens[t]
            let logpCtx = raw.logpCtx[t].finite(-1e4)
            let logpBare = raw.logpBare[t].finite(-1e4)
            let influence = logpCtx - logpBare
            let contentToken = isContent(token)
            let kind = classify(influence: influence, isContent: contentToken, threshold: configuration.groundingThreshold)
            let tuneNats = raw.tuneNats[t].finite()
            let kl = raw.contextKL[t].finite()
            let drift = max(0, kl - influence)
            let risk: Float = kind == .unsupported ? (1 - exp(logpCtx)).clamped(0, 1) : 0

            dependence += influence
            klSum += kl
            hCtx += raw.entropyCtx[t].finite()
            hBare += raw.entropyBare[t].finite()

            if contentToken {
                content += 1
                switch kind {
                case .grounded: grounded += 1
                case .unsupported: unsupported += 1
                case .contradicted: contradicted += 1
                case .function: break
                }
                driftSum += drift
                riskSum += risk
                if drift > configuration.driftThreshold {
                    drifting += 1
                    if firstDrift == nil { firstDrift = t }
                }
                let pushed = max(0, influence)
                outputKeys.insert(keyId(token))
                if let holders = shareIndex[keyId(token)] {
                    for holder in holders { attribution[holder.partition] += pushed * holder.share }
                } else {
                    unattributed += pushed
                }
                if drift > 0, let holders = shareIndex[keyId(raw.tune[t])] {
                    for holder in holders { missed[holder.partition] += drift * holder.share }
                }
            }

            var step = GroundingStep(
                index: t, token: token, text: detail == .off ? nil : decode(token),
                logpCtx: logpCtx, logpBare: logpBare, influence: influence,
                entropyCtx: raw.entropyCtx[t].finite(), entropyBare: raw.entropyBare[t].finite(),
                contextKL: kl, tune: raw.tune[t], tuneText: detail == .off ? nil : decode(raw.tune[t]),
                tuneNats: tuneNats, drift: drift, kind: kind, risk: risk, rankBare: nil, pushes: nil)
            if detail == .full {
                step.rankBare = raw.rankBare?[t]
                step.pushes = raw.pushes?[t].map { TokenLogprob(id: $0.id, text: decode($0.id), logprob: $0.nats.finite()) }
            }
            steps.append(step)
        }

        // Verbatim copying, per partition and overall.
        var copied = Set<Int>()
        var parrot = [Float](repeating: 0, count: partitions.count)
        for (p, partition) in partitions.enumerated() {
            let covered = copiedPositions(output: raw.tokens, source: partition.sequence, n: configuration.parrotNGram)
            parrot[p] = n > 0 ? Float(covered.count) / Float(n) : 0
            copied.formUnion(covered)
        }

        let influenceTotal = attribution.reduce(0, +) + unattributed
        let missedTotal = missed.reduce(0, +)
        let records: [PartitionAttribution] = partitions.enumerated().map { p, partition in
            let vocabulary = Set(keyedCounts[p].keys)
            return PartitionAttribution(
                partitionId: partition.id, documentId: partition.documentId,
                nats: attribution[p].rounded4,
                uptake: influenceTotal > 1e-6 ? (attribution[p] / influenceTotal).rounded4 : 0,
                missed: missed[p].rounded4,
                intent: missedTotal > 1e-6 ? (missed[p] / missedTotal).rounded4 : 0,
                coverage: vocabulary.isEmpty ? 0 : Float(vocabulary.intersection(outputKeys).count) / Float(vocabulary.count),
                parrot: parrot[p].rounded4,
                relevancy: partition.relevancyShare)
        }

        let c = Float(max(content, 1))
        let all = Float(max(n, 1))
        let summary = GroundingSummary(
            steps: n, contentTokens: content,
            grounding: content > 0 ? Float(grounded) / c : 0,
            unsupportedShare: content > 0 ? Float(unsupported) / c : 0,
            contradictedShare: content > 0 ? Float(contradicted) / c : 0,
            drift: content > 0 ? driftSum / c : 0,
            driftShare: content > 0 ? Float(drifting) / c : 0,
            firstDriftStep: firstDrift,
            contextDependence: dependence,
            meanContextKL: n > 0 ? klSum / all : 0,
            meanEntropyCtx: n > 0 ? hCtx / all : 0,
            meanEntropyBare: n > 0 ? hBare / all : 0,
            hallucinationRisk: content > 0 ? riskSum / c : 0,
            parrotShare: n > 0 ? Float(copied.count) / all : 0,
            unattributed: unattributed)

        return GroundingMeasurement(
            measured: true, skippedReason: nil, cacheReused: raw.cacheReused,
            promptTokens: raw.promptTokens, bareTokens: raw.bareTokens,
            sharedPrefixTokens: raw.sharedPrefixTokens, prefillMillis: raw.prefillMillis,
            scoreMillis: raw.scoreMillis, detail: detail == .off ? .summary : detail,
            summary: summary, attribution: records, steps: detail == .off ? [] : steps)
    }
}
