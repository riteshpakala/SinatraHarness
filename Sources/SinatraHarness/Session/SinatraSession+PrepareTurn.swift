//
//  SinatraSession+PrepareTurn.swift
//  SinatraHarness
//
//  WHAT: Everything that happens before the first token: sweep the band, featurise and
//        encode the retrieved partitions, weigh them from what earlier measurements
//        taught (priors, or the net once it is reliable), and build the injection.
//  PIN:  The band weight multiplies the injection once, in the builder. The prior is the
//        partition's mean steer target, not band-scaled here.
//

import Foundation

extension SinatraSession {

    /// Plan one generation. `dryRun` computes exactly what a real turn would without
    /// recording anything (paired comparisons, previews).
    public func prepareTurn(_ input: TurnInput, mode requestedMode: BiasMode? = nil, dryRun: Bool = false) throws -> InjectionPlan {
        let started = Date()
        let now = input.now
        let owner = input.owner
        let mode = requestedMode ?? configuration.biasMode
        let turnId = UUID()

        // 1. The band.
        var ledger: OwnerLedger
        if dryRun {
            ledger = store.ledger(owner, now: now)
            sweep(&ledger, now: now)
        } else {
            store.update(owner, now: now) { sweep(&$0, now: now) }
            ledger = store.ledger(owner, now: now)
        }

        // 2. The retrieved context — the only text the side model reads.
        var seen = Set<String>()
        let partitions = Array(input.retrieved.filter { seen.insert($0.id).inserted }.prefix(configuration.maxPartitionsPerTurn))
        let tokenized = partitions.map { $0.tokenIds ?? tokenizer.encode($0.text) }
        let rows = tokenized.map { TokenBatch.clip($0, maxTokens: configuration.maxTokensPerPartition) }
        // Hot tokens are counted over everything retrieved, then left out of what is biased
        // and attributed (counting only what survived the filter made them oscillate).
        let contentCounts = rows.map { filter.contentCounts($0, hot: nil, configuration: configuration) }
        let terms = contentCounts.map { counts in
            counts.filter {
                !ledger.hotTokens.isHot(
                    $0.key, ratio: configuration.hotTokenExclusionRatio,
                    minimumPartitions: configuration.hotTokenMinimumPartitions)
            }
        }

        // 3. Features.
        let index = LedgerIndex(ledger: ledger, now: now, configuration: configuration)
        let indicators = IndicatorSeries.features(
            points: ledger.series, periods: ledger.periods, minimumCount: configuration.minimumSeriesForIndicators)
        let scores = Self.normalisedScores(partitions.map(\.score), isDistance: configuration.scoreIsDistance)
        let periodics = TimeFeatures.periodics(now, timeZone: configuration.timeZone)
        let last = ledger.lastMeasured
        let queryGap: Float = ledger.lastTurnAt.map {
            TimeFeatures.lnNorm(seconds: now.timeIntervalSince($0), horizon: configuration.queryGapHorizon)
        } ?? 1

        var statics: [[Float]] = []
        var ageDays: [Double] = []
        var bands: [RelevancyBand] = []
        var bandWeights: [Float] = []
        var relevancy: [Float] = []
        var priors: [LedgerIndex.Stats] = []
        for (rank, partition) in partitions.enumerated() {
            let firstSeenDocument = ledger.firstSeenDocuments[partition.documentId]
            let created = partition.createdAt ?? partition.indexedAt ?? firstSeenDocument ?? now
            let docAge = TimeFeatures.days(now.timeIntervalSince(created))
            let modifiedAge = TimeFeatures.days(now.timeIntervalSince(partition.modifiedAt ?? created))
            let reference = TimeFeatures.recencyReference(partition, firstSeen: firstSeenDocument, now: now)
            let age = TimeFeatures.days(now.timeIntervalSince(reference))
            let band = TimeFeatures.band(ageDays: age, configuration: configuration)
            let bandWeight = TimeFeatures.bandWeight(band, configuration: configuration)
            let rel = Grounding.relevancy(bandWeight: bandWeight, score01: scores[rank])
            let stats = index.partition(partition.id)
            let document = index.document(partition.documentId)
            let firstSeen = ledger.firstSeenPartitions[partition.id]

            var values: [Float] = []
            values.reserveCapacity(FeatureVector.staticCount)
            values.append(TimeFeatures.lnNorm(days: docAge))
            values.append(partition.createdAt != nil ? 1 : 0)
            values.append(TimeFeatures.lnNorm(days: modifiedAge))
            values.append(partition.modifiedAt != nil ? 1 : 0)
            values.append(firstSeen.map { TimeFeatures.lnNorm(days: TimeFeatures.days(now.timeIntervalSince($0))) } ?? 0)
            values.append(firstSeen == nil ? 1 : 0)
            values.append(stats.lastRetrieved.map {
                TimeFeatures.lnNorm(days: TimeFeatures.days(now.timeIntervalSince($0)), horizon: configuration.retentionDays)
            } ?? 1)
            values.append(min(1, Float(stats.retrievals30) / 10))
            values.append(min(1, Float(stats.retrievals7) / 5))
            values.append(stats.steer)
            values.append(min(1, Float(stats.labelled) / 5))
            values.append(document.steer)
            values.append(stats.uptake)
            values.append(tanh(document.drift))
            values.append(partitions.count > 1 ? 1 - Float(rank) / Float(partitions.count - 1) : 1)
            values.append(scores[rank])
            values.append(min(1, Float(rows[rank].count) / Float(max(1, configuration.maxTokensPerPartition))))
            values.append(rel)
            values.append(last?.meanSteer ?? 0)
            values.append(last?.grounding?.grounding ?? 0.5)
            values.append(tanh(last?.grounding?.drift ?? 0))
            values.append(contentsOf: periodics)
            values.append(queryGap)
            statics.append(values.map { $0.finite().rounded4 })
            ageDays.append(age)
            bands.append(band)
            bandWeights.append(bandWeight)
            relevancy.append(rel)
            priors.append(stats.labelled > 0 ? stats : document)
        }
        let relevancyShares = Grounding.normalised(relevancy)
        let relevancyPeak = relevancy.max() ?? 0
        let relevancyWeights = relevancy.map { relevancyPeak > 0 ? $0 / relevancyPeak : 0 }

        // 4. Encode the context (never the prompt).
        let encodeStarted = Date()
        var pooled: [[Float]]?
        if let encoder, !rows.isEmpty {
            do {
                pooled = try encoder.encode(TokenBatch(rows: rows))
            } catch {
                log?.log(.warning, "context encoding failed: \(error)")
            }
        }
        let contextValid = pooled != nil
        let contexts: [[Float]] = pooled.map { vectors in
            let projection = contextProjection(inputDimension: encoder?.hiddenSize ?? vectors.first?.count ?? 1)
            return vectors.map { projection.project($0) }
        } ?? Array(repeating: Array(repeating: 0, count: configuration.contextDim), count: partitions.count)
        let encodeMillis = Date().timeIntervalSince(encodeStarted) * 1000

        // 5. Weights: learned where reliable, what earlier measurements taught otherwise.
        let model = weightModel(for: owner)
        let features = statics.map { FeatureVector.assemble(static: $0, indicators: indicators, contextValid: contextValid) }
        let gate: Float = model.isTrained ? ledger.training.reliability : 0
        var prediction: WeightPrediction?
        if gate > 0, !partitions.isEmpty {
            do {
                prediction = try model.predict(features: features, context: contexts)
            } catch {
                log?.log(.warning, "weight model prediction failed: \(error)")
            }
        }
        let effectiveGate: Float = prediction == nil ? 0 : gate
        var weights: [Float] = []
        var priorWeights: [Float] = []
        for p in partitions.indices {
            let prior = priors[p]
            let confidence = min(1, Float(prior.labelled) / 5)
            let priorWeight = configuration.coldStartPrior ? prior.steer * confidence : 0
            let netWeight = prediction?.steer[p] ?? 0
            priorWeights.append(priorWeight)
            weights.append(((1 - effectiveGate) * priorWeight + effectiveGate * netWeight).clamped(-1, 1))
        }
        if let override = input.weightOverride {
            for (p, value) in override.prefix(weights.count).enumerated() { weights[p] = value.clamped(-1, 1) }
        }

        // 6. The injection.
        let buildStarted = Date()
        let partitionIds = partitions.map(\.id)
        var built: (SparseBias, ImpactMask)?
        switch mode {
        case .off:
            built = nil
        case .lexical:
            built = lexical(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights)
        case .dense:
            if let pooled, let logits = try? encoder?.outputLogits(pooled) {
                built = DenseBiasCombiner.build(
                    headLogits: logits, partitionIds: partitionIds, weights: weights, bandWeights: bandWeights,
                    vocabularySize: vocabularySize, configuration: configuration,
                    isContent: { self.filter.isContent($0) })
            } else {
                log?.log(.info, "dense mode unavailable for this encoder; using lexical")
                built = lexical(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights)
            }
        }
        let buildMillis = Date().timeIntervalSince(buildStarted) * 1000
        let bias = built?.0
        var mask = built?.1
        if let ids = mask?.tokenIds {
            mask?.tokenTexts = ids.map { tokenizer.decode([Int($0)]) }
        }
        let weighted = weights.filter { abs($0) >= configuration.minimumWeight }.count
        let meanSteer = weights.isEmpty ? 0 : weights.map(abs).reduce(0, +) / Float(weights.count)

        // 7. Record the turn; it is labelled as soon as its output is measured.
        if !dryRun {
            store.update(owner, now: now) { ledger in
                ledger.turns.append(TurnRecord(
                    id: turnId, at: now, conversationId: input.conversationId, modelKey: modelKey, mode: mode,
                    partitionIds: partitionIds, meanSteer: meanSteer.rounded4,
                    injection: InjectionAggregates(
                        biasMaxAbs: bias?.maxAbs ?? 0, biasNonZero: bias?.nonZero ?? 0, biasL1: bias?.l1 ?? 0,
                        gate: effectiveGate, weightedPartitions: weighted)))
                for (p, partition) in partitions.enumerated() {
                    ledger.events.append(RetrievalEvent(
                        turnId: turnId, at: now, partitionId: partition.id, documentId: partition.documentId,
                        rank: p, staticFeatures: statics[p],
                        context: contextValid ? contexts[p] : nil,
                        contextModelKey: contextValid ? modelKey : nil,
                        relevancy: relevancyShares[p].rounded4, relevancyWeight: relevancyWeights[p].rounded4,
                        priorWeight: priorWeights[p].rounded4,
                        netWeight: prediction.map { $0.steer[p].rounded4 },
                        appliedWeight: weights[p].rounded4, bandWeight: bandWeights[p], label: nil))
                    if ledger.firstSeenPartitions[partition.id] == nil { ledger.firstSeenPartitions[partition.id] = now }
                    if ledger.firstSeenDocuments[partition.documentId] == nil {
                        ledger.firstSeenDocuments[partition.documentId] = partition.indexedAt ?? now
                    }
                    ledger.hotTokens.observe(Set(contentCounts[p].keys))
                }
                ledger.lastTurnAt = now
                ledger.lastBiasMagnitude = bias?.maxAbs ?? 0
            }
        }

        // 8. Diagnostics.
        let diagnostics = TurnDiagnostics(
            turnId: turnId, mode: mode,
            coldStart: ledger.labelledEvents < configuration.minimumLabelledEvents,
            labelledEvents: ledger.labelledEvents, measuredTurns: ledger.measuredTurns,
            observedTurns: ledger.turns.count + (dryRun ? 0 : 1),
            partitions: partitions.indices.map { p in
                PartitionDiagnostic(
                    id: partitions[p].id, documentId: partitions[p].documentId, rank: p,
                    docAgeDays: (ageDays[p] * 100).rounded() / 100, band: bands[p].rawValue,
                    bandWeight: bandWeights[p], relevancy: relevancy[p].rounded4,
                    priorWeight: priorWeights[p], priorUptake: priors[p].uptake,
                    netWeight: prediction?.steer[p], appliedWeight: weights[p],
                    contentTokens: terms[p].count, labelledBefore: priors[p].labelled)
            },
            weightedPartitions: weighted,
            biasNonZero: bias?.nonZero ?? 0, biasMaxAbs: bias?.maxAbs ?? 0, biasL1: bias?.l1 ?? 0,
            gate: effectiveGate,
            forecastUptake: prediction.map { $0.uptake.isEmpty ? 0 : $0.uptake.reduce(0, +) / Float($0.uptake.count) },
            periods: ledger.periods,
            topBiased: (bias?.top(12) ?? []).map {
                BiasedToken(id: $0.id, text: tokenizer.decode([$0.id]), bias: $0.bias)
            },
            encodeMillis: encodeMillis, buildMillis: buildMillis, dryRun: dryRun,
            features: features)
        log?.log(.debug, "prepareTurn \(owner) \(partitions.count) partitions, bias \(bias?.nonZero ?? 0) tokens in \(Int(Date().timeIntervalSince(started) * 1000)) ms")

        var weightMap: [String: Float] = [:]
        for (p, id) in partitionIds.enumerated() { weightMap[id] = weights[p] }
        let partitionTerms = partitions.indices.map { p in
            PartitionTerms(
                id: partitions[p].id, documentId: partitions[p].documentId, counts: terms[p],
                sequence: Array(tokenized[p].prefix(configuration.maxPartitionSequence)),
                relevancyShare: relevancyShares[p], relevancyWeight: relevancyWeights[p],
                appliedWeight: weights[p])
        }
        return InjectionPlan(
            turnId: turnId, owner: owner, mode: mode, bias: bias, mask: mask,
            perPartitionWeights: weightMap, gate: effectiveGate, diagnostics: diagnostics, dryRun: dryRun,
            terms: partitionTerms)
    }

    func lexical(partitionIds: [String], terms: [[Int: Int]], weights: [Float], bandWeights: [Float]) -> (SparseBias, ImpactMask)? {
        LexicalBiasBuilder.build(
            .init(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights),
            vocabularySize: vocabularySize, configuration: configuration,
            isExcluded: { self.filter.isControl($0) })
    }

    /// Min–max to [0, 1] within the turn, 1 = best. Equal scores → 0.5.
    static func normalisedScores(_ scores: [Float], isDistance: Bool) -> [Float] {
        guard let lo = scores.min(), let hi = scores.max(), hi > lo else {
            return Array(repeating: 0.5, count: scores.count)
        }
        return scores.map { value in
            let unit = (value - lo) / (hi - lo)
            return isDistance ? 1 - unit : unit
        }
    }
}

/// Per-partition and per-document statistics over the band, built once per turn from what
/// earlier measurements taught.
struct LedgerIndex {
    struct Stats {
        var retrievals30 = 0
        var retrievals7 = 0
        var lastRetrieved: Date?
        /// Labelled events behind the means below.
        var labelled = 0
        /// Mean steer target y_p.
        var steer: Float = 0
        /// Mean uptake.
        var uptake: Float = 0
        /// Mean drift (nats) of the turns it was retrieved into.
        var drift: Float = 0
    }

    private var partitions: [String: Stats] = [:]
    private var documents: [String: Stats] = [:]

    init(ledger: OwnerLedger, now: Date, configuration: SinatraConfiguration) {
        let horizon30 = now.addingTimeInterval(-configuration.retentionDays * 86_400)
        let horizon7 = now.addingTimeInterval(-configuration.freshBandDays * 86_400)
        struct Sums { var steer: Float = 0; var uptake: Float = 0; var drift: Float = 0; var n = 0 }
        var partitionSums: [String: Sums] = [:]
        var documentSums: [String: Sums] = [:]
        for event in ledger.events where event.at >= horizon30 {
            var stats = partitions[event.partitionId] ?? Stats()
            stats.retrievals30 += 1
            if event.at >= horizon7 { stats.retrievals7 += 1 }
            if stats.lastRetrieved.map({ event.at > $0 }) ?? true { stats.lastRetrieved = event.at }
            partitions[event.partitionId] = stats
            var documentStats = documents[event.documentId] ?? Stats()
            documentStats.retrievals30 += 1
            if event.at >= horizon7 { documentStats.retrievals7 += 1 }
            if documentStats.lastRetrieved.map({ event.at > $0 }) ?? true { documentStats.lastRetrieved = event.at }
            documents[event.documentId] = documentStats
            guard let label = event.label else { continue }
            partitionSums[event.partitionId, default: Sums()].steer += label.target
            partitionSums[event.partitionId, default: Sums()].uptake += label.uptake
            partitionSums[event.partitionId, default: Sums()].drift += label.turnDrift
            partitionSums[event.partitionId, default: Sums()].n += 1
            documentSums[event.documentId, default: Sums()].steer += label.target
            documentSums[event.documentId, default: Sums()].uptake += label.uptake
            documentSums[event.documentId, default: Sums()].drift += label.turnDrift
            documentSums[event.documentId, default: Sums()].n += 1
        }
        for (id, sum) in partitionSums where sum.n > 0 {
            let n = Float(sum.n)
            partitions[id, default: Stats()].labelled = sum.n
            partitions[id, default: Stats()].steer = sum.steer / n
            partitions[id, default: Stats()].uptake = sum.uptake / n
            partitions[id, default: Stats()].drift = sum.drift / n
        }
        for (id, sum) in documentSums where sum.n > 0 {
            let n = Float(sum.n)
            documents[id, default: Stats()].labelled = sum.n
            documents[id, default: Stats()].steer = sum.steer / n
            documents[id, default: Stats()].uptake = sum.uptake / n
            documents[id, default: Stats()].drift = sum.drift / n
        }
    }

    func partition(_ id: String) -> Stats { partitions[id] ?? Stats() }
    func document(_ id: String) -> Stats { documents[id] ?? Stats() }
}
