//
//  SinatraSession+Train.swift
//  SinatraHarness
//
//  WHAT: One training cycle: fit the weight model on the owner's labelled band, score it
//        on the most recent turns against the mean predictor, derive the reliability gate,
//        then let IMBHS propose new indicator windows on its cadence.
//  PIN:  Bounded by `trainingBudget` and `shouldAbort` (a generation is waiting). Never
//        called in front of a generation: the harness schedules it after one finishes.
//

import Foundation

extension SinatraSession {

    public func train(
        owner: OwnerID, budget: TimeInterval? = nil, force: Bool = false,
        now: Date = Date(), shouldAbort: @escaping @Sendable () -> Bool = { false }
    ) throws -> TrainingReport {
        let started = Date()
        let budget = budget ?? configuration.trainingBudget
        var ledger = store.ledger(owner, now: now)
        let labelled = ledger.labelledEvents
        let minimum = force ? 4 : configuration.minimumLabelledEvents
        guard labelled >= minimum else {
            return .skipped(
                "\(labelled) labelled events; \(minimum) needed", owner: owner, rows: labelled,
                reliability: ledger.training.reliability, cycle: ledger.training.cycles, periods: ledger.periods)
        }
        guard force || ledger.training.labelsSinceTrain >= configuration.trainEveryLabelledEvents else {
            return .skipped(
                "\(ledger.training.labelsSinceTrain) new labels since the last cycle", owner: owner, rows: labelled,
                reliability: ledger.training.reliability, cycle: ledger.training.cycles, periods: ledger.periods)
        }

        let model = weightModel(for: owner)
        var dataset = TrainingDataset.build(
            ledger: ledger, periods: ledger.periods, modelKey: modelKey, now: now, configuration: configuration)
        var (trainBatch, holdout) = dataset.split(holdoutFraction: configuration.holdoutFraction)
        guard !trainBatch.isEmpty else {
            return .skipped(
                "no weighted rows in the band", owner: owner, rows: 0,
                reliability: ledger.training.reliability, cycle: ledger.training.cycles, periods: ledger.periods)
        }

        var fit = try model.train(
            trainBatch, budget: budget, maxSteps: configuration.maxTrainingSteps, shouldAbort: shouldAbort)
        var (holdoutMAE, baselineMAE, reliability) = try score(model: model, train: trainBatch, holdout: holdout, labelled: labelled)

        // IMBHS: one improvisation per cadence, scored against the model just fitted.
        ledger.harmony.incrementGeneration()
        ledger.training.cycles += 1
        var harmonyRan = false
        var periodsChanged = false
        if ledger.harmony.shouldRun && !shouldAbort() {
            harmonyRan = true
            try ledger.harmony.recordFitness(ledger.harmony.activePeriods, fitness: Double(model.evaluate(dataset.batch)))
            var rng = SystemRandomNumberGenerator()
            let candidate = ledger.harmony.improvise(using: &rng)
            let candidateRows = TrainingDataset.build(
                ledger: ledger, periods: candidate, modelKey: modelKey, now: now, configuration: configuration)
            let fitness = try model.evaluate(candidateRows.batch)
            if ledger.harmony.update(candidate: candidate, candidateFitness: Double(fitness)) {
                periodsChanged = true
                ledger.periods = ledger.harmony.activePeriods
                log?.log(.info, "IMBHS \(owner): windows → \(ledger.periods.logDescription)")
                dataset = candidateRows
                (trainBatch, holdout) = dataset.split(holdoutFraction: configuration.holdoutFraction)
                let remaining = budget - Date().timeIntervalSince(started)
                if remaining > 0.05 && !shouldAbort() {
                    fit = try model.train(
                        trainBatch, budget: remaining, maxSteps: configuration.maxTrainingSteps, shouldAbort: shouldAbort)
                    (holdoutMAE, baselineMAE, reliability) = try score(
                        model: model, train: trainBatch, holdout: holdout, labelled: labelled)
                }
            }
        }

        let metadata = [
            "schema": (try? String(data: JSONEncoder().encode(schema), encoding: .utf8)) ?? "",
            "modelKey": modelKey,
            "owner": owner.rawValue,
            "trainedAt": ISO8601DateFormatter().string(from: now),
            "labelledEvents": String(labelled),
            "reliability": String(reliability),
            "periods": ledger.periods.logDescription,
        ]
        do {
            try FileManager.default.createDirectory(at: store.ownerDirectory(owner), withIntermediateDirectories: true)
            try model.save(to: store.modelURL(owner, modelKey: modelKey), metadata: metadata)
        } catch {
            log?.log(.error, "saving weight model for \(owner) failed: \(error)")
        }

        let report = TrainingReport(
            owner: owner.rawValue, skipped: nil, rows: dataset.batch.count, trainRows: trainBatch.count,
            holdoutRows: holdout.count, steps: fit.steps, initialLoss: fit.initialLoss, finalLoss: fit.finalLoss,
            holdoutMAE: holdoutMAE, baselineMAE: baselineMAE, reliability: reliability,
            elapsed: Date().timeIntervalSince(started), stoppedBy: fit.stoppedBy,
            cycle: ledger.training.cycles, harmonyRan: harmonyRan, periodsChanged: periodsChanged,
            periods: ledger.periods)
        store.update(owner, now: now) { stored in
            stored.harmony = ledger.harmony
            stored.periods = ledger.periods
            stored.training.cycles = ledger.training.cycles
            stored.training.lastTrainedAt = now
            stored.training.labelsSinceTrain = 0
            stored.training.reliability = model.isTrained ? reliability : 0
            stored.training.holdoutMAE = holdoutMAE
            stored.training.baselineMAE = baselineMAE
            stored.training.lastReport = report
        }
        log?.log(.info, "trained \(owner): \(fit.steps) steps, holdout MAE \(holdoutMAE.map { String(format: "%.3f", $0) } ?? "-") vs \(baselineMAE.map { String(format: "%.3f", $0) } ?? "-"), g=\(String(format: "%.2f", reliability))")
        return report
    }

    /// Holdout MAE, the mean predictor's MAE, and the reliability gate g.
    func score(model: any WeightModeling, train: TrainingBatch, holdout: TrainingBatch, labelled: Int) throws -> (Float?, Float?, Float) {
        guard !holdout.isEmpty else { return (nil, nil, 0) }
        let mae = try model.evaluate(holdout)
        let baseline = holdout.constantMAE(train.weightedMeanTarget)
        let skill = baseline > 1e-6 ? 1 - mae / baseline : 0
        let ramp = configuration.reliabilityRamp
        let progress = Float((Double(labelled) - ramp.lowerBound) / max(1, ramp.upperBound - ramp.lowerBound)).clamped(0, 1)
        let reliability = (skill / configuration.reliabilitySkillForFull).clamped(0, 1) * progress
        return (mae, baseline, reliability)
    }
}

/// One training row as the current weight model sees it (inspection).
public struct EvaluationRow: Sendable, Codable {
    public var turnId: UUID
    public var at: Date
    public var partitionId: String
    public var holdout: Bool
    public var target: Float
    public var prediction: Float
    public var sampleWeight: Float
    public var features: [Float]
}

extension SinatraSession {
    /// Every row of the owner's current dataset with the model's prediction and whether the
    /// last training split held it out.
    public func evaluationRows(owner: OwnerID, now: Date = Date()) throws -> [EvaluationRow] {
        let ledger = store.ledger(owner, now: now)
        let dataset = TrainingDataset.build(
            ledger: ledger, periods: ledger.periods, modelKey: modelKey, now: now, configuration: configuration)
        let model = weightModel(for: owner)
        let predictions = try model.predict(features: dataset.batch.features, context: dataset.batch.context).steer
        var order: [UUID] = []
        var seen = Set<UUID>()
        for turn in dataset.rowTurns where seen.insert(turn).inserted { order.append(turn) }
        let holdoutTurns = max(1, Int((Double(order.count) * configuration.holdoutFraction).rounded()))
        let held = Set(order.suffix(min(holdoutTurns, max(0, order.count - 1))))
        let partitionByRow = ledger.events
            .filter { $0.label != nil && $0.staticFeatures.count == FeatureVector.staticCount }
            .sorted { $0.at < $1.at }
        return dataset.batch.targets.indices.map { i in
            EvaluationRow(
                turnId: dataset.rowTurns[i], at: i < partitionByRow.count ? partitionByRow[i].at : now,
                partitionId: i < partitionByRow.count ? partitionByRow[i].partitionId : "?",
                holdout: held.contains(dataset.rowTurns[i]), target: dataset.batch.targets[i],
                prediction: i < predictions.count ? predictions[i] : 0,
                sampleWeight: dataset.batch.sampleWeights[i], features: dataset.batch.features[i])
        }
    }
}
