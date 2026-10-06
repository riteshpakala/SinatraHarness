//
//  TrainingDataset.swift
//  SinatraHarness
//
//  WHAT: Turns an owner's labelled events into weight-model rows, rebuilding the indicator
//        features as of each event's turn under any `IndicatorPeriods` — which is what lets
//        IMBHS score candidate windows against the current model.
//  PIN:  Targets are the steer targets y_p the grounding measurement produced
//        (`Grounding.steerTarget`), already corrections in [−1, 1]: nothing is centred here.
//

import Foundation

struct TrainingDataset {
    var batch: TrainingBatch
    /// Turn id of each row, rows in chronological order.
    var rowTurns: [UUID]

    static func build(
        ledger: OwnerLedger, periods: IndicatorPeriods, modelKey: String, now: Date,
        configuration: SinatraConfiguration
    ) -> TrainingDataset {
        let horizon = now.addingTimeInterval(-configuration.retentionDays * 86_400)
        let rows = ledger.events.indices
            .filter { ledger.events[$0].label != nil && ledger.events[$0].at >= horizon }
            .sorted { ledger.events[$0].at < ledger.events[$1].at }

        var indicatorCache: [Date: [Float]] = [:]
        var features: [[Float]] = []
        var context: [[Float]] = []
        var targets: [Float] = []
        var weights: [Float] = []
        var uptakes: [Float] = []
        var turns: [UUID] = []
        let zeros = [Float](repeating: 0, count: configuration.contextDim)

        for index in rows {
            let event = ledger.events[index]
            guard let label = event.label, event.staticFeatures.count == FeatureVector.staticCount else { continue }
            let ageDays = TimeFeatures.days(now.timeIntervalSince(event.at))
            let weight = Grounding.sampleWeight(ageDays: ageDays, configuration: configuration)
            guard weight > 0 else { continue }
            let indicators: [Float]
            if let cached = indicatorCache[event.at] {
                indicators = cached
            } else {
                indicators = IndicatorSeries.features(
                    points: IndicatorSeries.asOf(ledger.series, event.at), periods: periods,
                    minimumCount: configuration.minimumSeriesForIndicators)
                indicatorCache[event.at] = indicators
            }
            let valid = event.contextModelKey == modelKey && event.context?.count == configuration.contextDim
            features.append(FeatureVector.assemble(static: event.staticFeatures, indicators: indicators, contextValid: valid))
            context.append(valid ? event.context! : zeros)
            targets.append(label.target)
            uptakes.append(label.uptake)
            weights.append(weight)
            turns.append(event.turnId)
        }
        return TrainingDataset(
            batch: TrainingBatch(features: features, context: context, targets: targets, sampleWeights: weights, uptakes: uptakes),
            rowTurns: turns)
    }

    /// Chronological split by turn: the most recent `fraction` of turns is held out.
    func split(holdoutFraction: Double) -> (train: TrainingBatch, holdout: TrainingBatch) {
        var order: [UUID] = []
        var seen = Set<UUID>()
        for turn in rowTurns where seen.insert(turn).inserted { order.append(turn) }
        guard order.count >= 2 else { return (batch, batch.subset([Int]())) }
        let holdoutTurns = max(1, Int((Double(order.count) * holdoutFraction).rounded()))
        let held = Set(order.suffix(min(holdoutTurns, order.count - 1)))
        var train: [Int] = []
        var holdout: [Int] = []
        for (row, turn) in rowTurns.enumerated() {
            if held.contains(turn) { holdout.append(row) } else { train.append(row) }
        }
        return (batch.subset(train), batch.subset(holdout))
    }
}
