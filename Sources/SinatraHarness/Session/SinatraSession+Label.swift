//
//  SinatraSession+Label.swift
//  SinatraHarness
//
//  WHAT: The feedback channel. A turn is labelled the moment its grounding is measured:
//        each retrieved partition gets a steer target from what the context pushed for,
//        what the output took, how relevant recency made the partition, and whether the
//        output copied it. Nothing waits for the user's next message. The 30-day band and
//        the caps are enforced here too.
//

import Foundation

extension SinatraSession {

    /// Enforce the band and caps; drop turns whose generation never finished.
    func sweep(_ ledger: inout OwnerLedger, now: Date) {
        let abandoned = Set(ledger.turns.lazy
            .filter { !$0.isComplete && now.timeIntervalSince($0.at) > self.configuration.abandonedAfter }
            .map(\.id))
        if !abandoned.isEmpty {
            ledger.turns.removeAll { abandoned.contains($0.id) }
            ledger.events.removeAll { abandoned.contains($0.turnId) }
        }

        let horizon = now.addingTimeInterval(-configuration.retentionDays * 86_400)
        ledger.turns.removeAll { $0.at < horizon }
        ledger.events.removeAll { $0.at < horizon }
        ledger.series.removeAll { $0.at < horizon }

        if ledger.turns.count > configuration.maxTurnsPerOwner {
            let dropped = Set(ledger.turns.prefix(ledger.turns.count - configuration.maxTurnsPerOwner).map(\.id))
            ledger.turns.removeAll { dropped.contains($0.id) }
            ledger.events.removeAll { dropped.contains($0.turnId) }
        }
        if ledger.events.count > configuration.maxEventsPerOwner {
            ledger.events.removeFirst(ledger.events.count - configuration.maxEventsPerOwner)
        }
        if ledger.series.count > configuration.maxSeriesPoints {
            ledger.series.removeFirst(ledger.series.count - configuration.maxSeriesPoints)
        }
        Self.cap(&ledger.firstSeenPartitions, to: configuration.maxFirstSeenEntries)
        Self.cap(&ledger.firstSeenDocuments, to: configuration.maxFirstSeenEntries)
    }

    static func cap(_ dates: inout [String: Date], to limit: Int) {
        guard dates.count > limit else { return }
        let oldest = dates.sorted { $0.value < $1.value }.prefix(dates.count - limit).map(\.key)
        for key in oldest { dates[key] = nil }
    }

    /// Label one turn's events from its measurement. A turn whose output held no content
    /// tokens keeps its summary but teaches nothing.
    @discardableResult
    func label(_ ledger: inout OwnerLedger, turnIndex i: Int, measurement: GroundingMeasurement, at: Date) -> [String: Float] {
        let summary = measurement.summary
        ledger.turns[i].grounding = summary
        ledger.turns[i].measuredAt = at
        ledger.turns[i].unmeasuredReason = nil
        guard summary.contentTokens > 0 else { return [:] }

        let byPartition = Dictionary(measurement.attribution.map { ($0.partitionId, $0) }, uniquingKeysWith: { first, _ in first })
        var targets: [String: Float] = [:]
        let indices = ledger.eventIndices(turn: ledger.turns[i].id)
        for e in indices {
            let event = ledger.events[e]
            let part = byPartition[event.partitionId]
            let uptake = part?.uptake ?? 0
            let intent = part?.intent ?? 0
            let parrot = part?.parrot ?? 0
            let target = Grounding.steerTarget(
                applied: event.appliedWeight, relevancyWeight: event.relevancyWeight, intent: intent,
                uptake: uptake, driftShare: summary.driftShare, parrot: parrot, configuration: configuration)
            ledger.events[e].label = EventLabel(
                target: target.rounded4, uptake: uptake, intent: intent, attributionNats: part?.nats ?? 0,
                coverage: part?.coverage ?? 0, parrot: parrot, relevancy: event.relevancy,
                turnGrounding: summary.grounding, turnDrift: summary.drift, turnDriftShare: summary.driftShare,
                labelledAt: at)
            targets[event.partitionId] = target
        }
        ledger.training.labelsSinceTrain += indices.count

        let point = SeriesPoint(at: at, value: Double(summary.grounding), volume: summary.contentTokens)
        if let last = ledger.series.last, last.at > point.at {
            let insertAt = ledger.series.firstIndex { $0.at > point.at } ?? ledger.series.count
            ledger.series.insert(point, at: insertAt)
        } else {
            ledger.series.append(point)
        }
        return targets
    }

    // MARK: The measurement

    /// The turn's grounding from the scorer's numbers, read against the plan's partitions.
    public func groundingMeasurement(plan: InjectionPlan, raw: GroundingRaw, detail: TraceLevel = .summary) -> GroundingMeasurement {
        Grounding.measurement(
            raw: raw, partitions: plan.terms, isContent: { self.filter.isContent($0) },
            decode: { self.tokenizer.decode([$0]) }, detail: detail, configuration: configuration)
    }

    // MARK: Assistant side

    /// Record the assistant's side of a turn and, when it was measured, label it.
    /// Returns true when training is due.
    @discardableResult
    public func generationDidFinish(
        owner: OwnerID, turnId: UUID, assistantText: String, startedAt: Date, finishedAt: Date,
        trace: InjectionTrace? = nil, measurement: GroundingMeasurement? = nil
    ) -> Bool {
        store.update(owner, now: finishedAt) { ledger in
            guard let i = ledger.turnIndex(turnId) else { return }
            ledger.turns[i].assistantStartedAt = startedAt
            ledger.turns[i].assistantFinishedAt = finishedAt
            ledger.turns[i].assistantWords = Grounding.wordCount(assistantText)
            ledger.turns[i].assistantChars = assistantText.count
            if let trace {
                if trace.level != .off { ledger.turns[i].trace = TraceAggregates(trace) }
                ledger.lastTraceId = trace.traceId
            }
            if let measurement {
                if measurement.measured {
                    label(&ledger, turnIndex: i, measurement: measurement, at: finishedAt)
                } else {
                    ledger.turns[i].unmeasuredReason = measurement.skippedReason
                }
            }
        }
        if let trace {
            do {
                try store.saveTrace(trace, owner: owner)
            } catch {
                log?.log(.warning, "saving trace \(trace.traceId) failed: \(error)")
            }
        }
        return trainingDue(owner: owner)
    }

    /// Keep a trace without recording its turn (dry runs and paired comparisons), so it
    /// can still be read back by id.
    public func storeTrace(_ trace: InjectionTrace, owner: OwnerID) {
        do {
            try store.saveTrace(trace, owner: owner)
        } catch {
            log?.log(.warning, "saving trace \(trace.traceId) failed: \(error)")
        }
    }

    /// The generation was cancelled: forget the turn rather than label a partial answer.
    public func generationAbandoned(owner: OwnerID, turnId: UUID) {
        store.update(owner) { ledger in
            ledger.turns.removeAll { $0.id == turnId }
            ledger.events.removeAll { $0.turnId == turnId }
        }
    }

    /// Label a recorded turn with a measurement the caller made (replay, tests).
    public func observe(owner: OwnerID, turn observation: TurnObservation) throws -> ObservationReceipt {
        let now = observation.assistantFinishedAt ?? Date()
        return try store.update(owner, now: now) { ledger in
            guard let i = ledger.turnIndex(observation.turnId) else {
                throw SinatraError.unknownTurn(observation.turnId)
            }
            if ledger.turns[i].assistantFinishedAt == nil {
                let text = observation.assistantText ?? ""
                ledger.turns[i].assistantStartedAt = observation.assistantStartedAt ?? now
                ledger.turns[i].assistantFinishedAt = now
                ledger.turns[i].assistantWords = Grounding.wordCount(text)
                ledger.turns[i].assistantChars = text.count
            }
            let targets = observation.measurement.measured
                ? label(&ledger, turnIndex: i, measurement: observation.measurement, at: now) : [:]
            return ObservationReceipt(turnId: observation.turnId, summary: observation.measurement.summary, targets: targets)
        }
    }

    func trainingDue(owner: OwnerID) -> Bool {
        let ledger = store.ledger(owner)
        return ledger.labelledEvents >= configuration.minimumLabelledEvents
            && ledger.training.labelsSinceTrain >= configuration.trainEveryLabelledEvents
    }

    public func isTrainingDue(owner: OwnerID) -> Bool { trainingDue(owner: owner) }
}
