//
//  Render.swift
//  sinatra-harness
//
//  WHAT: Terminal views of plans, traces (injection and grounding), comparisons,
//        attributions and analyses.
//

import Foundation
import SinatraHarness

enum Render {

    static func f(_ value: Float, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", value) }
    static func f(_ value: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", value) }
    static func signed(_ value: Float, _ digits: Int = 3) -> String { String(format: "%+.\(digits)f", value) }

    static func token(_ text: String?) -> String {
        guard let text else { return "·" }
        let escaped = text.replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }

    static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? String(text.prefix(width)) : text + String(repeating: " ", count: width - text.count)
    }

    static func rule(_ title: String) {
        print("\n── \(title) " + String(repeating: "─", count: max(0, 72 - title.count)))
    }

    static func plan(_ d: TurnDiagnostics) {
        rule("Sinatra plan (\(d.mode.rawValue)\(d.coldStart ? ", cold start" : ""))")
        print("gate g=\(f(d.gate, 2))  labelled events \(d.labelledEvents)  measured turns \(d.measuredTurns)  observed \(d.observedTurns)")
        print("windows \(d.periods.logDescription)")
        print("bias: \(d.biasNonZero) tokens, max |b| \(f(d.biasMaxAbs, 2)), L1 \(f(d.biasL1, 1))  (encode \(f(d.encodeMillis, 1)) ms, build \(f(d.buildMillis, 1)) ms)")
        print("  rank  partition                         band    age(d)  rel    prior    uptake  net      applied  labels")
        for p in d.partitions {
            print("  \(pad(String(p.rank), 4))  \(pad(p.id, 32))  \(pad(p.band, 6))  \(pad(f(p.docAgeDays, 1), 6))  \(pad(f(p.relevancy, 2), 5))  \(pad(signed(p.priorWeight), 7))  \(pad(f(p.priorUptake, 2), 6))  \(pad(p.netWeight.map { signed($0) } ?? "   -", 7))  \(pad(signed(p.appliedWeight), 7))  \(p.labelledBefore)")
        }
        if !d.topBiased.isEmpty {
            print("  top biased: " + d.topBiased.prefix(10).map { "\(token($0.text)) \(signed($0.bias, 2))" }.joined(separator: "  "))
        }
    }

    static func summary(_ s: TraceSummary, title: String = "Trace summary") {
        rule(title)
        print("steps \(s.steps)   H \(f(s.meanEntropyPre)) → \(f(s.meanEntropyPost))  (ΔH \(signed(s.meanEntropyShift)))")
        print("KL total \(f(s.totalKL)) (mean \(f(s.meanKL, 4)))   JS mean \(f(s.meanJS, 4))   gain Σ \(signed(s.totalGain)) nats")
        print("divergence \(f(s.divergenceRate * 100, 1))% of steps (first at \(s.firstDivergenceStep.map(String.init) ?? "–"))   argmax flips \(s.flippedArgmaxSteps)")
        print("mass into mask \(signed(s.meanMassIntoMask, 4)) per step   sampled tokens inside the mask \(f(s.sampledInMaskShare * 100, 1))%")
        if !s.partitionAttribution.isEmpty {
            let parts = s.partitionAttribution.sorted { abs($0.value) > abs($1.value) }.prefix(6)
            print("attribution: " + parts.map { "\($0.key) \(signed($0.value, 2))" }.joined(separator: "  "))
        }
    }

    static func steps(_ trace: InjectionTrace, limit: Int = 80) {
        rule("Per-step impact (\(trace.level.rawValue), seed \(trace.seed.map(String.init) ?? "–"))")
        print("  step  sampled               counterfactual        H pre→post       KL       gain    rank   mask")
        for step in trace.steps.prefix(limit) {
            let cf = step.diverged ? token(step.counterfactualText) : "="
            let marker = step.inMask ? "●" : " "
            print("  \(pad(String(step.index), 4))  \(pad(token(step.sampledText), 20))  \(pad(cf, 20))  \(f(step.entropyPre, 2))→\(pad(f(step.entropyPost, 2), 6))  \(pad(f(step.kl, 4), 7))  \(pad(signed(step.gain, 2), 6))  \(pad("\(step.rankPre)→\(step.rankPost)", 6)) \(marker)\(signed(step.massIntoMask, 3))")
        }
        if trace.steps.count > limit { print("  … \(trace.steps.count - limit) more steps") }
        if trace.level == .full { topK(trace) ; heatmap(trace) }
    }

    static func topK(_ trace: InjectionTrace, steps: Int = 6) {
        rule("Top-k before → after (first \(steps) diverging steps)")
        for step in trace.steps.filter(\.diverged).prefix(steps) {
            let pre = (step.topPre ?? []).prefix(5).map { "\(token($0.text)) \(f($0.logprob, 2))" }.joined(separator: " ")
            let post = (step.topPost ?? []).prefix(5).map { "\(token($0.text)) \(f($0.logprob, 2))" }.joined(separator: " ")
            print("  step \(step.index):\n    before: \(pre)\n    after:  \(post)")
        }
    }

    /// Steps × the mask tokens whose probability moved most, one glyph per cell.
    static func heatmap(_ trace: InjectionTrace, tokens: Int = 16, steps: Int = 60) {
        guard let mask = trace.mask else { return }
        let rows = trace.steps.prefix(steps).compactMap { step in step.movement.map { (step, $0) } }
        guard !rows.isEmpty else { return }
        var totals = [Float](repeating: 0, count: mask.tokenIds.count)
        for (_, movement) in rows {
            for (i, value) in movement.enumerated() where i < totals.count { totals[i] += abs(value) }
        }
        let columns = totals.indices.sorted { totals[$0] > totals[$1] }.prefix(tokens).filter { totals[$0] > 0 }
        guard !columns.isEmpty else { return }
        rule("Where Sinatra acted: Δp of the mask tokens that moved most")
        print("  legend: '#' +10pp or more  '+' +1pp  '=' −10pp  '-' −1pp  '·' under 1pp")
        for (i, column) in columns.enumerated() {
            let text = mask.tokenTexts?[column]
            print("  col \(pad(String(i), 2)) \(pad(token(text), 16)) bias \(signed(mask.bias[column], 2))  Σ|Δp| \(f(totals[column], 3))")
        }
        let header = columns.indices.map { String($0 % 10) }.joined()
        print("  step \(header)  sampled")
        for (step, movement) in rows {
            let cells = columns.map { column -> Character in
                let value = column < movement.count ? movement[column] : 0
                switch value {
                case 0.1...: return "#"
                case 0.01..<0.1: return "+"
                case ...(-0.1): return "="
                case ...(-0.01): return "-"
                default: return "·"
                }
            }
            print("  \(pad(String(step.index), 4)) \(String(cells))  \(token(step.sampledText))\(step.diverged ? "  (was \(token(step.counterfactualText)))" : "")")
        }
    }

    static func comparison(_ report: ComparisonReport) {
        rule("Baseline (no injection)")
        print(report.baseline.text)
        rule("With Sinatra (\(report.injected.mode.rawValue))")
        print(report.injected.text)
        rule("Comparison (seed \(report.seed), temperature \(f(report.temperature, 2)))")
        print("common prefix \(report.commonPrefixTokens) tokens; first divergence at token \(report.firstDivergenceToken.map(String.init) ?? "– (identical)")")
        let hb = report.baseline.summary?.meanEntropyPost ?? 0
        let hi = report.injected.summary?.meanEntropyPost ?? 0
        print("mean entropy per step: baseline \(f(hb)) vs injected \(f(hi)) (ΔH \(signed(hi - hb)))")
        if let b = report.baseline.grounding, let i = report.injected.grounding {
            print("grounding: baseline \(f(b.grounding, 2)) vs injected \(f(i.grounding, 2));  drift \(f(b.drift, 2)) vs \(f(i.drift, 2)) nats;  risk \(f(b.hallucinationRisk, 2)) vs \(f(i.hallucinationRisk, 2))")
        }
        if let cross = report.crossLikelihood {
            print("log-likelihood of each output under each distribution (nats):")
            print("  baseline output: \(f(cross.baselineUnderBaseline, 2)) plain, \(f(cross.baselineUnderInjected, 2)) injected (\(signed(Float(cross.baselineShiftPerToken), 4))/token)")
            print("  injected output: \(f(cross.injectedUnderBaseline, 2)) plain, \(f(cross.injectedUnderInjected, 2)) injected (personalization \(signed(Float(cross.personalizationPerToken), 4))/token)")
        }
        print("entropy curves (per step, baseline | injected):")
        let n = max(report.baseline.entropy.count, report.injected.entropy.count)
        for i in stride(from: 0, to: min(n, 40), by: 1) {
            let b = i < report.baseline.entropy.count ? f(report.baseline.entropy[i], 2) : "  – "
            let j = i < report.injected.entropy.count ? f(report.injected.entropy[i], 2) : "  – "
            let bar = i < report.injected.entropy.count && i < report.baseline.entropy.count
                ? String(repeating: report.injected.entropy[i] < report.baseline.entropy[i] ? "<" : ">", count: min(20, Int(abs(report.injected.entropy[i] - report.baseline.entropy[i]) * 10)))
                : ""
            print("  \(pad(String(i), 4)) \(b) | \(j)  \(bar)")
        }
    }

    static func summary(_ s: OwnerSummary) {
        rule("Owner \(s.owner)")
        print("observed \(s.observations) turns in the band, \(s.measuredTurns) measured, \(s.labelledEvents) labelled events")
        if let grounding = s.meanGrounding {
            print("mean grounding \(f(grounding, 2)), drift \(s.meanDrift.map { f($0, 2) } ?? "–") nats, hallucination risk \(s.meanHallucinationRisk.map { f($0, 2) } ?? "–")")
        }
        print("training cycles \(s.trainingCycles), last \(s.trainedAt.map { ISO8601DateFormatter().string(from: $0) } ?? "never"), reliability g=\(f(s.reliability, 2))")
        print("holdout MAE \(s.holdoutMAE.map { f($0) } ?? "–") vs mean predictor \(s.baselineMAE.map { f($0) } ?? "–")")
        print("windows \(s.periods.logDescription)")
        print("store \(s.store)")
    }

    // MARK: - Grounding

    static func grounding(_ m: GroundingMeasurement, steps limit: Int = 60, baseline: GroundingMeasurement? = nil) {
        guard m.measured else {
            rule("Grounding")
            print("not measured: \(m.skippedReason ?? "unknown")")
            return
        }
        let s = m.summary
        rule("Grounding: what the context did to the output")
        print("\(s.steps) tokens, \(s.contentTokens) content  (prompt \(m.promptTokens), bare \(m.bareTokens), shared prefix \(m.sharedPrefixTokens); \(m.cacheReused ? "decode cache reused" : "fresh prefill"); prefill \(f(m.prefillMillis, 0)) ms, score \(f(m.scoreMillis, 0)) ms)")
        print("grounded \(f(s.grounding * 100, 1))%   unsupported \(f(s.unsupportedShare * 100, 1))%   contradicted \(f(s.contradictedShare * 100, 1))%")
        print("drift \(f(s.drift, 2)) nats per content token, \(f(s.driftShare * 100, 1))% drifting (first at \(s.firstDriftStep.map(String.init) ?? "–"))")
        print("context dependence Σι \(signed(s.contextDependence, 2)) nats   mean KL(ctx‖bare) \(f(s.meanContextKL, 3))   H \(f(s.meanEntropyBare, 2)) bare → \(f(s.meanEntropyCtx, 2)) ctx")
        print("hallucination risk \(f(s.hallucinationRisk, 3))   parroted \(f(s.parrotShare * 100, 1))%   unattributed influence \(f(s.unattributed, 2)) nats")
        if let b = baseline?.summary, baseline?.measured == true {
            print("baseline (no injection): grounded \(f(b.grounding * 100, 1))%, drift \(f(b.drift, 2)), risk \(f(b.hallucinationRisk, 3))")
        }
        if !m.attribution.isEmpty {
            print("  partition                         rel    nats    uptake  intent  cover   parrot")
            for a in m.attribution {
                print("  \(pad(a.partitionId, 32))  \(pad(f(a.relevancy, 2), 5))  \(pad(f(a.nats, 2), 6))  \(pad(f(a.uptake, 2), 6))  \(pad(f(a.intent, 2), 6))  \(pad(f(a.coverage, 2), 6))  \(f(a.parrot, 2))")
            }
        }
        guard !m.steps.isEmpty else { return }
        print("  step  token                 ι        KL      class         drift   the context pushed")
        for step in m.steps.prefix(limit) {
            let klass = step.kind == .function ? "·" : step.kind.rawValue
            let tune = step.drift > 0.05 ? "\(token(step.tuneText)) \(signed(step.tuneNats, 2))" : ""
            print("  \(pad(String(step.index), 4))  \(pad(token(step.text), 20))  \(pad(signed(step.influence, 2), 7))  \(pad(f(step.contextKL, 3), 6))  \(pad(klass, 12))  \(pad(f(step.drift, 2), 6))  \(tune)")
        }
        if m.steps.count > limit { print("  … \(m.steps.count - limit) more steps") }
        let pushed = m.steps.filter { ($0.pushes?.isEmpty == false) && $0.drift > 1 }.prefix(6)
        if !pushed.isEmpty {
            rule("Where the output left its context (full detail, drifting steps)")
            for step in pushed {
                let alternatives = (step.pushes ?? []).prefix(5).map { "\(token($0.text)) \(signed($0.logprob, 2))" }.joined(separator: " ")
                print("  step \(step.index): said \(token(step.text)) (ι \(signed(step.influence, 2)), bare rank \(step.rankBare.map(String.init) ?? "–")); context pushed \(alternatives)")
            }
        }
    }

    static func groundingReport(_ report: GroundingReport) {
        rule("Grounding for \(report.owner) (\(report.rows.count) measured turns)")
        for c in report.correlations {
            print("  r(\(pad(c.x, 15)), \(pad(c.y, 17))) = \(c.pearson.map { f($0, 3) } ?? "  n/a")  (n=\(c.n))")
        }
        for bin in report.bins {
            print("  \(pad(bin.label, 24)) n=\(pad(String(bin.count), 4)) grounding \(bin.meanGrounding.map { f($0, 3) } ?? "–")  drift \(bin.meanDrift.map { f($0, 3) } ?? "–")  risk \(bin.meanRisk.map { f($0, 3) } ?? "–")")
        }
        if !report.citations.isEmpty {
            print("  citations (nats of output attributed, across the band):")
            for c in report.citations.prefix(10) {
                print("    \(pad(c.documentId, 40)) \(pad(f(c.nats, 2), 8)) over \(c.turns) turns, uptake \(f(c.meanUptake, 2)), coverage \(f(c.meanCoverage, 2)), parrot \(f(c.meanParrot, 2))")
            }
        }
        if !report.mostDrifted.isEmpty {
            print("  most drifted turns:")
            for row in report.mostDrifted {
                print("    \(row.turnId.uuidString.prefix(8))  drift \(f(row.drift, 2)) (\(f(row.driftShare * 100, 0))%)  grounding \(f(row.grounding, 2))  risk \(f(row.hallucinationRisk, 3))  \(row.steered ? "steered |b|₁ \(f(row.biasL1, 1))" : "unsteered")")
            }
        }
        if !report.unmeasured.isEmpty {
            print("  unmeasured: " + report.unmeasured.map { "\($0.value)× \($0.key)" }.joined(separator: ", "))
        }
    }

    static func attribution(partitions: [Partition], lexical: [PartitionAttribution], leaveOneOut report: LeaveOneOutReport) {
        rule("Citation value per partition (nats the output owes it)")
        let bare = report.bare.map { "   without any context: \(f(report.full - $0, 2)) nats" } ?? ""
        print("output \(report.tokens) tokens, Σ log p = \(f(report.full, 2))\(bare)")
        print("  partition                         leave-one-out   lexical nats   lexical uptake")
        for (p, partition) in partitions.enumerated() {
            let exact = p < report.citation.count ? signed(Float(report.citation[p]), 2) : "–"
            let lex = lexical.first { $0.partitionId == partition.id }
            print("  \(pad(partition.id, 32))  \(pad(exact, 14))  \(pad(lex.map { f($0.nats, 2) } ?? "–", 13))  \(lex.map { f($0.uptake, 2) } ?? "–")")
        }
    }
}
