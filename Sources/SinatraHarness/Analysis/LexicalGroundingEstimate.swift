//
//  LexicalGroundingEstimate.swift
//  SinatraHarness
//
//  WHAT: A model-free stand-in for GroundingScorer, for replay without a model and for
//        tests. It fakes what the model would say from lexical overlap alone:
//          - a token found in a partition was pushed by the context with that partition's
//            emphasis (the largest, when several hold it); any other token was not;
//          - at every step the context lifts the model by the largest emphasis on average
//            (its KL), toward the most emphasised partition's most frequent content token.
//        Emphasis stands for "what the context would really have pushed for": the replay's
//        synthetic user supplies it; by default every partition gets `defaultEmphasis`.
//  PIN:  Never used on a real turn. Its numbers only have to exercise the loop.
//

import Foundation

public enum LexicalGroundingEstimate {

    public static func raw(
        output: [Int], plan: InjectionPlan, emphasis: [String: Float] = [:], defaultEmphasis: Float = 1
    ) -> GroundingRaw {
        let partitions = plan.terms
        func weight(_ p: Int) -> Float { emphasis[partitions[p].id] ?? defaultEmphasis }
        let strongest = partitions.indices.max { weight($0) < weight($1) }
        let tuneToken: Int = strongest.flatMap { p in
            partitions[p].counts.max { $0.value == $1.value ? $0.key > $1.key : $0.value < $1.value }?.key
        } ?? 0
        let tuneNats: Float = strongest.map(weight) ?? 0
        let bare: Float = -4

        var raw = GroundingRaw(
            tokens: output, logpCtx: [], logpBare: [], entropyCtx: [], entropyBare: [], contextKL: [],
            tune: [], tuneNats: [], promptTokens: 0, bareTokens: 0)
        for token in output {
            let pushed = partitions.indices
                .filter { partitions[$0].counts[token] != nil }
                .map(weight)
                .max() ?? 0
            raw.logpBare.append(bare)
            raw.logpCtx.append(bare + pushed)
            raw.entropyBare.append(2)
            raw.entropyCtx.append(max(0, 2 - 0.3 * pushed))
            raw.contextKL.append(tuneNats)
            raw.tune.append(tuneToken)
            raw.tuneNats.append(tuneNats)
        }
        return raw
    }
}
