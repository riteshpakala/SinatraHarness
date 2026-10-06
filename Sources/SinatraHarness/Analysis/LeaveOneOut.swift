//
//  LeaveOneOut.swift
//  SinatraHarness
//
//  WHAT: Exact citation values. The grounding measurement attributes the context's
//        influence to partitions lexically (a token belongs to the partitions it appears
//        in). Leave-one-out asks the model instead: score the same output under the full
//        prompt, under the prompt without each partition in turn, and without any context.
//
//          citation_p = Σ_t log p(y_t | full) − log p(y_t | full minus p)
//
//        P + 2 teacher-forced passes, so it is a diagnostic (CLI `attribute`), not something
//        a turn pays for.
//

import Foundation
import MLXLMCommon

public struct LeaveOneOutReport: Codable, Sendable {
    public var tokens: Int
    /// Σ log p(y | full prompt).
    public var full: Double
    /// Σ log p(y | no context), when a bare prompt was given.
    public var bare: Double?
    /// Σ log p(y | full prompt without partition p).
    public var dropped: [Double]
    /// Nats each partition adds to the output's likelihood: full − dropped[p].
    public var citation: [Double]
    /// Per step, per partition: log p(y_t | full) − log p(y_t | without p).
    public var perStep: [[Float]]
}

extension Harness {

    /// Score `output` under `full`, each of `ablated` (one per partition) and `bare`.
    public func leaveOneOut(full: UserInput, ablated: [UserInput], bare: UserInput?, output: [Int]) async throws -> LeaveOneOutReport {
        guard let context = currentContext else { throw SinatraError.notLoaded }
        func tokens(_ input: UserInput) async throws -> [Int] {
            try await context.processor.prepare(input: input).text.tokens
                .reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
        }
        let fullTokens = try await tokens(full)
        var ablatedTokens: [[Int]] = []
        for input in ablated { ablatedTokens.append(try await tokens(input)) }
        var bareTokens: [Int]?
        if let bare { bareTokens = try await tokens(bare) }
        return try await withContext { context in
            let base = try GroundingScorer.tokenLogprobs(model: context.model, prompt: fullTokens, continuation: output)
            var dropped: [Double] = []
            var perStep = Array(repeating: [Float](), count: output.count)
            for prompt in ablatedTokens {
                let logp = try GroundingScorer.tokenLogprobs(model: context.model, prompt: prompt, continuation: output)
                dropped.append(logp.reduce(0) { $0 + Double($1) })
                for t in logp.indices where t < base.count { perStep[t].append(base[t] - logp[t]) }
            }
            let bareTotal = try bareTokens.map {
                try GroundingScorer.tokenLogprobs(model: context.model, prompt: $0, continuation: output).reduce(0) { $0 + Double($1) }
            }
            let fullTotal = base.reduce(0) { $0 + Double($1) }
            return LeaveOneOutReport(
                tokens: output.count, full: fullTotal, bare: bareTotal, dropped: dropped,
                citation: dropped.map { fullTotal - $0 }, perStep: perStep)
        }
    }

    /// The loaded tokenizer's ids for `text` (no special tokens): the CLI turns a decoded
    /// output back into the tokens it scores.
    public func tokenize(_ text: String) async -> [Int]? {
        tokenizerAdapter?.encode(text)
    }
}
