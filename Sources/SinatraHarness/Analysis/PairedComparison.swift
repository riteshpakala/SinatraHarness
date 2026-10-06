//
//  PairedComparison.swift
//  SinatraHarness
//
//  WHAT: Decoding with and without Sinatra, side by side. The same input and seed are
//        decoded twice (injection off, then on), both traced, neither recorded. Then each
//        output is scored under both distributions with one teacher-forced pass per side:
//        how much likelier the injected output is under the injected model is the
//        personalization, in nats; how much less likely it is under the plain model is
//        what it cost. With a bare input, each side's grounding is measured too, so the
//        report says whether the injection made the output follow its context more closely.
//

import Foundation
import MLX
import MLXLMCommon

public struct ComparisonReport: Codable, Sendable {
    public struct Side: Codable, Sendable {
        public var mode: BiasMode
        public var text: String
        public var tokens: [Int]
        public var entropy: [Float]
        public var summary: TraceSummary?
        /// What the retrieved context did to this side's output.
        public var grounding: GroundingSummary?
        public var generationTokens: Int
        public var tokensPerSecond: Double?
    }

    public struct CrossLikelihood: Codable, Sendable {
        /// Σ log p(y | ·) for each output under each distribution.
        public var baselineUnderBaseline: Double
        public var baselineUnderInjected: Double
        public var injectedUnderBaseline: Double
        public var injectedUnderInjected: Double
        public var baselineTokens: Int
        public var injectedTokens: Int

        /// Nats per token the injection adds to its own output's likelihood.
        public var personalizationPerToken: Double {
            injectedTokens > 0 ? (injectedUnderInjected - injectedUnderBaseline) / Double(injectedTokens) : 0
        }
        /// Nats per token the plain model's output loses under the injection.
        public var baselineShiftPerToken: Double {
            baselineTokens > 0 ? (baselineUnderInjected - baselineUnderBaseline) / Double(baselineTokens) : 0
        }
    }

    public var seed: UInt64
    public var temperature: Float
    public var plan: TurnDiagnostics?
    public var baseline: Side
    public var injected: Side
    public var commonPrefixTokens: Int
    public var firstDivergenceToken: Int?
    public var crossLikelihood: CrossLikelihood?
}

extension Harness {

    /// Decode `request` with the injection off and on (same seed), trace both, record
    /// neither, and optionally score each output under both distributions.
    public func compare(_ request: GenerateRequest, injectedMode: BiasMode = .lexical, score: Bool = true) async throws -> ComparisonReport {
        var parameters = request.parameters
        let seed = parameters.seed ?? UInt64.random(in: 1...UInt64(Int64.max))
        parameters.seed = seed
        let level: TraceLevel = request.trace == .full ? .full : .summary

        var baselineRequest = request
        baselineRequest.parameters = parameters
        baselineRequest.mode = .off
        baselineRequest.trace = level
        baselineRequest.record = false
        let baseline = try await collect(try await generate(baselineRequest))

        var injectedRequest = request
        injectedRequest.parameters = parameters
        injectedRequest.mode = injectedMode
        injectedRequest.trace = level
        injectedRequest.record = false
        let injectedHandle = try await generate(injectedRequest)
        let injected = try await collect(injectedHandle)

        let baselineTokens = baseline.completion.trace?.steps.map(\.sampled) ?? []
        let injectedTokens = injected.completion.trace?.steps.map(\.sampled) ?? []
        let prefix = zip(baselineTokens, injectedTokens).prefix { $0 == $1 }.count
        let firstDivergence = prefix < max(baselineTokens.count, injectedTokens.count) ? prefix : nil

        var cross: ComparisonReport.CrossLikelihood?
        if score, let bias = injectedHandle.plan?.bias, injectedHandle.plan?.injects == true {
            cross = try await scoreBoth(
                input: request.input, bias: bias, baselineTokens: baselineTokens, injectedTokens: injectedTokens)
        }

        func side(_ mode: BiasMode, _ result: Collected) -> ComparisonReport.Side {
            ComparisonReport.Side(
                mode: mode, text: result.completion.text,
                tokens: result.completion.trace?.steps.map(\.sampled) ?? [],
                entropy: result.completion.trace?.steps.map(\.entropyPost) ?? [],
                summary: result.completion.trace?.summary,
                grounding: result.completion.grounding?.measured == true ? result.completion.grounding?.summary : nil,
                generationTokens: result.completion.info?.generationTokenCount ?? 0,
                tokensPerSecond: result.completion.info?.tokensPerSecond)
        }
        return ComparisonReport(
            seed: seed, temperature: parameters.temperature, plan: injectedHandle.plan?.diagnostics,
            baseline: side(.off, baseline), injected: side(injectedMode, injected),
            commonPrefixTokens: prefix, firstDivergenceToken: firstDivergence, crossLikelihood: cross)
    }

    struct Collected {
        var completion: TurnCompletion
    }

    private func collect(_ handle: GenerationHandle) async throws -> Collected {
        for await _ in handle.stream {}
        return Collected(completion: await handle.completion.value)
    }

    private func scoreBoth(input: UserInput, bias: SparseBias, baselineTokens: [Int], injectedTokens: [Int]) async throws -> ComparisonReport.CrossLikelihood {
        guard let prepared = try await preparedInput(input) else {
            throw SinatraError.notLoaded
        }
        return try await withContext { context in
            let dense = MLXArray(bias.denseVector(), [1, bias.vocabularySize])
            let baseline = try ContinuationScorer.logLikelihoods(
                input: prepared, continuation: baselineTokens, context: context, biases: [nil, dense])
            let injected = try ContinuationScorer.logLikelihoods(
                input: prepared, continuation: injectedTokens, context: context, biases: [nil, dense])
            return ComparisonReport.CrossLikelihood(
                baselineUnderBaseline: baseline[0], baselineUnderInjected: baseline[1],
                injectedUnderBaseline: injected[0], injectedUnderInjected: injected[1],
                baselineTokens: baselineTokens.count, injectedTokens: injectedTokens.count)
        }
    }

    private func preparedInput(_ input: UserInput) async throws -> LMInput? {
        guard let context = currentContext else { return nil }
        return try await context.processor.prepare(input: input)
    }
}

enum ContinuationScorer {

    /// Σ log p(continuation | prompt) under each bias (nil = no injection), one
    /// teacher-forced pass with the KV cache, logits evaluated in chunks.
    static func logLikelihoods(
        input: LMInput, continuation: [Int], context: ModelContext, biases: [MLXArray?], chunk: Int = 128
    ) throws -> [Double] {
        var totals = [Double](repeating: 0, count: biases.count)
        guard !continuation.isEmpty else { return totals }
        let model = context.model
        let cache = model.newCache(parameters: nil)

        var remaining: [Int] = []
        switch try model.prepare(input, cache: cache, windowSize: 512) {
        case .tokens(let text):
            remaining = text.tokens.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
        case .logits(let output):
            let last = output.logits[0..., -1, 0...].asType(.float32)
            accumulate(&totals, logits: last, targets: [continuation[0]], biases: biases)
        }

        // Position p of `feed` predicts continuation[p - n + 1] (n = remaining prompt tokens).
        let n = remaining.count
        let feed = remaining + continuation.dropLast()
        var offset = 0
        while offset < feed.count {
            let end = min(offset + chunk, feed.count)
            let tokens = MLXArray(feed[offset..<end].map { Int32($0) }, [1, end - offset])
            let logits = model(tokens, cache: cache)
            let firstLocal = max(0, (n - 1) - offset)
            if firstLocal < end - offset {
                let targetStart = offset + firstLocal - n + 1
                let count = end - offset - firstLocal
                let targets = Array(continuation[targetStart..<(targetStart + count)])
                let rows = logits[0, firstLocal..<(end - offset), 0...].asType(.float32)
                accumulate(&totals, logits: rows, targets: targets, biases: biases)
            } else {
                eval(logits)
            }
            offset = end
        }
        return totals
    }

    private static func accumulate(_ totals: inout [Double], logits: MLXArray, targets: [Int], biases: [MLXArray?]) {
        let rows = logits.ndim == 1 ? logits.reshaped([1, -1]) : logits
        let targetArray = MLXArray(targets.map { Int32($0) }, [targets.count, 1])
        for (i, bias) in biases.enumerated() {
            let z = bias.map { rows + $0.reshaped([1, -1]).asType(rows.dtype) } ?? rows
            let logprobs = z - logSumExp(z, axis: -1, keepDims: true)
            let picked = sum(takeAlong(logprobs, targetArray, axis: -1))
            totals[i] += Double(picked.item(Float.self))
        }
    }
}
