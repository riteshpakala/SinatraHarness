//
//  InjectedGeneration.swift
//  SinatraHarness
//
//  WHAT: Generation with the injection layer in the decode loop. Builds the token
//        iterator with [injection, penalties] (or the tracing wrappers) and hands it to
//        MLXLMCommon's own generate loop, so detokenisation, stop strings and tool-call
//        parsing are exactly what `MLXLMCommon.generate` does.
//  PIN:  With no injection, no trace and no measurement this IS `MLXLMCommon.generate`
//        (same iterator init, KV quantisation kept). The direct iterator init used otherwise
//        has no KV quantisation, so the cache is created here from the parameters (keeps
//        maxKVSize). A measured run keeps that cache and records every sampled token: the
//        grounding scorer re-scores the output from them after the decode.
//

import Foundation
import MLX
import MLXLMCommon

public enum InjectedGeneration {

    /// A running generation: its stream, its task, and its trace once it ends.
    public final class Run: @unchecked Sendable {
        public let stream: AsyncStream<Generation>
        public let task: Task<Void, Never>
        public let traceLevel: TraceLevel
        public let seed: UInt64?
        public let injected: Bool
        /// The decode's KV cache, kept for the grounding measurement (measured runs only).
        public let cache: [KVCache]?
        /// Prompt token ids, as the model saw them.
        public let promptTokens: [Int]
        let buffer: TraceBuffer?
        let recorder: TokenRecorder?

        init(
            stream: AsyncStream<Generation>, task: Task<Void, Never>, traceLevel: TraceLevel, seed: UInt64?,
            injected: Bool, buffer: TraceBuffer?, cache: [KVCache]? = nil, recorder: TokenRecorder? = nil,
            promptTokens: [Int] = []
        ) {
            self.stream = stream
            self.task = task
            self.traceLevel = traceLevel
            self.seed = seed
            self.injected = injected
            self.buffer = buffer
            self.cache = cache
            self.recorder = recorder
            self.promptTokens = promptTokens
        }

        public var measures: Bool { recorder != nil && cache != nil }

        /// Steps the loop consumed: the iterator computes one step ahead, and a stop keeps
        /// its deciding step.
        static func consumed(_ info: GenerateCompletionInfo?) -> Int? {
            guard let info else { return nil }
            switch info.stopReason {
            case .stop: return info.generationTokenCount + 1
            default: return info.generationTokenCount
            }
        }

        /// The tokens the decode produced, stop token included (measured runs only).
        public func sampledTokens(info: GenerateCompletionInfo?) -> [Int] {
            recorder?.drain(limit: Self.consumed(info)) ?? []
        }

        /// Evaluate the recorded steps, trimmed to the tokens the loop consumed.
        public func drainTrace(info: GenerateCompletionInfo?, tokenizer: (any SinatraTokenizing)?, mask: ImpactMask?) -> [StepTrace] {
            guard let buffer else { return [] }
            let raw = buffer.drain(limit: Self.consumed(info))
            func text(_ id: Int) -> String? { tokenizer?.decode([id]) }
            return raw.enumerated().map { index, step in
                StepTrace(
                    index: index, sampled: step.sampled, counterfactual: step.counterfactual,
                    sampledText: text(step.sampled),
                    counterfactualText: step.counterfactual == step.sampled ? nil : text(step.counterfactual),
                    entropyPre: step.entropyPre, entropyPost: step.entropyPost, kl: step.kl, js: step.js,
                    logprobPre: step.logprobPre, logprobPost: step.logprobPost,
                    rankPre: step.rankPre, rankPost: step.rankPost,
                    argmaxPre: step.argmaxPre, argmaxPost: step.argmaxPost,
                    massIntoMask: step.massIntoMask, inMask: mask?.position(of: step.sampled) != nil,
                    topPre: step.topPre?.map { TokenLogprob(id: $0.0, text: text($0.0), logprob: $0.1) },
                    topPost: step.topPost?.map { TokenLogprob(id: $0.0, text: text($0.0), logprob: $0.1) },
                    movement: step.movement)
            }
        }
    }

    /// Start a generation with `plan`'s injection. `trace` is resolved against whether the
    /// plan actually injects (`.automatic` → `.summary` only when it does). `measure` keeps
    /// the cache and the sampled tokens for the grounding measurement.
    public static func start(
        input: LMInput, parameters: GenerateParameters, context: ModelContext, plan: InjectionPlan?,
        trace: TraceLevel = .off, topK: Int = 8, cache: [KVCache]? = nil,
        tools: [[String: any Sendable]]? = nil, wiredMemoryTicket: WiredMemoryTicket? = nil,
        measure: Bool = false
    ) throws -> Run {
        let injects = plan?.injects ?? false
        let level = trace.resolved(injecting: injects)
        var parameters = parameters
        if level != .off && parameters.seed == nil {
            // The counterfactual needs an identically seeded shadow sampler.
            parameters.seed = UInt64.random(in: 1...UInt64(Int64.max))
        }
        let injection: InjectionProcessor? = injects ? plan?.bias.map { InjectionProcessor(bias: $0) } : nil
        let promptTokens = input.text.tokens.size

        if injection == nil && level == .off && !measure {
            let iterator = try TokenIterator(input: input, model: context.model, cache: cache, parameters: parameters)
            let (stream, task) = generateTask(
                promptTokenCount: promptTokens, modelConfiguration: context.configuration,
                tokenizer: context.tokenizer, iterator: iterator, wiredMemoryTicket: wiredMemoryTicket, tools: tools)
            return Run(stream: stream, task: task, traceLevel: .off, seed: parameters.seed, injected: false, buffer: nil)
        }

        let cache = cache ?? context.model.newCache(parameters: parameters)
        let penalty = parameters.processor()
        let recorder = measure ? TokenRecorder() : nil
        var processor: any LogitProcessor
        let sampler: any LogitSampler
        var buffer: TraceBuffer?
        if level == .off {
            processor = CompositeLogitProcessor([injection, penalty].compactMap { $0 })
            sampler = parameters.sampler()
        } else {
            var maskIndices: MLXArray?
            if injects, let bias = plan?.bias, !bias.isEmpty {
                maskIndices = MLXArray(bias.indices, [bias.nonZero])
            }
            let traceBuffer = TraceBuffer(level: level, mask: maskIndices, topK: topK)
            buffer = traceBuffer
            processor = TracingLogitProcessor(injection: injection, penalty: penalty, buffer: traceBuffer)
            sampler = TracingSampler(real: parameters.sampler(), shadow: parameters.sampler(), buffer: traceBuffer)
        }
        if let recorder { processor = CompositeLogitProcessor([processor, recorder]) }
        let iterator = try TokenIterator(
            input: input, model: context.model, cache: cache, processor: processor, sampler: sampler,
            prefillStepSize: parameters.prefillStepSize, maxTokens: parameters.maxTokens)
        let (stream, task) = generateTask(
            promptTokenCount: promptTokens, modelConfiguration: context.configuration,
            tokenizer: context.tokenizer, iterator: iterator, wiredMemoryTicket: wiredMemoryTicket, tools: tools)
        return Run(
            stream: stream, task: task, traceLevel: level, seed: parameters.seed, injected: injection != nil,
            buffer: buffer, cache: measure ? cache : nil, recorder: recorder,
            promptTokens: measure ? input.text.tokens.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init) : [])
    }

    /// Convenience: just the stream.
    public static func generate(
        input: LMInput, parameters: GenerateParameters, context: ModelContext, plan: InjectionPlan?,
        tools: [[String: any Sendable]]? = nil
    ) throws -> AsyncStream<Generation> {
        try start(input: input, parameters: parameters, context: context, plan: plan, trace: .off, tools: tools).stream
    }
}
