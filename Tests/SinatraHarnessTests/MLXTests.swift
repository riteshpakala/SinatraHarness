import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import SinatraHarness

/// Gated: needs `./scripts/metallib.sh debug` and FRIGATE_MLX_TESTS=1.
@Suite("MLX: injection, trace, weight model, encoder", .enabled(if: mlxAvailable))
struct MLXTests {

    @Test func injectionAddsTheBiasAndKeepsTheDtype() {
        let processor = InjectionProcessor(bias: SparseBias(vocabularySize: 8, entries: [3: 2.0, 5: -1.0]))
        let logits = MLXArray((0..<8).map { Float($0) * 0.1 }, [1, 8])
        let values = processor.process(logits: logits).asArray(Float.self)
        #expect(abs(values[3] - 2.3) < 1e-5)
        #expect(abs(values[5] - (-0.5)) < 1e-5)
        #expect(abs(values[0]) < 1e-6)
        #expect(processor.process(logits: logits.asType(.bfloat16)).dtype == .bfloat16)
        let wider = MLXArray.zeros([1, 9])
        #expect(processor.process(logits: wider).asArray(Float.self) == [Float](repeating: 0, count: 9))
    }

    @Test func traceMathIsZeroWithoutAnInjection() {
        let logits = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let y = argMax(logits, axis: -1)
        let step = TraceMath.step(
            base: logits, injected: logits, sampled: y, counterfactual: y,
            mask: MLXArray([Int32(1)], [1]), topK: 2, full: true)
        eval(step.arrays)
        #expect(abs(step.kl.item(Float.self)) < 1e-6)
        #expect(abs(step.entropyPre.item(Float.self) - step.entropyPost.item(Float.self)) < 1e-6)
        #expect(abs(step.massIntoMask.item(Float.self)) < 1e-6)
        // Entropy against a double-precision reference.
        let z: [Double] = [1.0, 2.0, 0.5, -1.0]
        let lse = log(z.map(exp).reduce(0, +))
        let reference = -z.map { exp($0 - lse) * ($0 - lse) }.reduce(0, +)
        #expect(abs(Double(step.entropyPre.item(Float.self)) - reference) < 1e-5)
    }

    @Test func traceMathMeasuresABoost() {
        let base = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let injected = base + MLXArray([0, 0, 3, 0] as [Float], [1, 4])
        let step = TraceMath.step(
            base: base, injected: injected, sampled: argMax(injected, axis: -1),
            counterfactual: argMax(base, axis: -1), mask: MLXArray([Int32(2)], [1]), topK: 2, full: true)
        eval(step.arrays)
        #expect(step.kl.item(Float.self) > 0.1)
        #expect(step.massIntoMask.item(Float.self) > 0.3)
        #expect(step.rankPre.item(Float.self) == 2)  // token 2 trailed 2.0 and 1.0 before the boost
        #expect(step.rankPost.item(Float.self) == 0)
        #expect(step.argmaxPre.item(Int32.self) == 1)
        #expect(step.argmaxPost.item(Int32.self) == 2)
        #expect((step.movement?.asArray(Float.self).first ?? 0) > 0.3)
    }

    @Test func identicallySeededSamplersAgreeOnEqualLogits() {
        let parameters = GenerateParameters(temperature: 0.7, seed: 99)
        let a = parameters.sampler()
        let b = parameters.sampler()
        let logits = MLXArray((0..<50).map { Float(sin(Double($0))) }, [1, 50])
        for _ in 0..<20 {
            #expect(a.sample(logits: logits).item(Int32.self) == b.sample(logits: logits).item(Int32.self))
        }
    }

    @Test func traceBufferRecordsTheCounterfactual() {
        let buffer = TraceBuffer(level: .summary, mask: MLXArray([Int32(2)], [1]), topK: 2)
        let processor = TracingLogitProcessor(
            injection: InjectionProcessor(bias: SparseBias(vocabularySize: 4, entries: [2: 3])),
            penalty: nil, buffer: buffer)
        let sampler = TracingSampler(real: ArgMaxSampler(), shadow: ArgMaxSampler(), buffer: buffer)
        let logits = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let token = sampler.sample(logits: processor.process(logits: logits))
        #expect(token.item(Int32.self) == 2)
        let steps = buffer.drain(limit: nil)
        #expect(steps.count == 1)
        #expect(steps[0].sampled == 2)
        #expect(steps[0].counterfactual == 1)
        #expect(steps[0].kl > 0)
    }

    @Test func weightModelStartsAtZeroLearnsAndRoundTrips() throws {
        let schema = ModelSchema(featureCount: 4, contextDim: 3, projection: "test", featureNames: ["a", "b", "c", "d"])
        let model = MLXWeightModel(schema: schema, learningRate: 1e-2, weightDecay: 0, uptakeWeight: 0.1)
        let features = (0..<64).map { i -> [Float] in [Float(i % 2), 0.5, Float(i % 3) / 3, 1] }
        let context = features.map { _ in [Float](repeating: 0.1, count: 3) }
        #expect(try model.predict(features: features, context: context).steer.allSatisfy { $0 == 0 })

        let targets = features.map { $0[0] > 0.5 ? Float(0.6) : Float(-0.6) }
        let batch = TrainingBatch(
            features: features, context: context, targets: targets,
            sampleWeights: Array(repeating: 1, count: 64), uptakes: Array(repeating: 0.5, count: 64))
        let report = try model.train(batch, budget: 10, maxSteps: 200, shouldAbort: { false })
        #expect(report.steps > 0)
        #expect(model.isTrained)
        let predictions = try model.predict(features: features, context: context).steer
        #expect(predictions[1] > predictions[0])

        let url = temporaryStore().appendingPathComponent("model.safetensors")
        try model.save(to: url, metadata: [:])
        let reloaded = MLXWeightModel(schema: schema)
        _ = try reloaded.load(from: url)
        #expect(reloaded.isTrained)
        for (a, b) in zip(predictions, try reloaded.predict(features: features, context: context).steer) {
            #expect(abs(a - b) < 1e-5)
        }
        let mismatched = MLXWeightModel(schema: ModelSchema(featureCount: 5, contextDim: 3, projection: "test", featureNames: []))
        #expect(throws: SinatraError.self) { _ = try mismatched.load(from: url) }
    }

    @Test func trainingStopsWhenAGenerationIsWaiting() throws {
        let schema = ModelSchema(featureCount: 2, contextDim: 2, projection: "test", featureNames: ["a", "b"])
        let model = MLXWeightModel(schema: schema)
        let batch = TrainingBatch(
            features: Array(repeating: [1, 0], count: 30), context: Array(repeating: [0, 0], count: 30),
            targets: Array(repeating: 0.5, count: 30), sampleWeights: Array(repeating: 1, count: 30),
            uptakes: Array(repeating: 0.5, count: 30))
        let report = try model.train(batch, budget: 10, maxSteps: 500, shouldAbort: { true })
        #expect(report.steps == 0)
        #expect(report.stoppedBy == "aborted")
    }

    @Test func encoderFindsTheEmbeddingTableByKeyPath() throws {
        let model = TinyLanguageModel(vocabulary: 12, hidden: 8)
        let encoder = try LanguageModelContextEncoder(model: model, modelKey: "tiny")
        #expect(encoder.hiddenSize == 8)
        #expect(encoder.vocabularySize == 12)
        #expect(encoder.embeddingPath == "model.embed_tokens")
        #expect(encoder.headPath == "lm_head")
        let pooled = try encoder.encode(TokenBatch(rows: [[1, 2, 3], [4]]))
        #expect(pooled.count == 2 && pooled[0].count == 8)
        let row = model.model.embedTokens.weight[4].asArray(Float.self)
        for (a, b) in zip(pooled[1], row) { #expect(abs(a - b) < 1e-5) }
        let logits = try #require(try encoder.outputLogits(pooled))
        #expect(logits.count == 2 && logits[0].count == 12)
        #expect(throws: SinatraError.self) { _ = try LanguageModelContextEncoder(model: Linear(2, 2), modelKey: "none") }
    }
}

/// A causal toy LM over one KVCacheSimple: each position's logits come from the running
/// mean of every embedding up to it. Trimming the cache wrongly changes its answers, which
/// is what the scorer tests need.
final class TinyCausalLM: Module, LanguageModel {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "head") var head: Linear

    init(vocabulary: Int, hidden: Int) {
        self._embed.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: hidden)
        self._head.wrappedValue = Linear(hidden, vocabulary)
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let length = inputs.dim(1)
        let e = embed(inputs).expandedDimensions(axis: 1)  // [1, 1, L, H]
        let layer = cache?.first ?? KVCacheSimple()
        let previous = layer.offset
        let (keys, _) = layer.update(keys: e, values: e)
        let total = previous + length
        let all = keys.squeezed(axis: 1)  // [1, total, H]
        let counts = MLXArray((1...total).map { Float($0) }, [1, total, 1])
        let means = cumsum(all, axis: 1) / counts
        return head(tanh(means[0..., previous..<total, 0...]) * 3)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
}

@Suite("MLX: the grounding scorer", .enabled(if: mlxAvailable))
struct GroundingScorerTests {
    let model = TinyCausalLM(vocabulary: 32, hidden: 16)
    let context = [1, 5, 6, 7, 8, 9, 2]
    let bare = [1, 5, 6, 2]
    let sampled = [10, 11, 12, 13, 14, 3]

    /// The cache a decode leaves behind: the prompt, then every sampled token fed back.
    func decodeCache() -> [KVCache] {
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(context.map { Int32($0) }, [1, context.count]), cache: cache)
        for token in sampled { _ = model(MLXArray([Int32(token)], [1, 1]), cache: cache) }
        eval(cache.flatMap { $0.innerState() })
        return cache
    }

    @Test func reusingTheDecodeCacheIsExact() throws {
        eval(model)
        let options = GroundingScorer.Options(topK: 8, chunk: 4)
        let reused = try GroundingScorer.score(
            model: model, decodeCache: decodeCache(), contextTokens: context, bareTokens: bare, sampled: sampled, options: options)
        let fresh = try GroundingScorer.score(
            model: model, decodeCache: nil, contextTokens: context, bareTokens: bare, sampled: sampled, options: options)
        #expect(reused.cacheReused && !fresh.cacheReused)
        #expect(reused.sharedPrefixTokens == 3)
        let ctxReference = try GroundingScorer.tokenLogprobs(model: model, prompt: context, continuation: sampled)
        let bareReference = try GroundingScorer.tokenLogprobs(model: model, prompt: bare, continuation: sampled)
        for t in sampled.indices {
            #expect(abs(reused.logpCtx[t] - fresh.logpCtx[t]) < 1e-4)
            #expect(abs(reused.logpBare[t] - fresh.logpBare[t]) < 1e-4)
            #expect(abs(reused.contextKL[t] - fresh.contextKL[t]) < 1e-4)
            #expect(abs(reused.logpCtx[t] - ctxReference[t]) < 1e-4)
            #expect(abs(reused.logpBare[t] - bareReference[t]) < 1e-4)
            #expect(reused.contextKL[t] >= -1e-5)
        }
        #expect(reused.tune.count == sampled.count && reused.tuneNats.allSatisfy { $0 >= -1e-5 })
    }

    @Test func noContextMovesNothing() throws {
        eval(model)
        let raw = try GroundingScorer.score(
            model: model, decodeCache: nil, contextTokens: bare, bareTokens: bare, sampled: sampled,
            options: GroundingScorer.Options(topK: 8, chunk: 4, detail: .full))
        for t in sampled.indices {
            #expect(abs(raw.logpCtx[t] - raw.logpBare[t]) < 1e-6)
            #expect(abs(raw.contextKL[t]) < 1e-6)
            #expect(abs(raw.tuneNats[t]) < 1e-6)
        }
        #expect(raw.rankBare?.count == sampled.count && raw.pushes?.count == sampled.count)
    }

    @Test func aWaitingGenerationStopsTheMeasurement() throws {
        eval(model)
        #expect(throws: GroundingScorer.Stop.waiting) {
            _ = try GroundingScorer.score(
                model: model, decodeCache: nil, contextTokens: context, bareTokens: bare, sampled: sampled,
                options: GroundingScorer.Options(), shouldAbort: { true })
        }
        #expect(throws: GroundingScorer.Stop.self) {
            _ = try GroundingScorer.score(
                model: model, decodeCache: nil, contextTokens: context, bareTokens: bare, sampled: [],
                options: GroundingScorer.Options())
        }
    }
}

final class TinyInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    init(vocabulary: Int, hidden: Int) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: hidden)
        super.init()
    }
}

final class TinyLanguageModel: Module {
    @ModuleInfo(key: "model") var model: TinyInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(vocabulary: Int, hidden: Int) {
        self._model.wrappedValue = TinyInner(vocabulary: vocabulary, hidden: hidden)
        self._lmHead.wrappedValue = Linear(hidden, vocabulary, bias: false)
        super.init()
    }
}
