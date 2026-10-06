//
//  LanguageModelContextEncoder.swift
//  SinatraHarness
//
//  WHAT: Encodes retrieved partitions with the loaded LLM's own input embedding table,
//        mean-pooled per partition — the "Encoder" in front of SinatraHarness in the diagram.
//        Also projects vectors through the model's output head for the dense mode.
//  PIN:  Found by key path through `Module.namedModules()` ("model.embed_tokens",
//        "lm_head"), so any MLXLLM model works without touching its source. A 4-bit
//        QuantizedEmbedding dequantises only the rows looked up. The prompt never comes here.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class LanguageModelContextEncoder: ContextEncoding {
    public let identifier: String
    public let hiddenSize: Int
    public let vocabularySize: Int?
    public let embeddingPath: String
    public let headPath: String?

    let embedding: Embedding
    let head: Linear?
    private var computeType: DType?

    /// `model` is any MLX module tree — a loaded `LanguageModel` in practice.
    public init(
        model: Module, modelKey: String,
        embeddingKeyPath: String? = nil, outputKeyPath: String? = nil
    ) throws {
        let named = model.namedModules()
        var byPath: [String: Module] = [:]
        for (path, module) in named where byPath[path] == nil { byPath[path] = module }

        var embedding: (String, Embedding)?
        for path in [embeddingKeyPath, "model.embed_tokens", "language_model.model.embed_tokens"].compactMap({ $0 }) {
            if let found = byPath[path] as? Embedding { embedding = (path, found); break }
        }
        if embedding == nil {
            let suffixes = ["embed_tokens", "wte", "tok_embeddings", "word_embeddings", "token_embedding"]
            if let match = named.first(where: { path, module in
                module is Embedding && suffixes.contains(where: { path.hasSuffix($0) })
            }), let found = match.1 as? Embedding {
                embedding = (match.0, found)
            }
        }
        guard let (embeddingPath, table) = embedding else {
            throw SinatraError.embeddingNotFound(
                "no Embedding at \(embeddingKeyPath ?? "model.embed_tokens") among \(named.count) modules")
        }

        var head: (String, Linear)?
        for path in [outputKeyPath, "lm_head", "language_model.lm_head"].compactMap({ $0 }) {
            if let found = byPath[path] as? Linear { head = (path, found); break }
        }
        if head == nil, let match = named.first(where: { $0.0.hasSuffix("lm_head") && $0.1 is Linear }),
            let found = match.1 as? Linear
        {
            head = (match.0, found)
        }

        let (rows, dimensions) = table.shape
        self.embedding = table
        self.embeddingPath = embeddingPath
        self.head = head?.1
        self.headPath = head?.0
        self.hiddenSize = dimensions
        self.vocabularySize = head.map { $0.1.shape.0 } ?? rows
        self.identifier = "\(modelKey)#\(embeddingPath)"
    }

    public func encode(_ batch: TokenBatch) throws -> [[Float]] {
        guard batch.count > 0 else { return [] }
        let padded = batch.padded()
        let ids = MLXArray(padded.ids, [padded.rows, padded.columns])
        let mask = MLXArray(padded.mask, [padded.rows, padded.columns, 1])
        let embedded = embedding(ids).asType(.float32)
        let summed = sum(embedded * mask, axis: 1)
        let lengths = MLXArray(batch.lengths.map { Float(max($0, 1)) }, [padded.rows, 1])
        let pooled = summed / lengths
        eval(pooled)
        let flat = pooled.asArray(Float.self)
        let h = pooled.dim(1)
        return (0..<padded.rows).map { Array(flat[($0 * h)..<(($0 + 1) * h)]) }
    }

    public func outputLogits(_ vectors: [[Float]]) throws -> [[Float]]? {
        guard !vectors.isEmpty else { return [] }
        let n = vectors.count
        let flatInput = vectors.flatMap { row -> [Float] in
            row.count == hiddenSize ? row : Array((row + [Float](repeating: 0, count: hiddenSize)).prefix(hiddenSize))
        }
        let x = MLXArray(flatInput, [n, hiddenSize]).asType(dtype())
        let logits = head.map { $0(x) } ?? embedding.asLinear(x)
        let out = logits.asType(.float32)
        eval(out)
        let v = out.dim(1)
        let flat = out.asArray(Float.self)
        return (0..<n).map { Array(flat[($0 * v)..<(($0 + 1) * v)]) }
    }

    /// The dtype the model computes in: whatever its embedding rows dequantise to.
    private func dtype() -> DType {
        if let computeType { return computeType }
        let probe = embedding(MLXArray([Int32(0)], [1]))
        computeType = probe.dtype
        return probe.dtype
    }
}
