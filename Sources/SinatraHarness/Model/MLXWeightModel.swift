//
//  MLXWeightModel.swift
//  SinatraHarness
//
//  WHAT: SinatraNet behind the `WeightModeling` contract: prediction, a bounded AdamW
//        training loop, and safetensors persistence with a schema check.
//  PIN:  Runs on the CPU device. The net is tiny, CPU avoids kernel-launch overhead, and
//        the LLM keeps the GPU to itself (the harness also never overlaps the two).
//        Loss: Σ s·(w − y)² / Σ s  +  0.25 · mean((û − u)²); weight decay via AdamW.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

public final class MLXWeightModel: WeightModeling {
    public let schema: ModelSchema
    public private(set) var isTrained = false
    let learningRate: Float
    let weightDecay: Float
    let uptakeWeight: Float
    private var net: SinatraNet
    private let device = Device.cpu

    public init(schema: ModelSchema, learningRate: Float = 2e-3, weightDecay: Float = 1e-2, uptakeWeight: Float = 0.25) {
        self.schema = schema
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.uptakeWeight = uptakeWeight
        self.net = Device.withDefaultDevice(Device.cpu) {
            SinatraNet(featureCount: schema.featureCount, contextDim: schema.contextDim)
        }
    }

    public static let factory: WeightModelFactory = { MLXWeightModel(schema: $0) }

    public static func factory(configuration: SinatraConfiguration) -> WeightModelFactory {
        let learningRate = configuration.learningRate
        let weightDecay = configuration.weightDecay
        let uptakeWeight = configuration.uptakeLossWeight
        return {
            MLXWeightModel(
                schema: $0, learningRate: learningRate, weightDecay: weightDecay, uptakeWeight: uptakeWeight)
        }
    }

    public func reset() {
        net = Device.withDefaultDevice(device) {
            SinatraNet(featureCount: schema.featureCount, contextDim: schema.contextDim)
        }
        isTrained = false
    }

    private func arrays(_ features: [[Float]], _ context: [[Float]]) -> (MLXArray, MLXArray) {
        let n = features.count
        func rows(_ rows: [[Float]], width: Int) -> [Float] {
            var flat = [Float]()
            flat.reserveCapacity(n * width)
            for row in rows {
                if row.count == width {
                    flat.append(contentsOf: row)
                } else {
                    flat.append(contentsOf: row.prefix(width))
                    if row.count < width { flat.append(contentsOf: [Float](repeating: 0, count: width - row.count)) }
                }
            }
            return flat
        }
        return (
            MLXArray(rows(features, width: schema.featureCount), [n, schema.featureCount]),
            MLXArray(rows(context, width: schema.contextDim), [n, schema.contextDim]))
    }

    public func predict(features: [[Float]], context: [[Float]]) throws -> WeightPrediction {
        guard !features.isEmpty else { return WeightPrediction(steer: [], uptake: []) }
        return Device.withDefaultDevice(device) {
            let (x, c) = arrays(features, context)
            net.train(false)
            let (w, u) = net(features: x, context: c)
            eval(w, u)
            return WeightPrediction(steer: w.asArray(Float.self), uptake: u.asArray(Float.self))
        }
    }

    public func train(_ batch: TrainingBatch, budget: TimeInterval, maxSteps: Int, shouldAbort: () -> Bool) throws -> ModelFitReport {
        guard !batch.isEmpty else {
            return ModelFitReport(steps: 0, initialLoss: nil, finalLoss: nil, elapsed: 0, stoppedBy: "empty")
        }
        let started = Date()
        // Warm-started every cycle, so early stopping is what keeps it from memorising:
        // the most recent 20% of rows (chronological) validate, the best weights are kept.
        let validationCount = batch.count >= 20 ? max(4, batch.count / 5) : 0
        let fitRows = Array(0..<(batch.count - validationCount))
        let validationRows = Array((batch.count - validationCount)..<batch.count)
        let fit = batch.subset(fitRows)
        let validation = batch.subset(validationRows)

        return Device.withDefaultDevice(device) {
            func tensors(_ b: TrainingBatch) -> [MLXArray] {
                let (x, c) = arrays(b.features, b.context)
                return [x, c, MLXArray(b.targets, [b.count]), MLXArray(b.sampleWeights, [b.count]),
                        MLXArray(b.uptakes, [b.count])]
            }
            let uptakeWeight = self.uptakeWeight
            func loss(_ model: SinatraNet, _ a: [MLXArray]) -> MLXArray {
                let (w, uptake) = model(features: a[0], context: a[1])
                let weighted = sum(square(w - a[2]) * a[3]) / (sum(a[3]) + 1e-6)
                let uptakeLoss = mean(square(uptake - a[4]))
                return weighted + uptakeWeight * uptakeLoss
            }
            let fitTensors = tensors(fit)
            let validationTensors = validation.isEmpty ? nil : tensors(validation)
            let lossAndGradient = valueAndGrad(model: net) { (model: SinatraNet, a: [MLXArray]) -> [MLXArray] in
                [loss(model, a)]
            }
            let optimizer = AdamW(learningRate: learningRate, weightDecay: weightDecay)

            var steps = 0
            var initial: Float?
            var last: Float?
            var stoppedBy = "steps"
            var best: (loss: Float, parameters: ModuleParameters)?
            var sinceBest = 0
            net.train(false)
            if let validationTensors {
                let start = loss(net, validationTensors)
                eval(start)
                best = (start.item(Float.self), net.parameters())
            }
            while steps < maxSteps {
                if shouldAbort() { stoppedBy = "aborted"; break }
                if Date().timeIntervalSince(started) >= budget { stoppedBy = "budget"; break }
                net.train(true)
                let (losses, gradients) = lossAndGradient(net, fitTensors)
                net.train(false)
                let (clipped, _) = clipGradNorm(gradients: gradients, maxNorm: 1.0)
                optimizer.update(model: net, gradients: clipped)
                eval(net, optimizer, losses[0])
                let value = losses[0].item(Float.self)
                if initial == nil { initial = value }
                last = value
                steps += 1
                if let validationTensors, steps % 5 == 0 {
                    let checked = loss(net, validationTensors)
                    eval(checked)
                    let validationLoss = checked.item(Float.self)
                    if validationLoss < (best?.loss ?? .infinity) - 1e-5 {
                        best = (validationLoss, net.parameters())
                        sinceBest = 0
                    } else {
                        sinceBest += 5
                        if sinceBest >= 25 { stoppedBy = "early-stop"; break }
                    }
                }
            }
            if let best, validationTensors != nil {
                net.update(parameters: best.parameters)
                eval(net)
            }
            if steps > 0 { isTrained = true }
            return ModelFitReport(
                steps: steps, initialLoss: initial, finalLoss: last,
                elapsed: Date().timeIntervalSince(started), stoppedBy: stoppedBy)
        }
    }

    public func save(to url: URL, metadata: [String: String]) throws {
        var meta = metadata
        meta["trained"] = isTrained ? "true" : "false"
        if meta["schema"] == nil, let data = try? JSONEncoder().encode(schema) {
            meta["schema"] = String(data: data, encoding: .utf8)
        }
        let parameters = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try MLX.save(arrays: parameters, metadata: meta, url: url)
    }

    public func load(from url: URL) throws -> [String: String] {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard let text = metadata["schema"], let data = text.data(using: .utf8),
            let saved = try? JSONDecoder().decode(ModelSchema.self, from: data)
        else {
            throw SinatraError.schemaMismatch("no schema in \(url.lastPathComponent)")
        }
        guard saved == schema else {
            throw SinatraError.schemaMismatch("saved \(saved.featureCount)×\(saved.contextDim) \(saved.projection), expected \(schema.featureCount)×\(schema.contextDim) \(schema.projection)")
        }
        try Device.withDefaultDevice(device) {
            let fresh = SinatraNet(featureCount: schema.featureCount, contextDim: schema.contextDim)
            try fresh.update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
            eval(fresh)
            net = fresh
        }
        isTrained = metadata["trained"] == "true"
        return metadata
    }
}
