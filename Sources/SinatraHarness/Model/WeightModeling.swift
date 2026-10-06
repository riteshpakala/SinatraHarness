//
//  WeightModeling.swift
//  SinatraHarness
//
//  WHAT: The time-series side model's contract: per-partition features and context in,
//        a steer w_p ∈ [−1, 1] and an uptake forecast out. `MLXWeightModel` is the real one
//        (SinatraNet on MLX); `PriorWeightModel` never learns, for model-free runs.
//

import Foundation

public struct ModelSchema: Codable, Sendable, Equatable {
    public var version: Int
    public var featureCount: Int
    public var contextDim: Int
    public var projection: String
    public var featureNames: [String]

    public init(featureCount: Int, contextDim: Int, projection: String, featureNames: [String], version: Int = 3) {
        self.version = version
        self.featureCount = featureCount
        self.contextDim = contextDim
        self.projection = projection
        self.featureNames = featureNames
    }

    public static func current(_ configuration: SinatraConfiguration) -> ModelSchema {
        ModelSchema(
            featureCount: FeatureVector.count, contextDim: configuration.contextDim,
            projection: CountSketchProjection.version, featureNames: FeatureVector.names)
    }
}

public struct WeightPrediction: Sendable, Equatable {
    /// w_net per partition, in [−1, 1].
    public var steer: [Float]
    /// The partition's forecast share of the context's influence on the output, in [0, 1].
    public var uptake: [Float]
}

/// Rows for one training or evaluation pass.
public struct TrainingBatch: Sendable {
    public var features: [[Float]]
    public var context: [[Float]]
    /// y_p ∈ [−1, 1], the steer head's target.
    public var targets: [Float]
    public var sampleWeights: [Float]
    /// uptake_p ∈ [0, 1], the uptake head's target.
    public var uptakes: [Float]

    public init(features: [[Float]], context: [[Float]], targets: [Float], sampleWeights: [Float], uptakes: [Float]) {
        self.features = features
        self.context = context
        self.targets = targets
        self.sampleWeights = sampleWeights
        self.uptakes = uptakes
    }

    public var count: Int { targets.count }
    public var isEmpty: Bool { targets.isEmpty }

    public func subset(_ rows: [Int]) -> TrainingBatch {
        TrainingBatch(
            features: rows.map { features[$0] }, context: rows.map { context[$0] },
            targets: rows.map { targets[$0] }, sampleWeights: rows.map { sampleWeights[$0] },
            uptakes: rows.map { uptakes[$0] })
    }

    /// Weighted MAE of constant `prediction` — the baseline a model must beat.
    public func constantMAE(_ prediction: Float) -> Float {
        weightedMAE(Array(repeating: prediction, count: count))
    }

    public func weightedMAE(_ predictions: [Float]) -> Float {
        var total: Float = 0
        var weight: Float = 0
        for i in 0..<min(count, predictions.count) {
            total += sampleWeights[i] * abs(predictions[i] - targets[i])
            weight += sampleWeights[i]
        }
        return weight > 0 ? total / weight : 0
    }

    public var weightedMeanTarget: Float {
        var total: Float = 0
        var weight: Float = 0
        for i in 0..<count {
            total += sampleWeights[i] * targets[i]
            weight += sampleWeights[i]
        }
        return weight > 0 ? total / weight : 0
    }
}

public struct ModelFitReport: Sendable, Codable, Equatable {
    public var steps: Int
    public var initialLoss: Float?
    public var finalLoss: Float?
    public var elapsed: TimeInterval
    public var stoppedBy: String
}

public protocol WeightModeling: AnyObject {
    var schema: ModelSchema { get }
    var isTrained: Bool { get }
    func predict(features: [[Float]], context: [[Float]]) throws -> WeightPrediction
    func train(_ batch: TrainingBatch, budget: TimeInterval, maxSteps: Int, shouldAbort: () -> Bool) throws -> ModelFitReport
    func save(to url: URL, metadata: [String: String]) throws
    /// Returns the saved metadata. Throws `SinatraError.schemaMismatch` for another schema.
    func load(from url: URL) throws -> [String: String]
    func reset()
}

extension WeightModeling {
    /// Weighted MAE of the steer against the targets.
    public func evaluate(_ batch: TrainingBatch) throws -> Float {
        guard !batch.isEmpty else { return 0 }
        return batch.weightedMAE(try predict(features: batch.features, context: batch.context).steer)
    }
}

public typealias WeightModelFactory = @Sendable (ModelSchema) -> any WeightModeling

/// Never learns: the session falls back to its ledger priors.
public final class PriorWeightModel: WeightModeling {
    public let schema: ModelSchema
    public let isTrained = false

    public init(schema: ModelSchema) { self.schema = schema }

    public static let factory: WeightModelFactory = { PriorWeightModel(schema: $0) }

    public func predict(features: [[Float]], context: [[Float]]) throws -> WeightPrediction {
        WeightPrediction(
            steer: Array(repeating: 0, count: features.count),
            uptake: Array(repeating: 0, count: features.count))
    }

    public func train(_ batch: TrainingBatch, budget: TimeInterval, maxSteps: Int, shouldAbort: () -> Bool) throws -> ModelFitReport {
        ModelFitReport(steps: 0, initialLoss: nil, finalLoss: nil, elapsed: 0, stoppedBy: "prior-only")
    }

    public func save(to url: URL, metadata: [String: String]) throws {}
    public func load(from url: URL) throws -> [String: String] { [:] }
    public func reset() {}
}
