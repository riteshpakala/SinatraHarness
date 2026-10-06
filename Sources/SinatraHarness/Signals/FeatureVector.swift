//
//  FeatureVector.swift
//  SinatraHarness
//
//  WHAT: The fixed per-partition feature layout (F = 38) the weight model reads.
//        26 static features captured at prepare time, 11 indicators over the owner's
//        grounding series rebuilt as of the turn under the active windows, and a
//        context-validity flag.
//  PIN:  Column order is part of the saved model's schema. Append, never reorder.
//        The relevancy band's weight is deliberately NOT an input: it multiplies the
//        injection and decays sample weights outside the model. As a step-function input
//        it made every document crossing day 7 an unseen region the net extrapolated into.
//        Recency enters as continuous ages instead (docAge, modifiedAge, firstSeenAge).
//

import Foundation

public enum FeatureVector {

    public static let staticNames: [String] = [
        "docAge", "docAgeKnown", "modifiedAge", "modifiedKnown",
        "firstSeenAge", "isNew", "retrievalRecency",
        "freq30", "freq7",
        "priorSteer", "priorConfidence", "docPriorSteer", "priorUptake", "docPriorDrift",
        "rank01", "score01", "partitionSize", "relevancy",
        "lastSteer", "lastGrounding", "lastDrift",
        "hourSin", "hourCos", "dowSin", "dowCos",
        "queryGap",
    ]

    public static let names: [String] = staticNames + IndicatorSeries.names + ["ctxValid"]

    public static var count: Int { names.count }
    public static var staticCount: Int { staticNames.count }

    /// Static features, indicators and the context flag, in schema order.
    static func assemble(static values: [Float], indicators: [Float], contextValid: Bool) -> [Float] {
        precondition(values.count == staticCount && indicators.count == IndicatorSeries.names.count)
        return values + indicators + [contextValid ? 1 : 0]
    }
}
