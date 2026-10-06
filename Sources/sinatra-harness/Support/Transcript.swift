//
//  Transcript.swift
//  sinatra-harness
//
//  WHAT: Transcript files for `replay`, and a deterministic synthetic owner whose context
//        keeps pointing at one document the stand-in model tends to leave out — to watch
//        the grounding loop learn to steer toward it.
//

import Foundation
import SinatraHarness

struct TranscriptTurn: Codable {
    var at: Date
    var conversationId: String?
    /// The user's message. It goes into the prompt; SinatraHarness never reads it.
    var user: String
    var context: [Partition]
    var assistant: String?
    var assistantAt: Date?
    /// Model-free replay only: how hard the context "really" pushes toward each partition
    /// (partition id → nats), standing in for the model's own measurement.
    var emphasis: [String: Float]?
}

enum Transcript {
    static func read(_ url: URL) throws -> [TranscriptTurn] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([TranscriptTurn].self, from: Data(contentsOf: url))
    }

    static func write(_ turns: [TranscriptTurn], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(turns).write(to: url)
    }
}

enum Synthesizer {
    static let documents: [(id: String, text: String)] = [
        ("garden", "The raised garden beds get morning sun, so tomatoes and basil thrive there. Compost from the kitchen scraps feeds the soil every spring, and drip irrigation keeps the roots evenly watered through August heat."),
        ("sourdough", "The sourdough starter needs feeding twice a day with equal weights of flour and water. A long cold proof overnight deepens the flavour and gives the crust its blistered, crackling finish."),
        ("cycling", "Interval sessions on the bike build threshold power: four blocks of eight minutes near the limit with short recoveries. Cadence around ninety keeps the legs fresh on long climbs."),
        ("budget", "The monthly budget splits income into fixed costs, savings and discretionary spending. Automating the transfer to savings on payday removes the temptation to skip it."),
        ("piano", "Scales in contrary motion strengthen finger independence at the piano. Slow practice with a metronome, then gradual tempo increases, makes difficult passages reliable."),
        ("travel", "Packing cubes keep a carry-on organised for a two week trip. Rolling clothes saves space, and a single pair of versatile shoes avoids overpacking."),
    ]

    /// The document the context keeps pointing at.
    static let preferred = "garden"

    static func transcript(turns count: Int, seed: UInt64, start: Date) -> [TranscriptTurn] {
        var rng = SplitMix64(seed: seed)
        var turns: [TranscriptTurn] = []
        var clock = start
        var conversation = 1
        for _ in 0..<count {
            var chosen = Set<String>()
            if Double.random(in: 0..<1, using: &rng) < 0.6 { chosen.insert(preferred) }
            while chosen.count < 3 { chosen.insert(documents[Int.random(in: 0..<documents.count, using: &rng)].id) }
            let context = chosen.sorted().enumerated().map { rank, id -> Partition in
                let text = documents.first { $0.id == id }!.text
                return Partition(id: "\(id)#0", documentId: "docs/\(id)", text: text, score: Float(rank) * 0.1)
            }
            var emphasis: [String: Float] = [:]
            for partition in context { emphasis[partition.id] = partition.id.hasPrefix(preferred) ? 2.0 : 0.3 }
            turns.append(TranscriptTurn(
                at: clock, conversationId: "c\(conversation)", user: "What should I focus on this week?",
                context: context, assistant: nil, assistantAt: nil, emphasis: emphasis))
            clock = clock.addingTimeInterval(Double.random(in: 60...900, using: &rng))
            if Double.random(in: 0..<1, using: &rng) < 0.1 {
                clock = clock.addingTimeInterval(Double.random(in: 7200...86_000, using: &rng))
                conversation += 1
            }
        }
        return turns
    }

    /// A stand-in answer: the first sentence of the two most strongly steered partitions
    /// (retrieval order breaks ties), so an unsteered model leaves the third one out.
    static func answer(for context: [Partition], weights: [String: Float]) -> String {
        context.enumerated()
            .sorted { lhs, rhs in
                let a = weights[lhs.element.id] ?? 0
                let b = weights[rhs.element.id] ?? 0
                return a == b ? lhs.offset < rhs.offset : a > b
            }
            .prefix(2)
            .compactMap { $0.element.text.split(separator: ".").first.map { String($0) + "." } }
            .joined(separator: " ")
    }
}
