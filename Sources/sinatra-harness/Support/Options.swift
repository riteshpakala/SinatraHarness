//
//  Options.swift
//  sinatra-harness
//

import ArgumentParser
import Foundation
import MLXLMCommon
import SinatraHarness

struct ModelOptions: ParsableArguments {
    @Option(help: "Hugging Face model id, e.g. mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit.")
    var model: String?

    @Option(help: "A model directory already on disk (instead of --model).")
    var modelDir: String?

    func load(_ harness: Harness) async throws {
        if let modelDir {
            let url = URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath)
            try await harness.load(directory: url, modelKey: model ?? url.lastPathComponent)
        } else if let model {
            var lastReported = -1
            try await harness.load(modelID: model) { fraction in
                let percent = Int(fraction * 100)
                if percent / 10 != lastReported / 10 {
                    lastReported = percent
                    FileHandle.standardError.write(Data("loading \(percent)%\n".utf8))
                }
            }
        } else {
            throw ValidationError("Pass --model or --model-dir.")
        }
    }
}

struct StoreOptions: ParsableArguments {
    @Option(help: "Store directory for ledgers, weight models and traces (default ~/.sinatra-harness).")
    var store: String?

    @Option(help: "Owner the personalization belongs to.")
    var owner: String = "demo"

    var storeURL: URL {
        let path = store ?? "~/.sinatra-harness"
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    var ownerID: OwnerID { OwnerID(owner) }
}

struct SamplingOptions: ParsableArguments {
    @Option(help: "Maximum tokens to generate.")
    var maxTokens: Int = 256

    @Option(help: "Sampling temperature (0 = greedy).")
    var temperature: Float = 0

    @Option(help: "Top-p.")
    var topP: Float = 1.0

    @Option(help: "Repetition penalty (unset = none).")
    var repetitionPenalty: Float?

    @Option(help: "Sampler seed (fixed seeds make decodes comparable).")
    var seed: UInt64?

    var parameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens, temperature: temperature, topP: topP,
            repetitionPenalty: repetitionPenalty, seed: seed ?? 1)
    }
}

struct InjectionOptions: ParsableArguments {
    @Option(help: "Injection mode: off, lexical, dense.")
    var mode: BiasMode = .lexical

    @Option(help: "Injection strength α.")
    var alpha: Float?

    @Option(help: "Cap on |bias| in nats.")
    var cap: Float?

    @Option(help: "Override per-partition weights, comma separated, e.g. \"1.0,-0.5\".")
    var forceWeights: String?

    var weights: [Float]? {
        forceWeights?.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
    }

    func apply(to configuration: inout SinatraConfiguration) {
        configuration.biasMode = mode
        if let alpha { configuration.alpha = alpha }
        if let cap { configuration.cap = cap }
    }
}

struct PromptOptions: ParsableArguments {
    @Option(help: "The user's question.")
    var prompt: String

    @Option(help: "A context file, optionally with its creation date: path or path:2026-09-20 (repeatable).")
    var context: [String] = []

    @Option(help: "Optional system prompt.")
    var system: String?

    func partitions() throws -> [Partition] {
        try context.flatMap { try ContextFile.partitions(from: $0) }
    }

    func userInput(partitions: [Partition]) -> UserInput {
        var messages: [Chat.Message] = []
        if let system { messages.append(.system(system)) }
        messages.append(.user(ContextFile.userMessage(prompt: prompt, partitions: partitions)))
        return UserInput(chat: messages)
    }

    /// The same chat without its context: what the grounding measurement scores against.
    func bareInput() -> UserInput {
        var messages: [Chat.Message] = []
        if let system { messages.append(.system(system)) }
        messages.append(.user(ContextFile.bareMessage(prompt: prompt)))
        return UserInput(chat: messages)
    }
}

enum ContextFile {
    /// `path` or `path:ISO-date`; the date may itself contain colons.
    static func parse(_ argument: String) -> (path: String, date: Date?) {
        var search = argument.startIndex
        while let colon = argument[search...].firstIndex(of: ":") {
            let suffix = String(argument[argument.index(after: colon)...])
            if let date = parseDate(suffix) {
                return (String(argument[..<colon]), date)
            }
            search = argument.index(after: colon)
        }
        return (argument, nil)
    }

    static func parseDate(_ text: String) -> Date? {
        let full = ISO8601DateFormatter()
        if let date = full.date(from: text) { return date }
        let day = ISO8601DateFormatter()
        day.formatOptions = [.withFullDate]
        return day.date(from: text)
    }

    /// Split a file into paragraph-aligned partitions of about `size` characters.
    static func partitions(from argument: String, size: Int = 1200) throws -> [Partition] {
        let (path, date) = parse(argument)
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let text = try String(contentsOf: url, encoding: .utf8)
        let paragraphs = text.components(separatedBy: "\n\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        var chunks: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if !current.isEmpty && current.count + paragraph.count > size {
                chunks.append(current)
                current = ""
            }
            current += current.isEmpty ? paragraph : "\n\n" + paragraph
        }
        if !current.isEmpty { chunks.append(current) }
        let name = url.lastPathComponent
        return chunks.enumerated().map { index, chunk in
            Partition(id: "\(name)#\(index)", documentId: url.path, text: chunk, score: Float(index) * 0.01, createdAt: date)
        }
    }

    static func userMessage(prompt: String, partitions: [Partition]) -> String {
        guard !partitions.isEmpty else { return bareMessage(prompt: prompt) }
        let context = partitions.enumerated().map { "[\($0.offset + 1)] \($0.element.text)" }.joined(separator: "\n\n")
        return "Use the context below to answer.\n\n<context>\n\(context)\n</context>\n\n" + bareMessage(prompt: prompt)
    }

    /// The question alone, framed as it is after the context.
    static func bareMessage(prompt: String) -> String {
        "Question: \(prompt)"
    }
}

func printJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    print(String(data: try encoder.encode(value), encoding: .utf8) ?? "")
}
