//
//  Readers.swift
//  sinatra-harness
//
//  WHAT: Read-only views of a store: traces, grounding analysis, ledgers, features, and
//        the encoder's view of a few texts — plus leave-one-out attribution.
//

import ArgumentParser
import Foundation
import SinatraHarness

/// A session that only reads the store (no model, no encoder).
func readerSession(_ store: StoreOptions) -> SinatraSession {
    SinatraSession(
        storeDirectory: store.storeURL, modelKey: "reader", tokenizer: SimpleTokenizer(),
        encoder: nil, vocabularySize: 0)
}

struct TraceCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "trace", abstract: "Render a stored turn trace — injection and grounding (the latest by default).")

    @OptionGroup var store: StoreOptions

    @Option(help: "Turn id of the trace.")
    var turn: String?

    @Option(help: "Steps to print.")
    var steps = 80

    @Flag(help: "Print JSON.")
    var json = false

    func run() async throws {
        let session = readerSession(store)
        let trace: InjectionTrace?
        if let turn {
            guard let id = UUID(uuidString: turn) else { throw ValidationError("--turn is not a UUID") }
            trace = await session.trace(owner: store.ownerID, turnId: id)
        } else {
            trace = await session.latestTrace(owner: store.ownerID)
        }
        guard let trace else {
            print("No trace for \(store.owner) in \(store.storeURL.path).")
            return
        }
        if json {
            try printJSON(trace)
        } else {
            if trace.level != .off {
                Render.summary(trace.summary)
                Render.steps(trace, limit: steps)
            }
            if let grounding = trace.grounding { Render.grounding(grounding, steps: steps) }
        }
    }
}

struct Analyze: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Grounding over time: does steering reduce drift, and which documents does the output cite?")

    @OptionGroup var store: StoreOptions

    @Flag(help: "Print JSON.")
    var json = false

    func run() async throws {
        let report = await readerSession(store).groundingReport(owner: store.ownerID)
        if json { try printJSON(report) } else { Render.groundingReport(report) }
    }
}

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Summarise an owner's ledger (or print it).")

    @OptionGroup var store: StoreOptions

    @Flag(help: "Print the full ledger JSON.")
    var ledger = false

    @Option(help: "Show the weight model's per-row targets and predictions for this model key.")
    var rows: String?

    func run() async throws {
        if let rows {
            let session = SinatraSession(
                storeDirectory: store.storeURL, modelKey: rows, tokenizer: SimpleTokenizer(), encoder: nil,
                vocabularySize: 0, weightModelFactory: MLXWeightModel.factory(configuration: SinatraConfiguration()))
            let evaluated = try await session.evaluationRows(owner: store.ownerID)
            var byPartition: [String: (fit: [Float], hold: [Float], targetsFit: [Float], targetsHold: [Float])] = [:]
            for row in evaluated {
                var entry = byPartition[row.partitionId] ?? ([], [], [], [])
                if row.holdout { entry.hold.append(row.prediction); entry.targetsHold.append(row.target) }
                else { entry.fit.append(row.prediction); entry.targetsFit.append(row.target) }
                byPartition[row.partitionId] = entry
            }
            let fitRows = evaluated.filter { !$0.holdout }
            let heldRows = evaluated.filter { $0.holdout }
            print("feature            fit mean   holdout mean   shift")
            for (f, name) in FeatureVector.names.enumerated() {
                let a = fitRows.map { $0.features[f] }.reduce(0, +) / Float(max(1, fitRows.count))
                let b = heldRows.map { $0.features[f] }.reduce(0, +) / Float(max(1, heldRows.count))
                if abs(b - a) > 0.05 { print("\(Render.pad(name, 18)) \(Render.signed(a))     \(Render.signed(b))       \(Render.signed(b - a))") }
            }
            func mean(_ v: [Float]) -> String { v.isEmpty ? "   -  " : Render.signed(v.reduce(0, +) / Float(v.count)) }
            print("partition        fit: target  pred    (n)   holdout: target  pred    (n)")
            for (id, e) in byPartition.sorted(by: { $0.key < $1.key }) {
                print("\(Render.pad(id, 16))      \(mean(e.targetsFit))  \(mean(e.fit))  (\(e.fit.count))           \(mean(e.targetsHold))  \(mean(e.hold))  (\(e.hold.count))")
            }
            return
        }
        let session = readerSession(store)
        if ledger {
            print(String(data: try await session.ledgerJSON(owner: store.ownerID), encoding: .utf8) ?? "")
        } else {
            Render.summary(await session.summary(owner: store.ownerID))
            let owners = await session.knownOwners()
            if !owners.isEmpty { print("owners in store: " + owners.map(\.rawValue).joined(separator: ", ")) }
        }
    }
}

struct Encode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Pool texts through the model's input embedding table and compare them (encoder smoke test).")

    @OptionGroup var modelOptions: ModelOptions

    @Option(help: "A text file to encode (repeatable).")
    var context: [String] = []

    func run() async throws {
        let harness = Harness(storeDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("sinatra-encode"))
        try await modelOptions.load(harness)
        let texts = try context.map { try String(contentsOf: URL(fileURLWithPath: ContextFile.parse($0).path), encoding: .utf8) }
        let vectors = try await harness.encode(texts: texts)
        let norms = vectors.map { $0.reduce(0) { $0 + $1 * $1 }.squareRoot() }
        print("hidden size \(vectors.first?.count ?? 0), vocabulary \(await harness.vocabularySize.map(String.init) ?? "?")")
        for (i, name) in context.enumerated() { print("  [\(i)] ‖v‖=\(Render.f(norms[i], 4))  \(name)") }
        print("cosine similarity:")
        for i in vectors.indices {
            let row = vectors.indices.map { j -> String in
                let dot = zip(vectors[i], vectors[j]).reduce(0) { $0 + $1.0 * $1.1 }
                return Render.f(dot / max(norms[i] * norms[j], 1e-12), 3)
            }
            print("  [\(i)] " + row.joined(separator: "  "))
        }
    }
}

struct Features: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show the side model's features and weights for some context against the owner's ledger (dry run).")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var injection: InjectionOptions

    @Option(help: "A context file, optionally with its creation date (repeatable).")
    var context: [String] = []

    @Option(help: "Evaluate as of this ISO-8601 time instead of now.")
    var now: String?

    func run() async throws {
        var configuration = SinatraConfiguration()
        injection.apply(to: &configuration)
        let partitions = try context.flatMap { try ContextFile.partitions(from: $0) }
        let at = now.flatMap(ContextFile.parseDate) ?? Date()
        let turn = TurnInput(owner: store.ownerID, retrieved: partitions, now: at, weightOverride: injection.weights)
        let plan: InjectionPlan
        if modelOptions.model != nil || modelOptions.modelDir != nil {
            let harness = Harness(storeDirectory: store.storeURL, configuration: configuration)
            try await modelOptions.load(harness)
            plan = try await harness.preview(turn, mode: injection.mode)
        } else {
            let session = SinatraSession(
                configuration: configuration, storeDirectory: store.storeURL, modelKey: "features/no-model",
                tokenizer: SimpleTokenizer(), encoder: HashingContextEncoder(), vocabularySize: 1 << 20)
            plan = try await session.prepareTurn(turn, mode: injection.mode, dryRun: true)
        }
        Render.plan(plan.diagnostics)
        Render.rule("Features (\(FeatureVector.count) per partition)")
        let header = plan.diagnostics.partitions.map { Render.pad(String($0.id.suffix(10)), 10) }.joined(separator: " ")
        print("  \(Render.pad("feature", 18)) \(header)")
        for (f, name) in FeatureVector.names.enumerated() {
            let values = plan.diagnostics.features.map { Render.pad(Render.f($0[f], 3), 10) }.joined(separator: " ")
            print("  \(Render.pad(name, 18)) \(values)")
        }
    }
}

struct Attribute: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode once, then score the output without each partition in turn: exact citation values beside the lexical attribution the grounding measurement makes.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var sampling: SamplingOptions
    @OptionGroup var prompt: PromptOptions

    @Flag(help: "Print JSON.")
    var json = false

    struct Output: Encodable {
        var text: String
        var grounding: GroundingSummary?
        var lexical: [PartitionAttribution]
        var leaveOneOut: LeaveOneOutReport
    }

    func run() async throws {
        let partitions = try prompt.partitions()
        guard !partitions.isEmpty else { throw ValidationError("Pass at least one --context.") }
        let harness = Harness(storeDirectory: store.storeURL)
        try await modelOptions.load(harness)
        let input = prompt.userInput(partitions: partitions)
        let bare = prompt.bareInput()
        let handle = try await harness.generate(GenerateRequest(
            input: input, parameters: sampling.parameters,
            turn: TurnInput(owner: store.ownerID, retrieved: partitions), bareInput: bare,
            mode: .off, trace: .summary, record: false))
        for await _ in handle.stream {}
        let completion = await handle.completion.value
        guard let grounding = completion.grounding, grounding.measured else {
            throw ValidationError("The output could not be measured: \(completion.grounding?.skippedReason ?? "no measurement")")
        }
        let output = grounding.steps.map(\.token)
        let ablated = partitions.indices.map { p in
            prompt.userInput(partitions: partitions.enumerated().filter { $0.offset != p }.map(\.element))
        }
        let report = try await harness.leaveOneOut(full: input, ablated: ablated, bare: bare, output: output)
        if json {
            try printJSON(Output(text: completion.text, grounding: grounding.summary, lexical: grounding.attribution, leaveOneOut: report))
            return
        }
        Render.rule("Output")
        print(completion.text)
        Render.attribution(partitions: partitions, lexical: grounding.attribution, leaveOneOut: report)
    }
}
