//
//  Generate.swift
//  sinatra-harness
//

import ArgumentParser
import Foundation
import SinatraHarness

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode with the Sinatra injection beside a baseline decode, trace its impact on the logits, and measure what the context did to each output.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var sampling: SamplingOptions
    @OptionGroup var injection: InjectionOptions
    @OptionGroup var prompt: PromptOptions

    @Option(help: "Trace level: summary or full (full adds top-k, the movement heatmap and the context's strongest pushes).")
    var trace: TraceLevel = .summary

    @Flag(help: "Skip the baseline decode.")
    var noBaseline = false

    @Flag(help: "Record the turn in the owner's ledger: its measurement labels the partitions.")
    var record = false

    @Option(help: "Trace steps to print.")
    var steps = 60

    @Flag(help: "Print JSON instead of tables.")
    var json = false

    @Flag(help: "Debug logging.")
    var verbose = false

    struct Output: Encodable {
        var baseline: String?
        var baselineSummary: TraceSummary?
        var baselineGrounding: GroundingSummary?
        var injected: String
        var plan: TurnDiagnostics?
        var trace: InjectionTrace?
    }

    func run() async throws {
        var configuration = SinatraConfiguration()
        injection.apply(to: &configuration)
        let harness = Harness(
            storeDirectory: store.storeURL, configuration: configuration,
            log: PrintLog(threshold: verbose ? .debug : .warning))
        try await modelOptions.load(harness)

        let partitions = try prompt.partitions()
        let input = prompt.userInput(partitions: partitions)
        let bare = partitions.isEmpty ? nil : prompt.bareInput()
        let turn = TurnInput(owner: store.ownerID, retrieved: partitions, weightOverride: injection.weights)

        var baseline: TurnCompletion?
        if !noBaseline {
            let handle = try await harness.generate(GenerateRequest(
                input: input, parameters: sampling.parameters, turn: turn, bareInput: bare, mode: .off,
                trace: trace, record: false))
            for await _ in handle.stream {}
            baseline = await handle.completion.value
            if !json {
                Render.rule("Baseline (no injection)")
                print(baseline?.text ?? "")
            }
        }

        let handle = try await harness.generate(GenerateRequest(
            input: input, parameters: sampling.parameters, turn: turn, bareInput: bare, mode: injection.mode,
            trace: trace, record: record))
        if !json {
            if let plan = handle.plan { Render.plan(plan.diagnostics) }
            Render.rule("With Sinatra (\(injection.mode.rawValue))")
        }
        for await item in handle.stream {
            if !json, case .chunk(let chunk) = item {
                print(chunk, terminator: "")
                fflush(stdout)
            }
        }
        let completion = await handle.completion.value
        await harness.flush()

        if json {
            try printJSON(Output(
                baseline: baseline?.text, baselineSummary: baseline?.trace?.summary,
                baselineGrounding: baseline?.grounding?.summary, injected: completion.text,
                plan: handle.plan?.diagnostics, trace: completion.trace))
            return
        }
        print()
        if let info = completion.info {
            print(String(format: "\n%d tokens, %.1f tok/s (prompt %d tokens, %.2f s)", info.generationTokenCount, info.tokensPerSecond, info.promptTokenCount, info.promptTime))
        }
        if let trace = completion.trace, trace.level != .off {
            if let baseline = baseline?.trace, baseline.level != .off {
                print("baseline mean entropy \(Render.f(baseline.summary.meanEntropyPost)) over \(baseline.summary.steps) steps")
            }
            Render.summary(trace.summary)
            Render.steps(trace, limit: steps)
        } else {
            print("(the plan did not inject — nothing learned yet for these partitions; try --force-weights)")
        }
        if let grounding = completion.grounding {
            Render.grounding(grounding, steps: steps, baseline: baseline?.grounding)
        } else if bare == nil {
            print("(no context, nothing to measure)")
        }
    }
}
