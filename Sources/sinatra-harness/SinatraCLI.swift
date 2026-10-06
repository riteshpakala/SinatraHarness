//
//  SinatraCLI.swift
//  sinatra-harness
//
//  WHAT: The harness from the command line: generate with and without the injection,
//        compare the two decodes, measure what the context did to each output, replay
//        transcripts through the grounding loop, attribute outputs to partitions, and read
//        back traces, grounding analysis and ledgers.
//

import ArgumentParser
import Foundation
import SinatraHarness

@main
struct SinatraCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sinatra-harness",
        abstract: "On-device LLM harness with a grounding-driven injection layer before decoding.",
        subcommands: [
            Generate.self, Compare.self, Replay.self, Attribute.self, TraceCommand.self, Analyze.self,
            Inspect.self, Encode.self, Features.self,
        ])
}

extension BiasMode: @retroactive ExpressibleByArgument {}
extension TraceLevel: @retroactive ExpressibleByArgument {}
