//
//  HarnessTypes.swift
//  SinatraHarness
//
//  WHAT: What callers hand the harness and what they get back.
//

import Foundation
import MLXLMCommon

public enum HarnessLoadState: Sendable, Equatable {
    case cold
    case loading(Double)
    case ready(String)
    case failed(String)

    public var name: String {
        switch self {
        case .cold: return "cold"
        case .loading: return "loading"
        case .ready: return "ready"
        case .failed: return "failed"
        }
    }
}

public struct GenerateRequest: @unchecked Sendable {
    /// Chat messages and tools; the harness applies the model's chat template.
    public var input: UserInput
    /// Sampling, penalties, max tokens, seed, KV settings.
    public var parameters: GenerateParameters
    /// nil: a plain generation — no injection, nothing observed or learned (utility passes).
    public var turn: TurnInput?
    /// The same chat without its retrieved context. With it, the turn's output is re-scored
    /// both ways after decoding (the grounding measurement); without it the turn is
    /// recorded but not measured, and teaches nothing.
    public var bareInput: UserInput?
    /// Override `SinatraConfiguration.biasMode` for this turn.
    public var mode: BiasMode?
    public var trace: TraceLevel
    /// false: dry run — the plan is computed but nothing is recorded or labelled.
    public var record: Bool
    /// Tool schemas for parsing tool-call arguments (as `MLXLMCommon.generate` takes them).
    public var tools: [[String: any Sendable]]?
    /// Replays and simulations: the assistant timeline to record instead of the wall clock.
    public var timeline: Timeline?

    public struct Timeline: Sendable {
        public var startedAt: Date
        public var finishedAt: Date
        public init(startedAt: Date, finishedAt: Date) {
            self.startedAt = startedAt
            self.finishedAt = finishedAt
        }
    }

    public init(
        input: UserInput, parameters: GenerateParameters, turn: TurnInput? = nil,
        bareInput: UserInput? = nil, mode: BiasMode? = nil, trace: TraceLevel = .automatic,
        record: Bool = true, tools: [[String: any Sendable]]? = nil, timeline: Timeline? = nil
    ) {
        self.timeline = timeline
        self.input = input
        self.parameters = parameters
        self.turn = turn
        self.bareInput = bareInput
        self.mode = mode
        self.trace = trace
        self.record = record
        self.tools = tools
    }
}

public struct GenerationHandle: @unchecked Sendable {
    /// MLXLMCommon's own events (.chunk, .toolCall, .info), forwarded unchanged.
    public let stream: AsyncStream<Generation>
    /// What the side model decided for this turn; nil for plain generations.
    public let plan: InjectionPlan?
    public var turnId: UUID? { plan?.turnId }
    /// Resolves once the stream has ended and the turn is recorded.
    public let completion: Task<TurnCompletion, Never>
}

public struct TurnCompletion: Sendable {
    public let text: String
    public let info: GenerateCompletionInfo?
    public let trace: InjectionTrace?
    /// What the retrieved context did to the output; nil when no measurement was attempted.
    public let grounding: GroundingMeasurement?
    public let summary: OwnerSummary?
    public let trainingScheduled: Bool
    public let cancelled: Bool
}
