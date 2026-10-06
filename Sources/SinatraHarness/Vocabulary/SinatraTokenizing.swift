//
//  SinatraTokenizing.swift
//  SinatraHarness
//
//  WHAT: The tokenizer surface the side model needs. `MLXLMTokenizer` adapts the loaded
//        LLM's tokenizer; `SimpleTokenizer` is a model-free stand-in for tests and replay.
//  PIN:  Token ids must be the LLM's own ids: the injection is added to its logits.
//

import Foundation

public protocol SinatraTokenizing: Sendable {
    /// Token ids without special tokens.
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    /// The raw vocabulary string, e.g. "<SPECIAL_20>" or "▁the".
    func tokenString(_ id: Int) -> String?
    /// BOS/EOS/UNK and any extra stop tokens: never biased.
    var specialTokenIds: Set<Int> { get }
}

/// Word-level tokenizer with a growing vocabulary. Deterministic within a process; for
/// tests, model-free replay and the `features` command only.
public final class SimpleTokenizer: SinatraTokenizing, @unchecked Sendable {
    private let lock = NSLock()
    private var ids: [String: Int] = [:]
    private var words: [String] = []
    /// Ids below this are reserved (0 = BOS, 1 = EOS, 2 = UNK).
    public static let reserved = 3

    public init() {}

    public var specialTokenIds: Set<Int> { [0, 1, 2] }

    public func encode(_ text: String) -> [Int] {
        let pieces = text.lowercased().split { !($0.isLetter || $0.isNumber) }
        lock.lock()
        defer { lock.unlock() }
        return pieces.map { piece in
            let word = String(piece)
            if let id = ids[word] { return id }
            let id = Self.reserved + words.count
            ids[word] = id
            words.append(word)
            return id
        }
    }

    public func decode(_ ids: [Int]) -> String {
        lock.lock()
        defer { lock.unlock() }
        return ids.map { id -> String in
            let index = id - Self.reserved
            return index >= 0 && index < words.count ? words[index] : "<\(id)>"
        }.joined(separator: " ")
    }

    public func tokenString(_ id: Int) -> String? {
        switch id {
        case 0: return "<s>"
        case 1: return "</s>"
        case 2: return "<unk>"
        default:
            lock.lock()
            defer { lock.unlock() }
            let index = id - Self.reserved
            return index >= 0 && index < words.count ? words[index] : nil
        }
    }

    public var vocabularySize: Int {
        lock.lock()
        defer { lock.unlock() }
        return Self.reserved + words.count
    }
}
