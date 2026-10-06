//
//  TokenFilter.swift
//  SinatraHarness
//
//  WHAT: Decides which token ids carry content — the only ids the injection may touch
//        and the only ids that count toward grounding shares and attribution.
//  PIN:  Excludes special and control tokens (Mistral reserves 0–999: "<SPECIAL_n>",
//        "[INST]", byte-fallback "<0x0A>"), punctuation, single characters and English
//        stopwords. Verdicts are memoised per id; only ids seen in context are ever
//        classified, never the whole vocabulary.
//

import Foundation

final class TokenFilter {
    private let tokenizer: any SinatraTokenizing
    private let useStopwords: Bool
    private var memo: [Int: Bool] = [:]
    private let special: Set<Int>

    init(tokenizer: any SinatraTokenizing, useStopwords: Bool = true) {
        self.tokenizer = tokenizer
        self.useStopwords = useStopwords
        self.special = tokenizer.specialTokenIds
    }

    /// Special or control token: never biased, whatever mode.
    func isControl(_ id: Int) -> Bool {
        if special.contains(id) { return true }
        guard let raw = tokenizer.tokenString(id) else { return false }
        return Self.looksLikeControl(raw)
    }

    func isContent(_ id: Int) -> Bool {
        if let cached = memo[id] { return cached }
        let verdict = classify(id)
        memo[id] = verdict
        return verdict
    }

    /// Term frequencies of content tokens, excluding owner-level hot tokens.
    func contentCounts(_ ids: [Int], hot: HotTokens? = nil, configuration: SinatraConfiguration) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for id in ids where isContent(id) {
            counts[id, default: 0] += 1
        }
        if let hot {
            for id in counts.keys where hot.isHot(
                id, ratio: configuration.hotTokenExclusionRatio,
                minimumPartitions: configuration.hotTokenMinimumPartitions)
            {
                counts[id] = nil
            }
        }
        return counts
    }

    func contentSet(_ ids: [Int], hot: HotTokens? = nil, configuration: SinatraConfiguration) -> Set<Int> {
        Set(contentCounts(ids, hot: hot, configuration: configuration).keys)
    }

    private func classify(_ id: Int) -> Bool {
        if id < 0 || isControl(id) { return false }
        let text = tokenizer.decode([id]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count >= 2 else { return false }
        guard text.unicodeScalars.contains(where: { CharacterSet.alphanumerics.contains($0) }) else { return false }
        if text.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) { return false }
        if useStopwords && Stopwords.english.contains(text.lowercased()) { return false }
        return true
    }

    static func looksLikeControl(_ raw: String) -> Bool {
        if raw.count >= 3, raw.hasPrefix("<"), raw.hasSuffix(">"), !raw.contains(" ") { return true }
        if raw.count >= 3, raw.hasPrefix("["), raw.hasSuffix("]") {
            let inner = raw.dropFirst().dropLast()
            if !inner.isEmpty && inner.allSatisfy({ $0.isLetter || $0 == "_" || $0 == "/" }) { return true }
        }
        return false
    }
}

enum Stopwords {
    static let english: Set<String> = [
        "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
        "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
        "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "however", "if", "in", "into", "is", "it", "its",
        "itself", "just", "may", "me", "might", "more", "most", "must", "my", "myself", "no", "nor",
        "not", "now", "of", "off", "on", "once", "one", "only", "or", "other", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shall", "she", "should", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "until", "up", "upon", "us",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
        "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't",
    ]
}
