//
//  MLXLMTokenizer.swift
//  SinatraHarness
//
//  WHAT: The loaded LLM's own tokenizer, adapted to `SinatraTokenizing`. Ids line up with
//        the logits the injection is added to.
//

import Foundation
import MLXLMCommon

public struct MLXLMTokenizer: SinatraTokenizing {
    let tokenizer: any MLXLMCommon.Tokenizer
    public let specialTokenIds: Set<Int>

    public init(_ tokenizer: any MLXLMCommon.Tokenizer, configuration: ModelConfiguration? = nil) {
        self.tokenizer = tokenizer
        var ids = Set<Int>()
        for token in [tokenizer.bosToken, tokenizer.eosToken, tokenizer.unknownToken].compactMap({ $0 }) {
            if let id = tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        if let configuration {
            ids.formUnion(configuration.eosTokenIds)
            for token in configuration.extraEOSTokens {
                if let id = tokenizer.convertTokenToId(token) { ids.insert(id) }
            }
        }
        self.specialTokenIds = ids
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }

    public func tokenString(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }
}
