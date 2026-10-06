import Foundation
@testable import SinatraHarness

/// MLX arrays abort on Darwin without the metallib beside the test bundle: build it with
/// `./scripts/metallib.sh debug` and run with FRIGATE_MLX_TESTS=1.
let mlxAvailable = ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"

func temporaryStore(_ name: String = #function) -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("sinatra-tests", isDirectory: true)
        .appendingPathComponent("\(name.filter { $0.isLetter || $0.isNumber })-\(UUID().uuidString)", isDirectory: true)
}

/// A vocabulary with Mistral-style control tokens.
final class StubTokenizer: SinatraTokenizing, @unchecked Sendable {
    let vocabulary: [Int: String] = [
        0: "<s>", 1: "</s>", 2: "<unk>", 5: "[INST]", 6: "<SPECIAL_20>", 7: "<0x0A>",
        8: "▁the", 9: "▁retrieval", 10: ",", 11: "▁a", 12: "▁feedback", 13: "▁garden", 14: "▁tomato",
    ]
    var specialTokenIds: Set<Int> { [0, 1, 2] }

    func encode(_ text: String) -> [Int] {
        text.lowercased().split(separator: " ").compactMap { word in
            vocabulary.first { $0.value == "▁" + word }?.key
        }
    }

    func decode(_ ids: [Int]) -> String {
        ids.compactMap { vocabulary[$0] }.joined().replacingOccurrences(of: "▁", with: " ")
    }

    func tokenString(_ id: Int) -> String? { vocabulary[id] }
}

func makeSession(
    store: URL, configuration: SinatraConfiguration = SinatraConfiguration(),
    factory: @escaping WeightModelFactory = PriorWeightModel.factory, tokenizer: SimpleTokenizer = SimpleTokenizer()
) -> SinatraSession {
    SinatraSession(
        configuration: configuration, storeDirectory: store, modelKey: "test-model",
        tokenizer: tokenizer, encoder: HashingContextEncoder(hiddenSize: 32),
        vocabularySize: 1 << 20, weightModelFactory: factory)
}

let gardenText = "Raised garden beds get morning sun so tomatoes and basil thrive with compost and drip irrigation."
let breadText = "The sourdough starter needs feeding twice a day with flour and water before a long cold proof."
let bikeText = "Interval sessions on the bike build threshold power with cadence near ninety on long climbs."

func partitions() -> [Partition] {
    [
        Partition(id: "garden#0", documentId: "docs/garden", text: gardenText, score: 0.1),
        Partition(id: "bread#0", documentId: "docs/bread", text: breadText, score: 0.2),
        Partition(id: "bike#0", documentId: "docs/bike", text: bikeText, score: 0.3),
    ]
}
