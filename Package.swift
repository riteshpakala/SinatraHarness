// swift-tools-version: 6.0
// SinatraHarness — an on-device LLM harness for Apple silicon whose retrieved context
// is scored by a small time-series side model and injected into the transformer's
// logits right before decoding.
import PackageDescription

let package = Package(
    name: "SinatraHarness",
    // Frigate's floors: macOS 15 (VisionAXCore), iOS/tvOS 17, visionOS 1.
    platforms: [.macOS("15.0"), .iOS(.v17), .tvOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "SinatraHarness", targets: ["SinatraHarness"]),
        .executable(name: "sinatra-harness", targets: ["sinatra-harness"]),
    ],
    dependencies: [
        // Frigate is the ONLY MLX in the graph: its targets are literally named
        // MLX, MLXNN, MLXOptimizers, ... Never add ml-explore/mlx-swift here — the
        // product and target names collide.
        //
        // Committed form (Frigate has no path dependencies of its own):
        //   .package(url: "https://github.com/rao-studios/Frigate.git", branch: "main"),
        // Local iteration against the sibling checkout (the same directory Sewn's
        // `../Frigate` resolves to, so Sewn's graph still holds one Frigate):
        .package(path: "../../rao/repositories/Frigate"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "SinatraHarness",
            dependencies: [
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "MLXOptimizers", package: "Frigate"),
                .product(name: "MLXRandom", package: "Frigate"),
                .product(name: "MLXLMCommon", package: "Frigate"),
                // Links the LLM factory trampoline that MLXLMCommon.loadModel finds by name.
                .product(name: "MLXLLM", package: "Frigate"),
                // HubDownloader / HubTokenizerLoader.
                .product(name: "FrigateBridge", package: "Frigate"),
            ],
            // Same language mode as MLXLMCommon/MLXLLM: MLXArray is not Sendable.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "sinatra-harness",
            dependencies: [
                "SinatraHarness",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SinatraHarnessTests",
            dependencies: ["SinatraHarness"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
