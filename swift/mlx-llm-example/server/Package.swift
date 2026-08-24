// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "mlx-llm-server",
  platforms: [
    .macOS(.v14)
  ],
  dependencies: [
    .package(
      url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0", traits: []),
    .package(url: "https://github.com/Joannis/mlx-swift-lm.git", branch: "jo/linux-cuda"),
    .package(url: "https://github.com/Joannis/swift-transformers.git", branch: "jo/localized-comments"),
    .package(url: "https://github.com/Joannis/mlx-swift.git", branch: "jo/linux-cuda"),
    .package(url: "https://github.com/apple/swift-container-plugin", from: "1.3.0"),
  ],
  targets: [
    .executableTarget(
      name: "mlx-llm-server",
      dependencies: [
        .product(name: "Hummingbird", package: "hummingbird"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
      ]
    )
  ],
  swiftLanguageModes: [.v5],
  cxxLanguageStandard: .gnucxx20
)
