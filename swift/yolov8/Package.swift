// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "yolov8-server",
  platforms: [
    .macOS("26.0")
  ],
  dependencies: [
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    .package(
      url: "https://github.com/hummingbird-project/hummingbird-websocket.git", from: "2.0.0"),
    .package(url: "https://github.com/wendylabsinc/gstreamer-swift.git", from: "0.0.3"),
    .package(url: "https://github.com/apple/swift-container-plugin", from: "1.0.0"),
  ],
  targets: [
    .systemLibrary(
      name: "CONNXRuntime",
      path: "Server/Sources/CONNXRuntime"
    ),
    .executableTarget(
      name: "YoloV8Server",
      dependencies: [
        .product(name: "Hummingbird", package: "hummingbird"),
        .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
        .product(name: "GStreamer", package: "gstreamer-swift"),
        "CONNXRuntime",
      ],
      path: "Server/Sources/YoloV8Server",
      swiftSettings: [
        .unsafeFlags(["-parse-as-library"])
      ],
      linkerSettings: [
        .unsafeFlags(["-L/opt/homebrew/Cellar/onnxruntime/1.24.1/lib"]),
        .linkedLibrary("onnxruntime")
      ],
      plugins: ["DownloadYOLOv8Model"]
    ),
    .plugin(
      name: "DownloadYOLOv8Model",
      capability: .buildTool(),
      path: "Plugins/DownloadYOLOv8Model"
    )
  ]
)
