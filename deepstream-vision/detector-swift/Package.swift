// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "DeepStreamDetectorSwift",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/tensorrt-swift", from: "0.0.1"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
    ],
    targets: [
        .systemLibrary(
            name: "CFFmpeg",
            pkgConfig: "libavformat libavcodec libavutil libswscale",
            providers: [
                .apt(["libavformat-dev", "libavcodec-dev", "libavutil-dev", "libswscale-dev"]),
            ]
        ),
        .systemLibrary(
            name: "CTurboJPEG",
            pkgConfig: "libturbojpeg",
            providers: [
                .apt(["libturbojpeg0-dev"]),
            ]
        ),
        .executableTarget(
            name: "Detector",
            dependencies: [
                .product(name: "TensorRT", package: "tensorrt-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                "CFFmpeg",
                "CTurboJPEG",
            ]
        ),
    ]
)
