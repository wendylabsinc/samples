// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "PersistentVolume",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-container-plugin", from: "1.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "PersistentVolume"
        ),
    ]
)
