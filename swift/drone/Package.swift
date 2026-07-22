// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "drone",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .target(name: "DroneCore"),
        .target(name: "DroneApp", dependencies: ["DroneCore"]),
        .testTarget(name: "DroneCoreTests", dependencies: ["DroneCore"]),
        .testTarget(name: "DroneAppTests", dependencies: ["DroneApp"]),
    ]
)
