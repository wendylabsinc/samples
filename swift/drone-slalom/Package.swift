// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "DroneRace",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/swift-mujoco.git", branch: "main"),
    ],
    targets: [
        .target(name: "SlalomCore", dependencies: [
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
        .executableTarget(name: "DroneRace", dependencies: [
            "SlalomCore",
            .product(name: "WendyMuJoCo", package: "swift-mujoco"),
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
        .testTarget(name: "SlalomCoreTests", dependencies: [
            "SlalomCore",
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
    ]
)
