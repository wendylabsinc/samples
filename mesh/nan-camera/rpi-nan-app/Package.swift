// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "rpi-nan-app",
    products: [.executable(name: "rpi-nan-app", targets: ["RPiNANCamera"])],
    targets: [
        .target(name: "CameraWire"),
        .target(name: "CameraC", publicHeadersPath: "include", linkerSettings: [.linkedLibrary("jpeg")]),
        .executableTarget(name: "RPiNANCamera", dependencies: ["CameraWire", "CameraC"]),
        .testTarget(name: "CameraControlTests", dependencies: ["RPiNANCamera"]),
        .testTarget(name: "CameraWireTests", dependencies: ["CameraWire", "CameraC"], resources: [.process("Fixtures")]),
    ]
)
