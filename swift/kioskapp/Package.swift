// swift-tools-version:6.1
import PackageDescription

let package = Package(
    name: "KioskApp",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/stackotter/swift-cross-ui", revision: "main")
    ],
    targets: [
        .executableTarget(
            name: "KioskApp",
            dependencies: [
                .product(name: "SwiftCrossUI", package: "swift-cross-ui"),
                .product(name: "GtkBackend", package: "swift-cross-ui")
            ]
        )
    ]
)
