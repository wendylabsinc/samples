// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SQLitePersistence",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.15.4"),
        .package(url: "https://github.com/apple/swift-container-plugin", from: "1.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "SQLitePersistence",
            dependencies: [
                .product(name: "SQLite", package: "SQLite.swift"),
            ]
        ),
    ]
)
