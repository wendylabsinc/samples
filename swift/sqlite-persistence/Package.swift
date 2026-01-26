// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SQLitePersistence",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", exact: "7.9.0"),
        .package(url: "https://github.com/apple/swift-container-plugin", from: "1.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "SQLitePersistence",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ]
        ),
    ]
)
