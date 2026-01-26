// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SQLitePersistence",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.15.0"),
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
