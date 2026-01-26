/// SQLite Persistence Example
///
/// Demonstrates using SQLite with a persistent volume that survives container restarts.

import Foundation
import GRDB

let dataDir = "/data"

struct Run: Codable, FetchableRecord, PersistableRecord {
    var id: Int64?
    var timestamp: String
    var message: String

    static let databaseTableName = "runs"
}

@main
struct SQLitePersistence {
    static func main() throws {
        print("SQLite Persistence Example")
        print(String(repeating: "=", count: 40))

        let fileManager = FileManager.default
        let dataURL = URL(fileURLWithPath: dataDir)
        let dbPath = dataURL.appendingPathComponent("app.db").path

        // Ensure the data directory exists
        try fileManager.createDirectory(at: dataURL, withIntermediateDirectories: true)

        // Connect to SQLite database
        print("\nConnecting to database: \(dbPath)")
        let dbQueue = try DatabaseQueue(path: dbPath)

        // Create table if it doesn't exist
        try dbQueue.write { db in
            try db.create(table: "runs", ifNotExists: true) { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("timestamp", .text).notNull()
                t.column("message", .text).notNull()
            }
        }

        // Insert a new record for this run
        let now = ISO8601DateFormatter().string(from: Date())
        let msg = "Application started at \(now)"
        let run = Run(id: nil, timestamp: now, message: msg)
        try dbQueue.write { db in
            try run.insert(db)
        }
        print("Inserted: \(msg)")

        // Query all records
        print("\n" + String(repeating: "-", count: 40))
        print("All recorded runs:")
        print(String(repeating: "-", count: 40))

        let runs = try dbQueue.read { db in
            try Run.order(Column("id")).fetchAll(db)
        }

        for run in runs {
            print("  [\(run.id ?? 0)] \(run.timestamp): \(run.message)")
        }

        print("\nTotal runs: \(runs.count)")
        print(String(repeating: "-", count: 40))

        print("\nSQLite persistence example complete!")
        print("Run this again to see the data persist across restarts.")
    }
}
