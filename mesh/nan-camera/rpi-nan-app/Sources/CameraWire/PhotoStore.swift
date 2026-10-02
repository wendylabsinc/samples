import Foundation

public enum PhotoStoreError: Error {
    case full
    case notFound
    case tooLarge
}

/// App-private persistent storage. A wire ID is always parsed as UUID before
/// deriving any path; clients never supply a filename.
public final class PhotoStore {
    public static let maxPhotos = 100
    public static let maxTotalBytes: UInt64 = 512 * 1024 * 1024
    private let directory: URL
    private let files: FileManager

    public convenience init(directory: URL) throws {
        try self.init(directory: directory, files: .default)
    }

    init(directory: URL, files: FileManager) throws {
        self.directory = directory
        self.files = files
        try files.createDirectory(at: directory, withIntermediateDirectories: true)
        try recoverIncompleteFiles()
    }

    private func photoURL(_ id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString.lowercased() + ".jpg")
    }

    private func thumbURL(_ id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString.lowercased() + ".thumb.jpg")
    }

    // Only UUID-named app files participate. Never remove a valid original
    // during recovery. Failed cleanup is visible and prevents another save.
    private func recoverIncompleteFiles() throws {
        for url in try files.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            let name = url.lastPathComponent
            if name.hasSuffix(".tmp") {
                let stem = String(name.dropLast(4))
                let id = stem.hasSuffix(".thumb") ? String(stem.dropLast(6)) : stem
                if UUID(uuidString: id) != nil { try files.removeItem(at: url) }
                continue
            }
            guard name.hasSuffix(".thumb.jpg"),
                  let id = UUID(uuidString: String(name.dropLast(10))) else { continue }
            if !files.fileExists(atPath: photoURL(id).path) {
                try files.removeItem(at: url)
            }
        }
    }

    private func storedBytes(_ records: [PhotoRecord]) throws -> UInt64 {
        var total: UInt64 = 0
        for record in records {
            total += record.jpegBytes
            let thumbnail = thumbURL(record.id)
            if files.fileExists(atPath: thumbnail.path) {
                let attributes = try files.attributesOfItem(atPath: thumbnail.path)
                total += (attributes[.size] as? NSNumber)?.uint64Value ?? 0
            }
        }
        return total
    }

    public func list() throws -> [PhotoRecord] {
        let paths = try files.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        return try paths.compactMap { url in
            guard url.pathExtension == "jpg", !url.lastPathComponent.hasSuffix(".thumb.jpg"),
                  let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent) else { return nil }
            return try record(id)
        }.sorted { $0.capturedAtUnixMs > $1.capturedAtUnixMs }
    }

    public func record(_ id: UUID) throws -> PhotoRecord {
        let url = photoURL(id)
        guard files.fileExists(atPath: url.path) else { throw PhotoStoreError.notFound }
        let attrs = try files.attributesOfItem(atPath: url.path)
        let size = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
        let date = (attrs[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
        return PhotoRecord(id: id, capturedAtUnixMs: UInt64(max(0, date.timeIntervalSince1970 * 1000)), jpegBytes: size)
    }

    public func save(jpeg: Data, thumbnail: Data) throws -> PhotoRecord {
        guard jpeg.count <= CameraHeader.maxPayload, thumbnail.count <= CameraHeader.maxPayload else { throw PhotoStoreError.tooLarge }
        try recoverIncompleteFiles()
        let current = try list()
        let total = try storedBytes(current)
        let incoming = UInt64(jpeg.count) + UInt64(thumbnail.count)
        guard current.count < Self.maxPhotos, total <= Self.maxTotalBytes,
              incoming <= Self.maxTotalBytes - total else { throw PhotoStoreError.full }
        let id = UUID()
        let temp = directory.appendingPathComponent(id.uuidString.lowercased() + ".tmp")
        let thumbnailTemp = directory.appendingPathComponent(id.uuidString.lowercased() + ".thumb.tmp")
        defer { try? files.removeItem(at: temp); try? files.removeItem(at: thumbnailTemp) }
        guard files.createFile(atPath: temp.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        let handle = try FileHandle(forWritingTo: temp)
        try handle.write(contentsOf: jpeg)
        try handle.synchronize()
        try handle.close()
        try thumbnail.write(to: thumbnailTemp)
        try files.moveItem(at: thumbnailTemp, to: thumbURL(id))
        do {
            try files.moveItem(at: temp, to: photoURL(id))
        } catch {
            // If removal also fails, recovery on the next save/start retries it
            // before admitting any further disk allocation.
            try? files.removeItem(at: thumbURL(id))
            throw error
        }
        return try record(id)
    }

    public func imageURL(_ id: UUID) throws -> URL {
        _ = try record(id)
        return photoURL(id)
    }

    public func thumbnail(_ id: UUID) throws -> Data {
        _ = try record(id)
        return try Data(contentsOf: thumbURL(id))
    }

    public func delete(_ id: UUID) throws {
        let original = photoURL(id)
        let thumbnail = thumbURL(id)
        let hasOriginal = files.fileExists(atPath: original.path)
        let hasThumbnail = files.fileExists(atPath: thumbnail.path)
        guard hasOriginal || hasThumbnail else { throw PhotoStoreError.notFound }
        if hasOriginal { try files.removeItem(at: original) }
        // Do not report success if only the original was deleted. A retry can
        // remove the remaining thumbnail even when its original is now absent.
        if hasThumbnail { try files.removeItem(at: thumbnail) }
    }
}
