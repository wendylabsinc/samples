import Foundation
import XCTest
@testable import CameraWire

final class PhotoStoreFailureTests: XCTestCase {
    final class FaultFiles: FileManager, @unchecked Sendable {
        var failOriginalRename = false
        var failThumbnailRemove = false
        var failTemporaryRemove = false
        override func moveItem(at source: URL, to destination: URL) throws {
            if failOriginalRename && destination.pathExtension == "jpg" && !destination.lastPathComponent.hasSuffix(".thumb.jpg") {
                throw CocoaError(.fileWriteUnknown)
            }
            try super.moveItem(at: source, to: destination)
        }
        override func removeItem(at url: URL) throws {
            if failTemporaryRemove && url.pathExtension == "tmp" { throw CocoaError(.fileWriteNoPermission) }
            if failThumbnailRemove && url.lastPathComponent.hasSuffix(".thumb.jpg") {
                throw CocoaError(.fileWriteNoPermission)
            }
            try super.removeItem(at: url)
        }
    }
    private func withDirectory(_ test: (URL) throws -> Void) throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try test(dir)
    }
    private func sparse(_ url: URL, _ size: UInt64) throws {
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: nil))
        let file = try FileHandle(forWritingTo: url)
        defer { try? file.close() }
        try file.truncate(atOffset: size)
    }
    private func names(_ dir: URL) throws -> [String] { try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted() }

    func testBudgetIncludesExistingAndIncomingThumbnailBytes() throws {
        try withDirectory { dir in
            let id = UUID().uuidString.lowercased()
            try sparse(dir.appendingPathComponent(id + ".jpg"), PhotoStore.maxTotalBytes - 8)
            try Data([1,2,3,4]).write(to: dir.appendingPathComponent(id + ".thumb.jpg"))
            let store = try PhotoStore(directory: dir)
            XCTAssertThrowsError(try store.save(jpeg: Data([1,2,3,4]), thumbnail: Data([5])))
            XCTAssertEqual(try names(dir).count, 2)
            let exact = try store.save(jpeg: Data([1,2,3]), thumbnail: Data([4]))
            XCTAssertEqual(try store.thumbnail(exact.id), Data([4]))
            XCTAssertThrowsError(try store.save(jpeg: Data([1]), thumbnail: Data()))
        }
    }
    func testFailedOriginalRenameRemovesCommittedThumbnailAndTemps() throws {
        try withDirectory { dir in
            let fs = FaultFiles(); fs.failOriginalRename = true
            let store = try PhotoStore(directory: dir, files: fs)
            XCTAssertThrowsError(try store.save(jpeg: Data([1]), thumbnail: Data([2])))
            XCTAssertEqual(try names(dir), [])
        }
    }
    func testUnremovableOrphanBlocksFurtherWritesThenRecovers() throws {
        try withDirectory { dir in
            let fs = FaultFiles(); fs.failOriginalRename = true; fs.failThumbnailRemove = true
            let store = try PhotoStore(directory: dir, files: fs)
            XCTAssertThrowsError(try store.save(jpeg: Data([1]), thumbnail: Data([2])))
            let orphan = try names(dir)
            XCTAssertEqual(orphan.count, 1); XCTAssertTrue(orphan[0].hasSuffix(".thumb.jpg"))
            fs.failOriginalRename = false
            XCTAssertThrowsError(try store.save(jpeg: Data([3]), thumbnail: Data([4])))
            XCTAssertEqual(try names(dir), orphan)
            fs.failThumbnailRemove = false
            let saved = try store.save(jpeg: Data([3]), thumbnail: Data([4]))
            XCTAssertEqual(try store.list().map(\.id), [saved.id])
            XCTAssertEqual(try names(dir).count, 2)
        }
    }
    func testDeleteReportsThumbnailFailureAndCanBeRetried() throws {
        try withDirectory { dir in
            let fs = FaultFiles()
            let store = try PhotoStore(directory: dir, files: fs)
            let saved = try store.save(jpeg: Data([1]), thumbnail: Data([2]))
            fs.failThumbnailRemove = true
            XCTAssertThrowsError(try store.delete(saved.id))
            XCTAssertEqual(try store.list(), [])
            XCTAssertEqual(try names(dir).count, 1)
            fs.failThumbnailRemove = false
            try store.delete(saved.id)
            XCTAssertEqual(try names(dir), [])
            XCTAssertThrowsError(try store.delete(saved.id))
        }
    }
    func testStartupRepairsOnlyOrphanThumbnailAndPreservesExistingPhotos() throws {
        try withDirectory { dir in
            let originalStore = try PhotoStore(directory: dir)
            let saved = try originalStore.save(jpeg: Data([1]), thumbnail: Data([2]))
            let orphan = dir.appendingPathComponent(UUID().uuidString.lowercased() + ".thumb.jpg")
            try Data([3]).write(to: orphan)
            try Data([4]).write(to: dir.appendingPathComponent("foreign.thumb.jpg"))
            let store = try PhotoStore(directory: dir)
            XCTAssertEqual(try store.list().map(\.id), [saved.id])
            XCTAssertEqual(try store.thumbnail(saved.id), Data([2]))
            XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("foreign.thumb.jpg").path))
        }
    }
    func testFailedTempCleanupBlocksSaveAndStartupUntilRecovered() throws {
        try withDirectory { dir in
            let fs = FaultFiles()
            let store = try PhotoStore(directory: dir, files: fs)
            let temp = dir.appendingPathComponent(UUID().uuidString.lowercased() + ".thumb.tmp")
            try Data([1,2,3]).write(to: temp)
            fs.failTemporaryRemove = true
            XCTAssertThrowsError(try store.save(jpeg: Data([4]), thumbnail: Data([5])))
            XCTAssertThrowsError(try PhotoStore(directory: dir, files: fs))
            XCTAssertEqual(try names(dir), [temp.lastPathComponent])
            fs.failTemporaryRemove = false
            let saved = try store.save(jpeg: Data([4]), thumbnail: Data([5]))
            XCTAssertEqual(try store.list().map(\.id), [saved.id])
            XCTAssertEqual(try names(dir).count, 2)
        }
    }

}
