#if os(Linux)
import CameraC
import CameraWire
import Foundation
import Glibc
import XCTest

private struct FixtureCamera: CameraFrameSource {
    let live: Data
    let photo: Data
    let thumb: Data
    func snapshot() throws -> Data { live }
    func take() throws -> (photo: Data, thumbnail: Data) { (photo, thumb) }
}

private struct LoopbackServer {
    let port: UInt16
    let finished: DispatchSemaphore
}

final class LoopbackIntegrationTests: XCTestCase {
    private func fixture() throws -> (live: Data, photo: Data, thumb: Data) {
        let live = try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: "source", withExtension: "jpg")))
        // Three legal JPEG COM segments make the saved photo larger than two
        // 64 KiB server writes while retaining a decodable image.
        var photo = Data([0xff, 0xd8])
        for index in 0..<3 {
            photo.append(contentsOf: [0xff, 0xfe] + UInt16(60_002).bigEndianBytes)
            photo.append(Data(repeating: UInt8(index), count: 60_000))
        }
        photo.append(live.dropFirst(2))
        var pointer: UnsafeMutablePointer<UInt8>?
        var size = 0
        var error = [CChar](repeating: 0, count: 256)
        let rc = photo.withUnsafeBytes { raw in
            wendy_camera_thumbnail(raw.bindMemory(to: UInt8.self).baseAddress, photo.count,
                                   &pointer, &size, &error, error.count)
        }
        XCTAssertEqual(rc, 0, "expanded JPEG must remain decodable")
        let bytes = try XCTUnwrap(pointer)
        defer { wendy_camera_free_bytes(bytes) }
        return (live, photo, Data(bytes: bytes, count: size))
    }

    private func startServer(directory: URL, fixture: (live: Data, photo: Data, thumb: Data),
                             connections: Int, requestTimeout: TimeInterval = 10,
                             responseTimeout: TimeInterval = 60,
                             sendBufferBytes: Int32? = nil) throws -> LoopbackServer {
        let listener = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        guard listener >= 0 else { throw POSIXError(.EIO) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = in_addr_t(0x7f00_0001).bigEndian
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0, listen(listener, 4) == 0 else {
            let saved = errno; close(listener); throw POSIXError(POSIXErrorCode(rawValue: saved) ?? .EIO)
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
        }
        guard named == 0 else { let saved = errno; close(listener); throw POSIXError(POSIXErrorCode(rawValue: saved) ?? .EIO) }
        let finished = DispatchSemaphore(value: 0)
        let live = fixture.live, photo = fixture.photo, thumb = fixture.thumb
        Thread.detachNewThread {
            defer { _ = Glibc.close(listener); finished.signal() }
            for _ in 0..<connections {
                let accepted = Glibc.accept(listener, nil, nil)
                if accepted < 0 { return }
                if var sendBufferBytes {
                    _ = setsockopt(accepted, SOL_SOCKET, SO_SNDBUF, &sendBufferBytes,
                                   socklen_t(MemoryLayout<Int32>.size))
                }
                let stream = SocketCameraConnection(fd: accepted, requestTimeoutSeconds: requestTimeout,
                                                    responseTimeoutSeconds: responseTimeout)
                if let store = try? PhotoStore(directory: directory) {
                    let handler = CameraCommandHandler(camera: FixtureCamera(live: live, photo: photo, thumb: thumb), photos: store)
                    handler.serve(stream)
                }
                stream.close()
            }
        }
        return LoopbackServer(port: UInt16(bigEndian: address.sin_port), finished: finished)
    }

    private func connect(_ port: UInt16, receiveBufferBytes: Int32? = nil) throws -> SocketCameraConnection {
        let fd = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        guard fd >= 0 else { throw POSIXError(.EIO) }
        if var receiveBufferBytes {
            _ = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receiveBufferBytes,
                           socklen_t(MemoryLayout<Int32>.size))
        }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = in_addr_t(0x7f00_0001).bigEndian
        let rc = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Glibc.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard rc == 0 else { let saved = errno; close(fd); throw POSIXError(POSIXErrorCode(rawValue: saved) ?? .EIO) }
        return SocketCameraConnection(fd: fd)
    }

    private func request(_ socket: SocketCameraConnection, op: UInt8, id: UInt32, payload: [UInt8] = []) throws {
        try socket.writeAll(try CameraHeader(operation: op, requestID: id, payloadLength: UInt32(payload.count)).bytes() + payload)
    }

    private func reply(_ socket: SocketCameraConnection) throws -> (CameraHeader, [UInt8]) {
        socket.beginRequest()
        let header = try CameraHeader.parse(socket.readExactly(CameraHeader.size))
        return (header, try socket.readExactly(Int(header.payloadLength)))
    }

    func testAllOperationsAndCatalogPersistAcrossConnections() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sample = try fixture()
        XCTAssertGreaterThan(sample.photo.count, 2 * 64 * 1024)
        let server = try startServer(directory: directory, fixture: sample, connections: 3)

        let first = try connect(server.port)
        try request(first, op: 1, id: 11)
        let snapshot = try reply(first)
        XCTAssertEqual(snapshot.0.operation, 0x81)
        XCTAssertEqual(snapshot.0.requestID, 11)
        XCTAssertEqual(Data(snapshot.1), sample.live)

        try request(first, op: 2, id: 12)
        let taken = try reply(first)
        XCTAssertEqual(taken.0.operation, 0x82)
        let record = try PhotoRecord.parse(taken.1)
        XCTAssertEqual(record.jpegBytes, UInt64(sample.photo.count))
        try request(first, op: 3, id: 13)
        let initialList = try reply(first)
        XCTAssertEqual(UInt16(bigEndianBytes: Array(initialList.1.prefix(2))), 1)
        XCTAssertEqual(try PhotoRecord.parse(Array(initialList.1.dropFirst(2))), record)
        first.close()

        // Each accepted connection constructs a fresh handler/PhotoStore.
        let second = try connect(server.port)
        try request(second, op: 3, id: 21)
        let persistedList = try reply(second)
        XCTAssertEqual(try PhotoRecord.parse(Array(persistedList.1.dropFirst(2))), record)

        let unknown = UUID()
        try request(second, op: 4, id: 22, payload: withUnsafeBytes(of: unknown.uuid) { Array($0) })
        let error = try reply(second)
        XCTAssertEqual(error.0.operation, 0x7f)
        XCTAssertEqual(error.0.requestID, 22)
        XCTAssertFalse(error.1.isEmpty)

        let idBytes = withUnsafeBytes(of: record.id.uuid) { Array($0) }
        try request(second, op: 4, id: 23, payload: idBytes)
        let thumbnail = try reply(second)
        XCTAssertEqual(thumbnail.0.operation, 0x84)
        XCTAssertEqual(Data(thumbnail.1), sample.thumb)

        try request(second, op: 5, id: 24, payload: idBytes)
        second.beginRequest()
        let downloadHeader = try CameraHeader.parse(second.readExactly(CameraHeader.size))
        XCTAssertEqual(downloadHeader.operation, 0x85)
        XCTAssertEqual(downloadHeader.requestID, 24)
        XCTAssertEqual(Int(downloadHeader.payloadLength), sample.photo.count)
        var downloaded = Data()
        while downloaded.count < sample.photo.count {
            let count = min(32 * 1024, sample.photo.count - downloaded.count)
            downloaded.append(contentsOf: try second.readExactly(count))
        }
        XCTAssertEqual(downloaded, sample.photo)

        try request(second, op: 6, id: 25, payload: idBytes)
        let removed = try reply(second)
        XCTAssertEqual(removed.0.operation, 0x86)
        XCTAssertEqual(removed.0.payloadLength, 0)
        second.close()

        let third = try connect(server.port)
        try request(third, op: 3, id: 31)
        let emptyList = try reply(third)
        XCTAssertEqual(emptyList.1, [0, 0])
        third.close()
        XCTAssertEqual(server.finished.wait(timeout: .now() + 3), .success)
    }

    func testMalformedAndOversizedHeadersCloseWithoutPayloadAllocation() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = try startServer(directory: directory, fixture: try fixture(), connections: 3)
        let oversized = try connect(server.port)
        let tooLarge = [UInt8]([87, 78, 67, 65, 1, 1]) + UInt32(1).bigEndianBytes + UInt32(CameraHeader.maxPayload + 1).bigEndianBytes
        try oversized.writeAll(tooLarge)
        XCTAssertThrowsError(try oversized.readExactly(1))
        oversized.close()

        let badMagic = try connect(server.port)
        try badMagic.writeAll([0, 78, 67, 65, 1, 1] + UInt32(2).bigEndianBytes + UInt32(0).bigEndianBytes)
        XCTAssertThrowsError(try badMagic.readExactly(1))
        badMagic.close()

        let badShape = try connect(server.port)
        try badShape.writeAll(try CameraHeader(operation: 1, requestID: 3, payloadLength: 16).bytes())
        XCTAssertThrowsError(try badShape.readExactly(1))
        badShape.close()
        XCTAssertEqual(server.finished.wait(timeout: .now() + 3), .success)
    }

    func testPartialHeaderDeadlineFreesSerialServerForNextClient() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sample = try fixture()
        let server = try startServer(directory: directory, fixture: sample, connections: 2, requestTimeout: 0.75)
        let stalled = try connect(server.port)
        try stalled.writeAll([87, 78, 67, 65, 1, 1]) // half a WNCA header
        let next = try connect(server.port)
        let started = ProcessInfo.processInfo.systemUptime
        try request(next, op: 1, id: 44)
        let response = try reply(next)
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(response.0.operation, 0x81)
        XCTAssertEqual(Data(response.1), sample.live)
        XCTAssertGreaterThan(elapsed, 0.5)
        XCTAssertLessThan(elapsed, 2.5)
        stalled.close()
        next.close()
        XCTAssertEqual(server.finished.wait(timeout: .now() + 3), .success)
    }

    func testStalledReaderDeadlineFreesSerialServerForNextClient() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var sample = try fixture()
        // This fits the protocol, but cannot fit in the deliberately small
        // socket buffers while the first client never reads its snapshot.
        sample.live = Data(repeating: 0x42, count: 8 * 1024 * 1024)
        let server = try startServer(directory: directory, fixture: sample, connections: 2,
                                     responseTimeout: 0.75, sendBufferBytes: 4096)
        let stalled = try connect(server.port, receiveBufferBytes: 4096)
        try request(stalled, op: 1, id: 51)

        let next = try connect(server.port)
        let started = ProcessInfo.processInfo.systemUptime
        try request(next, op: 3, id: 52)
        let response = try reply(next)
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(response.0.operation, 0x83)
        XCTAssertEqual(response.0.requestID, 52)
        XCTAssertEqual(response.1, [0, 0])
        XCTAssertGreaterThan(elapsed, 0.5)
        XCTAssertLessThan(elapsed, 3.5)
        stalled.close()
        next.close()
        XCTAssertEqual(server.finished.wait(timeout: .now() + 3), .success)
    }
}
#endif
