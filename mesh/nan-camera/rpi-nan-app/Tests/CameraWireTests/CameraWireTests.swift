import Foundation
import XCTest
import CameraC
@testable import CameraWire

final class CameraWireTests: XCTestCase {
    final class MemoryStream: CameraByteStream {
        var input: [UInt8]
        var output: [UInt8] = []
        init(_ input: [UInt8]) { self.input = input }
        func readExactly(_ count: Int) throws -> [UInt8] {
            guard input.count >= count else { throw CameraProtocolError.disconnected }
            let value = Array(input.prefix(count)); input.removeFirst(count); return value
        }
        func writeAll(_ bytes: [UInt8]) throws { output += bytes }
    }

    func testGoldenHeaderAndBounds() throws {
        let header = try CameraHeader(operation: 0x05, requestID: 0x01020304, payloadLength: 16)
        XCTAssertEqual(header.bytes(), [87,78,67,65,1,5,1,2,3,4,0,0,0,16])
        XCTAssertEqual(try CameraHeader.parse(header.bytes()), header)
        XCTAssertThrowsError(try CameraHeader.parse([0,0,0,0] + Array(header.bytes().dropFirst(4))))
        XCTAssertThrowsError(try CameraHeader(operation: 1, requestID: 1, payloadLength: UInt32(CameraHeader.maxPayload + 1)))
    }

    func testRequestAndReply() throws {
        let id = UUID(uuidString: "00112233-4455-6677-8899-aabbccddeeff")!
        let idBytes = withUnsafeBytes(of: id.uuid) { Array($0) }
        let input = try CameraHeader(operation: 5, requestID: 42, payloadLength: 16).bytes() + idBytes
        let stream = MemoryStream(input)
        let request = try readCameraRequest(from: stream)
        XCTAssertEqual(request.operation, .download)
        XCTAssertEqual(try parsePhotoID(request.payload), id)
        try writeCameraReply(to: stream, request: request, payload: [0xff, 0xd8, 0xff, 0xd9])
        XCTAssertEqual(stream.output, try CameraHeader(operation: 0x85, requestID: 42, payloadLength: 4).bytes() + [0xff,0xd8,0xff,0xd9])
        let malformed = MemoryStream(try CameraHeader(operation: 5, requestID: 1, payloadLength: 0).bytes())
        XCTAssertThrowsError(try readCameraRequest(from: malformed))
    }

    func testPhotoRecordRoundTripAndStorage() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try PhotoStore(directory: directory)
        let saved = try store.save(jpeg: Data([0xff,0xd8,0xff,0xd9]), thumbnail: Data([1,2,3]))
        XCTAssertEqual(try PhotoRecord.parse(saved.bytes()), saved)
        XCTAssertEqual(try store.list().map(\.id), [saved.id])
        XCTAssertEqual(try store.thumbnail(saved.id), Data([1,2,3]))
        XCTAssertEqual(try Data(contentsOf: store.imageURL(saved.id)), Data([0xff,0xd8,0xff,0xd9]))
        try store.delete(saved.id)
        XCTAssertEqual(try store.list(), [])
        XCTAssertThrowsError(try store.imageURL(saved.id))
    }

    func testThumbnailIsRealBoundedJPEG() throws {
        let fixture = try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: "source", withExtension: "jpg")))
        var output: UnsafeMutablePointer<UInt8>?
        var length = 0
        var error = [CChar](repeating: 0, count: 256)
        let result = fixture.withUnsafeBytes { raw in
            wendy_camera_thumbnail(raw.bindMemory(to: UInt8.self).baseAddress, fixture.count,
                                   &output, &length, &error, error.count)
        }
        XCTAssertEqual(result, 0)
        let bytes = try XCTUnwrap(output)
        defer { wendy_camera_free_bytes(bytes) }
        XCTAssertGreaterThan(length, 100)
        XCTAssertLessThan(length, 100_000)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: bytes, count: 2)), [0xff, 0xd8])
        XCTAssertEqual(Array(UnsafeBufferPointer(start: bytes.advanced(by: length - 2), count: 2)), [0xff, 0xd9])
        // The output is a 160x120 JPEG, rather than a low-res frame captured
        // at a different instant from the saved full-resolution photo.
        let jpeg = Array(UnsafeBufferPointer(start: bytes, count: length))
        guard let sof = jpeg.indices.first(where: { $0 + 8 < jpeg.count && jpeg[$0] == 0xff && jpeg[$0 + 1] == 0xc0 }) else {
            XCTFail("JPEG has no baseline SOF marker"); return
        }
        XCTAssertEqual(UInt16(bigEndianBytes: Array(jpeg[(sof + 5)..<(sof + 7)])), 120)
        XCTAssertEqual(UInt16(bigEndianBytes: Array(jpeg[(sof + 7)..<(sof + 9)])), 160)

        var rejected: UnsafeMutablePointer<UInt8>?
        var rejectedLength = 0
        XCTAssertNotEqual(wendy_camera_thumbnail([0, 1, 2, 3], 4, &rejected, &rejectedLength, &error, error.count), 0)
        XCTAssertNil(rejected)
    }

    func testOwnedNDPLifecycleUsesMACAndKeepsMultipleIDs() {
        let local = "e4:4a:e0:e7:6f:cd"
        let peer = "e4:4a:e0:e5:ba:65"
        let initiated = "e4:4a:e0:e5:ba:66"
        let first = "<3>NAN-NDP-CONNECTED peer=\(peer) ndp_id=4 local_ndi=\(local) peer_ndi=\(initiated)"
        let second = "<3>NAN-NDP-CONNECTED peer=\(peer) ndp_id=1024 local_ndi=\(local) peer_ndi=\(initiated)"
        guard case .connected(let a)? = parseOwnedNDPEvent(first, localNDIMAC: local),
              case .connected(let b)? = parseOwnedNDPEvent(second, localNDIMAC: local) else {
            XCTFail("failed to parse owned NDPs"); return
        }
        XCTAssertNotEqual(a.key, b.key)
        XCTAssertEqual(a.initiatorNDI, initiated)
        XCTAssertEqual(b.id, "1024")
        XCTAssertNil(parseOwnedNDPEvent(first, localNDIMAC: "00:00:00:00:00:01"))
        let disconnected = "<3>NAN-NDP-DISCONNECTED peer=\(peer) ndp_id=4 failure=0"
        XCTAssertEqual(parseOwnedNDPEvent(disconnected, localNDIMAC: local), .disconnected(peerNMI: peer, id: "4"))
    }
}
