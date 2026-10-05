import Foundation
import XCTest
@testable import CameraWire

final class CameraCommandHandlerTests: XCTestCase {
    private final class Stream: CameraByteStream {
        var input: [UInt8]
        var output: [UInt8] = []
        var writeCount = 0
        let failWrite: Int?

        init(failWrite: Int? = nil) throws {
            input = try CameraHeader(operation: 1, requestID: 1, payloadLength: 0).bytes()
                + CameraHeader(operation: 1, requestID: 2, payloadLength: 0).bytes()
            self.failWrite = failWrite
        }

        func readExactly(_ count: Int) throws -> [UInt8] {
            guard input.count >= count else { throw CameraProtocolError.disconnected }
            let bytes = Array(input.prefix(count))
            input.removeFirst(count)
            return bytes
        }

        func writeAll(_ bytes: [UInt8]) throws {
            writeCount += 1
            if writeCount == failWrite {
                output += bytes.prefix(1) // A failed write may still send bytes.
                throw CameraProtocolError.disconnected
            }
            output += bytes
        }
    }

    private final class Camera: CameraFrameSource {
        var failNextSnapshot = false
        func snapshot() throws -> Data {
            if failNextSnapshot {
                failNextSnapshot = false
                throw PhotoStoreError.full
            }
            return Data([1, 2, 3])
        }
        func take() throws -> (photo: Data, thumbnail: Data) { (Data(), Data()) }
    }

    func testPartialReplyClosesWithoutAppendingErrorFrame() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let handler = CameraCommandHandler(camera: Camera(), photos: try PhotoStore(directory: directory))
        for failedWrite in [1, 2] {
            let stream = try Stream(failWrite: failedWrite)
            handler.serve(stream)
            XCTAssertEqual(stream.writeCount, failedWrite)
            XCTAssertEqual(stream.input.count, CameraHeader.size, "Do not process another request on a broken stream")
            XCTAssertEqual(stream.output.count, failedWrite == 1 ? 1 : CameraHeader.size + 1)
        }
    }

    func testOperationErrorBeforeReplyAllowsNextRequest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let camera = Camera()
        camera.failNextSnapshot = true
        let handler = CameraCommandHandler(camera: camera, photos: try PhotoStore(directory: directory))
        let stream = try Stream()
        handler.serve(stream)
        let error = try CameraHeader.parse(Array(stream.output.prefix(CameraHeader.size)))
        XCTAssertEqual(error.operation, 0x7f)
        XCTAssertEqual(error.requestID, 1)
        let replyOffset = CameraHeader.size + Int(error.payloadLength)
        let reply = try CameraHeader.parse(Array(stream.output[replyOffset..<(replyOffset + CameraHeader.size)]))
        XCTAssertEqual(reply.operation, 0x81)
        XCTAssertEqual(reply.requestID, 2)
        XCTAssertEqual(Array(stream.output.suffix(3)), [1, 2, 3])
    }
}
