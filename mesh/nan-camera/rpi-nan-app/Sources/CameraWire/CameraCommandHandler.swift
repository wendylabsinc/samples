import Foundation

/// The production V4L2 camera and an in-memory fixture camera implement the
/// same request-facing seam. The handler never needs a radio or NDI.
public protocol CameraFrameSource {
    func snapshot() throws -> Data
    func take() throws -> (photo: Data, thumbnail: Data)
}

/// Serves one serialized WNCA TCP session. The owner closes its stream after
/// this method returns. A new handler can use the same PhotoStore directory
/// after an app restart without keeping any in-memory catalog state.
public final class CameraCommandHandler {
    private let camera: any CameraFrameSource
    private let photos: PhotoStore

    public init(camera: any CameraFrameSource, photos: PhotoStore) {
        self.camera = camera
        self.photos = photos
    }

    public func serve(_ client: CameraByteStream, shouldContinue: () -> Bool = { true }) {
        while shouldContinue() {
            let request: CameraRequest
            do { request = try readCameraRequest(from: client) }
            catch { return }
            let response = CameraResponseStream(client)
            do { try handle(request, response) }
            catch {
                // Once any reply bytes may have reached the peer, a second
                // header would be interpreted as part of the first payload.
                guard !response.started else { return }
                do { try writeCameraError(to: client, requestID: request.header.requestID, message: String(describing: error)) }
                catch { return }
            }
        }
    }

    private func handle(_ request: CameraRequest, _ client: CameraByteStream) throws {
        switch request.operation {
        case .snapshot:
            try writeCameraReply(to: client, request: request, payload: Array(camera.snapshot()))
        case .takePhoto:
            let captured = try camera.take()
            let saved = try photos.save(jpeg: captured.photo, thumbnail: captured.thumbnail)
            try writeCameraReply(to: client, request: request, payload: saved.bytes())
        case .listPhotos:
            let records = try photos.list()
            guard records.count <= UInt16.max else { throw CameraProtocolError.payloadTooLarge }
            var payload = UInt16(records.count).bigEndianBytes
            for record in records { payload += record.bytes() }
            try writeCameraReply(to: client, request: request, payload: payload)
        case .thumbnail:
            let id = try parsePhotoID(request.payload)
            try writeCameraReply(to: client, request: request, payload: Array(photos.thumbnail(id)))
        case .download:
            let id = try parsePhotoID(request.payload)
            let record = try photos.record(id)
            guard record.jpegBytes <= CameraHeader.maxPayload else { throw PhotoStoreError.tooLarge }
            let handle = try FileHandle(forReadingFrom: photos.imageURL(id))
            defer { try? handle.close() }
            let header = try CameraHeader(operation: request.operation.rawValue | 0x80,
                                          requestID: request.header.requestID, payloadLength: UInt32(record.jpegBytes))
            try client.writeAll(header.bytes())
            var remaining = record.jpegBytes
            while remaining > 0 {
                let chunk = try handle.read(upToCount: Int(min(remaining, 64 * 1024))) ?? Data()
                guard !chunk.isEmpty else { throw CameraProtocolError.disconnected }
                try client.writeAll(Array(chunk))
                remaining -= UInt64(chunk.count)
            }
        case .delete:
            try photos.delete(parsePhotoID(request.payload))
            try writeCameraReply(to: client, request: request, payload: [])
        }
    }
}

private final class CameraResponseStream: CameraByteStream {
    private let stream: any CameraByteStream
    private(set) var started = false

    init(_ stream: any CameraByteStream) { self.stream = stream }

    func readExactly(_ count: Int) throws -> [UInt8] { try stream.readExactly(count) }

    func writeAll(_ bytes: [UInt8]) throws {
        if !bytes.isEmpty { started = true }
        try stream.writeAll(bytes)
    }
}
