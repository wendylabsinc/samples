import Foundation

public protocol CameraByteStream: AnyObject {
    func beginRequest()
    func readExactly(_ count: Int) throws -> [UInt8]
    func writeAll(_ bytes: [UInt8]) throws
}

public extension CameraByteStream {
    func beginRequest() {}
}

public struct CameraRequest {
    public let header: CameraHeader
    public let operation: CameraOperation
    public let payload: [UInt8]
}

public func readCameraRequest(from stream: CameraByteStream) throws -> CameraRequest {
    stream.beginRequest()
    let header = try CameraHeader.parse(stream.readExactly(CameraHeader.size))
    guard let operation = CameraOperation(rawValue: header.operation) else { throw CameraProtocolError.invalidOperation }
    let required = switch operation {
    case .snapshot, .takePhoto, .listPhotos: 0
    case .thumbnail, .download, .delete: 16
    }
    guard header.payloadLength == required else { throw CameraProtocolError.malformedPayload }
    return CameraRequest(header: header, operation: operation, payload: try stream.readExactly(required))
}

public func writeCameraReply(to stream: CameraByteStream, request: CameraRequest, payload: [UInt8]) throws {
    let header = try CameraHeader(operation: request.operation.rawValue | 0x80,
                                  requestID: request.header.requestID, payloadLength: UInt32(payload.count))
    try stream.writeAll(header.bytes())
    try stream.writeAll(payload)
}

public func writeCameraError(to stream: CameraByteStream, requestID: UInt32, message: String) throws {
    let payload = Array(message.utf8.prefix(240))
    let header = try CameraHeader(operation: 0x7f, requestID: requestID, payloadLength: UInt32(payload.count))
    try stream.writeAll(header.bytes())
    try stream.writeAll(payload)
}
