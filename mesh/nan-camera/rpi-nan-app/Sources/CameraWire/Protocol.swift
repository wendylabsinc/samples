import Foundation

public enum CameraProtocolError: Error, Equatable {
    case invalidMagic
    case invalidVersion
    case invalidOperation
    case payloadTooLarge
    case malformedPayload
    case disconnected
    case requestTimedOut
    case responseTimedOut
}

public enum CameraOperation: UInt8, CaseIterable {
    case snapshot = 0x01
    case takePhoto = 0x02
    case listPhotos = 0x03
    case thumbnail = 0x04
    case download = 0x05
    case delete = 0x06
}

public struct CameraHeader: Equatable {
    public static let size = 14
    public static let maxPayload = 32 * 1024 * 1024
    public let operation: UInt8
    public let requestID: UInt32
    public let payloadLength: UInt32

    public init(operation: UInt8, requestID: UInt32, payloadLength: UInt32) throws {
        guard Int(payloadLength) <= Self.maxPayload else { throw CameraProtocolError.payloadTooLarge }
        self.operation = operation
        self.requestID = requestID
        self.payloadLength = payloadLength
    }

    public func bytes() -> [UInt8] {
        [87, 78, 67, 65, 1, operation] + requestID.bigEndianBytes + payloadLength.bigEndianBytes
    }

    public static func parse(_ bytes: [UInt8]) throws -> Self {
        guard bytes.count == size else { throw CameraProtocolError.malformedPayload }
        guard bytes[0..<4].elementsEqual([87, 78, 67, 65]) else { throw CameraProtocolError.invalidMagic }
        guard bytes[4] == 1 else { throw CameraProtocolError.invalidVersion }
        return try Self(operation: bytes[5], requestID: UInt32(bigEndianBytes: Array(bytes[6..<10])), payloadLength: UInt32(bigEndianBytes: Array(bytes[10..<14])))
    }
}

public struct PhotoRecord: Equatable {
    public static let size = 32
    public let id: UUID
    public let capturedAtUnixMs: UInt64
    public let jpegBytes: UInt64

    public init(id: UUID, capturedAtUnixMs: UInt64, jpegBytes: UInt64) {
        self.id = id
        self.capturedAtUnixMs = capturedAtUnixMs
        self.jpegBytes = jpegBytes
    }

    public func bytes() -> [UInt8] {
        let idBytes = withUnsafeBytes(of: id.uuid) { Array($0) }
        return idBytes + capturedAtUnixMs.bigEndianBytes + jpegBytes.bigEndianBytes
    }

    public static func parse(_ bytes: [UInt8]) throws -> Self {
        guard bytes.count == size else { throw CameraProtocolError.malformedPayload }
        let id = UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ))
        return Self(id: id, capturedAtUnixMs: UInt64(bigEndianBytes: Array(bytes[16..<24])), jpegBytes: UInt64(bigEndianBytes: Array(bytes[24..<32])))
    }
}

public func parsePhotoID(_ bytes: [UInt8]) throws -> UUID {
    guard bytes.count == 16 else { throw CameraProtocolError.malformedPayload }
    return UUID(uuid: (
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    ))
}

public extension FixedWidthInteger {
    var bigEndianBytes: [UInt8] {
        (0..<MemoryLayout<Self>.size).reversed().map { UInt8(truncatingIfNeeded: self >> ($0 * 8)) }
    }

    init(bigEndianBytes bytes: [UInt8]) {
        self = bytes.reduce(0) { ($0 << 8) | Self($1) }
    }
}
