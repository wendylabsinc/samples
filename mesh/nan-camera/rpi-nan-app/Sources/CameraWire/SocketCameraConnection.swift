#if os(Linux)
import Foundation
import Glibc

/// The same bounded TCP channel is used by the NDI listener and the loopback
/// integration test. Monotonic deadlines span the entire request and response,
/// so a stalled reader or writer cannot monopolize the serial server forever.
public final class SocketCameraConnection: CameraByteStream {
    private let fd: Int32
    private let requestTimeoutSeconds: TimeInterval
    private let responseTimeoutSeconds: TimeInterval
    private let shouldStop: () -> Bool
    private var requestDeadline: TimeInterval
    private var responseDeadline: TimeInterval
    private var closed = false

    public init(fd: Int32, requestTimeoutSeconds: TimeInterval = 10,
                responseTimeoutSeconds: TimeInterval = 60,
                shouldStop: @escaping () -> Bool = { false }) {
        self.fd = fd
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.responseTimeoutSeconds = responseTimeoutSeconds
        self.shouldStop = shouldStop
        let now = ProcessInfo.processInfo.systemUptime
        requestDeadline = now + requestTimeoutSeconds
        responseDeadline = now + responseTimeoutSeconds
        var receiveTimeout = timeval(tv_sec: 1, tv_usec: 0)
        var sendTimeout = timeval(tv_sec: 1, tv_usec: 0)
        _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receiveTimeout, socklen_t(MemoryLayout<timeval>.size))
        _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sendTimeout, socklen_t(MemoryLayout<timeval>.size))
        var noDelay: Int32 = 1
        _ = setsockopt(fd, Int32(IPPROTO_TCP), TCP_NODELAY, &noDelay, socklen_t(MemoryLayout<Int32>.size))
    }

    deinit { close() }

    public func close() {
        guard !closed else { return }
        closed = true
        _ = shutdown(fd, Int32(SHUT_RDWR))
        _ = Glibc.close(fd)
    }

    public func beginRequest() {
        let now = ProcessInfo.processInfo.systemUptime
        requestDeadline = now + requestTimeoutSeconds
        responseDeadline = now + responseTimeoutSeconds
    }

    public func readExactly(_ count: Int) throws -> [UInt8] {
        if count == 0 { return [] }
        var bytes = [UInt8](repeating: 0, count: count)
        var offset = 0
        while offset < count {
            if shouldStop() { throw CameraProtocolError.disconnected }
            if ProcessInfo.processInfo.systemUptime >= requestDeadline { throw CameraProtocolError.requestTimedOut }
            let n = bytes.withUnsafeMutableBytes { raw in
                recv(fd, raw.baseAddress!.advanced(by: offset), count - offset, 0)
            }
            if n == 0 { throw CameraProtocolError.disconnected }
            if n < 0 {
                if errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK { continue }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            offset += n
        }
        return bytes
    }

    public func writeAll(_ bytes: [UInt8]) throws {
        var offset = 0
        try bytes.withUnsafeBytes { raw in
            while offset < bytes.count {
                if shouldStop() { throw CameraProtocolError.disconnected }
                if ProcessInfo.processInfo.systemUptime >= responseDeadline { throw CameraProtocolError.responseTimedOut }
                let n = send(fd, raw.baseAddress!.advanced(by: offset), bytes.count - offset, Int32(MSG_NOSIGNAL))
                if n < 0 {
                    if errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK { continue }
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
                if n == 0 { throw CameraProtocolError.disconnected }
                offset += n
            }
        }
    }
}
#endif
