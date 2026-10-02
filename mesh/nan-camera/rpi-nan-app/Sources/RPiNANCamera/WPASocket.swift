import Foundation
import Glibc
import CameraC
import CameraWire

enum NANError: Error, CustomStringConvertible {
    case failed(String)
    /// The datagram was sent, but no trustworthy response arrived. Its side
    /// effect is unknown: callers must not treat this as a definitive FAIL.
    case ambiguous(String)
    var outcomeUnknown: Bool { if case .ambiguous = self { return true }; return false }
    var description: String {
        switch self {
        case .failed(let value): return value
        case .ambiguous(let value): return value + " (command outcome unknown)"
        }
    }
}

/// wpa control replies have no request IDs. An ambiguous command retires its
/// local pathname, so a late response cannot become the next command's reply.
/// The separate ATTACH socket keeps its endpoint during ordinary event timeouts.
final class WPASocket {
    private var fd: Int32 = -1
    private var localPath: String?
    private let remotePath: String
    private let clientDirectory: String
    private let timeoutSeconds: Int32
    private let commandLock = NSLock()

    init(remotePath: String, clientDirectory: String, receiveTimeoutSeconds: Int32) throws {
        guard receiveTimeoutSeconds > 0 else { throw NANError.failed("wpa timeout must be positive") }
        self.remotePath = remotePath
        self.clientDirectory = clientDirectory
        timeoutSeconds = receiveTimeoutSeconds
        try openEndpoint()
    }

    deinit { retireEndpoint() }

    private func retireEndpoint() {
        if fd >= 0 { close(fd); fd = -1 }
        if let localPath { unlink(localPath); self.localPath = nil }
    }

    private func openEndpoint() throws {
        guard fd < 0 else { return }
        let path = clientDirectory + "/cam-" + UUID().uuidString.lowercased().prefix(20)
        let opened = socket(AF_UNIX, Int32(SOCK_DGRAM.rawValue), 0)
        guard opened >= 0 else { throw NANError.failed("wpa socket: \(errno)") }
        var bound = false
        do {
            var timeout = timeval(tv_sec: Int(timeoutSeconds), tv_usec: 0)
            for option in [SO_RCVTIMEO, SO_SNDTIMEO] {
                guard setsockopt(opened, SOL_SOCKET, option, &timeout, socklen_t(MemoryLayout<timeval>.size)) == 0 else {
                    throw NANError.failed("wpa timeout setup: \(errno)")
                }
            }
            var local = try Self.address(path)
            let bindResult = withUnsafePointer(to: &local) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(opened, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
            }
            guard bindResult == 0 else { throw NANError.failed("wpa bind: \(errno)") }
            bound = true
            var remote = try Self.address(remotePath)
            let connectResult = withUnsafePointer(to: &remote) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(opened, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
            }
            guard connectResult == 0 else { throw NANError.failed("wpa connect: \(errno)") }
            fd = opened
            localPath = path
        } catch {
            close(opened)
            if bound { unlink(path) }
            throw error
        }
    }

    private static func address(_ path: String) throws -> sockaddr_un {
        var value = sockaddr_un()
        value.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8) + [0]
        guard bytes.count <= MemoryLayout.size(ofValue: value.sun_path) else { throw NANError.failed("wpa socket pathname too long") }
        withUnsafeMutableBytes(of: &value.sun_path) { raw in raw.copyBytes(from: bytes) }
        return value
    }

    func command(_ text: String) throws -> String {
        commandLock.lock()
        defer { commandLock.unlock() }
        try openEndpoint()
        let bytes = Array(text.utf8)
        let sent = bytes.withUnsafeBytes { send(fd, $0.baseAddress, bytes.count, Int32(MSG_NOSIGNAL)) }
        guard sent == bytes.count else {
            let code = errno
            retireEndpoint()
            throw NANError.failed("wpa send \(text.prefix(20)): \(code)")
        }
        let reply: String
        do { reply = try receive() }
        catch {
            retireEndpoint()
            // No retry: a timed-out publication may exist until its finite TTL.
            throw NANError.ambiguous("wpa \(text.prefix(20)): \(error)")
        }
        guard !reply.hasPrefix("FAIL"), reply != "UNKNOWN COMMAND" else {
            throw NANError.failed("wpa \(text.prefix(20)): \(reply)")
        }
        return reply
    }

    func receive() throws -> String {
        guard fd >= 0 else { throw NANError.failed("wpa endpoint unavailable") }
        var buffer = [UInt8](repeating: 0, count: 16 * 1024)
        let capacity = buffer.count
        let size = buffer.withUnsafeMutableBytes { recv(fd, $0.baseAddress, capacity, Int32(MSG_TRUNC)) }
        guard size >= 0 else { throw NANError.failed("wpa recv: \(errno)") }
        guard size <= capacity else { throw NANError.failed("wpa oversized datagram") }
        return String(decoding: buffer.prefix(size), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Startup has no established TCP/NDP session yet. Keep ambiguous-publish
/// cooldown in-process so a restart policy cannot immediately replay it.
func waitForNANPublishRetry(delay: TimeInterval,
                           now: () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
                           pause: (TimeInterval) -> Void = { Thread.sleep(forTimeInterval: $0) },
                           stopped: () -> Bool = { wendy_stop_requested() != 0 }) throws {
    let until = now() + delay
    while now() < until {
        if stopped() { throw NANError.failed("NAN publication stopped") }
        pause(min(0.1, max(0, until - now())))
    }
    if stopped() { throw NANError.failed("NAN publication stopped") }
}

/// Definitive rejections retain the existing six-attempt startup policy. An
/// uncertain publish waits in this process even on the final attempt, before
/// returning an error that could trigger an automatic container restart.
func initialNANPublication(command: () throws -> String,
                           now: () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
                           wait: (TimeInterval) throws -> Void = { try waitForNANPublishRetry(delay: $0) }) throws -> (String, TimeInterval) {
    for attempt in 0..<6 {
        let started = now()
        do { return (try command(), started) }
        catch {
            if let error = error as? NANError, error.outcomeUnknown {
                print("[nan-camera] initial publish outcome unknown; waiting for finite lease cooldown")
                try wait(NANPublicationLease.uncertainRetryDelay)
            }
            if attempt == 5 { throw error }
            try wait(1)
        }
    }
    throw NANError.failed("initial publish attempts exhausted")
}
