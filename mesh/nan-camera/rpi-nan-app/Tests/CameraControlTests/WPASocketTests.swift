import XCTest
import Foundation
import Glibc
@testable import RPiNANCamera

private final class DatagramServer: @unchecked Sendable {
    let directory: String
    let path: String
    let fd: Int32
    private let lock = NSLock()
    private var seen: [String] = []
    private var paths: [String] = []
    private var failure: String?
    let done = DispatchSemaphore(value: 0)

    init() throws {
        directory = "/tmp/wpa-test-" + UUID().uuidString
        path = directory + "/server"
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        fd = socket(AF_UNIX, Int32(SOCK_DGRAM.rawValue), 0)
        guard fd >= 0 else { throw NANError.failed("test socket") }
        var timeout = timeval(tv_sec: 5, tv_usec: 0)
        _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        var address = Self.address(path)
        let result = withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) } }
        guard result == 0 else { throw NANError.failed("test bind") }
    }
    deinit { close(fd); try? FileManager.default.removeItem(atPath: directory) }
    static func address(_ path: String) -> sockaddr_un {
        var a = sockaddr_un(); a.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8) + [0]
        withUnsafeMutableBytes(of: &a.sun_path) { $0.copyBytes(from: bytes) }
        return a
    }
    func serve(_ replies: [(String, TimeInterval)]) {
        DispatchQueue.global().async { [self] in
            defer { done.signal() }
            for (reply, delay) in replies {
                var remote = sockaddr_un(); var count = socklen_t(MemoryLayout<sockaddr_un>.size)
                var bytes = [UInt8](repeating: 0, count: 1024)
                let size = withUnsafeMutablePointer(to: &remote) { rp in rp.withMemoryRebound(to: sockaddr.self, capacity: 1) { sp in bytes.withUnsafeMutableBytes { recvfrom(fd, $0.baseAddress, $0.count, 0, sp, &count) } } }
                guard size > 0 else { lock.lock(); failure = "server recv \(errno)"; lock.unlock(); return }
                let pathname = withUnsafeBytes(of: remote.sun_path) { raw in String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self) }
                lock.lock(); seen.append(String(decoding: bytes.prefix(size), as: UTF8.self)); paths.append(pathname); lock.unlock()
                Thread.sleep(forTimeInterval: delay)
                let data = Array(reply.utf8)
                _ = withUnsafePointer(to: &remote) { rp in rp.withMemoryRebound(to: sockaddr.self, capacity: 1) { sp in data.withUnsafeBytes { sendto(fd, $0.baseAddress, $0.count, Int32(MSG_NOSIGNAL), sp, count) } } }
                // A response to an intentionally retired pathname may fail.
            }
        }
    }
    func results() -> ([String], [String], String?) {
        lock.lock(); defer { lock.unlock() }; return (seen, paths, failure)
    }
}

final class WPASocketTests: XCTestCase {
    func testDelayedReplyCannotPoisonNextCommand() throws {
        let server = try DatagramServer()
        server.serve([("101", 1.25), ("OK", 0)])
        var client: WPASocket? = try WPASocket(remotePath: server.path, clientDirectory: server.directory, receiveTimeoutSeconds: 1)
        XCTAssertThrowsError(try client!.command("NAN_PUBLISH service_name=fixture ttl=180"))
        XCTAssertEqual(try client!.command("NAN_CANCEL_PUBLISH publish_id=known-owned"), "OK")
        XCTAssertEqual(server.done.wait(timeout: .now()+3), .success)
        let (commands, paths, error) = server.results()
        XCTAssertNil(error); XCTAssertEqual(commands.count, 2) // no automatic replay
        XCTAssertNotEqual(paths[0], paths[1])
        XCTAssertFalse(FileManager.default.fileExists(atPath: paths[0]))
        client = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: paths[1]))
    }

    func testDefinitiveFailureKeepsCommandEndpoint() throws {
        let server = try DatagramServer(); server.serve([("FAIL-BUSY", 0), ("OK", 0)])
        let client = try WPASocket(remotePath: server.path, clientDirectory: server.directory, receiveTimeoutSeconds: 1)
        XCTAssertThrowsError(try client.command("FAIL_TEST"))
        XCTAssertEqual(try client.command("PING"), "OK")
        XCTAssertEqual(server.done.wait(timeout: .now()+3), .success)
        let (_, paths, error) = server.results(); XCTAssertNil(error); XCTAssertEqual(paths[0], paths[1])
    }

    func testEventTimeoutRetainsAttachEndpoint() throws {
        let server = try DatagramServer(); server.serve([("OK", 0)])
        let client = try WPASocket(remotePath: server.path, clientDirectory: server.directory, receiveTimeoutSeconds: 1)
        XCTAssertEqual(try client.command("ATTACH"), "OK")
        XCTAssertEqual(server.done.wait(timeout: .now()+3), .success)
        let (_, paths, _) = server.results()
        XCTAssertThrowsError(try client.receive())
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths[0]))
        var remote = DatagramServer.address(paths[0]); let event = Array("<3>NAN-DISCOVERY-RESULT".utf8)
        let sent = withUnsafePointer(to: &remote) { rp in rp.withMemoryRebound(to: sockaddr.self, capacity: 1) { sp in event.withUnsafeBytes { sendto(server.fd, $0.baseAddress, $0.count, 0, sp, socklen_t(MemoryLayout<sockaddr_un>.size)) } } }
        XCTAssertEqual(sent, event.count)
        XCTAssertEqual(try client.receive(), "<3>NAN-DISCOVERY-RESULT")
    }

    func testFullServerQueueHasBoundedSendAndCanRecover() throws {
        let server = try DatagramServer()
        let filler = socket(AF_UNIX, Int32(SOCK_DGRAM.rawValue), 0); defer { close(filler) }
        var remote = DatagramServer.address(server.path)
        let connected = withUnsafePointer(to: &remote) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(filler, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) } }
        XCTAssertEqual(connected, 0)
        var byte: UInt8 = 65
        var filled = 0
        while filled < 1000 && send(filler, &byte, 1, Int32(MSG_DONTWAIT)) == 1 { filled += 1 }
        XCTAssertGreaterThan(filled, 0); XCTAssertLessThan(filled, 1000)
        let client = try WPASocket(remotePath: server.path, clientDirectory: server.directory, receiveTimeoutSeconds: 1)
        let start = ProcessInfo.processInfo.systemUptime
        XCTAssertThrowsError(try client.command("PING"))
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime-start, 2.5)
        while recv(server.fd, &byte, 1, Int32(MSG_DONTWAIT)) > 0 {}
        server.serve([("PONG", 0)])
        XCTAssertEqual(try client.command("PING"), "PONG")
        XCTAssertEqual(server.done.wait(timeout: .now()+3), .success)
        XCTAssertEqual(server.results().0, ["PING"])
    }

    func testFailedConnectDoesNotLeakClientPath() throws {
        let server = try DatagramServer()
        XCTAssertThrowsError(try WPASocket(remotePath: server.path+"-missing", clientDirectory: server.directory, receiveTimeoutSeconds: 1))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: server.directory), ["server"])
    }

    func testOversizedReplyRetiresEndpoint() throws {
        let server = try DatagramServer(); server.serve([(String(repeating: "x", count: 17*1024), 0), ("OK", 0)])
        let client = try WPASocket(remotePath: server.path, clientDirectory: server.directory, receiveTimeoutSeconds: 1)
        XCTAssertThrowsError(try client.command("OVERSIZE_TEST"))
        XCTAssertEqual(try client.command("PING"), "OK")
        XCTAssertEqual(server.done.wait(timeout: .now()+3), .success)
        XCTAssertNotEqual(server.results().1[0], server.results().1[1])
    }
}
