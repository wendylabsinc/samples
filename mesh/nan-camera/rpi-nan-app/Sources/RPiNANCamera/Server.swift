import CameraWire
import Foundation
import Glibc
import CameraC

enum ServerError: Error, CustomStringConvertible {
    case failed(String)
    var description: String { switch self { case .failed(let text): return text } }
}

/// Bind to the NDI's own scoped link-local address. Binding to :: would expose
/// the camera on the host's LAN because this container has host networking.
final class NDIListener {
    private let fd: Int32

    init(interface name: String, port: UInt16) throws {
        let index = name.withCString { if_nametoindex($0) }
        guard index != 0 else {
            throw ServerError.failed("NAN NDI \(name) does not exist")
        }
        var selected: in6_addr?
        // A newly created NDI is UP/NO-CARRIER and has no IPv6 address until
        // an Android client establishes an NDP. Continue waiting while our
        // NAN publish/event responder runs on its own thread.
        while selected == nil && wendy_stop_requested() == 0 {
            var root: UnsafeMutablePointer<ifaddrs>?
            guard getifaddrs(&root) == 0 else { throw ServerError.failed("getifaddrs: \(errno)") }
            var item = root
            while let current = item {
                let iface = current.pointee
                if String(cString: iface.ifa_name) == name,
                   let addr = iface.ifa_addr, Int32(addr.pointee.sa_family) == AF_INET6 {
                    let ip = UnsafeRawPointer(addr).assumingMemoryBound(to: sockaddr_in6.self).pointee.sin6_addr
                    let octets = withUnsafeBytes(of: ip) { Array($0) }
                    if octets[0] == 0xfe && (octets[1] & 0xc0) == 0x80 { selected = ip; break }
                }
                item = iface.ifa_next
            }
            freeifaddrs(root)
            if selected == nil { Thread.sleep(forTimeInterval: 0.25) }
        }
        if wendy_stop_requested() != 0 { throw ServerError.failed("shutdown while waiting for NDP address") }
        guard let address = selected else { throw ServerError.failed("NAN NDI \(name) has no IPv6 link-local address") }
        let fd = socket(AF_INET6, Int32(SOCK_STREAM.rawValue), 0)
        guard fd >= 0 else { throw ServerError.failed("TCP socket: \(errno)") }
        self.fd = fd
        var one: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
        _ = setsockopt(fd, Int32(IPPROTO_IPV6), IPV6_V6ONLY, &one, socklen_t(MemoryLayout<Int32>.size))
        var socketAddress = sockaddr_in6()
        socketAddress.sin6_family = sa_family_t(AF_INET6)
        socketAddress.sin6_port = port.bigEndian
        socketAddress.sin6_addr = address
        socketAddress.sin6_scope_id = index
        var bound = false
        for _ in 0..<40 {
            let result = withUnsafePointer(to: &socketAddress) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in6>.size)) }
            }
            if result == 0 { bound = true; break }
            if errno != EADDRNOTAVAIL { break }
            // The newly created NDI may still be finishing IPv6 DAD.
            Thread.sleep(forTimeInterval: 0.25)
        }
        guard bound else { let e = errno; close(fd); throw ServerError.failed("NDI bind: \(e)") }
        guard listen(fd, 4) == 0 else { let e = errno; close(fd); throw ServerError.failed("NDI listen: \(e)") }
        var printable = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
        withUnsafePointer(to: address) { _ = inet_ntop(AF_INET6, $0, &printable, socklen_t(printable.count)) }
        print("[nan-camera] TCP listening on [\(String(cString: printable))%\(name)]:\(port)")
    }

    deinit { close(fd) }

    func acceptConnection() throws -> SocketCameraConnection? {
        var pending = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        let ready = poll(&pending, 1, 1000)
        if ready == 0 || (ready < 0 && errno == EINTR) { return nil }
        guard ready > 0 else { throw ServerError.failed("TCP poll: \(errno)") }
        let client = accept(fd, nil, nil)
        guard client >= 0 else { throw ServerError.failed("TCP accept: \(errno)") }
        return SocketCameraConnection(fd: client, shouldStop: { wendy_stop_requested() != 0 })
    }
}

final class CameraServer {
    private let handler: CameraCommandHandler
    private let listener: NDIListener

    init(ndi: String, directory: URL) throws {
        handler = CameraCommandHandler(camera: Webcam(), photos: try PhotoStore(directory: directory))
        listener = try NDIListener(interface: ndi, port: 9091)
    }

    func run() {
        while wendy_stop_requested() == 0 {
            do {
                guard let client = try listener.acceptConnection() else { continue }
                print("[nan-camera] TCP client connected")
                handler.serve(client) { wendy_stop_requested() == 0 }
                print("[nan-camera] TCP client disconnected")
            } catch {
                print("[nan-camera] accept: \(error)")
                Thread.sleep(forTimeInterval: 0.2)
            }
        }
    }

}
