import Foundation
import Glibc
import CameraC

do {
    wendy_install_stop_handlers()
    guard let ndi = ProcessInfo.processInfo.environment["WENDY_NAN_NDI"] else {
        throw ServerError.failed("WENDY_NAN_NDI missing; add nan entitlement")
    }
    // The NDI has no IPv6 link-local address until a peer establishes an NDP.
    // Publish and respond first; the listener waits for the resulting address.
    let publisher = try NANPublisher()
    defer { publisher.stop() }
    Thread.detachNewThread { publisher.runEvents() }
    let server = try CameraServer(ndi: ndi, directory: URL(fileURLWithPath: "/photos", isDirectory: true))
    server.run()
} catch {
    if wendy_stop_requested() != 0 { exit(0) }
    print("[nan-camera] fatal: \(error)")
    exit(1)
}
