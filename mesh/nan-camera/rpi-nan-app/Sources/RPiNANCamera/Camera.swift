import CameraC
import CameraWire
import Foundation

enum CameraError: Error, CustomStringConvertible {
    case unavailable(String)
    var description: String {
        switch self { case .unavailable(let detail): return detail }
    }
}

/// V4L2 stays streaming between live-view requests. A full-resolution still
/// briefly switches camera mode, after which the next snapshot reopens low-res.
final class Webcam: CameraFrameSource {
    private var device: String?
    private var live: OpaquePointer?

    deinit { if let live { wendy_camera_close(live) } }

    private func candidates() -> [String] {
        if let configured = ProcessInfo.processInfo.environment["WENDY_CAMERA_DEVICE"] { return [configured] }
        let paths = (try? FileManager.default.contentsOfDirectory(atPath: "/dev")) ?? []
        return paths.filter { $0.range(of: #"^video[0-9]+$"#, options: .regularExpression) != nil }
            .sorted().map { "/dev/" + $0 }
    }

    private func open(width: UInt32, height: UInt32) throws -> OpaquePointer {
        let available = candidates()
        let paths = device.map { previous in
            [previous] + available.filter { $0 != previous }
        } ?? available
        var lastError = "no USB UVC /dev/video* capture device"
        for path in paths {
            var handle: OpaquePointer?
            var error = [CChar](repeating: 0, count: 256)
            if path.withCString({ wendy_camera_open($0, width, height, &handle, &error, error.count) }) == 0,
                let handle {
                device = path
                return handle
            }
            lastError = String(cString: error)
        }
        throw CameraError.unavailable(lastError)
    }

    private func grab(_ handle: OpaquePointer) throws -> Data {
        var bytes: UnsafeMutablePointer<UInt8>?
        var length = 0
        var error = [CChar](repeating: 0, count: 256)
        guard wendy_camera_grab_jpeg(handle, &bytes, &length, &error, error.count) == 0,
              let bytes else { throw CameraError.unavailable(String(cString: error)) }
        defer { wendy_camera_free_bytes(bytes) }
        return Data(bytes: bytes, count: length)
    }

    func snapshot() throws -> Data {
        if live == nil { live = try open(width: 320, height: 240) }
        do { return try grab(live!) }
        catch {
            wendy_camera_close(live)
            live = nil
            throw error
        }
    }

    func take() throws -> (photo: Data, thumbnail: Data) {
        if let live { wendy_camera_close(live); self.live = nil }
        let still = try open(width: 1920, height: 1080)
        defer { wendy_camera_close(still) }
        let photo = try grab(still)
        var thumbBytes: UnsafeMutablePointer<UInt8>?
        var thumbLength = 0
        var error = [CChar](repeating: 0, count: 256)
        let rc = photo.withUnsafeBytes { raw in
            wendy_camera_thumbnail(raw.bindMemory(to: UInt8.self).baseAddress, photo.count,
                                   &thumbBytes, &thumbLength, &error, error.count)
        }
        guard rc == 0, let thumbBytes else { throw CameraError.unavailable(String(cString: error)) }
        defer { wendy_camera_free_bytes(thumbBytes) }
        return (photo, Data(bytes: thumbBytes, count: thumbLength))
    }
}
