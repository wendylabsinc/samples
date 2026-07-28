import Foundation
import GStreamer
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import Logging
import NIOCore
import WSCore

// MARK: - Response Types

struct SoundsResponse: ResponseEncodable {
    let sounds: [String]
}

struct PlayResponse: ResponseEncodable {
    let success: Bool
    let sound: String?
    let error: String?
}

struct StopResponse: ResponseEncodable {
    let success: Bool
}

// MARK: - Audio Manager

actor AudioManager {
    private var isPlaying = false

    /// Play a sound file using GStreamer.
    func playSound(_ soundName: String, soundsPath: String) async -> PlayResponse {
        guard !isPlaying else {
            return PlayResponse(success: false, sound: nil, error: "Already playing")
        }

        let soundFile = "\(soundsPath)/\(soundName).wav"
        guard FileManager.default.fileExists(atPath: soundFile) else {
            return PlayResponse(success: false, sound: nil, error: "Sound file not found: \(soundName)")
        }

        isPlaying = true
        defer { isPlaying = false }

        do {
            // Use GStreamer to play the audio file
            let pipelineDesc = """
                filesrc location=\(soundFile) ! \
                wavparse ! \
                audioconvert ! \
                audioresample ! \
                autoaudiosink
                """

            let pipeline = try Pipeline(pipelineDesc)
            try pipeline.play()

            // Wait for playback to complete
            for await message in pipeline.bus.messages() {
                switch message {
                case .eos:
                    pipeline.stop()
                    return PlayResponse(success: true, sound: soundName, error: nil)
                case .error(let msg, _):
                    pipeline.stop()
                    return PlayResponse(success: false, sound: nil, error: "Playback error: \(msg)")
                default:
                    continue
                }
            }

            return PlayResponse(success: true, sound: soundName, error: nil)
        } catch {
            return PlayResponse(success: false, sound: nil, error: "GStreamer error: \(error)")
        }
    }
}

// MARK: - Main

@main
struct AudioServer {
    static let audioManager = AudioManager()

    static func main() async throws {
        let hostname = ProcessInfo.processInfo.environment["WENDY_HOSTNAME"] ?? "localhost"
        let port = 3005
        let logger = Logger(label: "audio-server")

        // Initialize GStreamer
        logger.info("GStreamer version: \(GStreamer.versionString)")

        // Determine sounds path
        let containerPath = "/app/sounds"
        let bundlePath = Bundle.module.resourcePath.map { "\($0)/sounds" }
        let cwdPath = FileManager.default.currentDirectoryPath + "/Sources/audio-server/sounds"

        let soundsPath: String
        if FileManager.default.fileExists(atPath: containerPath) {
            soundsPath = containerPath
        } else if let bp = bundlePath, FileManager.default.fileExists(atPath: bp) {
            soundsPath = bp
        } else {
            soundsPath = cwdPath
        }

        logger.info("Audio files from: \(soundsPath)")

        // Determine frontend dist path
        let envPath = ProcessInfo.processInfo.environment["FRONTEND_DIST"]
        let containerFrontend = "/app/frontend/dist"
        let cwdFrontend = FileManager.default.currentDirectoryPath + "/../frontend/dist"

        let frontendDist: String
        if let env = envPath {
            frontendDist = env
        } else if FileManager.default.fileExists(atPath: containerFrontend + "/index.html") {
            frontendDist = containerFrontend
        } else {
            frontendDist = cwdFrontend
        }

        logger.info("Serving frontend from: \(frontendDist)")

        // Capture soundsPath for closures
        let capturedSoundsPath = soundsPath

        // Set up router
        let router = Router()

        // API: List available sounds
        router.get("/api/sounds") { _, _ in
            SoundsResponse(sounds: ["cat", "dog", "cow"])
        }

        // API: Play a sound
        router.post("/api/play/{sound}") { request, _ in
            guard let sound = request.uri.path.split(separator: "/").last.map(String.init) else {
                return PlayResponse(success: false, sound: nil, error: "Invalid sound")
            }

            let validSounds = ["cat", "dog", "cow"]
            guard validSounds.contains(sound) else {
                return PlayResponse(success: false, sound: nil, error: "Invalid sound")
            }

            return await audioManager.playSound(sound, soundsPath: capturedSoundsPath)
        }

        // API: Stop microphone (for cleanup)
        router.post("/api/stop-microphone") { _, _ in
            StopResponse(success: true)
        }

        // Serve static files from frontend dist
        router.add(middleware: FileMiddleware(frontendDist, searchForIndexHtml: true))

        // Create HTTP application with WebSocket upgrade
        let app = Application(
            router: router,
            server: .http1WebSocketUpgrade { request, _, logger in
                // Only upgrade /ws/microphone
                guard request.path == "/ws/microphone" else {
                    return .dontUpgrade
                }

                logger.info("WebSocket upgrade request for microphone")

                return .upgrade([:]) { inbound, outbound, _ in
                    await handleMicrophoneWebSocket(inbound: inbound, outbound: outbound, logger: logger)
                }
            },
            configuration: .init(address: .hostname("0.0.0.0", port: port))
        )

        logger.info("Server running on http://\(hostname):\(port)")
        try await app.runService()
    }
}

// MARK: - Audio Buffer Helper

/// Extract bytes from an AudioBuffer (sync helper to avoid compiler crash)
func extractAudioBytes(from buffer: AudioBuffer) -> Data {
    var audioBytes = [UInt8]()
    buffer.bytes.withUnsafeBytes { bytes in
        audioBytes = Array(UnsafeRawBufferPointer(start: bytes.baseAddress!, count: bytes.count))
    }
    return Data(audioBytes)
}

func readAlsaCaptureDevices() -> [String] {
    let pcmPath = "/proc/asound/pcm"
    guard let contents = try? String(contentsOfFile: pcmPath, encoding: .utf8) else {
        return []
    }

    var devices: [String] = []
    for rawLine in contents.split(separator: "\n") {
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard line.contains("capture") else {
            continue
        }

        let idPart = line.split(separator: ":").first?.trimmingCharacters(in: .whitespaces) ?? ""
        let idPieces = idPart.split(separator: "-")
        guard idPieces.count == 2,
              let card = Int(idPieces[0]),
              let device = Int(idPieces[1])
        else {
            continue
        }

        devices.append("hw:\(card),\(device)")
    }

    return devices
}

// MARK: - WebSocket Handler

func handleMicrophoneWebSocket(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    logger: Logger
) async {
    logger.info("Microphone WebSocket connected")

    do {
        // Create a microphone capture pipeline with GStreamer
        // Try different audio backends in order of preference
        let envBackend = ProcessInfo.processInfo.environment["AUDIO_BACKEND"]
        let envDevice = ProcessInfo.processInfo.environment["AUDIO_DEVICE"]
        let alsaDevices = readAlsaCaptureDevices()
        if !alsaDevices.isEmpty {
            logger.debug("ALSA capture devices detected: \(alsaDevices.joined(separator: ", "))")
        }

        let alsaBackend: String
        if let device = envDevice, !device.isEmpty {
            alsaBackend = "alsasrc device=\(device)"
        } else if let firstDevice = alsaDevices.first {
            alsaBackend = "alsasrc device=\(firstDevice)"
        } else {
            alsaBackend = "alsasrc"
        }

        let backends: [String]
        if let backend = envBackend, !backend.isEmpty {
            backends = [backend]
        } else {
            // Prefer ALSA (direct /dev/snd) inside the container
            backends = [
                alsaBackend,
                "pulsesrc",
                "pipewiresrc",
            ]
        }

        var pipeline: Pipeline?
        var sink: AudioBufferSink?

        for backend in backends {
            let pipelineDesc = """
                \(backend) ! \
                audioconvert ! \
                audioresample ! \
                audio/x-raw,format=S16LE,rate=16000,channels=1 ! \
                appsink name=sink
                """

            do {
                pipeline = try Pipeline(pipelineDesc)
                sink = try pipeline!.audioBufferSink(named: "sink")
                try pipeline!.play()
                logger.info("Using audio backend: \(backend)")
                break
            } catch {
                logger.debug("Backend \(backend) not available: \(error)")
                continue
            }
        }

        guard let pipeline, let sink else {
            let errorMsg = #"{"type":"error","message":"No audio backend available"}"#
            try await outbound.write(.text(errorMsg))
            return
        }

        defer { pipeline.stop() }

        // Stream audio buffers to the client
        for await buffer in sink.buffers() {
            if Task.isCancelled {
                break
            }

            // Extract bytes using sync helper
            let data = extractAudioBytes(from: buffer)

            // Base64 encode the audio data
            let base64Data = data.base64EncodedString()

            // Send as JSON
            let json = """
                {"type":"audio","data":"\(base64Data)","sampleRate":16000,"channels":1}
                """

            do {
                try await outbound.write(.text(json))
            } catch {
                break
            }
        }

        logger.info("Microphone WebSocket disconnected")

    } catch {
        logger.error("Microphone stream error: \(error)")
        let errorMsg = #"{"type":"error","message":"\#(error)"}"#
        try? await outbound.write(.text(errorMsg))
    }
}
