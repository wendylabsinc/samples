import Foundation
import GStreamer
import Hummingbird
import HummingbirdWebSocket
import Logging
import NIOCore
import Synchronization

// MARK: - Video Settings

let frameWidth = 1280
let frameHeight = 720
let framerate = 30
let jpegQuality = 85

// MARK: - Detection Types

struct Detection: Codable, Sendable {
  let classId: Int
  let label: String
  let confidence: Float
  let bbox: BoundingBox

  struct BoundingBox: Codable, Sendable {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
  }
}

struct DetectionFrame: Codable, Sendable {
  let timestamp: Double
  let detections: [Detection]
  let frameNumber: Int
}

// MARK: - Response Types

struct StatusResponse: ResponseEncodable {
  let connectedClients: Int
  let cameraActive: Bool
  let modelLoaded: Bool
  let settings: Settings

  struct Settings: Codable, Sendable {
    let width: Int
    let height: Int
    let framerate: Int
    let jpegQuality: Int
    let modelType: String
  }
}

// MARK: - YOLOv8 Manager

actor YoloV8Manager {
  private var pipeline: Pipeline?
  private var frameTask: Task<Void, Never>?
  private var videoClients: [UUID: AsyncStream<[UInt8]>.Continuation] = [:]
  private var detectionClients: [UUID: AsyncStream<DetectionFrame>.Continuation] = [:]
  private let logger = Logger(label: "yolov8")
  private var frameCounter = 0
  private var modelType: String = "none"
  private var inference: YOLOv8Inference?

  init() {
    // Try to load the model from various locations
    let modelPaths = [
      "/app/yolov8n.onnx",
      FileManager.default.currentDirectoryPath + "/yolov8n.onnx",
      ".build/plugins/outputs/yolov8-server/YoloV8Server/DownloadYOLOv8Model/models/yolov8n.onnx"
    ]

    for path in modelPaths {
      if FileManager.default.fileExists(atPath: path) {
        do {
          self.inference = try YOLOv8Inference(modelPath: path)
          logger.info("Loaded YOLOv8 model from: \(path)")
          break
        } catch {
          logger.error("Failed to load model from \(path): \(error)")
        }
      }
    }

    if inference == nil {
      logger.warning("YOLOv8 model not found, will use mock detections")
    }
  }

  func addVideoClient() -> (UUID, AsyncStream<[UInt8]>) {
    let id = UUID()
    let (stream, continuation) = AsyncStream<[UInt8]>.makeStream(
      bufferingPolicy: .bufferingNewest(2)
    )
    videoClients[id] = continuation

    if pipeline == nil {
      startPipeline()
    }

    return (id, stream)
  }

  func addDetectionClient() -> (UUID, AsyncStream<DetectionFrame>) {
    let id = UUID()
    let (stream, continuation) = AsyncStream<DetectionFrame>.makeStream(
      bufferingPolicy: .bufferingNewest(2)
    )
    detectionClients[id] = continuation

    if pipeline == nil {
      startPipeline()
    }

    return (id, stream)
  }

  func removeVideoClient(_ id: UUID) {
    videoClients[id]?.finish()
    videoClients.removeValue(forKey: id)
    checkShutdown()
  }

  func removeDetectionClient(_ id: UUID) {
    detectionClients[id]?.finish()
    detectionClients.removeValue(forKey: id)
    checkShutdown()
  }

  private func checkShutdown() {
    if videoClients.isEmpty && detectionClients.isEmpty {
      stopPipeline()
    }
  }

  private func startPipeline() {
    // NVIDIA DeepStream pipeline with YOLOv8 nvinfer
    let deepstreamPipeline = """
      v4l2src device=/dev/video0 ! \
      video/x-raw,width=\(frameWidth),height=\(frameHeight),framerate=\(framerate)/1 ! \
      nvvidconv ! \
      video/x-raw(memory:NVMM),format=NV12 ! \
      m.sink_0 nvinfer config-file-path=/app/yolov8n.txt ! \
      nvvidconv ! \
      nvdsosd ! \
      tee name=t ! \
      queue ! nvjpegenc quality=\(jpegQuality) ! appsink name=sink max-buffers=2 drop=true \
      t. ! queue ! nvvidconv ! video/x-raw,format=RGBA ! appsink name=meta max-buffers=2 drop=true
      """

    // macOS/Linux CPU pipeline with basic detection placeholder
    // Note: For production, you'd integrate CoreML (macOS) or ONNX Runtime
    let cpuPipeline = """
      avfvideosrc device-index=0 ! \
      videoconvert ! videoscale method=lanczos ! aspectratiocrop aspect-ratio=16/9 ! videorate ! \
      video/x-raw,width=\(frameWidth),height=\(frameHeight),framerate=\(framerate)/1,format=I420 ! \
      tee name=t ! \
      queue ! jpegenc quality=\(jpegQuality) ! appsink name=sink max-buffers=2 drop=true \
      t. ! queue ! videoconvert ! appsink name=meta max-buffers=2 drop=true
      """

    // Linux V4L2 with software JPEG encoding
    let v4l2Pipeline = """
      v4l2src device=/dev/video0 ! \
      videoconvert ! videoscale add-borders=true ! videorate ! \
      video/x-raw,width=\(frameWidth),height=\(frameHeight),framerate=\(framerate)/1,format=I420 ! \
      tee name=t ! \
      queue ! jpegenc quality=\(jpegQuality) ! appsink name=sink max-buffers=2 drop=true \
      t. ! queue ! videoconvert ! appsink name=meta max-buffers=2 drop=true
      """

    func hasFactory(_ name: String) -> Bool {
      do {
        try GStreamer.initialize()
        _ = try Element.make(factory: name)
        return true
      } catch {
        return false
      }
    }

    // Check for YOLOv8 model file
    func hasModelFile() -> Bool {
      FileManager.default.fileExists(atPath: "/app/yolov8n.txt") ||
      FileManager.default.fileExists(atPath: "/app/yolov8n.engine")
    }

    // Prefer DeepStream (NVIDIA), then macOS, then V4L2 software
    let candidates: [(name: String, desc: String, requires: [String], needsModel: Bool)] = [
      ("DeepStream YOLOv8", deepstreamPipeline, ["v4l2src", "nvvidconv", "nvinfer", "nvjpegenc"], true),
      ("macOS (CPU)", cpuPipeline, ["avfvideosrc", "jpegenc"], false),
      ("Linux V4L2 (CPU)", v4l2Pipeline, ["v4l2src", "jpegenc"], false)
    ]

    let available = candidates.filter { candidate in
      let hasElements = candidate.requires.allSatisfy(hasFactory)
      let hasModel = !candidate.needsModel || hasModelFile()
      return hasElements && hasModel
    }
    let pipelines = (available.isEmpty ? candidates : available).map { ($0.name, $0.desc) }

    for (name, desc) in pipelines {
      do {
        let p = try Pipeline(desc)
        let sink = try p.appSink(named: "sink")
        let metaSink = try? p.appSink(named: "meta")
        try p.play()
        pipeline = p
        modelType = name
        logger.info("Using \(name) pipeline")

        frameTask = Task { [weak self] in
          await self?.processPipeline(jpegSink: sink, metaSink: metaSink)
        }

        return
      } catch {
        logger.debug("\(name) pipeline failed: \(error)")
      }
    }

    logger.error("No working GStreamer pipeline found")
  }

  private func processPipeline(jpegSink: AppSink, metaSink: AppSink?) async {
    // If we have a meta sink, process detections in parallel
    if let metaSink = metaSink {
      await withTaskGroup(of: Void.self) { group in
        // Process JPEG frames
        group.addTask { [weak self] in
          do {
            for try await frame in jpegSink.frames() {
              if Task.isCancelled { break }
              let jpegData = try frame.withUnsafeBytes { buffer in
                Array(UnsafeRawBufferPointer(start: buffer.baseAddress, count: buffer.count))
              }
              await self?.broadcastFrame(jpegData)
            }
          } catch {
            await self?.logError("JPEG frame loop error: \(error)")
          }
        }

        // Process detection metadata
        group.addTask { [weak self] in
          do {
            for try await metaBuffer in metaSink.frames() {
              if Task.isCancelled { break }
              // Run YOLOv8 inference on raw video frames
              let detections = await self?.parseDetections(from: metaBuffer) ?? []
              let frameNum = await self?.incrementFrame() ?? 0
              let detectionFrame = DetectionFrame(
                timestamp: Date().timeIntervalSince1970,
                detections: detections,
                frameNumber: frameNum
              )
              await self?.broadcastDetections(detectionFrame)
            }
          } catch {
            await self?.logError("Meta frame loop error: \(error)")
          }
        }
      }
    } else {
      // No metadata sink, just process JPEG frames with mock detections
      do {
        for try await frame in jpegSink.frames() {
          if Task.isCancelled { break }
          let jpegData = try frame.withUnsafeBytes { buffer in
            Array(UnsafeRawBufferPointer(start: buffer.baseAddress, count: buffer.count))
          }
          self.broadcastFrame(jpegData)

          // Generate mock detections for demo purposes
          let mockDetections = generateMockDetections()
          let frameNum = incrementFrame()
          let detectionFrame = DetectionFrame(
            timestamp: Date().timeIntervalSince1970,
            detections: mockDetections,
            frameNumber: frameNum
          )
          self.broadcastDetections(detectionFrame)
        }
      } catch {
        self.logError("Frame loop error: \(error)")
      }
    }
  }

  private func parseDetections(from frame: VideoFrame) -> [Detection] {
    // Use ONNX Runtime inference if available
    if let inference = inference {
      do {
        return try inference.detect(frame: frame)
      } catch {
        logError("Inference error: \(error)")
        return []
      }
    }

    // Fallback to mock detections
    return generateMockDetections()
  }

  private func generateMockDetections() -> [Detection] {
    // Generate 0-3 random detections for demo
    let count = Int.random(in: 0...3)
    let labels = ["person", "car", "dog", "cat", "bicycle", "bottle"]

    return (0..<count).map { i in
      Detection(
        classId: i,
        label: labels.randomElement() ?? "object",
        confidence: Float.random(in: 0.5...0.99),
        bbox: Detection.BoundingBox(
          x: Float.random(in: 0.1...0.6),
          y: Float.random(in: 0.1...0.6),
          width: Float.random(in: 0.1...0.3),
          height: Float.random(in: 0.1...0.3)
        )
      )
    }
  }

  private func incrementFrame() -> Int {
    frameCounter += 1
    return frameCounter
  }

  private func broadcastFrame(_ data: [UInt8]) {
    for (_, continuation) in videoClients {
      continuation.yield(data)
    }
  }

  private func broadcastDetections(_ frame: DetectionFrame) {
    for (_, continuation) in detectionClients {
      continuation.yield(frame)
    }
  }

  private func logError(_ message: String) {
    logger.error("\(message)")
  }

  private func stopPipeline() {
    frameTask?.cancel()
    frameTask = nil
    pipeline?.stop()
    pipeline = nil
    logger.info("YOLOv8 pipeline stopped")
  }

  var status: StatusResponse {
    StatusResponse(
      connectedClients: videoClients.count + detectionClients.count,
      cameraActive: pipeline != nil,
      modelLoaded: inference != nil,
      settings: .init(
        width: frameWidth,
        height: frameHeight,
        framerate: framerate,
        jpegQuality: jpegQuality,
        modelType: inference != nil ? "YOLOv8 (ONNX Runtime)" : modelType
      )
    )
  }
}

// MARK: - WebSocket Handlers

func handleVideoStream(
  inbound: WebSocketInboundStream,
  outbound: WebSocketOutboundWriter,
  logger: Logger,
  yolov8: YoloV8Manager
) async {
  logger.info("Video WebSocket client connected")
  let (clientId, frameStream) = await yolov8.addVideoClient()

  await withTaskGroup(of: Void.self) { group in
    // Send frames to client
    group.addTask {
      for await frameData in frameStream {
        do {
          var buffer = ByteBufferAllocator().buffer(capacity: frameData.count)
          buffer.writeBytes(frameData)
          try await outbound.write(.binary(buffer))
        } catch {
          break
        }
      }
    }

    // Detect client disconnect
    group.addTask {
      do {
        for try await _ in inbound {}
      } catch {
        // Client disconnected
      }
    }

    _ = await group.next()
    group.cancelAll()
  }

  await yolov8.removeVideoClient(clientId)
  logger.info("Video WebSocket client disconnected")
}

func handleDetectionStream(
  inbound: WebSocketInboundStream,
  outbound: WebSocketOutboundWriter,
  logger: Logger,
  yolov8: YoloV8Manager
) async {
  logger.info("Detection WebSocket client connected")
  let (clientId, detectionStream) = await yolov8.addDetectionClient()

  await withTaskGroup(of: Void.self) { group in
    // Send detections to client
    group.addTask {
      let encoder = JSONEncoder()
      for await detectionFrame in detectionStream {
        do {
          let jsonData = try encoder.encode(detectionFrame)
          guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            continue
          }
          try await outbound.write(.text(jsonString))
        } catch {
          break
        }
      }
    }

    // Detect client disconnect
    group.addTask {
      do {
        for try await _ in inbound {}
      } catch {
        // Client disconnected
      }
    }

    _ = await group.next()
    group.cancelAll()
  }

  await yolov8.removeDetectionClient(clientId)
  logger.info("Detection WebSocket client disconnected")
}

// MARK: - Main

@main
struct YoloV8Server {
  static let yolov8 = YoloV8Manager()

  static func main() async throws {
    let hostname = ProcessInfo.processInfo.environment["WENDY_HOSTNAME"] ?? "localhost"
    let port = 3004
    let logger = Logger(label: "yolov8-server")

    logger.info("GStreamer version: \(GStreamer.versionString)")

    // Determine static file paths
    let containerPath = "/app"
    let cwdPath = FileManager.default.currentDirectoryPath
    let staticRoot: String
    if FileManager.default.fileExists(atPath: containerPath + "/index.html") {
      staticRoot = containerPath
    } else {
      staticRoot = cwdPath
    }

    logger.info("Serving static files from: \(staticRoot)")

    let capturedYolov8 = yolov8

    let router = Router()

    // Serve static files with FileMiddleware
    router.middlewares.add(FileMiddleware(staticRoot, searchForIndexHtml: true))
    router.middlewares.add(LogRequestsMiddleware(.info))

    router.get("/status") { _, _ in
      await capturedYolov8.status
    }

    let app = Application(
      router: router,
      server: .http1WebSocketUpgrade { request, _, _ in
        switch request.path {
        case "/stream":
          return .upgrade([:]) { inbound, outbound, _ in
            await handleVideoStream(
              inbound: inbound,
              outbound: outbound,
              logger: logger,
              yolov8: capturedYolov8
            )
          }
        case "/detections":
          return .upgrade([:]) { inbound, outbound, _ in
            await handleDetectionStream(
              inbound: inbound,
              outbound: outbound,
              logger: logger,
              yolov8: capturedYolov8
            )
          }
        default:
          return .dontUpgrade
        }
      },
      configuration: .init(address: .hostname("0.0.0.0", port: port))
    )

    logger.info("Server running on http://\(hostname):\(port)")
    try await app.runService()
  }
}
