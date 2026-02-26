// Detector.swift
// Main entry point for the Swift YOLO11n object detector.
//
// Ties together:
//   - TensorRT engine loading (DetectorEngine)
//   - RTSP frame decoding (RTSPFrameReader via FFmpeg subprocess)
//   - YOLO preprocessing, inference, postprocessing, NMS
//   - IoU-based multi-object tracking (IOUTracker)
//   - Bounding box rendering and JPEG encoding (FrameRenderer)
//   - Hummingbird HTTP server on :9090 (metrics, MJPEG, health)
//   - Prometheus metrics (Metrics.swift)

import ArgumentParser
import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

@main
struct Detector: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        abstract: "YOLO11n object detector — Swift/TensorRT edition"
    )

    @Option(name: .long, help: "Path to serialised TensorRT engine file")
    var enginePath: String = "/app/model_b2_gpu0_fp16.engine"

    @Option(name: .long, help: "Path to ONNX model (fallback if engine missing)")
    var onnxPath: String = "/app/yolo11n.onnx"

    @Option(name: .long, help: "Path to labels.txt (one class name per line)")
    var labelsPath: String = "/app/labels.txt"

    @Option(name: .long, help: "Path to streams.json configuration")
    var streamsPath: String = "/app/streams.json"

    @Option(name: .long, help: "HTTP server port")
    var port: Int = 9090

    func run() async throws {
        var logger = Logger(label: "detector")
        logger.logLevel = .info

        // ---------------------------------------------------------------
        // 1. Load stream configuration
        // ---------------------------------------------------------------

        let streamsConfig = try StreamsConfig.load(from: streamsPath)
        let enabledStreams = streamsConfig.streams.filter(\.enabled)

        guard !enabledStreams.isEmpty else {
            logger.error("No enabled streams found in \(streamsPath)")
            return
        }

        logger.info("Loaded \(enabledStreams.count) stream(s)", metadata: [
            "streams": "\(enabledStreams.map(\.name))",
        ])

        // ---------------------------------------------------------------
        // 2. Initialise the TensorRT detector engine
        // ---------------------------------------------------------------

        logger.info("Loading TensorRT engine...")
        let engine = try await DetectorEngine(
            enginePath: enginePath,
            onnxPath: onnxPath,
            labelsPath: labelsPath
        )
        logger.info("DetectorEngine ready")

        // ---------------------------------------------------------------
        // 3. Shared state for the HTTP server
        // ---------------------------------------------------------------

        let detectorState = DetectorState()

        // ---------------------------------------------------------------
        // 4. Start the HTTP server in the background
        // ---------------------------------------------------------------

        let httpTask = Task {
            try await startHTTPServer(
                state: detectorState,
                metrics: metrics,
                port: port
            )
        }

        logger.info("HTTP server starting on port \(port)")

        // ---------------------------------------------------------------
        // 5. Run one detection loop per enabled stream
        // ---------------------------------------------------------------

        let taskLogger = logger
        await withTaskGroup(of: Void.self) { group in

            for stream in enabledStreams {
                group.addTask {
                    await runStreamDetectionLoop(
                        stream: stream,
                        engine: engine,
                        state: detectorState,
                        logger: taskLogger
                    )
                }
            }

            // Wait for SIGTERM/SIGINT — the task group keeps running until
            // all children complete (which they won't unless the streams
            // are stopped or the process is signalled).
            await group.waitForAll()
        }

        // If we get here the streams have stopped; cancel the HTTP server.
        httpTask.cancel()
    }
}

// MARK: - Per-stream detection loop

/// Reads frames from an RTSP stream, runs YOLO detection + tracking, updates
/// metrics, and pushes rendered frames to the MJPEG server.
private func runStreamDetectionLoop(
    stream: StreamConfig,
    engine: DetectorEngine,
    state: DetectorState,
    logger: Logger
) async {
    var logger = logger
    logger[metadataKey: "stream"] = "\(stream.name)"

    // Pre-create metric handles for this stream.
    let fps = fpsGauge(stream: stream.name)
    let framesProcessed = framesProcessedCounter(stream: stream.name)
    let inferenceLatency = inferenceLatencyHistogram(stream: stream.name)
    let totalLatency = totalLatencyHistogram(stream: stream.name)

    // Tracker state (mutated per frame).
    var tracker = IOUTracker()

    // FPS calculation state.
    var frameCount: UInt64 = 0
    var fpsWindowStart = ContinuousClock.now

    // Frame reader (spawns ffmpeg).
    let reader = FrameReader(stream: stream)

    logger.info("Starting detection loop")

    for await frame in reader.frames() {
        let frameStart = ContinuousClock.now

        do {
            // --- Inference ---
            let inferStart = ContinuousClock.now
            var detections = try await engine.detect(frame: frame)
            let inferEnd = ContinuousClock.now
            inferenceLatency.observe(durationMs(inferEnd - inferStart))

            // --- Tracking ---
            detections = tracker.update(detections: detections)

            // --- Metrics ---
            frameCount += 1
            framesProcessed.inc()

            // Record per-class detection counts.
            var classCounts: [String: Int] = [:]
            for det in detections {
                classCounts[det.label, default: 0] += 1
            }
            for (className, count) in classCounts {
                detectionsTotalCounter(stream: stream.name, class_: className)
                    .inc(by: Double(count))
            }

            // FPS: compute over a sliding 1-second window.
            let elapsed = (ContinuousClock.now - fpsWindowStart)
            let elapsedSeconds = durationSeconds(elapsed)
            if elapsedSeconds >= 1.0 {
                fps.set(Double(frameCount) / elapsedSeconds)
                frameCount = 0
                fpsWindowStart = ContinuousClock.now
            }

            // --- Frame rendering (only if MJPEG clients connected) ---
            if await state.shouldExtractFrames {
                if let jpeg = FrameRenderer.renderFrame(
                    frame.data,
                    width: frame.width,
                    height: frame.height,
                    detections: detections
                ) {
                    await state.setFrame(jpeg)
                }
            }

            // --- Total latency ---
            let frameEnd = ContinuousClock.now
            totalLatency.observe(durationMs(frameEnd - frameStart))

        } catch {
            logger.error("Detection error: \(error)")
        }
    }

    logger.warning("Detection loop ended for stream \(stream.name)")
}

// MARK: - Duration conversion helpers

/// Converts a Duration to milliseconds, accounting for both the seconds
/// and attoseconds components. `.components.attoseconds` only returns the
/// fractional part — without adding the whole-seconds portion the result
/// silently loses any duration >= 1 second.
private func durationMs(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) * 1000.0 + Double(c.attoseconds) / 1_000_000_000_000_000
}

/// Converts a Duration to seconds (Double).
private func durationSeconds(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) + Double(c.attoseconds) / 1_000_000_000_000_000_000
}
