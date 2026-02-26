// HTTPServer.swift
// HTTP server for the Swift detector, matching the Flask server from detector.py.
//
// Exposes:
//   GET /metrics               - Prometheus text format metrics
//   GET /health                - JSON health status
//   GET /stream                - MJPEG video stream (multipart/x-mixed-replace)
//   GET /api/vlm_descriptions  - Recent VLM descriptions (placeholder)
//   GET /api/vlm_status        - VLM service status (placeholder)
//
// Port: 9090 (matching Python detector default)

import Hummingbird
import Logging
import NIOCore

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - DetectorState

/// Shared mutable state that the HTTP server reads from the detection pipeline.
///
/// The actor serialises all access to `latestJPEGFrame` and
/// `mjpegClientCount`. New frames are broadcast to every waiting MJPEG handler
/// through per-client `AsyncStream` continuations so each connected client
/// wakes up independently and at its own pace.
actor DetectorState: Sendable {

    // MARK: Stored state

    /// The most recently produced JPEG-encoded frame (bboxes already rendered).
    private(set) var latestJPEGFrame: [UInt8]?

    /// Number of browsers / players currently consuming the MJPEG stream.
    private(set) var mjpegClientCount: Int = 0

    // MARK: Frame notification

    /// Active continuations keyed by client UUID - one entry per connected MJPEG
    /// client. Each continuation receives a copy of the JPEG bytes whenever a
    /// new frame is stored.
    private var frameContinuations: [UUID: AsyncStream<[UInt8]>.Continuation] = [:]

    // MARK: Public interface

    /// Returns true while at least one MJPEG client is connected.
    ///
    /// The detection pipeline can read this to decide whether it is worth
    /// spending CPU time JPEG-encoding frames.
    var shouldExtractFrames: Bool {
        mjpegClientCount > 0
    }

    /// Stores a new JPEG frame and wakes every waiting MJPEG client.
    func setFrame(_ jpeg: [UInt8]) {
        latestJPEGFrame = jpeg
        for continuation in frameContinuations.values {
            continuation.yield(jpeg)
        }
    }

    /// Returns the latest stored frame, or nil if none has arrived yet.
    func getFrame() -> [UInt8]? {
        latestJPEGFrame
    }

    // MARK: MJPEG client lifecycle

    /// Registers a new MJPEG client and returns a stream of JPEG frames.
    ///
    /// - Returns: A tuple of a unique token (used to unregister later) and an
    ///   `AsyncStream` that yields JPEG-encoded bytes whenever `setFrame` is
    ///   called.
    func connectMJPEGClient() -> (id: UUID, frames: AsyncStream<[UInt8]>) {
        mjpegClientCount += 1
        let id = UUID()
        let (stream, continuation) = AsyncStream<[UInt8]>.makeStream()
        frameContinuations[id] = continuation
        return (id, stream)
    }

    /// Unregisters an MJPEG client and finishes its frame stream.
    func disconnectMJPEGClient(id: UUID) {
        guard mjpegClientCount > 0 else { return }
        mjpegClientCount -= 1
        frameContinuations[id]?.finish()
        frameContinuations.removeValue(forKey: id)
    }
}

// MARK: - AllOriginsMiddleware

/// Adds CORS headers to every response regardless of whether an `Origin`
/// request header is present.
///
/// This matches the Python detector's `add_cors_headers` after_request hook
/// which unconditionally sets:
///   Access-Control-Allow-Origin: *
///   Access-Control-Allow-Methods: GET, POST, OPTIONS
///   Access-Control-Allow-Headers: Content-Type
///
/// Named `AllOriginsMiddleware` rather than `CORSMiddleware` to avoid a name
/// collision with Hummingbird's built-in `CORSMiddleware` (which only fires
/// when an `Origin` request header is present).
struct AllOriginsMiddleware<Context: RequestContext>: RouterMiddleware {

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        var response = try await next(request, context)
        response.headers[.accessControlAllowOrigin] = "*"
        response.headers[.accessControlAllowMethods] = "GET, POST, OPTIONS"
        response.headers[.accessControlAllowHeaders] = "Content-Type"
        return response
    }
}

// MARK: - Router builder

/// Builds the Hummingbird router with all detector HTTP routes.
///
/// - Parameters:
///   - state:   The shared actor that holds the current frame and client count.
///   - metrics: The registry whose `render()` output is served at /metrics.
/// - Returns: A configured `Router<BasicRequestContext>` ready to be passed
///   to `Application`.
func buildRouter(
    state: DetectorState,
    metrics: MetricsRegistry
) -> Router<BasicRequestContext> {

    let router = Router(context: BasicRequestContext.self)

    // Apply CORS headers unconditionally to all responses.
    router.middlewares.add(AllOriginsMiddleware())

    // ------------------------------------------------------------------
    // GET /metrics  - Prometheus text exposition format
    // ------------------------------------------------------------------
    router.get("metrics") { _, _ -> Response in
        let body = metrics.render()
        var headers = HTTPFields()
        headers[.contentType] = "text/plain; version=0.0.4; charset=utf-8"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: body))
        )
    }

    // ------------------------------------------------------------------
    // GET /health  - JSON health check
    //
    // Matches Python: {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}
    // ------------------------------------------------------------------
    router.get("health") { _, _ -> Response in
        let timestamp = Date().ISO8601Format()
        let json = "{\"status\":\"healthy\",\"timestamp\":\"\(timestamp)\"}"
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: json))
        )
    }

    // ------------------------------------------------------------------
    // GET /stream  - MJPEG multipart stream
    //
    // Registers the caller as an MJPEG client, then streams JPEG frames
    // as they arrive from the detection pipeline. The `shouldExtractFrames`
    // flag tells the pipeline whether it is worth spending CPU time encoding
    // frames.
    //
    // Frames and 1-second keepalive ticks are merged into a single
    // AsyncStream<[UInt8]?> so the writer loop is a plain `for await`
    // without touching any non-Sendable iterators from multiple tasks.
    // ------------------------------------------------------------------
    router.get("stream") { _, _ -> Response in
        let (clientID, frameStream) = await state.connectMJPEGClient()

        var headers = HTTPFields()
        headers[.contentType] = "multipart/x-mixed-replace; boundary=frame"
        // Ask the client not to buffer; each part should render immediately.
        headers[.cacheControl] = "no-cache, no-store, must-revalidate"

        // Merge the frame stream with a 1-second keepalive clock into a single
        // AsyncStream<[UInt8]?> where Some([UInt8]) is a frame and nil is a tick.
        // AsyncStream.Continuation is Sendable so both tasks may safely share it.
        let (mergedStream, mergedContinuation) = AsyncStream<[UInt8]?>.makeStream()

        // Frame forwarder: relays real JPEG frames and finishes the merged
        // stream when the detection pipeline closes.
        let forwarderTask = Task {
            for await jpeg in frameStream {
                mergedContinuation.yield(jpeg)
            }
            mergedContinuation.finish()
        }

        // Keepalive clock: yields nil every second so the writer can send an
        // empty boundary marker when the pipeline is idle.
        let tickerTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                mergedContinuation.yield(nil)
            }
        }

        // Build a streaming ResponseBody. Hummingbird calls the closure with a
        // ResponseBodyWriter and keeps the TCP connection open until the closure
        // returns or throws.
        let body = ResponseBody { [state] (writer: inout any ResponseBodyWriter) in
            defer {
                // Cancel background tasks and update the actor. The defer block
                // is synchronous, so Task{} is used for the async actor call.
                forwarderTask.cancel()
                tickerTask.cancel()
                Task { await state.disconnectMJPEGClient(id: clientID) }
            }

            // Send the most-recent frame immediately so the browser displays
            // something without waiting for the next detection cycle.
            if let existing = await state.getFrame() {
                try await writer.write(mjpegPart(jpeg: existing))
            }

            // Relay frames and keepalive ticks. The loop exits when the merged
            // stream finishes (pipeline shutdown) or the task is cancelled
            // (client disconnected / server shutting down).
            for await element in mergedStream {
                if Task.isCancelled { break }
                if let jpeg = element {
                    // Real detection frame - encode as a MJPEG part.
                    try await writer.write(mjpegPart(jpeg: jpeg))
                } else {
                    // Keepalive tick - send an empty boundary so browsers and
                    // reverse proxies do not close the idle connection.
                    var buf = ByteBuffer()
                    buf.writeString("--frame\r\n\r\n")
                    try await writer.write(buf)
                }
            }

            // Signal end-of-body (only reached on pipeline shutdown or
            // task cancellation after the loop exits normally).
            try await writer.finish(nil)
        }

        return Response(status: .ok, headers: headers, body: body)
    }

    // ------------------------------------------------------------------
    // GET /api/vlm_descriptions  - placeholder for VLM integration
    //
    // When VLM is integrated this will return recent scene descriptions
    // generated by the language model. Returns an empty list for now.
    // ------------------------------------------------------------------
    router.get("api/vlm_descriptions") { _, _ -> Response in
        let json = "{\"count\":0,\"descriptions\":[]}"
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: json))
        )
    }

    // ------------------------------------------------------------------
    // GET /api/vlm_status  - placeholder for VLM integration
    //
    // When VLM is integrated this will reflect the real availability of the
    // language model service. Returns unavailable for now.
    // ------------------------------------------------------------------
    router.get("api/vlm_status") { _, _ -> Response in
        let json = "{\"available\":false}"
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: json))
        )
    }

    return router
}

// MARK: - MJPEG frame helper

/// Builds a single MJPEG multipart body part from raw JPEG bytes.
///
/// Wire format (matching Python detector.py generate() yield):
///
///     --frame\r\n
///     Content-Type: image/jpeg\r\n
///     Content-Length: <N>\r\n
///     \r\n
///     <JPEG bytes>
///     \r\n
private func mjpegPart(jpeg: [UInt8]) -> ByteBuffer {
    var buf = ByteBuffer()
    buf.writeString("--frame\r\n")
    buf.writeString("Content-Type: image/jpeg\r\n")
    buf.writeString("Content-Length: \(jpeg.count)\r\n")
    buf.writeString("\r\n")
    buf.writeBytes(jpeg)
    buf.writeString("\r\n")
    return buf
}

// MARK: - Server entry point

/// Creates and runs the Hummingbird HTTP server on `0.0.0.0:port`.
///
/// This function suspends until the server exits (e.g. on SIGTERM).
/// When using `ServiceGroup` for graceful shutdown, pass the returned
/// `Application` as a service rather than calling this function directly.
///
/// - Parameters:
///   - state:   Shared actor the MJPEG stream endpoint reads frames from.
///   - metrics: Registry whose text is served at GET /metrics.
///   - port:    TCP port to bind. Defaults to 9090 to match the Python detector.
func startHTTPServer(
    state: DetectorState,
    metrics: MetricsRegistry,
    port: Int = 9090
) async throws {
    var logger = Logger(label: "detector.http")
    logger.logLevel = .info

    let router = buildRouter(state: state, metrics: metrics)

    let app = Application(
        router: router,
        configuration: .init(
            address: .hostname("0.0.0.0", port: port)
        ),
        logger: logger
    )

    logger.info(
        "HTTP server listening",
        metadata: ["address": "0.0.0.0:\(port)"]
    )

    try await app.runService()
}
