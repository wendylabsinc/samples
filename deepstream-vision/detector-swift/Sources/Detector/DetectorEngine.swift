// DetectorEngine.swift
// High-level YOLO11n detection interface backed by a TensorRT engine.
//
// Engine configuration (mirrors the Python detector):
//   - Pre-built engine: model_b2_gpu0_fp16.engine  (FP16, batch size 2)
//   - ONNX fallback:    yolo11n.onnx
//   - Input shape:      [batch, 3, 640, 640]  Float32
//   - Output shape:     [batch, 84, 8400]     Float32  (YOLO11 transposed)
//
// The actor serialises all engine state access so that concurrent callers
// cannot corrupt the TensorRT execution context. CPU-heavy work (preprocessing,
// postprocessing) runs inside the actor calls; GPU work is launched from the
// context's enqueue methods and waits for completion before returning.

import TensorRT
import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - DetectorError

/// Errors raised during engine lifecycle or inference.
enum DetectorError: Error, Sendable {
    /// Neither enginePath nor onnxPath was provided, or neither file existed.
    case noModelSource(String)
    /// The TensorRT runtime could not deserialise or build an engine.
    case engineLoadFailed(String)
    /// Engine returned an output tensor of unexpected size.
    case unexpectedOutputSize(got: Int, expected: Int)
    /// Inference failed at the TensorRT layer.
    case inferenceFailed(String)
    /// An empty batch was submitted to `detectBatch`.
    case emptyBatch
}

// MARK: - DetectorEngine

/// Actor that owns a TensorRT engine and execution context and exposes a
/// high-level detection API.
///
/// All TensorRT state is confined to this actor. Preprocessing and
/// postprocessing are stateless and cheap to call; the actor's serial executor
/// prevents concurrent use of the execution context, which is not thread-safe.
actor DetectorEngine {

    // MARK: Stored properties

    /// Loaded TensorRT engine (owns the plan memory).
    let engine: Engine

    /// Execution context bound to `engine`.
    ///
    /// `ExecutionContext` is not thread-safe; actor isolation guarantees that
    /// only one call drives it at a time.
    let context: ExecutionContext

    /// Stateless YOLO preprocessor (letterbox resize + CHW conversion).
    let preprocessor: YOLOPreprocessor

    /// Stateless YOLO postprocessor (anchor decode + per-class NMS).
    let postprocessor: YOLOPostprocessor

    /// Side length of the square model input (640 for YOLO11n).
    let modelSize: Int

    // MARK: Private properties

    private let logger: Logger

    /// Maximum batch size the engine was built for.
    private let maxBatchSize: Int

    // MARK: Init

    /// Load or build a TensorRT engine and prepare it for inference.
    ///
    /// - Parameters:
    ///   - enginePath: Path to a serialised `.engine` file. Loaded first when the
    ///     file exists.
    ///   - onnxPath: Path to an ONNX model used to build an engine when
    ///     `enginePath` is absent or points to a non-existent file. The engine is
    ///     built with FP16 precision.
    ///   - labelsPath: Path to a text file with one COCO class name per line
    ///     (80 lines for standard YOLO11n).
    ///
    /// - Throws: `DetectorError` when neither source produces a valid engine, or
    ///   when engine introspection/warmup fails.
    init(enginePath: String?, onnxPath: String?, labelsPath: String) async throws {
        self.logger = Logger(label: "DetectorEngine")
        self.modelSize = 640
        self.preprocessor = YOLOPreprocessor()

        // ------------------------------------------------------------------
        // 1. Load or build the TensorRT engine
        // ------------------------------------------------------------------

        let runtime = TensorRTRuntime()
        let loadedEngine: Engine

        if let ep = enginePath, FileManager.default.fileExists(atPath: ep) {
            // Preferred path: deserialise a pre-built plan (fast startup).
            logger.info("Loading pre-built TensorRT engine", metadata: [
                "path": "\(ep)",
            ])
            do {
                loadedEngine = try runtime.deserializeEngine(from: ep)
            } catch {
                throw DetectorError.engineLoadFailed(
                    "Failed to deserialise engine at \(ep): \(error)"
                )
            }
        } else if let op = onnxPath {
            // Fallback: build from ONNX with FP16 precision (slow, ~minutes).
            logger.warning(
                "Engine file not found; building from ONNX (this may take several minutes)",
                metadata: ["onnxPath": "\(op)"]
            )
            let onnxURL = URL(fileURLWithPath: op)
            // TODO: Verify exact EngineBuildOptions initialiser against the
            //       installed tensorrt-swift version. The type and field names
            //       here follow the documented API surface.
            let options = EngineBuildOptions(precision: [.fp16])
            do {
                loadedEngine = try await runtime.buildEngine(onnxURL: onnxURL, options: options)
            } catch {
                throw DetectorError.engineLoadFailed(
                    "Failed to build engine from ONNX at \(op): \(error)"
                )
            }
        } else {
            throw DetectorError.noModelSource(
                "Provide enginePath (preferred) or onnxPath to load YOLO11n."
            )
        }

        self.engine = loadedEngine

        // ------------------------------------------------------------------
        // 2. Create an execution context
        // ------------------------------------------------------------------

        // TODO: Verify the exact method name against the installed
        //       tensorrt-swift version; it may be makeExecutionContext() or
        //       createExecutionContext().
        self.context = try engine.makeExecutionContext()

        // ------------------------------------------------------------------
        // 3. Determine max batch size from the engine
        // ------------------------------------------------------------------

        // TODO: Replace with the actual API to query the engine's max batch
        //       size. tensorrt-swift may expose this via engine.maxBatchSize,
        //       engine.profile, or a binding descriptor. Default to 2 to match
        //       the pre-built model_b2_gpu0_fp16.engine.
        self.maxBatchSize = 2

        // ------------------------------------------------------------------
        // 4. Build the postprocessor (loads labels from disk)
        // ------------------------------------------------------------------

        self.postprocessor = YOLOPostprocessor(labelsPath: labelsPath)

        logger.info("DetectorEngine ready", metadata: [
            "modelSize": "\(self.modelSize)",
            "maxBatchSize": "\(self.maxBatchSize)",
            "labels": "\(self.postprocessor.labels.count)",
        ])

        // ------------------------------------------------------------------
        // 5. Warm-up: a few dummy inferences to prime GPU kernels
        // ------------------------------------------------------------------

        try await warmup()
    }

    // MARK: - Public API

    /// Run YOLO11n inference on a single video frame.
    ///
    /// The call is fully serialised through the actor's executor, so concurrent
    /// calls will queue naturally.
    ///
    /// - Parameter frame: An RGB24 frame decoded from an RTSP stream.
    /// - Returns: Detections in original-image coordinate space (normalised 0–1).
    /// - Throws: `DetectorError` on inference failure.
    func detect(frame: Frame) async throws -> [Detection] {
        // Preprocess: letterbox → normalise → CHW float tensor.
        let (inputData, letterbox) = frame.data.withUnsafeBufferPointer { ptr in
            YOLOPreprocessor.preprocess(ptr, width: frame.width, height: frame.height)
        }

        // Output layout: [1, 84, 8400] for a single-image batch.
        let numRows    = 84       // 4 bbox + 80 classes
        let numAnchors = 8400
        let outputCount = numRows * numAnchors
        var outputBuffer = [Float](repeating: 0.0, count: outputCount)

        // Run inference on the GPU via the execution context.
        //
        // TODO: Verify the exact enqueueF32 / enqueue signature for the
        //       installed tensorrt-swift version. The expected call pattern is:
        //
        //   try context.enqueueF32(
        //       inputName:  "images",
        //       input:      inputPtr,
        //       outputName: "output0",
        //       output:     outputPtr
        //   )
        //
        // If the library requires TensorDescriptor / TensorValue wrappers,
        // construct them here and pass them instead of raw buffer pointers.
        let inferenceSucceeded = try inputData.withUnsafeBufferPointer { inputPtr in
            try outputBuffer.withUnsafeMutableBufferPointer { outputPtr in
                try context.enqueueF32(
                    inputName:  "images",
                    input:      inputPtr,
                    outputName: "output0",
                    output:     outputPtr
                )
            }
        }

        if !inferenceSucceeded {
            throw DetectorError.inferenceFailed("enqueueF32 returned false for single frame")
        }

        // Postprocess: decode anchors and apply per-class NMS.
        var detections = outputBuffer.withUnsafeBufferPointer { ptr in
            postprocessor.process(output: ptr, batchSize: 1)
        }

        // Remap from 640x640 model space back to original image coordinates.
        YOLOPreprocessor.remapBoxes(&detections, letterbox: letterbox)

        return detections
    }

    /// Run YOLO11n inference on a batch of frames in a single GPU call.
    ///
    /// This is more efficient than calling `detect(frame:)` repeatedly when
    /// multiple camera streams need to be processed in lock-step. The batch
    /// size matches the engine's compile-time maximum (2 for the default
    /// `model_b2_gpu0_fp16.engine`).
    ///
    /// - Parameter frames: 1 to `maxBatchSize` frames. When more frames are
    ///   provided than `maxBatchSize`, the excess frames are ignored.
    ///   TODO: Add chunked execution for stream counts larger than maxBatchSize.
    /// - Returns: One `[Detection]` array per input frame, in the same order.
    /// - Throws: `DetectorError.emptyBatch` when `frames` is empty, or
    ///   `DetectorError.inferenceFailed` on GPU errors.
    func detectBatch(frames: [Frame]) async throws -> [[Detection]] {
        guard !frames.isEmpty else {
            throw DetectorError.emptyBatch
        }

        // Cap at the engine's maximum batch size.
        // TODO: For stream counts > maxBatchSize, split into chunks and run
        //       multiple inference calls, then concatenate the results.
        let batch = Array(frames.prefix(maxBatchSize))
        let batchSize = batch.count

        // ------------------------------------------------------------------
        // 1. Preprocess all frames and concatenate into a single batch tensor.
        // ------------------------------------------------------------------

        // Each preprocessed image is a flat CHW float array of size
        // 3 * modelSize * modelSize. The batch tensor layout is
        // [batchSize, 3, 640, 640] in row-major (C-contiguous) order.
        let imagePlaneSize = 3 * modelSize * modelSize
        var batchInput  = [Float](repeating: 0.0, count: batchSize * imagePlaneSize)
        var letterboxes = [LetterboxInfo]()
        letterboxes.reserveCapacity(batchSize)

        for (i, frame) in batch.enumerated() {
            let (imageData, lb) = frame.data.withUnsafeBufferPointer { ptr in
                YOLOPreprocessor.preprocess(ptr, width: frame.width, height: frame.height)
            }
            letterboxes.append(lb)

            // Write this image's CHW data into its contiguous slot in the
            // batch tensor.
            let dstOffset = i * imagePlaneSize
            batchInput.withUnsafeMutableBufferPointer { dst in
                imageData.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: dstOffset)
                        .update(from: src.baseAddress!, count: imagePlaneSize)
                }
            }
        }

        // ------------------------------------------------------------------
        // 2. Single batched inference call.
        // ------------------------------------------------------------------

        // Output layout: [batchSize, 84, 8400].
        let numRows          = 84
        let numAnchors       = 8400
        let elementsPerImage = numRows * numAnchors
        var batchOutput      = [Float](repeating: 0.0, count: batchSize * elementsPerImage)

        // TODO: Verify that context.enqueueF32 accepts a batched input when the
        //       engine was built with batchSize > 1. Some tensorrt-swift
        //       versions require a setInputShape() call before enqueue to set
        //       the dynamic batch dimension at runtime.
        let inferenceSucceeded = try batchInput.withUnsafeBufferPointer { inputPtr in
            try batchOutput.withUnsafeMutableBufferPointer { outputPtr in
                try context.enqueueF32(
                    inputName:  "images",
                    input:      inputPtr,
                    outputName: "output0",
                    output:     outputPtr
                )
            }
        }

        if !inferenceSucceeded {
            throw DetectorError.inferenceFailed(
                "enqueueF32 returned false for batch of \(batchSize)"
            )
        }

        // ------------------------------------------------------------------
        // 3. Split output tensor and postprocess per frame.
        // ------------------------------------------------------------------

        var results = [[Detection]]()
        results.reserveCapacity(batchSize)

        for i in 0 ..< batchSize {
            // Slice the flat output buffer to this frame's contiguous region.
            let offset      = i * elementsPerImage
            var detections  = batchOutput.withUnsafeBufferPointer { fullBuf -> [Detection] in
                let slice = UnsafeBufferPointer(
                    start: fullBuf.baseAddress!.advanced(by: offset),
                    count: elementsPerImage
                )
                return postprocessor.process(output: slice, batchSize: 1)
            }

            // Remap from 640x640 model space back to original image coordinates.
            YOLOPreprocessor.remapBoxes(&detections, letterbox: letterboxes[i])
            results.append(detections)
        }

        return results
    }

    // MARK: - Private helpers

    /// Run a small number of dummy inferences to amortise GPU kernel
    /// compilation and CUDA stream creation before live traffic arrives.
    ///
    /// Uses a gray-filled frame (pixel value 114, the standard YOLO padding
    /// colour) at the model's native 640x640 resolution.
    private func warmup() async throws {
        logger.info("Warming up TensorRT engine (3 iterations)")

        let dummyData  = [UInt8](repeating: 114, count: modelSize * modelSize * 3)
        let dummyFrame = Frame(data: dummyData, width: modelSize, height: modelSize)

        let warmupIterations = 3
        for iteration in 1 ... warmupIterations {
            _ = try await detect(frame: dummyFrame)
            logger.debug("Warmup iteration \(iteration)/\(warmupIterations) complete")
        }

        logger.info("Warmup complete")
    }
}
