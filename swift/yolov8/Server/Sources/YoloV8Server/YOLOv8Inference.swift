import Foundation
import CONNXRuntime
import GStreamer
import Synchronization

/// COCO class labels for YOLOv8
let COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

/// YOLOv8 inference engine using ONNX Runtime
final class YOLOv8Inference: @unchecked Sendable {
    private let api: UnsafePointer<OrtApi>
    private let env: OpaquePointer
    private let session: OpaquePointer
    private let memoryInfo: OpaquePointer
    private let lock = Mutex<Void>(())

    private let inputWidth: Int = 640
    private let inputHeight: Int = 640
    private let confThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45

    init(modelPath: String) throws {
        // Get ONNX Runtime API
        guard let apiBase = OrtGetApiBase() else {
            throw YOLOv8Error.initializationFailed("Failed to get ONNX Runtime API base")
        }

        guard let apiPtr = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw YOLOv8Error.initializationFailed("Failed to get ONNX Runtime API")
        }

        // Use local variable until all properties are initialized
        let api = apiPtr

        // Create environment
        var envPtr: OpaquePointer?
        let envStatus = api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YOLOv8", &envPtr)
        guard envStatus == nil, let envValue = envPtr else {
            throw YOLOv8Error.initializationFailed("Failed to create ONNX Runtime environment")
        }

        // Create session options
        var sessionOptionsPtr: OpaquePointer?
        let optStatus = api.pointee.CreateSessionOptions(&sessionOptionsPtr)
        guard optStatus == nil, let sessionOptions = sessionOptionsPtr else {
            throw YOLOv8Error.initializationFailed("Failed to create ONNX Runtime environment")
        }
        defer {
            api.pointee.ReleaseSessionOptions(sessionOptions)
        }

        // Set graph optimization level
        let _ = api.pointee.SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_ALL)

        // Create session
        var sessionPtr: OpaquePointer?
        let sessionStatus = modelPath.withCString { pathCStr in
            api.pointee.CreateSession(envValue, pathCStr, sessionOptions, &sessionPtr)
        }
        guard sessionStatus == nil, let sessionValue = sessionPtr else {
            throw YOLOv8Error.initializationFailed("Failed to create ONNX Runtime session")
        }

        // Create memory info
        var memInfoPtr: OpaquePointer?
        let memStatus = api.pointee.CreateCpuMemoryInfo(
            OrtArenaAllocator,
            OrtMemTypeDefault,
            &memInfoPtr
        )
        guard memStatus == nil, let memInfoValue = memInfoPtr else {
            throw YOLOv8Error.initializationFailed("Failed to create memory info")
        }

        // Initialize all properties
        self.api = apiPtr
        self.env = envValue
        self.session = sessionValue
        self.memoryInfo = memInfoValue

        print("✅ YOLOv8 inference engine initialized with ONNX Runtime")
    }

    deinit {
        api.pointee.ReleaseSession(session)
        api.pointee.ReleaseMemoryInfo(memoryInfo)
        api.pointee.ReleaseEnv(env)
    }

    /// Run YOLOv8 inference on a video frame
    func detect(frame: VideoFrame) throws -> [Detection] {
        try lock.withLock { _ in
            try _detect(frame: frame)
        }
    }

    private func _detect(frame: VideoFrame) throws -> [Detection] {
        // Extract frame data
        let data = try frame.withUnsafeBytes { rawSpan in
            Data(bytes: rawSpan.baseAddress!, count: rawSpan.count)
        }

        // Get frame dimensions
        let width = frame.width
        let height = frame.height
        let format = frame.format

        // Preprocess image
        let inputTensor = try preprocess(
            data: data,
            width: width,
            height: height,
            format: format
        )

        // Run inference
        let output = try runInference(input: inputTensor)

        // Postprocess results
        let detections = try postprocess(output: output)

        return detections
    }

    private func preprocess(data: Data, width: Int, height: Int, format: PixelFormat) throws -> [Float] {
        var inputData = [Float](repeating: 0, count: 3 * inputHeight * inputWidth)

        // Simple bilinear resize and normalization
        let scaleX = Float(width) / Float(inputWidth)
        let scaleY = Float(height) / Float(inputHeight)

        // Determine bytes per pixel and channel order based on format
        let (bytesPerPixel, getRGB): (Int, (Data, Int) -> (UInt8, UInt8, UInt8)) = {
            switch format {
            case .rgba:
                return (4, { data, idx in (data[idx], data[idx + 1], data[idx + 2]) })
            case .bgra:
                return (4, { data, idx in (data[idx + 2], data[idx + 1], data[idx]) })
            case .nv12, .i420, .gray8:
                // YUV/Grayscale formats would need proper color space conversion
                // For now, just attempt basic extraction
                return (1, { data, idx in
                    let y = data[idx]
                    return (y, y, y)  // Treat as grayscale
                })
            case .unknown:
                // Fallback: assume RGBA
                return (4, { data, idx in (data[idx], data[idx + 1], data[idx + 2]) })
            }
        }()

        for y in 0..<inputHeight {
            for x in 0..<inputWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIdx = (srcY * width + srcX) * bytesPerPixel

                guard srcIdx + bytesPerPixel - 1 < data.count else { continue }

                // Get RGB values and normalize to [0, 1]
                let (r, g, b) = getRGB(data, srcIdx)

                // CHW layout: [C, H, W]
                inputData[0 * inputHeight * inputWidth + y * inputWidth + x] = Float(r) / 255.0
                inputData[1 * inputHeight * inputWidth + y * inputWidth + x] = Float(g) / 255.0
                inputData[2 * inputHeight * inputWidth + y * inputWidth + x] = Float(b) / 255.0
            }
        }

        return inputData
    }

    private func runInference(input: [Float]) throws -> [Float] {

        // Input tensor shape: [1, 3, 640, 640]
        var inputShape: [Int64] = [1, 3, Int64(inputHeight), Int64(inputWidth)]
        let inputCount = input.count

        // Create input tensor
        var inputTensor: OpaquePointer?
        let createStatus = input.withUnsafeBytes { bufferPtr in
            api.pointee.CreateTensorWithDataAsOrtValue(
                memoryInfo,
                UnsafeMutableRawPointer(mutating: bufferPtr.baseAddress!),
                inputCount * MemoryLayout<Float>.stride,
                &inputShape,
                4,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &inputTensor
            )
        }

        guard createStatus == nil, let tensor = inputTensor else {
            throw YOLOv8Error.processingFailed("Failed to create input tensor")
        }
        defer {
            api.pointee.ReleaseValue(tensor)
        }

        // Run inference
        let inputName = "images"
        let outputName = "output0"

        let inputNamePtr = inputName.withCString { strdup($0) }!
        let outputNamePtr = outputName.withCString { strdup($0) }!

        var inputNames: [UnsafePointer<CChar>?] = [UnsafePointer(inputNamePtr)]
        var outputNames: [UnsafePointer<CChar>?] = [UnsafePointer(outputNamePtr)]

        defer {
            free(inputNamePtr)
            free(outputNamePtr)
        }

        var inputTensors: [OpaquePointer?] = [tensor]
        var outputTensor: OpaquePointer?

        let runStatus = api.pointee.Run(
            session,
            nil,
            &inputNames,
            &inputTensors,
            1,
            &outputNames,
            1,
            &outputTensor
        )

        guard runStatus == nil, let output = outputTensor else {
            throw YOLOv8Error.processingFailed("Failed to run inference")
        }
        defer {
            api.pointee.ReleaseValue(output)
        }

        // Get output tensor data
        var tensorData: UnsafeMutableRawPointer?
        let dataStatus = api.pointee.GetTensorMutableData(output, &tensorData)
        guard dataStatus == nil, let data = tensorData else {
            throw YOLOv8Error.processingFailed("Failed to get output tensor data")
        }

        // Get output shape
        var outputShapeInfo: OpaquePointer?
        let shapeStatus = api.pointee.GetTensorTypeAndShape(output, &outputShapeInfo)
        guard shapeStatus == nil, let shapeInfo = outputShapeInfo else {
            throw YOLOv8Error.processingFailed("Failed to get output shape")
        }
        defer {
            api.pointee.ReleaseTensorTypeAndShapeInfo(shapeInfo)
        }

        var dimCount: Int = 0
        let _ = api.pointee.GetDimensionsCount(shapeInfo, &dimCount)

        var elementCount: Int = 0
        let _ = api.pointee.GetTensorShapeElementCount(shapeInfo, &elementCount)

        // Copy output data
        let floatData = data.bindMemory(to: Float.self, capacity: elementCount)
        var outputArray = [Float](repeating: 0, count: elementCount)
        for i in 0..<elementCount {
            outputArray[i] = floatData[i]
        }

        return outputArray
    }

    private func postprocess(output: [Float]) throws -> [Detection] {
        // YOLOv8 output shape: [1, 84, 8400]
        // First 4 values are box coordinates (cx, cy, w, h)
        // Next 80 values are class scores

        let numPredictions = 8400
        let numClasses = 80
        let boxSize = 4

        var boxes: [[Float]] = []
        var scores: [Float] = []
        var classIds: [Int] = []

        // Parse predictions
        for i in 0..<numPredictions {
            var maxScore: Float = 0
            var maxClassId = 0

            // Find max class score
            for c in 0..<numClasses {
                let score = output[(boxSize + c) * numPredictions + i]
                if score > maxScore {
                    maxScore = score
                    maxClassId = c
                }
            }

            // Filter by confidence threshold
            if maxScore > confThreshold {
                let cx = output[0 * numPredictions + i]
                let cy = output[1 * numPredictions + i]
                let w = output[2 * numPredictions + i]
                let h = output[3 * numPredictions + i]

                // Convert to normalized coordinates
                let x = (cx - w / 2) / Float(inputWidth)
                let y = (cy - h / 2) / Float(inputHeight)
                let normW = w / Float(inputWidth)
                let normH = h / Float(inputHeight)

                boxes.append([x, y, normW, normH])
                scores.append(maxScore)
                classIds.append(maxClassId)
            }
        }

        // Apply NMS
        let keepIndices = nms(boxes: boxes, scores: scores, iouThreshold: iouThreshold)

        // Build final detections
        var detections: [Detection] = []
        for i in keepIndices {
            let box = boxes[i]
            let classId = classIds[i]
            let label = classId < COCO_CLASSES.count ? COCO_CLASSES[classId] : "class_\(classId)"

            detections.append(Detection(
                classId: classId,
                label: label,
                confidence: scores[i],
                bbox: Detection.BoundingBox(
                    x: box[0],
                    y: box[1],
                    width: box[2],
                    height: box[3]
                )
            ))
        }

        return detections
    }

    private func nms(boxes: [[Float]], scores: [Float], iouThreshold: Float) -> [Int] {
        guard !boxes.isEmpty else { return [] }

        // Sort by score descending
        let indices = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }

        var keep: [Int] = []
        var remaining = indices

        while !remaining.isEmpty {
            let current = remaining.removeFirst()
            keep.append(current)

            let currentBox = boxes[current]

            remaining = remaining.filter { idx in
                let iou = calculateIoU(box1: currentBox, box2: boxes[idx])
                return iou < iouThreshold
            }
        }

        return keep
    }

    private func calculateIoU(box1: [Float], box2: [Float]) -> Float {
        let x1 = box1[0], y1 = box1[1], w1 = box1[2], h1 = box1[3]
        let x2 = box2[0], y2 = box2[1], w2 = box2[2], h2 = box2[3]

        // Convert to corner coordinates
        let x1Min = x1, y1Min = y1
        let x1Max = x1 + w1, y1Max = y1 + h1
        let x2Min = x2, y2Min = y2
        let x2Max = x2 + w2, y2Max = y2 + h2

        // Intersection
        let interXMin = max(x1Min, x2Min)
        let interYMin = max(y1Min, y2Min)
        let interXMax = min(x1Max, x2Max)
        let interYMax = min(y1Max, y2Max)

        let interArea = max(0, interXMax - interXMin) * max(0, interYMax - interYMin)

        // Union
        let box1Area = w1 * h1
        let box2Area = w2 * h2
        let unionArea = box1Area + box2Area - interArea

        return unionArea > 0 ? interArea / unionArea : 0
    }
}

enum YOLOv8Error: Error {
    case initializationFailed(String)
    case processingFailed(String)
}
