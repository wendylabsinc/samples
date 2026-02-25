// YOLOPostprocessor.swift
// Parses YOLO11n output tensors and applies per-class Non-Maximum Suppression.
//
// YOLO11n output tensor shape: [1, 84, 8400]
//   - Axis 1: 4 bbox coordinates (cx, cy, w, h) followed by 80 class scores
//   - Axis 2: 8400 anchor predictions
//
// This transposed layout differs from YOLOv5 ([batch, anchors, 85]).
// For anchor `a`, element `r` lives at flat index: r * numAnchors + a.
//
// Matching nvinfer_config.txt settings:
//   num-detected-classes = 80  (COCO)
//   nms-iou-threshold    = 0.45
//   pre-cluster-threshold = 0.4
//   topk                 = 300
//   cluster-mode         = 2   (NMS)

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - Detection

/// A single object detection in model coordinate space (0–640).
struct Detection: Sendable {
    /// Top-left corner x coordinate (pixels, 0–640).
    var x: Float
    /// Top-left corner y coordinate (pixels, 0–640).
    var y: Float
    /// Bounding box width (pixels).
    var width: Float
    /// Bounding box height (pixels).
    var height: Float
    /// COCO class index (0-based).
    var classId: Int
    /// Highest class score after sigmoid (0–1).
    var confidence: Float
    /// Human-readable class name from labels.txt.
    var label: String
    /// Optional tracker-assigned identity; populated downstream.
    var trackId: Int?
}

// MARK: - YOLOPostprocessor

/// Parses raw YOLO11n output and runs per-class NMS.
///
/// Designed for repeated calls on the hot path (20+ FPS). All allocations are
/// bounded by `topK` and `numAnchors` and are reused across calls where
/// possible through preallocated temporary arrays.
struct YOLOPostprocessor: Sendable {
    let confidenceThreshold: Float
    let nmsThreshold: Float
    let topK: Int
    let labels: [String]

    // MARK: Initialisation

    /// Creates a postprocessor by loading class labels from a text file.
    ///
    /// - Parameter labelsPath: Path to a labels.txt with one class name per line
    ///   (80 lines for COCO). Empty lines are skipped.
    init(
        labelsPath: String,
        confidenceThreshold: Float = 0.4,
        nmsThreshold: Float = 0.45,
        topK: Int = 300
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.nmsThreshold = nmsThreshold
        self.topK = topK

        // Load labels, tolerating a trailing newline (labels.txt has 81 lines
        // where line 81 is empty).
        let raw = (try? String(contentsOfFile: labelsPath, encoding: .utf8)) ?? ""
        self.labels = raw.split(separator: "\n").map(String.init)
    }

    // MARK: Processing

    /// Parses the raw YOLO11n output buffer and returns NMS-filtered detections.
    ///
    /// - Parameters:
    ///   - output: Pointer to the flat Float tensor. Expected layout:
    ///     `[batchSize, numClasses + 4, numAnchors]` in row-major order.
    ///     Only batch index 0 is processed.
    ///   - batchSize: Number of images in the batch (default 1).
    ///   - numClasses: Number of object classes (default 80 for COCO).
    ///   - numAnchors: Number of anchor predictions (default 8400).
    /// - Returns: Detections after NMS, sorted by confidence descending,
    ///   capped at `topK`.
    func process(
        output: UnsafeBufferPointer<Float>,
        batchSize: Int = 1,
        numClasses: Int = 80,
        numAnchors: Int = 8400
    ) -> [Detection] {
        let numRows = 4 + numClasses          // 84 for COCO
        let elementsPerBatch = numRows * numAnchors

        guard output.count >= elementsPerBatch else { return [] }

        // Work on batch index 0. For multi-batch callers, run once per image.
        let batchBase = output.baseAddress!   // UnsafePointer<Float>

        // Inline helpers for the transposed layout.
        // Element (row, anchor) lives at: row * numAnchors + anchor
        @inline(__always)
        func value(_ row: Int, _ anchor: Int) -> Float {
            batchBase[row &* numAnchors &+ anchor]
        }

        // ------------------------------------------------------------------
        // Phase 1: Candidate filtering
        //
        // Walk all 8400 anchors. For each, find the maximum class score.
        // If it clears the confidence threshold, record a candidate.
        // ------------------------------------------------------------------

        // Pre-allocate to avoid repeated heap growth. We expect far fewer than
        // numAnchors candidates in practice, but reserve conservatively.
        var candidates: [Detection] = []
        candidates.reserveCapacity(512)

        for anchor in 0 ..< numAnchors {
            // bbox (cx, cy, w, h) – transposed rows 0–3
            let cx = value(0, anchor)
            let cy = value(1, anchor)
            let w  = value(2, anchor)
            let h  = value(3, anchor)

            // Find argmax over the 80 class scores (rows 4–83).
            // Inline the search to avoid allocating a slice.
            var maxScore: Float = -Float.infinity
            var maxClass: Int  = 0

            for cls in 0 ..< numClasses {
                let score = value(4 &+ cls, anchor)
                if score > maxScore {
                    maxScore = score
                    maxClass = cls
                }
            }

            guard maxScore >= confidenceThreshold else { continue }

            // Convert center format → top-left format.
            let x = cx - w * 0.5
            let y = cy - h * 0.5

            let label = maxClass < labels.count ? labels[maxClass] : "\(maxClass)"

            candidates.append(Detection(
                x: x,
                y: y,
                width: w,
                height: h,
                classId: maxClass,
                confidence: maxScore,
                label: label
            ))
        }

        guard !candidates.isEmpty else { return [] }

        // ------------------------------------------------------------------
        // Phase 2: Per-class NMS
        //
        // Group candidates by classId and run greedy NMS within each group.
        // This matches DeepStream cluster-mode=2 behaviour.
        // ------------------------------------------------------------------

        // Find the class-id range present in candidates (avoids full 0..79
        // iteration when only a few classes fire).
        var minClass = candidates[0].classId
        var maxClass = candidates[0].classId
        for c in candidates {
            if c.classId < minClass { minClass = c.classId }
            if c.classId > maxClass { maxClass = c.classId }
        }

        var results: [Detection] = []
        results.reserveCapacity(min(topK, candidates.count))

        // Scratch buffer reused across class iterations.
        var classGroup: [Detection] = []

        for cls in minClass ... maxClass {
            classGroup.removeAll(keepingCapacity: true)
            for c in candidates where c.classId == cls {
                classGroup.append(c)
            }
            guard !classGroup.isEmpty else { continue }

            // Sort descending by confidence (in-place, avoids extra allocation).
            classGroup.sort { $0.confidence > $1.confidence }

            // Greedy NMS: keep a box, suppress anything that overlaps it.
            var kept = [Bool](repeating: true, count: classGroup.count)

            for i in 0 ..< classGroup.count {
                guard kept[i] else { continue }
                results.append(classGroup[i])
                if results.count >= topK { break }

                for j in (i + 1) ..< classGroup.count {
                    guard kept[j] else { continue }
                    if computeIoU(classGroup[i], classGroup[j]) > nmsThreshold {
                        kept[j] = false
                    }
                }
            }

            if results.count >= topK { break }
        }

        // ------------------------------------------------------------------
        // Phase 3: Global topK cap and final sort
        // ------------------------------------------------------------------
        if results.count > topK {
            results = Array(results.prefix(topK))
        }

        results.sort { $0.confidence > $1.confidence }
        return results
    }
}

// MARK: - IoU helper

/// Computes Intersection over Union for two detections in (x, y, w, h) format,
/// where x/y are the top-left corner.
///
/// Both boxes are assumed to be axis-aligned rectangles in model pixel space.
@inline(__always)
private func computeIoU(_ a: Detection, _ b: Detection) -> Float {
    // Convert to (x1, y1, x2, y2) for easier intersection calculation.
    let ax1 = a.x
    let ay1 = a.y
    let ax2 = a.x + a.width
    let ay2 = a.y + a.height

    let bx1 = b.x
    let by1 = b.y
    let bx2 = b.x + b.width
    let by2 = b.y + b.height

    // Intersection rectangle.
    let ix1 = max(ax1, bx1)
    let iy1 = max(ay1, by1)
    let ix2 = min(ax2, bx2)
    let iy2 = min(ay2, by2)

    let iw = max(0, ix2 - ix1)
    let ih = max(0, iy2 - iy1)

    let intersection = iw * ih
    guard intersection > 0 else { return 0 }

    let areaA = a.width * a.height
    let areaB = b.width * b.height
    let union = areaA + areaB - intersection

    guard union > 0 else { return 0 }
    return intersection / union
}
