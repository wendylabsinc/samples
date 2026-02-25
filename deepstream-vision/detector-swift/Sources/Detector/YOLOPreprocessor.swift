// YOLOPreprocessor.swift
// Handles YOLO11n image preprocessing for TensorRT inference.
//
// Matches the nvinfer configuration:
//   infer-dims=3;640;640          → CHW, 640x640 target
//   maintain-aspect-ratio=1       → letterbox resize
//   symmetric-padding=1           → center padding
//   net-scale-factor=0.003921...  → pixel / 255.0
//   model-color-format=0          → RGB input expected

/// Stores the letterbox parameters needed to map model-space coordinates
/// back to the original image coordinate system.
struct LetterboxInfo: Sendable {
    /// Uniform scale applied to the original image before padding.
    /// Equals min(targetSize / originalWidth, targetSize / originalHeight).
    let scale: Float

    /// Horizontal pixel offset of the image content within the padded canvas.
    let xOffset: Int

    /// Vertical pixel offset of the image content within the padded canvas.
    let yOffset: Int

    /// Width of the original source image in pixels.
    let originalWidth: Int

    /// Height of the original source image in pixels.
    let originalHeight: Int
}

/// Stateless utility that preprocesses RGB frames for YOLO11n TensorRT inference
/// and remaps output detections back to original image coordinates.
///
/// The preprocessing pipeline:
///   1. Letterbox resize with bilinear interpolation (maintain aspect ratio, center pad).
///   2. Gray padding (value 114) — standard YOLO padding color.
///   3. Normalize: pixel / 255.0.
///   4. HWC → CHW transpose.
///
/// Output layout: [Float] with 3 * targetSize * targetSize elements
/// representing a single image in planar CHW order (R plane, G plane, B plane).
struct YOLOPreprocessor: Sendable {

    // MARK: - Preprocessing

    /// Preprocess a packed RGB (HWC, 3 bytes per pixel) buffer for YOLO inference.
    ///
    /// - Parameters:
    ///   - rgbPixels: Pointer to the raw RGB byte data (width * height * 3 bytes).
    ///   - width: Width of the source image in pixels.
    ///   - height: Height of the source image in pixels.
    ///   - targetSize: Square side length of the model input (default: 640).
    /// - Returns: A tuple of the normalized CHW float array and the letterbox
    ///   parameters required for coordinate remapping.
    static func preprocess(
        _ rgbPixels: UnsafeBufferPointer<UInt8>,
        width: Int,
        height: Int,
        targetSize: Int = 640
    ) -> (data: [Float], letterbox: LetterboxInfo) {

        // --- Letterbox geometry ---
        let scaleX = Float(targetSize) / Float(width)
        let scaleY = Float(targetSize) / Float(height)
        let scale = min(scaleX, scaleY)

        let scaledWidth = Int(Float(width) * scale)
        let scaledHeight = Int(Float(height) * scale)

        // Center the scaled image on the padded canvas.
        let xOffset = (targetSize - scaledWidth) / 2
        let yOffset = (targetSize - scaledHeight) / 2

        let letterbox = LetterboxInfo(
            scale: scale,
            xOffset: xOffset,
            yOffset: yOffset,
            originalWidth: width,
            originalHeight: height
        )

        // Allocate the output buffer in HWC layout first, then transpose.
        // Using a flat CHW buffer directly and writing via channel stride is
        // equivalent and avoids an extra allocation.
        //
        // Layout: [channel][row][col]
        //   channel 0 (R): indices [0 ..< targetSize*targetSize]
        //   channel 1 (G): indices [targetSize*targetSize ..< 2*targetSize*targetSize]
        //   channel 2 (B): indices [2*targetSize*targetSize ..< 3*targetSize*targetSize]

        let planeSize = targetSize * targetSize
        let totalElements = 3 * planeSize

        // Fill with normalized gray padding (114 / 255.0 ≈ 0.44706).
        let paddingValue = Float(114) / 255.0
        var output = [Float](repeating: paddingValue, count: totalElements)

        // Reciprocal scale for mapping canvas pixels → source image pixels.
        let invScale = 1.0 / scale

        output.withUnsafeMutableBufferPointer { buf in
            let rPlane = buf.baseAddress!
            let gPlane = buf.baseAddress! + planeSize
            let bPlane = buf.baseAddress! + 2 * planeSize

            for canvasY in yOffset ..< (yOffset + scaledHeight) {
                for canvasX in xOffset ..< (xOffset + scaledWidth) {

                    // Map canvas pixel to source image coordinates.
                    let srcXf = (Float(canvasX - xOffset) + 0.5) * invScale - 0.5
                    let srcYf = (Float(canvasY - yOffset) + 0.5) * invScale - 0.5

                    // Bilinear interpolation corners, clamped to valid range.
                    let x0 = max(0, Int(srcXf))
                    let y0 = max(0, Int(srcYf))
                    let x1 = min(x0 + 1, width - 1)
                    let y1 = min(y0 + 1, height - 1)

                    // Fractional weights.
                    let dx = max(0.0, srcXf - Float(x0))
                    let dy = max(0.0, srcYf - Float(y0))
                    let dx1 = 1.0 - dx
                    let dy1 = 1.0 - dy

                    // Source pixel byte offsets (3 bytes per pixel, packed RGB).
                    let idx00 = (y0 * width + x0) * 3
                    let idx10 = (y0 * width + x1) * 3
                    let idx01 = (y1 * width + x0) * 3
                    let idx11 = (y1 * width + x1) * 3

                    let src = rgbPixels

                    // Bilinear interpolation for R, G, B channels.
                    let r = dy1 * (dx1 * Float(src[idx00])     + dx * Float(src[idx10]))
                          + dy  * (dx1 * Float(src[idx01])     + dx * Float(src[idx11]))

                    let g = dy1 * (dx1 * Float(src[idx00 + 1]) + dx * Float(src[idx10 + 1]))
                          + dy  * (dx1 * Float(src[idx01 + 1]) + dx * Float(src[idx11 + 1]))

                    let b = dy1 * (dx1 * Float(src[idx00 + 2]) + dx * Float(src[idx10 + 2]))
                          + dy  * (dx1 * Float(src[idx01 + 2]) + dx * Float(src[idx11 + 2]))

                    // CHW write: normalize by 1/255.
                    let dstIdx = canvasY * targetSize + canvasX
                    rPlane[dstIdx] = r * (1.0 / 255.0)
                    gPlane[dstIdx] = g * (1.0 / 255.0)
                    bPlane[dstIdx] = b * (1.0 / 255.0)
                }
            }
        }

        return (data: output, letterbox: letterbox)
    }

    // MARK: - Coordinate Remapping

    /// Convert detection bounding boxes from 640x640 model space back to the
    /// coordinate system of the original source image (pixel coordinates).
    ///
    /// The model outputs boxes in the letterboxed canvas space (0–640 pixels).
    /// This function subtracts the padding offset and divides by the letterbox
    /// scale to recover source-image pixel coordinates.
    ///
    /// - Parameters:
    ///   - detections: Detection array to remap in place.
    ///   - letterbox: The letterbox parameters returned by `preprocess`.
    static func remapBoxes(
        _ detections: inout [Detection],
        letterbox: LetterboxInfo
    ) {
        let invScale = 1.0 / letterbox.scale
        let xOff = Float(letterbox.xOffset)
        let yOff = Float(letterbox.yOffset)
        let origW = Float(letterbox.originalWidth)
        let origH = Float(letterbox.originalHeight)

        for i in detections.indices {
            // Detections are in 640x640 canvas pixel coordinates.
            // Remove letterbox offset and undo scale to get source-image pixels.
            let srcX = (detections[i].x - xOff) * invScale
            let srcY = (detections[i].y - yOff) * invScale
            let srcW = detections[i].width * invScale
            let srcH = detections[i].height * invScale

            // Clamp to original image bounds.
            detections[i].x = max(0.0, min(origW, srcX))
            detections[i].y = max(0.0, min(origH, srcY))
            detections[i].width = max(0.0, min(origW - detections[i].x, srcW))
            detections[i].height = max(0.0, min(origH - detections[i].y, srcH))
        }
    }
}
