// FrameRenderer.swift
// Draws bounding boxes on RGB frames and JPEG-encodes them for the MJPEG stream.
//
// Matches the Python detector's OpenCV drawing:
//   - Green (0,255,0) rectangles, 2px thick
//   - Label text: "person: 0.85" or "car #4: 0.92" (with track ID)
//   - JPEG quality: 80

import CTurboJPEG

// MARK: - FrameRenderer

/// Draws detection bounding boxes on RGB frames.
struct FrameRenderer: Sendable {

    /// Draw detection bounding boxes on an RGB24 frame buffer.
    ///
    /// - Parameters:
    ///   - frame: Mutable RGB24 pixel buffer (width * height * 3 bytes, row-major).
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - detections: Detections with coordinates in original image pixel space.
    static func drawDetections(
        on frame: inout [UInt8],
        width: Int,
        height: Int,
        detections: [Detection]
    ) {
        for det in detections {
            let x0 = max(0, min(width - 1, Int(det.x)))
            let y0 = max(0, min(height - 1, Int(det.y)))
            let x1 = max(0, min(width - 1, Int(det.x + det.width)))
            let y1 = max(0, min(height - 1, Int(det.y + det.height)))

            // Draw green rectangle (2px thick).
            let green: (UInt8, UInt8, UInt8) = (0, 255, 0)
            drawRect(on: &frame, width: width, height: height,
                     x0: x0, y0: y0, x1: x1, y1: y1,
                     color: green, thickness: 2)

            // Build label text.
            var label = "\(det.label): \(String(format: "%.2f", det.confidence))"
            if let trackId = det.trackId {
                label = "\(det.label) #\(trackId): \(String(format: "%.2f", det.confidence))"
            }

            // Draw a filled green background for the label.
            let charW = 6
            let charH = 9
            let labelW = label.count * charW + 2
            let labelH = charH + 4
            let labelY0 = max(0, y0 - labelH)
            let labelY1 = y0
            let labelX1 = min(width - 1, x0 + labelW)
            fillRect(on: &frame, width: width, height: height,
                     x0: x0, y0: labelY0, x1: labelX1, y1: labelY1,
                     color: green)

            // Draw white text on the green background.
            drawText(on: &frame, width: width, height: height,
                     text: label, x: x0 + 1, y: labelY0 + 2,
                     color: (0, 0, 0))
        }
    }

    /// Render detections on a frame and JPEG-encode the result.
    ///
    /// - Returns: JPEG bytes, or nil if encoding fails.
    static func renderFrame(
        _ frame: [UInt8],
        width: Int,
        height: Int,
        detections: [Detection]
    ) -> [UInt8]? {
        var mutableFrame = frame
        drawDetections(on: &mutableFrame, width: width, height: height,
                       detections: detections)
        return JPEGEncoder.encode(rgb: mutableFrame, width: width, height: height)
    }
}

// MARK: - Drawing primitives

/// Draw a rectangle outline on an RGB24 buffer.
private func drawRect(
    on frame: inout [UInt8],
    width: Int, height: Int,
    x0: Int, y0: Int, x1: Int, y1: Int,
    color: (UInt8, UInt8, UInt8),
    thickness: Int
) {
    for t in 0..<thickness {
        // Top edge
        let ty = y0 + t
        if ty < height {
            for x in x0...x1 where x < width {
                setPixel(on: &frame, width: width, x: x, y: ty, color: color)
            }
        }
        // Bottom edge
        let by = y1 - t
        if by >= 0, by < height {
            for x in x0...x1 where x < width {
                setPixel(on: &frame, width: width, x: x, y: by, color: color)
            }
        }
        // Left edge
        let lx = x0 + t
        if lx < width {
            for y in y0...y1 where y < height {
                setPixel(on: &frame, width: width, x: lx, y: y, color: color)
            }
        }
        // Right edge
        let rx = x1 - t
        if rx >= 0, rx < width {
            for y in y0...y1 where y < height {
                setPixel(on: &frame, width: width, x: rx, y: y, color: color)
            }
        }
    }
}

/// Fill a rectangle on an RGB24 buffer.
private func fillRect(
    on frame: inout [UInt8],
    width: Int, height: Int,
    x0: Int, y0: Int, x1: Int, y1: Int,
    color: (UInt8, UInt8, UInt8)
) {
    for y in max(0, y0)...min(height - 1, y1) {
        for x in max(0, x0)...min(width - 1, x1) {
            setPixel(on: &frame, width: width, x: x, y: y, color: color)
        }
    }
}

@inline(__always)
private func setPixel(
    on frame: inout [UInt8],
    width: Int,
    x: Int, y: Int,
    color: (UInt8, UInt8, UInt8)
) {
    let idx = (y * width + x) * 3
    frame[idx]     = color.0
    frame[idx + 1] = color.1
    frame[idx + 2] = color.2
}

// MARK: - Bitmap font (5x7 pixel glyphs for ASCII 32–126)

/// Draw a string using a minimal bitmap font.
private func drawText(
    on frame: inout [UInt8],
    width: Int, height: Int,
    text: String,
    x: Int, y: Int,
    color: (UInt8, UInt8, UInt8)
) {
    var cursorX = x
    for ch in text.utf8 {
        let glyphIndex = Int(ch) - 32
        guard glyphIndex >= 0, glyphIndex < font5x7.count else {
            cursorX += 6
            continue
        }
        let glyph = font5x7[glyphIndex]
        for row in 0..<7 {
            let bits = glyph[row]
            for col in 0..<5 {
                if bits & (1 << (4 - col)) != 0 {
                    let px = cursorX + col
                    let py = y + row
                    if px >= 0, px < width, py >= 0, py < height {
                        setPixel(on: &frame, width: width, x: px, y: py, color: color)
                    }
                }
            }
        }
        cursorX += 6
    }
}

/// Minimal 5x7 bitmap font for printable ASCII (space through tilde).
/// Each glyph is 7 rows of UInt8; the top 5 bits of each byte form the pixels.
private let font5x7: [[UInt8]] = {
    // Space through common chars needed for detection labels.
    // Format: each row is a bitmask where bit 4 = leftmost pixel, bit 0 = rightmost.
    var glyphs = [[UInt8]](repeating: [0,0,0,0,0,0,0], count: 95)

    // ' ' (32)
    glyphs[0]  = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    // '!' (33)
    glyphs[1]  = [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04]
    // '#' (35)
    glyphs[3]  = [0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00]
    // '.' (46)
    glyphs[14] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04]
    // '0' (48)
    glyphs[16] = [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E]
    // '1' (49)
    glyphs[17] = [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E]
    // '2' (50)
    glyphs[18] = [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F]
    // '3' (51)
    glyphs[19] = [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E]
    // '4' (52)
    glyphs[20] = [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02]
    // '5' (53)
    glyphs[21] = [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]
    // '6' (54)
    glyphs[22] = [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E]
    // '7' (55)
    glyphs[23] = [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08]
    // '8' (56)
    glyphs[24] = [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E]
    // '9' (57)
    glyphs[25] = [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C]
    // ':' (58)
    glyphs[26] = [0x00, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00]
    // 'A'-'Z' (65-90)
    glyphs[33] = [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11] // A
    glyphs[34] = [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E] // B
    glyphs[35] = [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E] // C
    glyphs[36] = [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E] // D
    glyphs[37] = [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F] // E
    glyphs[38] = [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10] // F
    glyphs[39] = [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F] // G
    glyphs[40] = [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11] // H
    glyphs[41] = [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E] // I
    glyphs[42] = [0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E] // J
    glyphs[43] = [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11] // K
    glyphs[44] = [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F] // L
    glyphs[45] = [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11] // M
    glyphs[46] = [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11] // N
    glyphs[47] = [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E] // O
    glyphs[48] = [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10] // P
    glyphs[49] = [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D] // Q
    glyphs[50] = [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11] // R
    glyphs[51] = [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E] // S
    glyphs[52] = [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04] // T
    glyphs[53] = [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E] // U
    glyphs[54] = [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04] // V
    glyphs[55] = [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11] // W
    glyphs[56] = [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11] // X
    glyphs[57] = [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04] // Y
    glyphs[58] = [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F] // Z
    // 'a'-'z' (97-122)
    glyphs[65] = [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F] // a
    glyphs[66] = [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E] // b
    glyphs[67] = [0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E] // c
    glyphs[68] = [0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F] // d
    glyphs[69] = [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E] // e
    glyphs[70] = [0x06, 0x08, 0x1E, 0x08, 0x08, 0x08, 0x08] // f
    glyphs[71] = [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E] // g
    glyphs[72] = [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11] // h
    glyphs[73] = [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E] // i
    glyphs[74] = [0x02, 0x00, 0x02, 0x02, 0x02, 0x12, 0x0C] // j
    glyphs[75] = [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12] // k
    glyphs[76] = [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E] // l
    glyphs[77] = [0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15] // m
    glyphs[78] = [0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11] // n
    glyphs[79] = [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E] // o
    glyphs[80] = [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10] // p
    glyphs[81] = [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01] // q
    glyphs[82] = [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10] // r
    glyphs[83] = [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E] // s
    glyphs[84] = [0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06] // t
    glyphs[85] = [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D] // u
    glyphs[86] = [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04] // v
    glyphs[87] = [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A] // w
    glyphs[88] = [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11] // x
    glyphs[89] = [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E] // y
    glyphs[90] = [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F] // z
    // '_' (95)
    glyphs[63] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F]
    // '-' (45)
    glyphs[13] = [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00]

    return glyphs
}()

// MARK: - JPEGEncoder

/// JPEG encoding via libturbojpeg C interop.
struct JPEGEncoder: Sendable {

    /// Encode an RGB24 frame to JPEG.
    ///
    /// - Parameters:
    ///   - frame: Raw RGB24 pixel data (width * height * 3 bytes).
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - quality: JPEG quality (1–100, default 80).
    /// - Returns: JPEG bytes, or nil on failure.
    static func encode(rgb frame: [UInt8], width: Int, height: Int, quality: Int = 80) -> [UInt8]? {
        guard let handle = tjInitCompress() else { return nil }
        defer { tjDestroy(handle) }

        var jpegBuf: UnsafeMutablePointer<UInt8>? = nil
        var jpegSize: UInt = 0

        let result = frame.withUnsafeBufferPointer { srcPtr -> Int32 in
            guard let baseAddr = srcPtr.baseAddress else { return -1 }
            return tjCompress2(
                handle,
                baseAddr,
                Int32(width),
                0,                          // pitch (0 = width * pixelSize)
                Int32(height),
                Int32(TJPF_RGB.rawValue),
                &jpegBuf,
                &jpegSize,
                Int32(TJSAMP_420.rawValue),
                Int32(quality),
                Int32(TJFLAG_FASTDCT)
            )
        }

        guard result == 0, let buf = jpegBuf else {
            if let buf = jpegBuf { tjFree(buf) }
            return nil
        }

        let bytes = Array(UnsafeBufferPointer(start: buf, count: Int(jpegSize)))
        tjFree(buf)
        return bytes
    }
}
