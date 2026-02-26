// FrameRenderer.swift
// Draws bounding boxes on RGB frames and JPEG-encodes them for the MJPEG stream.
//
// Rendering pipeline (matches Python detector using OpenCV):
//   1. Draw green (0, 255, 0) rectangle border, 2 px thick, for each detection.
//   2. Draw a filled green rectangle behind the label text at the top of the bbox.
//   3. Render white ASCII text (5x7 bitmap font) onto the label background.
//   4. JPEG-encode the annotated frame at quality 80 via libturbojpeg.
//
// Detection coordinates arrive already remapped from YOLO model space (640x640)
// to original-image pixel space by YOLOPreprocessor.remapBoxes.

import CTurboJPEG

// MARK: - FrameRenderer

/// Stateless utility that draws YOLO detections onto a packed RGB pixel buffer.
///
/// All drawing is integer arithmetic directly on the `[UInt8]` slice — no
/// external graphics library required.
struct FrameRenderer: Sendable {

    // MARK: Public API

    /// Convenience entry point: copy `frame`, annotate with detections, and
    /// JPEG-encode the result.
    ///
    /// - Parameters:
    ///   - frame: Source RGB pixel data (width × height × 3 bytes, row-major).
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - detections: Detections whose coordinates are in source-image pixel
    ///     space (top-left origin, already remapped from model space).
    /// - Returns: JPEG-encoded bytes, or `nil` when encoding fails.
    static func renderFrame(
        _ frame: [UInt8],
        width: Int,
        height: Int,
        detections: [Detection]
    ) -> [UInt8]? {
        var mutable = frame
        drawDetections(on: &mutable, width: width, height: height, detections: detections)
        return JPEGEncoder.encode(rgb: mutable, width: width, height: height)
    }

    // MARK: Bounding box drawing

    /// Draw green bounding boxes and label overlays for every detection.
    ///
    /// For each detection the function draws:
    ///   - A 2-pixel-thick green rectangle that outlines the bounding box.
    ///   - A solid green rectangle behind the label (anchored to the top edge).
    ///   - White 5×7 bitmap-font text inside the green label background.
    ///
    /// All coordinates are clamped to image bounds before any pixel write.
    ///
    /// - Parameters:
    ///   - frame: RGB pixel buffer, mutated in place (3 bytes per pixel, HWC).
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - detections: Detections to render.
    static func drawDetections(
        on frame: inout [UInt8],
        width: Int,
        height: Int,
        detections: [Detection]
    ) {
        for det in detections {
            // Convert Float coordinates to integer pixel space and clamp.
            let x0 = clamp(Int(det.x),              lo: 0, hi: width  - 1)
            let y0 = clamp(Int(det.y),              lo: 0, hi: height - 1)
            let x1 = clamp(Int(det.x + det.width),  lo: 0, hi: width  - 1)
            let y1 = clamp(Int(det.y + det.height), lo: 0, hi: height - 1)

            guard x1 > x0, y1 > y0 else { continue }

            // --- 2-pixel-thick green rectangle border ---
            drawRect(
                on: &frame, width: width, height: height,
                x0: x0, y0: y0, x1: x1, y1: y1,
                thickness: 2, r: 0, g: 255, b: 0
            )

            // --- Label string: "person: 0.85" or "car #4: 0.92" ---
            let label = labelString(for: det)

            // Estimate background dimensions.
            // Each glyph cell is (fontW + 1) px wide; add 4 px horizontal padding.
            let bgWidth  = label.unicodeScalars.count * (BitmapFont.glyphWidth + 1) + 4
            let bgHeight = BitmapFont.glyphHeight + 4

            // Place the label background directly above the top edge of the bbox.
            // If the bbox is too close to the frame top, flip the label inside.
            let bgY0 = y0 >= bgHeight ? y0 - bgHeight : y0
            let bgY1 = y0 >= bgHeight ? y0             : min(y0 + bgHeight, height - 1)
            let bgX0 = x0
            let bgX1 = min(width - 1, x0 + bgWidth)

            // Filled green rectangle behind the label.
            fillRect(
                on: &frame, width: width, height: height,
                x0: bgX0, y0: bgY0, x1: bgX1, y1: bgY1,
                r: 0, g: 255, b: 0
            )

            // White text rendered 2 px from the top-left of the background.
            drawText(
                on: &frame, width: width, height: height,
                text: label, x: bgX0 + 2, y: bgY0 + 2
            )
        }
    }

    // MARK: Text rendering

    /// Render an ASCII string into the frame using the embedded 5×7 bitmap font.
    ///
    /// Characters outside the font's covered range (ASCII 32–126) are rendered
    /// as blank space.  The coordinate (x, y) is the top-left pixel of the
    /// first glyph.  Text color is always white (255, 255, 255).
    ///
    /// - Parameters:
    ///   - frame: RGB pixel buffer, mutated in place.
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - text: ASCII string to render.
    ///   - x: Horizontal start position (pixels from left).
    ///   - y: Vertical start position (pixels from top).
    ///   - scale: Integer pixel magnification — each font pixel becomes an
    ///     `scale × scale` block (default: 1).
    static func drawText(
        on frame: inout [UInt8],
        width: Int,
        height: Int,
        text: String,
        x: Int,
        y: Int,
        scale: Int = 1
    ) {
        let s = max(1, scale)
        var cursorX = x

        for scalar in text.unicodeScalars {
            let glyph = BitmapFont.glyph(for: scalar)

            for row in 0 ..< BitmapFont.glyphHeight {
                let bits = glyph[row]
                for col in 0 ..< BitmapFont.glyphWidth {
                    // Bit 4 is the leftmost column.
                    guard bits & (0x10 >> col) != 0 else { continue }
                    for dy in 0 ..< s {
                        for dx in 0 ..< s {
                            let px = cursorX + col * s + dx
                            let py = y       + row * s + dy
                            guard px >= 0, px < width, py >= 0, py < height else { continue }
                            let idx = (py * width + px) * 3
                            frame[idx]     = 255   // R – white text
                            frame[idx + 1] = 255   // G
                            frame[idx + 2] = 255   // B
                        }
                    }
                }
            }

            // Advance cursor by glyph width + 1 px inter-character gap.
            cursorX += (BitmapFont.glyphWidth + 1) * s
        }
    }

    // MARK: Drawing primitives

    /// Draw an axis-aligned unfilled rectangle with `thickness` concentric
    /// single-pixel outlines.
    private static func drawRect(
        on frame: inout [UInt8],
        width: Int,
        height: Int,
        x0: Int, y0: Int, x1: Int, y1: Int,
        thickness: Int,
        r: UInt8, g: UInt8, b: UInt8
    ) {
        for t in 0 ..< thickness {
            let lx0 = x0 + t
            let ly0 = y0 + t
            let lx1 = x1 - t
            let ly1 = y1 - t
            guard lx1 > lx0, ly1 > ly0 else { break }
            hLine(on: &frame, width: width, height: height, x0: lx0, x1: lx1, y: ly0, r: r, g: g, b: b)
            hLine(on: &frame, width: width, height: height, x0: lx0, x1: lx1, y: ly1, r: r, g: g, b: b)
            vLine(on: &frame, width: width, height: height, x: lx0, y0: ly0, y1: ly1, r: r, g: g, b: b)
            vLine(on: &frame, width: width, height: height, x: lx1, y0: ly0, y1: ly1, r: r, g: g, b: b)
        }
    }

    /// Fill an axis-aligned rectangle with a solid color.
    private static func fillRect(
        on frame: inout [UInt8],
        width: Int,
        height: Int,
        x0: Int, y0: Int, x1: Int, y1: Int,
        r: UInt8, g: UInt8, b: UInt8
    ) {
        let cx0 = clamp(x0, lo: 0, hi: width  - 1)
        let cy0 = clamp(y0, lo: 0, hi: height - 1)
        let cx1 = clamp(x1, lo: 0, hi: width  - 1)
        let cy1 = clamp(y1, lo: 0, hi: height - 1)
        guard cx1 >= cx0, cy1 >= cy0 else { return }
        for py in cy0 ... cy1 {
            for px in cx0 ... cx1 {
                let idx = (py * width + px) * 3
                frame[idx]     = r
                frame[idx + 1] = g
                frame[idx + 2] = b
            }
        }
    }

    /// Write a horizontal run of pixels.
    private static func hLine(
        on frame: inout [UInt8],
        width: Int, height: Int,
        x0: Int, x1: Int, y: Int,
        r: UInt8, g: UInt8, b: UInt8
    ) {
        guard y >= 0, y < height else { return }
        let cx0 = clamp(x0, lo: 0, hi: width - 1)
        let cx1 = clamp(x1, lo: 0, hi: width - 1)
        guard cx1 >= cx0 else { return }
        for px in cx0 ... cx1 {
            let idx = (y * width + px) * 3
            frame[idx]     = r
            frame[idx + 1] = g
            frame[idx + 2] = b
        }
    }

    /// Write a vertical run of pixels.
    private static func vLine(
        on frame: inout [UInt8],
        width: Int, height: Int,
        x: Int, y0: Int, y1: Int,
        r: UInt8, g: UInt8, b: UInt8
    ) {
        guard x >= 0, x < width else { return }
        let cy0 = clamp(y0, lo: 0, hi: height - 1)
        let cy1 = clamp(y1, lo: 0, hi: height - 1)
        guard cy1 >= cy0 else { return }
        for py in cy0 ... cy1 {
            let idx = (py * width + x) * 3
            frame[idx]     = r
            frame[idx + 1] = g
            frame[idx + 2] = b
        }
    }

    // MARK: Label helpers

    /// Build the human-readable label string for a detection.
    ///
    /// Without track ID: `"person: 0.85"`
    /// With track ID:    `"car #4: 0.92"`
    private static func labelString(for det: Detection) -> String {
        let conf = String(format: "%.2f", det.confidence)
        if let tid = det.trackId {
            return "\(det.label) #\(tid): \(conf)"
        } else {
            return "\(det.label): \(conf)"
        }
    }

    // MARK: Utilities

    @inline(__always)
    private static func clamp(_ v: Int, lo: Int, hi: Int) -> Int {
        min(max(v, lo), hi)
    }
}

// MARK: - JPEGEncoder

/// Encodes an RGB pixel buffer to JPEG using libturbojpeg.
///
/// Matches the Python detector's `cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])`
/// at the same default quality (80) and YUV 4:2:0 chroma subsampling.
///
/// Memory contract:
///   - `tjCompress2` allocates its output buffer via the internal `tjAlloc`.
///   - That buffer is freed with `tjFree` before the function returns, in
///     both the success and error paths (via `defer`).
///   - The compressor handle is always released via `tjDestroy` (also `defer`).
///   - The caller receives a plain Swift `[UInt8]`; no C pointers escape.
struct JPEGEncoder: Sendable {

    /// Encode an RGB frame to JPEG bytes.
    ///
    /// - Parameters:
    ///   - frame: Packed RGB pixel data (width × height × 3 bytes).
    ///   - width: Frame width in pixels.
    ///   - height: Frame height in pixels.
    ///   - quality: JPEG quality 1–100 (default: 80).
    /// - Returns: JPEG-encoded bytes on success, `nil` on any failure.
    static func encode(
        rgb frame: [UInt8],
        width: Int,
        height: Int,
        quality: Int = 80
    ) -> [UInt8]? {
        guard width > 0, height > 0, frame.count == width * height * 3 else {
            return nil
        }

        guard let handle = tjInitCompress() else { return nil }
        defer { tjDestroy(handle) }

        // libturbojpeg manages this buffer; freed below via tjFree.
        var jpegBuf: UnsafeMutablePointer<UInt8>? = nil
        var jpegSize: CUnsignedLong = 0

        // Always free the turbojpeg output buffer before returning,
        // regardless of whether we successfully copied it to Swift memory.
        defer {
            if let buf = jpegBuf { tjFree(buf) }
        }

        let rc: Int32 = frame.withUnsafeBufferPointer { srcBuf in
            guard let src = srcBuf.baseAddress else { return -1 }
            return tjCompress2(
                handle,
                src,
                Int32(width),
                0,              // pitch – 0 means tightly packed rows
                Int32(height),
                TJPF_RGB.rawValue,
                &jpegBuf,
                &jpegSize,
                Int32(TJSAMP_420.rawValue),
                Int32(quality),
                0               // flags
            )
        }

        guard rc == 0, let buf = jpegBuf, jpegSize > 0 else { return nil }

        // Copy into Swift-owned memory before the defer block frees `buf`.
        return Array(UnsafeBufferPointer(start: buf, count: Int(jpegSize)))
    }
}

// MARK: - BitmapFont

/// Complete 5×7 monochrome bitmap font covering printable ASCII (U+0020–U+007E).
///
/// Glyph encoding: each glyph is 7 bytes, one per row (top to bottom).
/// Within each byte, bits [4..0] represent pixel columns 0–4 (left to right):
///   bit 4 → column 0 (leftmost pixel)
///   bit 0 → column 4 (rightmost pixel)
///
/// Test for pixel (row, col): `glyph[row] & (0x10 >> col) != 0`
private enum BitmapFont {
    static let glyphWidth  = 5
    static let glyphHeight = 7

    /// Return the 7-byte glyph for `scalar`, falling back to a blank glyph
    /// for characters outside the printable ASCII range.
    static func glyph(for scalar: Unicode.Scalar) -> [UInt8] {
        let v = scalar.value
        guard v >= 32, v <= 126 else { return glyphs[0] }
        return glyphs[Int(v) - 32]
    }

    // Table index = ASCII codepoint - 32.  95 entries for U+0020..U+007E.
    //
    // Each sub-array is 7 bytes.  Binary literals make the pixel patterns
    // visually verifiable: a 1 bit is a lit (white) pixel, 0 is dark.
    static let glyphs: [[UInt8]] = [
        // 32: space
        [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        // 33: !
        [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        // 34: "
        [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000],
        // 35: #
        [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
        // 36: $
        [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
        // 37: %
        [0b11001, 0b11010, 0b00100, 0b00100, 0b01011, 0b10011, 0b00000],
        // 38: &
        [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        // 39: '
        [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        // 40: (
        [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        // 41: )
        [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        // 42: *
        [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
        // 43: +
        [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        // 44: ,
        [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000],
        // 45: -
        [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        // 46: .
        [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
        // 47: /
        [0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b00000],
        // 48: 0
        [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b11001, 0b01110],
        // 49: 1
        [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        // 50: 2
        [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        // 51: 3
        [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        // 52: 4
        [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        // 53: 5
        [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        // 54: 6
        [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        // 55: 7
        [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        // 56: 8
        [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        // 57: 9
        [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        // 58: :
        [0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000],
        // 59: ;
        [0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00100, 0b01000],
        // 60: <
        [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        // 61: =
        [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        // 62: >
        [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        // 63: ?
        [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
        // 64: @
        [0b01110, 0b10001, 0b00001, 0b01101, 0b10101, 0b10110, 0b01110],
        // 65: A
        [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        // 66: B
        [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        // 67: C
        [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        // 68: D
        [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        // 69: E
        [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        // 70: F
        [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        // 71: G
        [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        // 72: H
        [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        // 73: I
        [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        // 74: J
        [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        // 75: K
        [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        // 76: L
        [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        // 77: M
        [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        // 78: N
        [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        // 79: O
        [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        // 80: P
        [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        // 81: Q
        [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        // 82: R
        [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        // 83: S
        [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        // 84: T
        [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        // 85: U
        [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        // 86: V
        [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
        // 87: W
        [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        // 88: X
        [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        // 89: Y
        [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        // 90: Z
        [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        // 91: [
        [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        // 92: backslash
        [0b10000, 0b01000, 0b00100, 0b00100, 0b00010, 0b00001, 0b00000],
        // 93: ]
        [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        // 94: ^
        [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
        // 95: _
        [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        // 96: `
        [0b01000, 0b00100, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000],
        // 97: a
        [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        // 98: b
        [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        // 99: c
        [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        // 100: d
        [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
        // 101: e
        [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        // 102: f
        [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
        // 103: g
        [0b00000, 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        // 104: h
        [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        // 105: i
        [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        // 106: j
        [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
        // 107: k
        [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        // 108: l
        [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        // 109: m
        [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
        // 110: n
        [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        // 111: o
        [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        // 112: p
        [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000],
        // 113: q
        [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001],
        // 114: r
        [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        // 115: s
        [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
        // 116: t
        [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
        // 117: u
        [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        // 118: v
        [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        // 119: w
        [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b11011, 0b10001],
        // 120: x
        [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        // 121: y
        [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        // 122: z
        [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        // 123: {
        [0b00110, 0b00100, 0b00100, 0b01100, 0b00100, 0b00100, 0b00110],
        // 124: |
        [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        // 125: }
        [0b01100, 0b00100, 0b00100, 0b00110, 0b00100, 0b00100, 0b01100],
        // 126: ~
        [0b00000, 0b01000, 0b11010, 0b00101, 0b00010, 0b00000, 0b00000],
    ]
}
