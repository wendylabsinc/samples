// Binary WebSocket frame format (little-endian), matching server/manager.py:
//   [ 4 bytes magic "WCB1" ][ uint16 meta_len ][ meta JSON ][ JPEG bytes ]

export interface FrameMeta {
  seq: number
  send_ts_ms: number
  src_fps: number
  w: number
  h: number
  fmt: string
}

export interface ParsedFrame {
  meta: FrameMeta
  jpeg: Blob
}

const decoder = new TextDecoder()

export function parseFrame(buf: ArrayBuffer): ParsedFrame | null {
  if (buf.byteLength < 6) return null
  const dv = new DataView(buf)
  // magic: 'W' 'C' 'B' '1'
  if (
    dv.getUint8(0) !== 0x57 ||
    dv.getUint8(1) !== 0x43 ||
    dv.getUint8(2) !== 0x42 ||
    dv.getUint8(3) !== 0x31
  ) {
    return null
  }
  const metaLen = dv.getUint16(4, true)
  const metaBytes = new Uint8Array(buf, 6, metaLen)
  let meta: FrameMeta
  try {
    meta = JSON.parse(decoder.decode(metaBytes)) as FrameMeta
  } catch {
    return null
  }
  const jpeg = new Blob([new Uint8Array(buf, 6 + metaLen)], { type: "image/jpeg" })
  return { meta, jpeg }
}

/** Simple percentile over an array (0..1). Returns null if empty. */
export function percentile(values: number[], q: number): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(q * sorted.length))
  return sorted[idx]
}
