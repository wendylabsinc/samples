export type CameraKind = "usb" | "csi"

export interface LatencySummary {
  p50: number
  p95: number
  avg: number
}

export interface CameraMetrics {
  kind: string
  label: string
  name: string | null
  mode: string
  device: string | null
  synthetic: boolean
  online: boolean
  resolution: string | null
  format: string | null
  server_fps: number
  startup_ms: number | null
  cpu_pct: number | null
  rss_mb: number | null
  sharpness: number | null
  brightness: number | null
  pipeline_latency_ms: LatencySummary | null
}

export interface BoardPower {
  available: boolean
  power_w: number | null
  reason?: string | null
}

export interface MetricsSnapshot {
  ts: number
  board: BoardPower
  cameras: Record<string, CameraMetrics>
}

/** Client-side measurements produced by useCameraStream (per camera). */
export interface StreamStats {
  connected: boolean
  clientFps: number
  e2eP50: number | null
  e2eP95: number | null
}
