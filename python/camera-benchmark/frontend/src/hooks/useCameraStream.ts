import { useEffect, useState, type RefObject } from "react"

import { ClockSync } from "@/lib/clock"
import { parseFrame, percentile } from "@/lib/framing"
import type { StreamStats } from "@/lib/types"

const INITIAL_STATS: StreamStats = {
  connected: false,
  clientFps: 0,
  e2eP50: null,
  e2eP95: null,
}

/**
 * Opens a WebSocket MJPEG stream for one camera, renders frames straight into
 * `imgRef` (no per-frame React re-render), runs the clock-sync handshake, and
 * reports client-measured FPS + end-to-end latency once per second.
 */
export function useCameraStream(
  kind: string,
  imgRef: RefObject<HTMLImageElement | null>,
): StreamStats {
  const [stats, setStats] = useState<StreamStats>(INITIAL_STATS)

  useEffect(() => {
    const clock = new ClockSync()
    let ws: WebSocket | null = null
    let closed = false
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined
    let reconnectDelay = 1000

    let frameCount = 0
    let latencies: number[] = []
    let prevUrl: string | null = null

    const sendPing = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", t0: Date.now() }))
      }
    }

    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
      ws = new WebSocket(`${proto}//${window.location.host}/stream/${kind}`)
      ws.binaryType = "arraybuffer"

      ws.onopen = () => {
        reconnectDelay = 1000
        clock.relax()
        // A short burst of pings nails down the offset quickly.
        for (let i = 0; i < 5; i++) setTimeout(sendPing, i * 120)
        setStats((s) => ({ ...s, connected: true }))
      }

      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          try {
            const msg = JSON.parse(ev.data)
            if (msg.type === "pong") clock.onPong(msg.t0, msg.t1)
          } catch {
            /* ignore */
          }
          return
        }
        const parsed = parseFrame(ev.data as ArrayBuffer)
        if (!parsed) return
        const url = URL.createObjectURL(parsed.jpeg)
        if (imgRef.current) imgRef.current.src = url
        if (prevUrl) URL.revokeObjectURL(prevUrl)
        prevUrl = url
        frameCount += 1
        const lat = clock.latencyMs(parsed.meta.send_ts_ms)
        if (Number.isFinite(lat)) {
          latencies.push(lat)
          if (latencies.length > 150) latencies.shift()
        }
      }

      ws.onclose = () => {
        setStats((s) => ({ ...s, connected: false }))
        if (!closed) {
          reconnectTimer = setTimeout(connect, reconnectDelay)
          reconnectDelay = Math.min(reconnectDelay * 1.5, 8000)
        }
      }

      ws.onerror = () => ws?.close()
    }

    // Once per second: publish stats and refresh the clock offset.
    const statsTimer = setInterval(() => {
      const p50 = percentile(latencies, 0.5)
      const p95 = percentile(latencies, 0.95)
      setStats({
        connected: ws?.readyState === WebSocket.OPEN,
        clientFps: frameCount,
        e2eP50: p50 === null ? null : Math.round(p50),
        e2eP95: p95 === null ? null : Math.round(p95),
      })
      frameCount = 0
      sendPing()
    }, 1000)

    connect()

    return () => {
      closed = true
      clearInterval(statsTimer)
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (prevUrl) URL.revokeObjectURL(prevUrl)
      ws?.close()
    }
  }, [kind, imgRef])

  return stats
}
