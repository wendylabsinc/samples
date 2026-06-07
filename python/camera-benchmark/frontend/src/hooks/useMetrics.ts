import { useEffect, useState } from "react"

import type { MetricsSnapshot } from "@/lib/types"

/** Polls GET /metrics on an interval and returns the latest snapshot. */
export function useMetrics(intervalMs = 1000): MetricsSnapshot | null {
  const [data, setData] = useState<MetricsSnapshot | null>(null)

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch("/metrics")
        if (!res.ok) return
        const json = (await res.json()) as MetricsSnapshot
        if (alive) setData(json)
      } catch {
        /* transient — keep last snapshot */
      }
    }
    tick()
    const id = setInterval(tick, intervalMs)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [intervalMs])

  return data
}
