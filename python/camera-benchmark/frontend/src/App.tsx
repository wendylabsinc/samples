import { useCallback, useState } from "react"

import { BenchmarkTable } from "@/components/BenchmarkTable"
import { CameraPanel } from "@/components/CameraPanel"
import { Header } from "@/components/Header"
import { useMetrics } from "@/hooks/useMetrics"
import type { StreamStats } from "@/lib/types"

export default function App() {
  const snapshot = useMetrics(1000)
  const [streamStats, setStreamStats] = useState<Record<string, StreamStats>>({})
  const [restarting, setRestarting] = useState(false)

  const onStats = useCallback((kind: string, stats: StreamStats) => {
    setStreamStats((prev) => ({ ...prev, [kind]: stats }))
  }, [])

  const handleRestart = useCallback(async () => {
    setRestarting(true)
    try {
      await fetch("/restart", { method: "POST" })
    } catch {
      /* ignore */
    }
    // Give the children a moment to respawn and report a fresh startup time.
    setTimeout(() => setRestarting(false), 1500)
  }, [])

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
      <Header onRestart={handleRestart} restarting={restarting} />
      <div className="grid gap-4 md:grid-cols-2">
        <CameraPanel kind="usb" metrics={snapshot?.cameras.usb} onStats={onStats} />
        <CameraPanel kind="csi" metrics={snapshot?.cameras.csi} onStats={onStats} />
      </div>
      <BenchmarkTable snapshot={snapshot} stream={streamStats} />
    </div>
  )
}
