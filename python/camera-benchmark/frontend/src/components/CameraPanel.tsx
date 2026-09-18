import { useEffect, useRef } from "react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useCameraStream } from "@/hooks/useCameraStream"
import type { CameraMetrics, StreamStats } from "@/lib/types"

interface CameraPanelProps {
  kind: string
  metrics?: CameraMetrics
  onStats: (kind: string, stats: StreamStats) => void
}

export function CameraPanel({ kind, metrics, onStats }: CameraPanelProps) {
  const imgRef = useRef<HTMLImageElement>(null)
  const stats = useCameraStream(kind, imgRef)

  useEffect(() => {
    onStats(kind, stats)
  }, [kind, stats, onStats])

  const title = metrics?.label ?? kind.toUpperCase()

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-base">{title}</CardTitle>
          <div className="flex flex-wrap items-center justify-end gap-1.5">
            {metrics?.synthetic && <Badge variant="secondary">Synthetic</Badge>}
            <Badge variant={stats.connected ? "default" : "destructive"}>
              {stats.connected ? "Live" : "Offline"}
            </Badge>
            <Badge variant="outline">{stats.clientFps} fps</Badge>
            {stats.e2eP50 !== null && (
              <Badge variant="outline">{stats.e2eP50} ms</Badge>
            )}
          </div>
        </div>
        <p className="truncate text-xs text-muted-foreground">
          {metrics?.name ?? "…"} · {metrics?.resolution ?? "—"} ·{" "}
          {metrics?.mode ?? "—"}
        </p>
      </CardHeader>
      <CardContent>
        <div className="relative aspect-video w-full overflow-hidden rounded-md bg-black">
          <img
            ref={imgRef}
            alt={`${title} feed`}
            className="h-full w-full object-contain"
          />
          {!stats.connected && (
            <div className="absolute inset-0 grid place-items-center text-sm text-white/70">
              Connecting…
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
