import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import type { MetricsSnapshot, StreamStats } from "@/lib/types"

type Dir = "low" | "high" | null
type Value = number | string | null | undefined

interface Row {
  label: string
  usb: Value
  csi: Value
  unit?: string
  better?: Dir
  hint?: string
}

interface BenchmarkTableProps {
  snapshot: MetricsSnapshot | null
  stream: Record<string, StreamStats>
}

function fmt(v: Value, unit?: string): string {
  if (v === null || v === undefined) return "—"
  if (typeof v === "number") return unit ? `${v} ${unit}` : `${v}`
  return v
}

function winner(usb: Value, csi: Value, better?: Dir): "usb" | "csi" | null {
  if (!better || typeof usb !== "number" || typeof csi !== "number") return null
  if (usb === csi) return null
  const usbWins = better === "low" ? usb < csi : usb > csi
  return usbWins ? "usb" : "csi"
}

function delta(usb: Value, csi: Value, unit?: string): string {
  if (typeof usb !== "number" || typeof csi !== "number") return "—"
  const d = Math.abs(usb - csi)
  const rounded = Math.round(d * 10) / 10
  return unit ? `${rounded} ${unit}` : `${rounded}`
}

export function BenchmarkTable({ snapshot, stream }: BenchmarkTableProps) {
  const usbM = snapshot?.cameras.usb
  const csiM = snapshot?.cameras.csi
  const usbS = stream.usb
  const csiS = stream.csi
  const usbLabel = usbM?.label ?? "USB Webcam"
  const csiLabel = csiM?.label ?? "Ribbon Cam (CSI)"

  const rows: Row[] = [
    { label: "End-to-end latency (p50)", usb: usbS?.e2eP50, csi: csiS?.e2eP50, unit: "ms", better: "low" },
    { label: "End-to-end latency (p95)", usb: usbS?.e2eP95, csi: csiS?.e2eP95, unit: "ms", better: "low" },
    { label: "Frame rate (client)", usb: usbS?.clientFps, csi: csiS?.clientFps, unit: "fps", better: "high" },
    { label: "Frame rate (capture)", usb: usbM?.server_fps, csi: csiM?.server_fps, unit: "fps", better: "high" },
    { label: "Startup time", usb: usbM?.startup_ms, csi: csiM?.startup_ms, unit: "ms", better: "low" },
    { label: "CPU usage", usb: usbM?.cpu_pct, csi: csiM?.cpu_pct, unit: "%", better: "low", hint: "per process · 100% = 1 core" },
    { label: "Memory (RSS)", usb: usbM?.rss_mb, csi: csiM?.rss_mb, unit: "MB", better: "low" },
    { label: "Pipeline latency (p50)", usb: usbM?.pipeline_latency_ms?.p50, csi: csiM?.pipeline_latency_ms?.p50, unit: "ms", better: "low", hint: "capture → server" },
    { label: "Sharpness", usb: usbM?.sharpness, csi: csiM?.sharpness, better: "high", hint: "variance of Laplacian" },
    { label: "Brightness", usb: usbM?.brightness, csi: csiM?.brightness, hint: "mean luma 0–255" },
    { label: "Resolution", usb: usbM?.resolution, csi: csiM?.resolution },
    { label: "Format", usb: usbM?.format, csi: csiM?.format },
    { label: "Source", usb: usbM?.mode, csi: csiM?.mode },
  ]

  const board = snapshot?.board

  return (
    <Card>
      <CardHeader>
        <CardTitle>Benchmark comparison</CardTitle>
        <CardDescription>
          Live metrics, updated once per second. Highlighted cells are the better
          result for that row.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[40%]">Metric</TableHead>
              <TableHead className="text-right">{usbLabel}</TableHead>
              <TableHead className="text-right">{csiLabel}</TableHead>
              <TableHead className="text-right">Δ</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const win = winner(r.usb, r.csi, r.better)
              return (
                <TableRow key={r.label}>
                  <TableCell className="font-medium">
                    {r.label}
                    {r.hint && (
                      <span className="block text-xs font-normal text-muted-foreground">
                        {r.hint}
                      </span>
                    )}
                  </TableCell>
                  <TableCell
                    className={cn(
                      "text-right tabular-nums",
                      win === "usb" && "font-semibold text-emerald-600 dark:text-emerald-400"
                    )}
                  >
                    {fmt(r.usb, r.unit)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      "text-right tabular-nums",
                      win === "csi" && "font-semibold text-emerald-600 dark:text-emerald-400"
                    )}
                  >
                    {fmt(r.csi, r.unit)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {delta(r.usb, r.csi, r.unit)}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
      <CardFooter className="border-t text-sm text-muted-foreground">
        <span>
          Board power (both cameras active):{" "}
          {board?.available && board.power_w !== null ? (
            <span className="font-medium text-foreground tabular-nums">
              {board.power_w} W
            </span>
          ) : (
            <span className="italic">unavailable (best-effort, Raspberry Pi 5 only)</span>
          )}
        </span>
      </CardFooter>
    </Card>
  )
}
