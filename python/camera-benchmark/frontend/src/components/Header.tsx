import { RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"

interface HeaderProps {
  onRestart: () => void
  restarting: boolean
}

export function Header({ onRestart, restarting }: HeaderProps) {
  return (
    <header className="flex items-center justify-between gap-4 border-b pb-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
          Camera Benchmark
        </h1>
        <p className="text-sm text-muted-foreground">
          USB webcam vs Raspberry Pi ribbon (CSI) camera
        </p>
      </div>
      <div className="flex items-center gap-4">
        <Button
          variant="outline"
          size="sm"
          onClick={onRestart}
          disabled={restarting}
        >
          <RefreshCw className={restarting ? "animate-spin" : ""} />
          {restarting ? "Restarting…" : "Restart cameras"}
        </Button>
        {/* Wendy wordmark, top-right */}
        <img src="/logo.svg" alt="Wendy" className="h-8 w-auto" />
      </div>
    </header>
  )
}
