import { useState, useEffect, useCallback } from "react"

export interface AudioDevice {
  id: string
  name: string
  isDefault?: boolean
}

export interface AudioDevices {
  inputs: AudioDevice[]
  outputs: AudioDevice[]
}

export function useAudioDevices() {
  const [devices, setDevices] = useState<AudioDevices>({ inputs: [], outputs: [] })
  const [selectedInput, setSelectedInput] = useState<string>("")
  const [selectedOutput, setSelectedOutput] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchDevices = useCallback(async () => {
    setLoading(true)
    setError(null)

    const fallbackDevices: AudioDevices = {
      inputs: [{ id: "default", name: "Default Microphone", isDefault: true }],
      outputs: [{ id: "default", name: "Default Speaker", isDefault: true }],
    }

    const mediaDevices =
      typeof navigator !== "undefined" ? navigator.mediaDevices : undefined

    let browserInputs: AudioDevice[] = []
    let browserOutputs: AudioDevice[] = []

    if (mediaDevices?.enumerateDevices) {
      try {
        const browserDevices = await mediaDevices.enumerateDevices()

        browserInputs = browserDevices
          .filter((d) => d.kind === "audioinput")
          .map((d, i) => ({
            id: d.deviceId || `input-${i}`,
            name: d.label || `Microphone ${i + 1}`,
            isDefault: d.deviceId === "default",
          }))

        browserOutputs = browserDevices
          .filter((d) => d.kind === "audiooutput")
          .map((d, i) => ({
            id: d.deviceId || `output-${i}`,
            name: d.label || `Speaker ${i + 1}`,
            isDefault: d.deviceId === "default",
          }))
      } catch (err) {
        setError("Failed to enumerate browser audio devices")
        console.error("Error fetching devices:", err)
      }
    } else {
      setError("Browser audio devices are not available")
    }

    let mergedInputs = [...browserInputs]
    let mergedOutputs = [...browserOutputs]

    // Also fetch server-side devices (for Jetson audio devices)
    try {
      const response = await fetch("/api/devices")
      if (response.ok) {
        const serverDevices: AudioDevices = await response.json()
        const shouldPrefix = mergedInputs.length > 0 || mergedOutputs.length > 0
        const serverName = (name: string) =>
          shouldPrefix ? `[Server] ${name}` : name

        for (const device of serverDevices.inputs) {
          const name = serverName(device.name)
          if (
            !mergedInputs.some(
              (d) => d.name === device.name || d.name === name
            )
          ) {
            mergedInputs.push({ ...device, name })
          }
        }
        for (const device of serverDevices.outputs) {
          const name = serverName(device.name)
          if (
            !mergedOutputs.some(
              (d) => d.name === device.name || d.name === name
            )
          ) {
            mergedOutputs.push({ ...device, name })
          }
        }
      }
    } catch {
      // Server devices not available, use browser devices only
    }

    if (mergedInputs.length === 0) {
      mergedInputs = fallbackDevices.inputs
    }
    if (mergedOutputs.length === 0) {
      mergedOutputs = fallbackDevices.outputs
    }

    setDevices({ inputs: mergedInputs, outputs: mergedOutputs })
    setLoading(false)
  }, [])

  // Set defaults when devices are loaded
  useEffect(() => {
    if (devices.inputs.length > 0 && !selectedInput) {
      const defaultInput = devices.inputs.find((d) => d.isDefault) || devices.inputs[0]
      setSelectedInput(defaultInput.id)
    }
    if (devices.outputs.length > 0 && !selectedOutput) {
      const defaultOutput = devices.outputs.find((d) => d.isDefault) || devices.outputs[0]
      setSelectedOutput(defaultOutput.id)
    }
  }, [devices, selectedInput, selectedOutput])

  // Fetch devices on mount and when devices change
  useEffect(() => {
    fetchDevices()

    // Listen for device changes
    const mediaDevices =
      typeof navigator !== "undefined" ? navigator.mediaDevices : undefined
    if (mediaDevices?.addEventListener) {
      mediaDevices.addEventListener("devicechange", fetchDevices)
      return () => {
        mediaDevices.removeEventListener("devicechange", fetchDevices)
      }
    }

    if (mediaDevices) {
      const handler = () => fetchDevices()
      mediaDevices.ondevicechange = handler
      return () => {
        if (mediaDevices.ondevicechange === handler) {
          mediaDevices.ondevicechange = null
        }
      }
    }

    return () => {
      // no-op
    }
  }, [fetchDevices])

  return {
    devices,
    selectedInput,
    selectedOutput,
    setSelectedInput,
    setSelectedOutput,
    loading,
    error,
    refresh: fetchDevices,
  }
}
