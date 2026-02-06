import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import {
  AutoSizer,
  CellMeasurer,
  CellMeasurerCache,
  List,
} from "react-virtualized"
import type { ListRowRenderer } from "react-virtualized"
import { ShaderAnimation, type ShaderAnimationRef } from "@/components/ui/shader-animation"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { useAudioDevices } from "@/hooks/useAudioDevices"
import { useAnimatedText } from "@/hooks/useAnimatedText"
import { Mic, MicOff, Volume2, Loader2, Bot, User, Copy, Check } from "lucide-react"

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error"

interface Message {
  id: string
  type: "user" | "assistant"
  text: string
  timestamp: Date
}

function AnimatedText({ text }: { text: string }) {
  const animated = useAnimatedText(text)
  return <>{animated}</>
}

type ListItem =
  | (Message & { kind: "message" })
  | {
      id: string
      kind: "transcript"
      text: string
    }
  | {
      id: string
      kind: "status"
      status: string
    }

function getStatusMessage(status: string): string {
  switch (status) {
    case "transcribing":
      return "Transcribing..."
    case "thinking":
      return "Thinking..."
    case "speaking":
      return "Speaking..."
    default:
      return ""
  }
}

function MessageList({ messages, currentTranscript, pipelineStatus }: { messages: Message[]; currentTranscript: string; pipelineStatus: string }) {
  const listRef = useRef<List>(null)
  const [cache] = useState(
    () => new CellMeasurerCache({
      defaultHeight: 96,
      fixedWidth: true,
    })
  )

  const items = useMemo<ListItem[]>(() => {
    const base: ListItem[] = messages.map((message) => ({
      ...message,
      kind: "message" as const,
    }))
    if (currentTranscript) {
      base.push({
        id: "transcript",
        kind: "transcript",
        text: currentTranscript,
      })
    }
    // Show status indicator when processing
    const statusMessage = getStatusMessage(pipelineStatus)
    if (statusMessage && pipelineStatus !== "transcribing") {
      base.push({
        id: "status",
        kind: "status",
        status: pipelineStatus,
      })
    }
    return base
  }, [messages, currentTranscript, pipelineStatus])

  // Scroll to bottom when items change
  useEffect(() => {
    if (items.length > 0) {
      // Use setTimeout to ensure DOM has updated
      setTimeout(() => {
        listRef.current?.scrollToRow(items.length - 1)
      }, 0)
    }
  }, [items.length])

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
        <Bot className="w-12 h-12 opacity-50" />
        <p>Press the microphone button to start talking</p>
      </div>
    )
  }

  const rowRenderer: ListRowRenderer = ({ index, key, parent, style }) => {
    const item = items[index]

    if (item.kind === "transcript") {
      return (
        <CellMeasurer
          cache={cache}
          columnIndex={0}
          parent={parent}
          rowIndex={index}
          key={key}
        >
          {({ registerChild }) => (
              <div ref={registerChild as React.LegacyRef<HTMLDivElement>} style={style} className="px-4 py-2">
              <div className="flex gap-3 flex-row-reverse">
                <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-primary text-primary-foreground">
                  <User className="w-4 h-4" />
                </div>
                <div className="max-w-[75%] rounded-2xl px-4 py-3 bg-primary/50 text-primary-foreground rounded-br-md">
                  <p className="text-sm leading-relaxed italic">
                    <AnimatedText text={item.text} />...
                  </p>
                </div>
              </div>
            </div>
          )}
        </CellMeasurer>
      )
    }

    if (item.kind === "status") {
      const statusMessage = getStatusMessage(item.status)
      return (
        <CellMeasurer
          cache={cache}
          columnIndex={0}
          parent={parent}
          rowIndex={index}
          key={key}
        >
          {({ registerChild }) => (
              <div ref={registerChild as React.LegacyRef<HTMLDivElement>} style={style} className="px-4 py-2">
                <div className="flex gap-3 flex-row">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-blue-500 text-white">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm italic">{statusMessage}</span>
                  </div>
                </div>
              </div>
          )}
        </CellMeasurer>
      )
    }

    const msg = item
    return (
      <CellMeasurer
        cache={cache}
        columnIndex={0}
        parent={parent}
        rowIndex={index}
        key={key}
      >
        {({ registerChild }) => (
            <div ref={registerChild as React.LegacyRef<HTMLDivElement>} style={style} className="px-4 py-2">
            <div
              className={`flex gap-3 ${msg.type === "user" ? "flex-row-reverse" : "flex-row"}`}
            >
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.type === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-blue-500 text-white"
                }`}
              >
                {msg.type === "user" ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </div>
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                  msg.type === "user"
                    ? "bg-primary text-primary-foreground rounded-br-md"
                    : "bg-muted text-foreground rounded-bl-md"
                }`}
              >
                <p className="text-sm leading-relaxed">
                  {msg.type === "assistant" ? <AnimatedText text={msg.text} /> : msg.text}
                </p>
                <span className="text-xs opacity-60 mt-1 block">
                  {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
            </div>
            </div>
        )}
      </CellMeasurer>
    )
  }

  return (
    <div className="h-full">
      <AutoSizer>
        {({ height, width }) => (
          <List
            ref={listRef}
            height={height}
            width={width}
            rowCount={items.length}
            rowHeight={cache.rowHeight}
            deferredMeasurementCache={cache}
            rowRenderer={rowRenderer}
            overscanRowCount={6}
          />
        )}
      </AutoSizer>
    </div>
  )
}

function App() {
  const {
    devices,
    selectedInput,
    selectedOutput,
    setSelectedInput,
    setSelectedOutput,
    loading: devicesLoading,
  } = useAudioDevices()

  const [status, setStatus] = useState<ConnectionStatus>("disconnected")
  const [isListening, setIsListening] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [currentTranscript, setCurrentTranscript] = useState("")
  const [pipelineStatus, setPipelineStatus] = useState<string>("idle")
  const [audioLevel, setAudioLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [errorDialogOpen, setErrorDialogOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [canUseMic, setCanUseMic] = useState<boolean | null>(null)
  const [isPlayingSound, setIsPlayingSound] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationFrameRef = useRef<number>(0)
  const nextPlayTimeRef = useRef<number>(0)
  const shaderRef = useRef<ShaderAnimationRef>(null)

  const addMessage = useCallback((type: "user" | "assistant", text: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        type,
        text,
        timestamp: new Date(),
      },
    ])
  }, [])

  const showError = useCallback((message: string) => {
    setError(message)
    setErrorDialogOpen(true)
    setCopied(false)
  }, [])

  const copyErrorToClipboard = useCallback(async () => {
    if (error) {
      await navigator.clipboard.writeText(error)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [error])

  useEffect(() => {
    const mediaDevices =
      typeof navigator !== "undefined" ? navigator.mediaDevices : undefined
    const supported = !!mediaDevices?.getUserMedia
    setCanUseMic(supported)
  }, [])

  // Audio level analysis for visualization - runs continuously while analyser exists
  const updateAudioLevel = useCallback(() => {
    if (!analyserRef.current) return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)

    // Calculate average volume level (0-1)
    const sum = dataArray.reduce((a, b) => a + b, 0)
    const average = sum / dataArray.length / 255

    setAudioLevel(average)

    // Update shader ref directly for smoother animation
    if (shaderRef.current) {
      shaderRef.current.setAudioLevel(average)
    }

    // Continue the animation loop - stopRecording will cancel this
    animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
  }, [])

  // Start recording when button is pressed
  const startRecording = useCallback(async () => {
    // Close any existing connection first
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    setStatus("connecting")
    setIsRecording(true)
    setPipelineStatus("idle")

    try {
      const useServerMic = canUseMic === false

      let stream: MediaStream | null = null

      if (!useServerMic) {
        const mediaDevices =
          typeof navigator !== "undefined" ? navigator.mediaDevices : undefined
        if (!mediaDevices?.getUserMedia) {
          showError("Microphone access is not available in this browser.")
          setStatus("error")
          setIsRecording(false)
          return
        }

        // Request microphone access
        stream = await mediaDevices.getUserMedia({
          audio: {
            deviceId: selectedInput ? { exact: selectedInput } : undefined,
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        })
        mediaStreamRef.current = stream
      }

      // Create audio context
      const audioContext = new AudioContext({ sampleRate: 16000 })
      audioContextRef.current = audioContext

      // Connect WebSocket
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/voice`)
      wsRef.current = ws

      ws.onopen = () => {
        console.log("WebSocket connected")
      }

      ws.onmessage = async (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === "ready") {
            setStatus("connected")
            setIsListening(true)

            if (useServerMic) {
              ws.send(
                JSON.stringify({
                  type: "start_server_mic",
                  deviceId: selectedInput || "default",
                })
              )
            } else if (stream) {
              // Start capturing audio
              const source = audioContext.createMediaStreamSource(stream)

              // Create analyser for visualization
              const analyser = audioContext.createAnalyser()
              analyser.fftSize = 256
              analyserRef.current = analyser
              source.connect(analyser)

              // Start audio level updates
              animationFrameRef.current = requestAnimationFrame(updateAudioLevel)

              const processor = audioContext.createScriptProcessor(1024, 1, 1)
              processorRef.current = processor

              processor.onaudioprocess = (e) => {
                if (ws.readyState !== WebSocket.OPEN) return

                const inputData = e.inputBuffer.getChannelData(0)

                // Convert to 16-bit PCM
                const pcmData = new Int16Array(inputData.length)
                for (let i = 0; i < inputData.length; i++) {
                  const s = Math.max(-1, Math.min(1, inputData[i]))
                  pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff
                }

                // Send to server as base64
                const base64 = btoa(
                  String.fromCharCode(...new Uint8Array(pcmData.buffer))
                )
                ws.send(
                  JSON.stringify({
                    type: "audio",
                    data: base64,
                    sampleRate: 16000,
                    channels: 1,
                  })
                )
              }

              source.connect(processor)
              processor.connect(audioContext.destination)
            }
          } else if (data.type === "audio") {
            // Play received audio
            const binaryString = atob(data.data)
            const bytes = new Uint8Array(binaryString.length)
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i)
            }

            const samples = new Int16Array(bytes.buffer)
            const sampleRate = data.sampleRate || 16000

            const audioBuffer = audioContext.createBuffer(
              1,
              samples.length,
              sampleRate
            )
            const channelData = audioBuffer.getChannelData(0)
            for (let i = 0; i < samples.length; i++) {
              channelData[i] = samples[i] / 32768
            }

            const source = audioContext.createBufferSource()
            source.buffer = audioBuffer
            source.connect(audioContext.destination)

            const currentTime = audioContext.currentTime
            const startTime = Math.max(currentTime, nextPlayTimeRef.current)
            source.start(startTime)
            nextPlayTimeRef.current = startTime + audioBuffer.duration

            if (nextPlayTimeRef.current - currentTime > 0.5) {
              nextPlayTimeRef.current = currentTime + 0.05
            }
          } else if (data.type === "audio_level") {
            // Audio level from server microphone
            setAudioLevel(data.level)
            if (shaderRef.current) {
              shaderRef.current.setAudioLevel(data.level)
            }
          } else if (data.type === "status") {
            // Pipeline status update
            setPipelineStatus(data.status)
          } else if (data.type === "interim_transcription") {
            // Show what user is saying in real-time
            setCurrentTranscript(data.text)
            setPipelineStatus("transcribing")
          } else if (data.type === "transcription") {
            // Final transcription - add as user message
            addMessage("user", data.text)
            setCurrentTranscript("")
          } else if (data.type === "assistant_message") {
            // Full LLM response - add as assistant message
            addMessage("assistant", data.text)
          } else if (data.type === "error") {
            showError(data.message)
            setStatus("error")
          }
        } catch (err) {
          console.error("Error processing message:", err)
        }
      }

      ws.onerror = () => {
        showError("WebSocket connection error")
        setStatus("error")
      }

      ws.onclose = () => {
        setStatus("disconnected")
        setIsListening(false)
        setIsRecording(false)
        setPipelineStatus("idle")
        setAudioLevel(0)
        if (shaderRef.current) {
          shaderRef.current.setAudioLevel(0)
        }
      }
    } catch (err) {
      console.error("Error starting session:", err)
      showError(err instanceof Error ? err.message : "Failed to start session")
      setStatus("error")
    }
  }, [selectedInput, addMessage, showError, canUseMic, updateAudioLevel])

  const playTestSound = useCallback(async () => {
    setIsPlayingSound(true)
    try {
      const response = await fetch("/api/sounds/dog", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          deviceId: selectedOutput || "default",
        }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => null)
        const details = data?.details ? ` (${data.details})` : ""
        throw new Error(`${data?.error || "Failed to play sound"}${details}`)
      }
    } catch (err) {
      showError(err instanceof Error ? err.message : "Failed to play sound")
    } finally {
      setIsPlayingSound(false)
    }
  }, [selectedOutput, showError])

  // Stop recording but keep websocket open for response
  const stopRecording = useCallback(() => {
    setIsRecording(false)
    setAudioLevel(0)

    // Stop audio level animation
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = 0
    }

    // Reset shader animation
    if (shaderRef.current) {
      shaderRef.current.setAudioLevel(0)
    }

    // Stop media stream (microphone)
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }

    // Disconnect processor
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    // Clear analyser
    analyserRef.current = null

    // Tell server to stop server mic if using it
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop_server_mic" }))
      // Send end of recording signal
      wsRef.current.send(JSON.stringify({ type: "stop_recording" }))
    }

    setIsListening(false)
  }, [])

  // Full session cleanup
  const stopSession = useCallback(() => {
    stopRecording()

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "end" }))
      wsRef.current.close()
      wsRef.current = null
    }

    nextPlayTimeRef.current = 0
    setStatus("disconnected")
  }, [stopRecording])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSession()
    }
  }, [stopSession])

  const getStatusText = () => {
    switch (status) {
      case "connecting":
        return "Connecting..."
      case "connected":
        return isListening ? "Listening" : "Connected"
      case "error":
        return "Error"
      default:
        return "Ready"
    }
  }

  return (
    <TooltipProvider>
    <div className="flex flex-col h-screen w-full bg-background">
      {/* Header */}
      <header className="relative h-[180px] flex-shrink-0">
        <ShaderAnimation ref={shaderRef} audioLevel={audioLevel} />
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <h1 className="text-4xl font-semibold tracking-tight text-foreground">
            Wendy
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Voice Assistant</p>

          {/* Status indicator */}
          <div className="flex items-center gap-2 mt-3">
            <div
              className={`w-2 h-2 rounded-full ${
                status === "connected"
                  ? "bg-green-500 animate-pulse"
                  : status === "connecting"
                    ? "bg-yellow-500 animate-pulse"
                    : status === "error"
                      ? "bg-red-500"
                      : "bg-muted-foreground/50"
              }`}
            />
            <span className="text-xs text-muted-foreground">{getStatusText()}</span>
          </div>
        </div>

      </header>

      {/* Chat messages */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-2xl mx-auto">
          <MessageList messages={messages} currentTranscript={currentTranscript} pipelineStatus={pipelineStatus} />
        </div>
      </main>

      {/* Error Dialog */}
      <AlertDialog open={errorDialogOpen} onOpenChange={setErrorDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Error</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground break-words whitespace-pre-wrap">
                  {error}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyErrorToClipboard}
                  className="w-full"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 mr-2" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy Error
                    </>
                  )}
                </Button>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction>Close</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Bottom control bar */}
      <footer className="flex-shrink-0 border-t border-border bg-card/80 backdrop-blur-sm p-4">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-6">
          <div className="w-full sm:w-64">
            <label className="text-xs text-muted-foreground flex items-center gap-2 mb-2">
              <Mic className="w-3 h-3" />
              Microphone
            </label>
            <Select
              value={selectedInput}
              onValueChange={setSelectedInput}
              disabled={isListening || devicesLoading}
            >
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="Select microphone" />
              </SelectTrigger>
              <SelectContent>
                {devices.inputs.map((device) => (
                  <SelectItem key={device.id} value={device.id}>
                    {device.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onPointerDown={() => {
                  // Allow recording when disconnected, error, or connected but idle
                  if (!isRecording && (status === "disconnected" || status === "error" || (status === "connected" && pipelineStatus === "idle"))) {
                    startRecording()
                  }
                }}
                onPointerUp={() => {
                  if (isRecording) {
                    stopRecording()
                  }
                }}
                onPointerLeave={() => {
                  if (isRecording) {
                    stopRecording()
                  }
                }}
                onContextMenu={(e) => e.preventDefault()}
                size="lg"
                disabled={status === "connecting" || devicesLoading || (status === "connected" && !isRecording && pipelineStatus !== "idle")}
                className={`rounded-full w-16 h-16 select-none touch-none ${
                  isRecording
                    ? "!bg-red-500 hover:!bg-red-600 animate-pulse"
                    : ""
                }`}
              >
                {status === "connecting" ? (
                  <Loader2 className="w-6 h-6 animate-spin" />
                ) : isRecording ? (
                  <MicOff className="w-6 h-6" />
                ) : (
                  <Mic className="w-6 h-6" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Hold to record, release to send</p>
            </TooltipContent>
          </Tooltip>

          <div className="w-full sm:w-64 flex items-end gap-2">
            <div className="flex-1">
            <label className="text-xs text-muted-foreground flex items-center gap-2 mb-2">
              <Volume2 className="w-3 h-3" />
              Speaker
            </label>
            <Select
              value={selectedOutput}
              onValueChange={setSelectedOutput}
              disabled={isListening || devicesLoading}
            >
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="Select speaker" />
              </SelectTrigger>
              <SelectContent>
                {devices.outputs.map((device) => (
                  <SelectItem key={device.id} value={device.id}>
                    {device.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={playTestSound}
                  size="sm"
                  variant="secondary"
                  disabled={devicesLoading || isPlayingSound}
                  className="h-9 px-3"
                >
                  {isPlayingSound ? "Playing..." : "Test Sound"}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Play a test sound to verify the speaker is working</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
        <p className="text-center text-xs text-muted-foreground mt-3">
          {canUseMic === false
            ? "Hold to record (server mic)"
            : isRecording
              ? "Release to send"
              : pipelineStatus !== "idle" && status === "connected"
                ? getStatusMessage(pipelineStatus) || "Processing..."
                : "Hold to record"}
        </p>
      </footer>
    </div>
    </TooltipProvider>
  )
}

export default App
