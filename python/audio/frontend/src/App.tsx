import { useState, useRef, useEffect, useCallback } from "react";
import { ShaderAnimation } from "@/components/ui/shader-animation";
import { Button } from "@/components/ui/button";
import { Cat, Dog, Mic, MicOff } from "lucide-react";

// Cow icon component since lucide doesn't have one
function CowIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 2c-2 0-3.5 1.5-3.5 3.5 0 .5.1 1 .3 1.5H6c-1.5 0-3 1-3 2.5 0 1 .5 2 1.5 2.5-.5.5-1 1.5-1 2.5 0 2 1.5 3.5 3.5 3.5h10c2 0 3.5-1.5 3.5-3.5 0-1-.5-2-1-2.5 1-.5 1.5-1.5 1.5-2.5 0-1.5-1.5-2.5-3-2.5h-2.8c.2-.5.3-1 .3-1.5C15.5 3.5 14 2 12 2z" />
      <circle cx="8.5" cy="11" r="1" />
      <circle cx="15.5" cy="11" r="1" />
      <path d="M9 16h6" />
    </svg>
  );
}

function AudioVisualizer({
  audioData,
  isListening,
}: {
  audioData: number[];
  isListening: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas with dark background (actual color, not CSS variable)
    ctx.fillStyle = "#27272a"; // zinc-800
    ctx.fillRect(0, 0, width, height);

    if (!isListening || audioData.length === 0) {
      // Draw flat line when not listening
      ctx.strokeStyle = "#52525b"; // zinc-600
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    // Draw frequency bars first (behind waveform)
    const barCount = 32;
    const barWidth = width / barCount - 2;
    ctx.fillStyle = "rgba(34, 197, 94, 0.3)"; // green-500 with opacity

    for (let i = 0; i < barCount; i++) {
      const dataIndex = Math.floor((i / barCount) * audioData.length);
      const barHeight = Math.abs(audioData[dataIndex]) * height * 0.8;
      const x = i * (barWidth + 2);
      const y = height - barHeight;
      ctx.fillRect(x, y, barWidth, barHeight);
    }

    // Draw waveform on top
    ctx.strokeStyle = "#22c55e"; // green-500
    ctx.lineWidth = 2;
    ctx.beginPath();

    const sliceWidth = width / audioData.length;
    let x = 0;

    for (let i = 0; i < audioData.length; i++) {
      const v = audioData[i];
      const y = (v * height) / 2 + height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.stroke();
  }, [audioData, isListening]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={150}
      className="w-full max-w-full rounded-lg border border-border"
      style={{ maxWidth: "600px" }}
    />
  );
}

function App() {
  const [playingSound, setPlayingSound] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [audioData, setAudioData] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);

  const playSound = async (sound: string) => {
    setPlayingSound(sound);
    setError(null);
    try {
      const response = await fetch(`/api/play/${sound}`, { method: "POST" });
      const result = await response.json();
      if (!result.success) {
        setError(result.error);
      }
    } catch (err) {
      setError("Failed to play sound");
      console.error("Play sound error:", err);
    } finally {
      setPlayingSound(null);
    }
  };

  const startListening = useCallback(() => {
    setError(null);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/microphone`);
    wsRef.current = ws;

    // Create audio context for playback
    audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    nextPlayTimeRef.current = 0;

    ws.onopen = () => {
      setIsListening(true);
    };

    ws.onmessage = async (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === "audio") {
          // Decode base64 audio data
          const binaryString = atob(message.data);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }

          // Convert to Int16 samples
          const samples = new Int16Array(bytes.buffer);

          // Normalize to -1 to 1 range for visualization
          const normalizedData = Array.from(samples).map((s) => s / 32768);
          setAudioData(normalizedData);

          // Play audio in browser with proper scheduling
          const ctx = audioContextRef.current;
          if (ctx && ctx.state === "running") {
            const sampleRate = message.sampleRate || 16000;
            const audioBuffer = ctx.createBuffer(1, samples.length, sampleRate);
            const channelData = audioBuffer.getChannelData(0);
            for (let i = 0; i < samples.length; i++) {
              channelData[i] = samples[i] / 32768;
            }

            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            // Schedule this chunk to play right after the previous one
            const currentTime = ctx.currentTime;
            const startTime = Math.max(currentTime, nextPlayTimeRef.current);
            source.start(startTime);

            // Update next play time to end of this buffer
            nextPlayTimeRef.current = startTime + audioBuffer.duration;

            // Reset if we've fallen too far behind (catches up to live)
            if (nextPlayTimeRef.current - currentTime > 0.5) {
              nextPlayTimeRef.current = currentTime + 0.05;
            }
          }
        } else if (message.type === "error") {
          setError(message.message);
        }
      } catch (err) {
        console.error("Error processing audio message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setError("WebSocket connection error");
      setIsListening(false);
    };

    ws.onclose = () => {
      setIsListening(false);
      setAudioData([]);
    };
  }, []);

  const stopListening = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    nextPlayTimeRef.current = 0;
    setIsListening(false);
    setAudioData([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <div className="relative flex min-h-screen w-full flex-col items-center overflow-hidden bg-background">
      <div className="relative flex h-[300px] w-full flex-col items-center justify-center">
        <ShaderAnimation />
        <span className="absolute pointer-events-none z-10 text-center text-6xl leading-none font-semibold tracking-tighter whitespace-pre-wrap text-foreground">
          Audio Demo
        </span>
      </div>

      <div className="z-10 flex flex-col items-center gap-8 p-8 w-full max-w-2xl">
        {/* Sound Buttons */}
        <div className="flex flex-col items-center gap-4">
          <h2 className="text-xl font-semibold text-foreground">Play Sounds on Jetson</h2>
          <div className="flex gap-4">
            <Button
              onClick={() => playSound("cat")}
              disabled={playingSound !== null}
              size="lg"
              variant="secondary"
              className="text-lg gap-2"
            >
              <Cat className="w-5 h-5" />
              {playingSound === "cat" ? "Playing..." : "Cat"}
            </Button>
            <Button
              onClick={() => playSound("dog")}
              disabled={playingSound !== null}
              size="lg"
              variant="secondary"
              className="text-lg gap-2"
            >
              <Dog className="w-5 h-5" />
              {playingSound === "dog" ? "Playing..." : "Dog"}
            </Button>
            <Button
              onClick={() => playSound("cow")}
              disabled={playingSound !== null}
              size="lg"
              variant="secondary"
              className="text-lg gap-2"
            >
              <CowIcon className="w-5 h-5" />
              {playingSound === "cow" ? "Playing..." : "Cow"}
            </Button>
          </div>
        </div>

        {/* Microphone Section */}
        <div className="flex flex-col items-center gap-4 w-full">
          <h2 className="text-xl font-semibold text-foreground">Microphone Stream</h2>
          <Button
            onClick={isListening ? stopListening : startListening}
            size="lg"
            variant={isListening ? "destructive" : "secondary"}
            className="text-lg gap-2"
          >
            {isListening ? (
              <>
                <MicOff className="w-5 h-5" />
                Stop Listening
              </>
            ) : (
              <>
                <Mic className="w-5 h-5" />
                Listen
              </>
            )}
          </Button>

          {/* Audio Visualizer */}
          <div className="w-full bg-muted rounded-lg p-4">
            <AudioVisualizer audioData={audioData} isListening={isListening} />
            <p className="text-muted-foreground text-sm mt-2 text-center">
              {isListening
                ? "Streaming audio from Jetson microphone..."
                : "Click Listen to stream microphone audio"}
            </p>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="w-full bg-destructive/20 border border-destructive/50 rounded-lg p-4">
            <p className="text-destructive text-center">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
