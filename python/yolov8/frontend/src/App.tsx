import { useState, useEffect, useRef } from "react";

interface Detection {
  label: string;
  confidence: number;
  timestamp: string;
}

type ConnectionStatus = "connected" | "reconnecting";

function App() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const logRef = useRef<HTMLDivElement>(null);
  const [apiStatus, setApiStatus] = useState<ConnectionStatus>("reconnecting");
  const [apiError, setApiError] = useState<string | null>(null);
  const [apiRetryMs, setApiRetryMs] = useState<number | null>(null);
  const [videoStatus, setVideoStatus] = useState<ConnectionStatus>("reconnecting");
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoRetryMs, setVideoRetryMs] = useState<number | null>(null);
  const [videoSrc, setVideoSrc] = useState(
    `/api/video-feed?ts=${Date.now()}`
  );
  const isMountedRef = useRef(true);
  const videoRetryRef = useRef(500);
  const videoTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pollIntervalMs = 500;
  const pollMaxBackoffMs = 8000;
  const videoMaxBackoffMs = 8000;

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (videoTimeoutRef.current) {
        clearTimeout(videoTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    let isCancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let retryDelay = pollIntervalMs;

    const scheduleNext = (delay: number) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      timeoutId = setTimeout(fetchDetections, delay);
    };

    const fetchDetections = async () => {
      try {
        const response = await fetch("/api/detections", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data: Detection[] = await response.json();
        if (isCancelled) {
          return;
        }
        setDetections(data);
        setApiStatus("connected");
        setApiError(null);
        setApiRetryMs(null);
        retryDelay = pollIntervalMs;
        scheduleNext(pollIntervalMs);
      } catch (error) {
        if (isCancelled) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        console.error("Failed to fetch detections:", error);
        setApiStatus("reconnecting");
        setApiError(message);
        setApiRetryMs(retryDelay);
        scheduleNext(retryDelay);
        retryDelay = Math.min(retryDelay * 2, pollMaxBackoffMs);
      }
    };

    fetchDetections();
    return () => {
      isCancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  const scheduleVideoReconnect = (message: string) => {
    if (!isMountedRef.current) {
      return;
    }
    if (videoTimeoutRef.current) {
      clearTimeout(videoTimeoutRef.current);
    }
    setVideoStatus("reconnecting");
    setVideoError(message);
    const delay = videoRetryRef.current;
    setVideoRetryMs(delay);
    videoTimeoutRef.current = setTimeout(() => {
      if (!isMountedRef.current) {
        return;
      }
      setVideoSrc(`/api/video-feed?ts=${Date.now()}`);
    }, delay);
    videoRetryRef.current = Math.min(delay * 2, videoMaxBackoffMs);
  };

  const handleVideoLoad = () => {
    if (!isMountedRef.current) {
      return;
    }
    setVideoStatus("connected");
    setVideoError(null);
    setVideoRetryMs(null);
    videoRetryRef.current = 500;
  };

  const handleVideoError = () => {
    scheduleVideoReconnect(
      "Video feed disconnected. Attempting to reconnect..."
    );
  };

  useEffect(() => {
    // Auto-scroll to bottom of log
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [detections]);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      <div className="absolute left-4 top-4 z-20 flex flex-col gap-2 text-xs font-mono">
        <div className="rounded-md bg-black/70 px-3 py-2 text-white/80 shadow">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                apiStatus === "connected" ? "bg-green-400" : "bg-yellow-400"
              }`}
            />
            <span>
              Backend: {apiStatus === "connected" ? "connected" : "disconnected"}
            </span>
          </div>
          {apiStatus !== "connected" && (
            <div className="mt-1 text-white/60">
              Disconnected... attempting reconnect
              {apiRetryMs ? ` in ${Math.ceil(apiRetryMs / 1000)}s` : ""}.
              {apiError && (
                <div className="text-red-300 break-all">{apiError}</div>
              )}
            </div>
          )}
        </div>
        <div className="rounded-md bg-black/70 px-3 py-2 text-white/80 shadow">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                videoStatus === "connected" ? "bg-green-400" : "bg-yellow-400"
              }`}
            />
            <span>
              Video: {videoStatus === "connected" ? "connected" : "disconnected"}
            </span>
          </div>
          {videoStatus !== "connected" && (
            <div className="mt-1 text-white/60">
              Disconnected... attempting reconnect
              {videoRetryMs ? ` in ${Math.ceil(videoRetryMs / 1000)}s` : ""}.
              {videoError && (
                <div className="text-red-300 break-all">{videoError}</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Full-screen video feed */}
      <img
        src={videoSrc}
        alt="YOLOv8 Detection Feed"
        className="absolute inset-0 h-full w-full object-contain"
        onLoad={handleVideoLoad}
        onError={handleVideoError}
      />

      {/* Bottom overlay for detection log */}
      <div className="absolute bottom-0 left-0 right-0 h-1/5 bg-black/70 backdrop-blur-sm z-10">
        <div className="h-full flex flex-col">
          <div className="px-4 py-2 border-b border-white/20 flex items-center justify-between">
            <h2 className="text-white text-sm font-semibold tracking-wide uppercase">
              COCO Object Detections
            </h2>
            <span
              className={`text-xs font-mono ${
                apiStatus === "connected" ? "text-green-400" : "text-yellow-300"
              }`}
            >
              {apiStatus === "connected" ? "connected" : "reconnecting"}
            </span>
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-y-auto px-4 py-2 font-mono text-sm"
          >
            {apiStatus !== "connected" ? (
              <p className="text-white/60">
                Disconnected from backend... attempting reconnect.
              </p>
            ) : detections.length === 0 ? (
              <p className="text-white/50">Waiting for detections...</p>
            ) : (
              detections.map((detection, index) => (
                <div
                  key={`${detection.timestamp}-${index}`}
                  className="text-white/90 py-0.5"
                >
                  <span className="text-green-400">[{detection.confidence}%]</span>{" "}
                  <span className="text-yellow-300">{detection.label}</span>{" "}
                  <span className="text-white/50 text-xs">
                    {new Date(detection.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
