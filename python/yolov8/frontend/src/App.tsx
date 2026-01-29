import { useState, useEffect, useRef } from "react";

interface Detection {
  label: string;
  confidence: number;
  timestamp: string;
}

function App() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Poll for detections
    const fetchDetections = async () => {
      try {
        const response = await fetch("/api/detections");
        const data: Detection[] = await response.json();
        setDetections(data);
      } catch (error) {
        console.error("Failed to fetch detections:", error);
      }
    };

    fetchDetections();
    const interval = setInterval(fetchDetections, 500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom of log
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [detections]);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Full-screen video feed */}
      <img
        src="/api/video-feed"
        alt="YOLOv8 Detection Feed"
        className="absolute inset-0 h-full w-full object-contain"
      />

      {/* Bottom overlay for detection log */}
      <div className="absolute bottom-0 left-0 right-0 h-1/5 bg-black/70 backdrop-blur-sm z-10">
        <div className="h-full flex flex-col">
          <div className="px-4 py-2 border-b border-white/20">
            <h2 className="text-white text-sm font-semibold tracking-wide uppercase">
              COCO Object Detections
            </h2>
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-y-auto px-4 py-2 font-mono text-sm"
          >
            {detections.length === 0 ? (
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
