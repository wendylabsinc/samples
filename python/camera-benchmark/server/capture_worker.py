"""Capture child process: one GStreamer pipeline per camera.

Each camera (USB or CSI) runs in its OWN process so the parent can attribute
CPU% and RSS to it precisely via ``psutil.Process(child.pid)``. The child:

- builds a GStreamer pipeline for its ``mode`` (``v4l2`` | ``libcamera`` | ``synthetic``),
- runs a GLib main loop with a JPEG ``appsink``,
- timestamps every frame and ships ``{type: "frame", ...}`` messages (plus periodic
  ``stat``/one-shot ``startup``) over a ``multiprocessing.Queue`` to the parent,
- measures startup time (pipeline bring-up → first frame) and a rolling capture FPS,
- samples image-quality metrics ~1 Hz.

GStreamer / GLib are imported lazily inside ``run_worker`` (the child entry point),
never at module import time, so the parent can import ``run_worker`` without pulling
GStreamer into its address space — keeping its per-PID accounting clean. The process
is started with the ``spawn`` method (``fork`` + GLib is unsafe).
"""
from __future__ import annotations

import logging
import signal
import time

logger = logging.getLogger(__name__)

JPEG_QUALITY = 70
QUALITY_INTERVAL_S = 1.0  # how often to recompute fps + image-quality

_APPSINK = "appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"

# Populated by run_worker() in the child only.
Gst = None
GLib = None
imagequality = None


def _candidate_pipelines(mode: str, device, width: int, height: int,
                         pattern: str, label: str) -> list[str]:
    """Ordered list of pipeline strings to try, most-preferred first."""
    if mode == "v4l2":
        dev = device or "/dev/video0"
        src = f"v4l2src device={dev}"
        return [
            # USB webcams usually emit MJPEG natively — zero re-encode.
            f"{src} ! image/jpeg ! {_APPSINK}",
            f"{src} ! image/jpeg,width=640,height=480 ! {_APPSINK}",
            # Raw sensor → encode to JPEG.
            f"{src} ! videoconvert ! jpegenc quality={JPEG_QUALITY} ! {_APPSINK}",
        ]
    if mode == "libcamera":
        cam = f"camera-name={device}" if device else ""
        src = f"libcamerasrc {cam}".strip()
        w = width or 1280
        h = height or 720
        # CSI sensors emit raw Bayer/RGB → libcamera ISP → encode to JPEG.
        return [
            f"{src} ! video/x-raw,width={w},height={h} ! videoconvert ! jpegenc quality={JPEG_QUALITY} ! {_APPSINK}",
            f"{src} ! videoconvert ! jpegenc quality={JPEG_QUALITY} ! {_APPSINK}",
        ]
    # synthetic — no hardware, no libcamera required.
    w = width or 640
    h = height or 480
    overlay = (
        f'textoverlay text="{label}" valignment=top halignment=left '
        f'font-desc="Sans, 22" shaded-background=true'
    )
    return [
        f"videotestsrc pattern={pattern} is-live=true "
        f"! video/x-raw,width={w},height={h},framerate=30/1 "
        f"! {overlay} ! videoconvert ! jpegenc quality={JPEG_QUALITY} ! {_APPSINK}"
    ]


class _CaptureWorker:
    def __init__(self, cfg: dict, frame_queue):
        self.kind: str = cfg["kind"]
        self.mode: str = cfg["mode"]
        self.device = cfg.get("device")
        self.width = int(cfg.get("width") or 0)
        self.height = int(cfg.get("height") or 0)
        self.pattern = cfg.get("pattern", "smpte")
        self.label = cfg.get("label", self.kind.upper())
        self.q = frame_queue

        self.loop = GLib.MainLoop()
        self.pipeline = None
        self.seq = 0
        self.start_ns: int | None = None
        self.first_frame = False
        self.resolution: str | None = None
        self.fmt = "MJPEG"
        self._w = 0
        self._h = 0
        self._fps_count = 0
        self._fps_t0 = time.monotonic()

    # -- IPC ---------------------------------------------------------------
    def _send(self, msg: dict) -> None:
        try:
            self.q.put_nowait(msg)
        except Exception:
            # Queue full (parent briefly behind) → drop. Frames are lossy by
            # design; stats recur; startup is re-derivable from first frame.
            pass

    # -- pipeline ----------------------------------------------------------
    def _build(self):
        # Start the clock before bring-up so startup_ms includes camera-open cost.
        self.start_ns = time.monotonic_ns()
        for p_str in _candidate_pipelines(
            self.mode, self.device, self.width, self.height, self.pattern, self.label
        ):
            try:
                pipeline = Gst.parse_launch(p_str)
            except Exception as exc:
                logger.info("[%s] pipeline parse failed: %s — %s", self.kind, p_str, exc)
                continue
            ret = pipeline.set_state(Gst.State.PAUSED)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                logger.info("[%s] pipeline failed: %s", self.kind, p_str)
                continue
            if ret == Gst.StateChangeReturn.ASYNC:
                ret, _, _ = pipeline.get_state(5 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.FAILURE:
                    pipeline.set_state(Gst.State.NULL)
                    logger.info("[%s] pipeline preroll failed: %s", self.kind, p_str)
                    continue
            logger.info("[%s] pipeline ready: %s", self.kind, p_str)
            return pipeline
        return None

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        data = bytes(mi.data)
        buf.unmap(mi)

        cap_mono_ns = time.monotonic_ns()
        self.seq += 1

        if self.resolution is None:
            caps = sample.get_caps()
            if caps and caps.get_size() > 0:
                st = caps.get_structure(0)
                okw, w = st.get_int("width")
                okh, h = st.get_int("height")
                self._w = w if okw else 0
                self._h = h if okh else 0
                self.resolution = f"{w}x{h}" if okw and okh else "unknown"
                self.fmt = "MJPEG" if "jpeg" in st.get_name() else st.get_name()

        if not self.first_frame:
            self.first_frame = True
            self._send({
                "type": "startup",
                "startup_ms": round((cap_mono_ns - self.start_ns) / 1e6, 1),
                "resolution": self.resolution or "unknown",
                "format": self.fmt,
                "synthetic": self.mode == "synthetic",
                "mode": self.mode,
                "device": self.device,
            })

        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= QUALITY_INTERVAL_S:
            stat = {"type": "stat", "src_fps": round(self._fps_count / elapsed, 1)}
            sharp, bright = imagequality.measure(data)
            if sharp is not None:
                stat["sharpness"] = sharp
                stat["brightness"] = bright
            self._send(stat)
            self._fps_count = 0
            self._fps_t0 = now

        self._send({
            "type": "frame",
            "seq": self.seq,
            "cap_mono_ns": cap_mono_ns,
            "w": self._w,
            "h": self._h,
            "fmt": self.fmt,
            "jpeg": data,
        })
        return Gst.FlowReturn.OK

    # -- lifecycle ---------------------------------------------------------
    def _stop(self, *_):
        if self.loop.is_running():
            self.loop.quit()
        return GLib.SOURCE_REMOVE

    def run(self):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, self._stop)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, self._stop)

        self.pipeline = self._build()
        if self.pipeline is None:
            logger.warning("[%s] no working pipeline for mode=%s", self.kind, self.mode)
            self._send({"type": "error", "kind": self.kind, "mode": self.mode})
            return

        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)
        self._send({"type": "started", "mode": self.mode,
                    "synthetic": self.mode == "synthetic"})
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)


def run_worker(cfg: dict, frame_queue) -> None:
    """Process entry point (must be a top-level, picklable callable for spawn).

    All GStreamer imports happen here so the parent never loads GStreamer.
    """
    global Gst, GLib, imagequality
    logging.basicConfig(level=logging.INFO)

    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib as _GLib
    from gi.repository import Gst as _Gst

    from . import imagequality as _iq

    Gst, GLib, imagequality = _Gst, _GLib, _iq
    Gst.init(None)
    _CaptureWorker(cfg, frame_queue).run()
