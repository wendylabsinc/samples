"""Parent-side orchestration of the two camera capture children.

``CameraManager`` owns one child process per benchmark slot (``usb`` / ``csi``),
drains each child's frame queue on the event loop, fans frames out to that slot's
WebSocket clients, samples per-PID CPU/RSS, and exposes a metrics snapshot.

Key choices:
- ``spawn`` start method (``fork`` + GLib/GStreamer is unsafe).
- Frames travel as pre-encoded JPEG over a bounded ``multiprocessing.Queue``; the
  drain runs ``queue.get`` in a thread executor and processes the result back on
  the loop thread, so the per-client ``asyncio.Queue`` fan-out is loop-thread-safe.
- A failed real pipeline (e.g. libcamera missing) is auto-replaced by a synthetic
  source, so the comparison view always has two live panels.
"""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import queue as _queue
import struct
import time

from . import enumerate as cam_enum
from . import metrics as metrics_mod
from .capture_worker import run_worker

logger = logging.getLogger(__name__)

_MAGIC = b"WCB1"
_GET_TIMEOUT = 0.5
_EMPTY = object()


def _frame_payload(meta: dict, jpeg: bytes) -> bytes:
    meta_b = json.dumps(meta, separators=(",", ":")).encode()
    return _MAGIC + struct.pack("<H", len(meta_b)) + meta_b + jpeg


def _blocking_get(q):
    try:
        return q.get(timeout=_GET_TIMEOUT)
    except _queue.Empty:
        return _EMPTY


class _Slot:
    def __init__(self, kind: str, cfg: dict):
        self.kind = kind
        self.cfg = cfg
        self.proc: mp.Process | None = None
        self.queue = None
        self.sampler: metrics_mod.ProcessSampler | None = None
        self.clients: set[asyncio.Queue] = set()
        self.drain_task: asyncio.Task | None = None
        self.stopping = False
        self.latency = metrics_mod.Rolling()
        self.metrics = self._blank_metrics(cfg)

    @staticmethod
    def _blank_metrics(cfg: dict) -> dict:
        return {
            "kind": cfg["kind"],
            "label": cfg["label"],
            "name": cfg.get("name"),
            "mode": cfg["mode"],
            "device": cfg.get("device"),
            "synthetic": cfg.get("synthetic", cfg["mode"] == "synthetic"),
            "online": False,
            "resolution": None,
            "format": None,
            "server_fps": 0.0,
            "startup_ms": None,
            "cpu_pct": None,
            "rss_mb": None,
            "sharpness": None,
            "brightness": None,
        }


class CameraManager:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self.slots: dict[str, _Slot] = {}
        self.board: dict = {"available": False, "power_w": None}
        self._sampler_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ start
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        cfgs = cam_enum.assign_slots()
        for kind, cfg in cfgs.items():
            self.slots[kind] = _Slot(kind, cfg)
            await self._spawn(self.slots[kind])
        self._sampler_task = asyncio.create_task(self._sample_loop())
        logger.info("CameraManager started: %s", {k: s.cfg["mode"] for k, s in self.slots.items()})

    async def _spawn(self, slot: _Slot) -> None:
        slot.stopping = False
        slot.queue = self._ctx.Queue(maxsize=8)
        slot.proc = self._ctx.Process(
            target=run_worker, args=(slot.cfg, slot.queue), daemon=True,
            name=f"capture-{slot.kind}",
        )
        slot.proc.start()
        slot.sampler = metrics_mod.ProcessSampler(slot.proc.pid)
        slot.metrics = _Slot._blank_metrics(slot.cfg)
        slot.latency = metrics_mod.Rolling()
        slot.drain_task = asyncio.create_task(self._drain(slot))
        logger.info("[%s] spawned pid=%s mode=%s device=%s",
                    slot.kind, slot.proc.pid, slot.cfg["mode"], slot.cfg.get("device"))

    # ------------------------------------------------------------------ drain
    async def _drain(self, slot: _Slot) -> None:
        loop = asyncio.get_running_loop()
        while not slot.stopping:
            msg = await loop.run_in_executor(None, _blocking_get, slot.queue)
            if msg is _EMPTY:
                continue
            try:
                self._handle(slot, msg)
            except Exception as exc:  # never let one bad message kill the drain
                logger.debug("[%s] handle error: %s", slot.kind, exc)

    def _handle(self, slot: _Slot, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "frame":
            slot.metrics["online"] = True
            recv_mono = time.monotonic_ns()
            cap = msg.get("cap_mono_ns")
            if cap:
                slot.latency.add((recv_mono - cap) / 1e6)
            meta = {
                "seq": msg["seq"],
                "send_ts_ms": time.time() * 1000.0,
                "src_fps": slot.metrics["server_fps"],
                "w": msg.get("w", 0),
                "h": msg.get("h", 0),
                "fmt": msg.get("fmt", "MJPEG"),
            }
            payload = _frame_payload(meta, msg["jpeg"])
            for q in slot.clients:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
        elif kind == "stat":
            slot.metrics["server_fps"] = msg.get("src_fps", slot.metrics["server_fps"])
            if msg.get("sharpness") is not None:
                slot.metrics["sharpness"] = msg["sharpness"]
                slot.metrics["brightness"] = msg["brightness"]
        elif kind == "startup":
            slot.metrics.update({
                "online": True,
                "startup_ms": msg.get("startup_ms"),
                "resolution": msg.get("resolution"),
                "format": msg.get("format"),
                "synthetic": msg.get("synthetic", slot.metrics["synthetic"]),
                "mode": msg.get("mode", slot.metrics["mode"]),
                "device": msg.get("device", slot.metrics["device"]),
            })
        elif kind == "started":
            slot.metrics["online"] = True
        elif kind == "error":
            logger.warning("[%s] capture error (mode=%s)", slot.kind, slot.cfg["mode"])
            slot.metrics["online"] = False
            if slot.cfg["mode"] != "synthetic":
                # Fall back to a synthetic source so the panel stays alive.
                asyncio.create_task(self._fallback_synthetic(slot.kind))

    # --------------------------------------------------------------- sampling
    async def _sample_loop(self) -> None:
        while True:
            for slot in self.slots.values():
                if slot.sampler is not None:
                    cpu, rss = slot.sampler.sample()
                    slot.metrics["cpu_pct"] = cpu
                    slot.metrics["rss_mb"] = rss
            self.board = metrics_mod.read_board_power()
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------ lifecycle
    async def _stop_slot(self, slot: _Slot) -> None:
        slot.stopping = True
        if slot.drain_task:
            try:
                await asyncio.wait_for(slot.drain_task, timeout=_GET_TIMEOUT + 1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                slot.drain_task.cancel()
        if slot.proc and slot.proc.is_alive():
            slot.proc.terminate()
            slot.proc.join(timeout=3)
            if slot.proc.is_alive():
                slot.proc.kill()
        if slot.queue is not None:
            slot.queue.close()

    async def _fallback_synthetic(self, kind: str) -> None:
        slot = self.slots[kind]
        await self._stop_slot(slot)
        slot.cfg = cam_enum._synthetic(kind)
        await self._spawn(slot)
        logger.info("[%s] fell back to synthetic source", kind)

    async def restart(self) -> dict:
        """Tear down and respawn both children (re-measures startup time)."""
        # Re-detect so a freshly attached camera is picked up on restart.
        cfgs = cam_enum.assign_slots()
        for kind, slot in self.slots.items():
            await self._stop_slot(slot)
            slot.cfg = cfgs.get(kind, slot.cfg)
            await self._spawn(slot)
        logger.info("Restarted all cameras")
        return {k: s.cfg["mode"] for k, s in self.slots.items()}

    async def shutdown(self) -> None:
        if self._sampler_task:
            self._sampler_task.cancel()
        for slot in self.slots.values():
            await self._stop_slot(slot)

    # ------------------------------------------------------------- clients
    def register(self, kind: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self.slots[kind].clients.add(q)
        return q

    def unregister(self, kind: str, q: asyncio.Queue) -> None:
        self.slots[kind].clients.discard(q)

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        cams = {}
        for kind, slot in self.slots.items():
            m = dict(slot.metrics)
            m["pipeline_latency_ms"] = slot.latency.summary()
            cams[kind] = m
        return {"ts": time.time() * 1000.0, "board": self.board, "cameras": cams}
