"""Floor calibrations: taking them, keeping them, and saying what state they are in.

One FloorCalibrationManager per web process. The ROS executor thread offers
it every depth frame's points (observe); an HTTP handler thread or the
startup thread asks it for a calibration (calibrate), which waits for the
next frames, fits, validates and installs. Geometry lives in floor_model.py;
this module adds the clock, the threads and the file. No ROS imports, and
the clock, sleep and store are injected so the tests drive it directly.

Why a startup calibration must match a reference height, and why a camera
that moves is reported rather than silently re-learned: see
docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from floor_model import (
    NO_REFERENCE_REASON,
    CalibrationLimits,
    CameraIntrinsics,
    FloorFit,
    FloorPlane,
    HealthConfig,
    HealthMonitor,
    fit_floor,
    validate_calibration,
)

FILE_VERSION = 1
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# Everything a malformed calibration entry can raise while it is parsed:
# json accepts Infinity, and int() of it overflows.
CORRUPT_ERRORS = (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError)


def iso_utc(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).strftime(ISO_FORMAT)


def parse_iso_utc(text: str) -> float:
    return datetime.strptime(text, ISO_FORMAT).replace(tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True)
class Calibration:
    plane: FloorPlane
    reference_height_m: float
    source: str
    created_at: float
    inliers: int
    inlier_ratio: float
    floor_span_m: tuple[float, float]

    @classmethod
    def from_fit(cls, fit: FloorFit, reference_height_m: float, source: str, created_at: float) -> "Calibration":
        return cls(fit.plane, reference_height_m, source, created_at, fit.inliers, fit.inlier_ratio, fit.floor_span_m)

    def to_json(self) -> dict:
        return {
            "plane": {"normal": [round(v, 6) for v in self.plane.normal], "offset_m": round(self.plane.offset_m, 4)},
            "height_m": round(self.plane.height_m, 3),
            "pitch_deg": round(self.plane.pitch_deg, 1),
            "roll_deg": round(self.plane.roll_deg, 1),
            "reference_height_m": round(self.reference_height_m, 3),
            "source": self.source,
            "created_at": iso_utc(self.created_at),
            "inliers": int(self.inliers),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "floor_span_m": [round(self.floor_span_m[0], 3), round(self.floor_span_m[1], 3)],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Calibration":
        """Raises one of CORRUPT_ERRORS on anything malformed."""
        plane = FloorPlane.from_normal_offset(data["plane"]["normal"], float(data["plane"]["offset_m"]))
        reference = float(data["reference_height_m"])
        source = str(data["source"])
        if source not in {"operator", "startup"}:
            raise ValueError(f"unknown source {source!r}")
        span = data["floor_span_m"]
        return cls(
            plane,
            reference,
            source,
            parse_iso_utc(str(data["created_at"])),
            int(data["inliers"]),
            float(data["inlier_ratio"]),
            (float(span[0]), float(span[1])),
        )


class CalibrationStore:
    """One JSON file holding a calibration per camera, written atomically."""

    def __init__(self, path: Path | str, log=print) -> None:
        self.path = Path(path)
        self._log = log

    def load(self) -> dict[str, Calibration]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._log(f"FLOOR_CALIBRATION_UNREADABLE path={self.path} {type(exc).__name__}: {exc}")
            return {}
        try:
            data = json.loads(raw)
            if data.get("version") != FILE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            cameras = data["cameras"]
            if not isinstance(cameras, dict):
                raise TypeError("cameras is not an object")
        except CORRUPT_ERRORS as exc:
            self._log(f"FLOOR_CALIBRATION_CORRUPT path={self.path} {type(exc).__name__}: {exc}")
            return {}
        loaded = {}
        for camera, entry in cameras.items():
            try:
                loaded[str(camera)] = Calibration.from_json(entry)
            except CORRUPT_ERRORS as exc:
                self._log(f"FLOOR_CALIBRATION_CORRUPT path={self.path} camera={camera} {type(exc).__name__}: {exc}")
        return loaded

    def save(self, calibrations: dict[str, Calibration]) -> bool:
        """Write every camera's calibration; False when the file could not be written.

        The directory is never created here. It is the persist volume's mount
        point, so a missing one means the volume is missing, and writing into
        the container's own filesystem would report "saved" for a file the
        next restart throws away.
        """
        body = json.dumps(
            {"version": FILE_VERSION, "cameras": {name: cal.to_json() for name, cal in sorted(calibrations.items())}},
            indent=2,
            sort_keys=True,
        )
        try:
            fd, temp = tempfile.mkstemp(prefix=".floor_calibration.", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
                raise
        except OSError as exc:
            self._log(f"FLOOR_CALIBRATION_NOT_SAVED path={self.path} {type(exc).__name__}: {exc}")
            return False
        return True


@dataclass(frozen=True)
class CalibrationSettings:
    frames: int = 10
    min_depth_m: float = 0.2
    max_depth_m: float = 3.0
    inlier_m: float = 0.015
    collect_timeout_s: float = 2.5
    # How long a second request waits for a calibration already running.
    # With collect_timeout_s and a fit, it keeps the worst-case Recalibrate
    # under the page's 4 s fetch timeout.
    busy_timeout_s: float = 1.0
    health_period_s: float = 0.5
    startup_retry_s: float = 10.0
    startup_window_s: float = 600.0
    limits: CalibrationLimits = field(default_factory=CalibrationLimits)
    health: HealthConfig = field(default_factory=HealthConfig)


class _Collection:
    def __init__(self, camera: str, frames: int) -> None:
        self.camera = camera
        self.frames = frames
        self.points: list[np.ndarray] = []
        self.done = threading.Event()


class FloorCalibrationManager:
    def __init__(
        self,
        store: CalibrationStore,
        settings: CalibrationSettings = CalibrationSettings(),
        clock=time.monotonic,
        wall_clock=time.time,
        sleep=time.sleep,
        log=print,
    ) -> None:
        self._store = store
        self._settings = settings
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._log = log
        self._lock = threading.Lock()
        # One calibration run at a time, whoever asked for it.
        self._run_lock = threading.Lock()
        self._calibrations = store.load()
        self._saved = {camera: True for camera in self._calibrations}
        self._health = {camera: HealthMonitor(cal.plane, settings.health) for camera, cal in self._calibrations.items()}
        self._health_at: dict[str, float] = {}
        self._intrinsics: dict[str, CameraIntrinsics] = {}
        self._last_result: dict[str, dict] = {}
        self._accepted: set[str] = set()
        self._startup_closed = False
        self._collection: _Collection | None = None
        self._calibrating: dict[str, str] = {}

    # The executor thread's side: cheap, and never waits on a calibration.

    def set_intrinsics(self, camera: str, intrinsics: CameraIntrinsics) -> None:
        with self._lock:
            self._intrinsics[camera] = intrinsics

    def intrinsics(self, camera: str) -> CameraIntrinsics | None:
        with self._lock:
            return self._intrinsics.get(camera)

    def plane(self, camera: str) -> FloorPlane | None:
        with self._lock:
            calibration = self._calibrations.get(camera)
        return calibration.plane if calibration else None

    def observe(self, camera: str, points: np.ndarray) -> None:
        """One depth frame's points: feed a run in progress, and the health check when it is due."""
        now = self._clock()
        s = self._settings
        with self._lock:
            collection = self._collection
            if collection is not None and collection.camera == camera and not collection.done.is_set():
                depth = points[:, 2]
                collection.points.append(points[(depth >= s.min_depth_m) & (depth <= s.max_depth_m)])
                if len(collection.points) >= collection.frames:
                    collection.done.set()
            monitor = self._health.get(camera)
            due = monitor is not None and now - self._health_at.get(camera, float("-inf")) >= s.health_period_s
            if due:
                self._health_at[camera] = now
        if due:
            before = monitor.state
            after = monitor.update(points)
            if after == "stale" and before != "stale":
                self._log(
                    f"FLOOR_CALIBRATION_STALE camera={camera} angle_deg={monitor.last_angle_deg} "
                    f"height_diff_m={monitor.last_height_diff_m}"
                )

    # Anyone's side.

    def status(self, camera: str) -> dict:
        now = self._wall_clock()
        with self._lock:
            calibration = self._calibrations.get(camera)
            monitor = self._health.get(camera)
            has_intrinsics = camera in self._intrinsics
            calibrating = self._calibrating.get(camera)
            last = dict(self._last_result[camera]) if camera in self._last_result else None
            saved = self._saved.get(camera)
        health = monitor.state if monitor else None
        if not has_intrinsics:
            state = "no_camera_info"
        elif calibrating == "operator":
            state = "calibrating"
        elif calibration is None:
            state = "missing"
        elif health == "stale":
            state = "stale"
        else:
            state = "ok"
        plane = calibration.plane if calibration else None
        return {
            "camera": camera,
            "state": state,
            "calibrated": calibration is not None,
            "health": health,
            "usable": has_intrinsics and calibration is not None and health != "stale",
            "height_m": round(plane.height_m, 3) if plane else None,
            "pitch_deg": round(plane.pitch_deg, 1) if plane else None,
            "roll_deg": round(plane.roll_deg, 1) if plane else None,
            "reference_height_m": round(calibration.reference_height_m, 3) if calibration else None,
            "source": calibration.source if calibration else None,
            "created_at": iso_utc(calibration.created_at) if calibration else None,
            "age_s": round(max(0.0, now - calibration.created_at), 1) if calibration else None,
            "saved": saved if calibration else None,
            "health_angle_deg": monitor.last_angle_deg if monitor else None,
            "health_height_diff_m": monitor.last_height_diff_m if monitor else None,
            "last_result": last,
        }

    def end_startup_window(self) -> None:
        """No more startup calibrations in this process: the car has driven.

        A startup calibration is for adjustments made while the car was off.
        Once it has moved, a floor that no longer matches means the camera
        moved mid-session, and that is reported (stale) and left for an
        operator's Recalibrate, never re-learned without anyone asking.
        """
        with self._lock:
            self._startup_closed = True

    def calibrate(self, camera: str, source: str) -> dict:
        """Take a calibration from this camera's next frames: {accepted, reason, calibration}.

        Blocks the calling thread for up to collect_timeout_s while the frames
        arrive, then fits on it, so an HTTP handler thread or the startup
        thread pays for the fit and the ROS executor never does.
        """
        s = self._settings
        if not self._run_lock.acquire(timeout=s.busy_timeout_s):
            return {"accepted": False, "reason": "a calibration is already running", "calibration": self.status(camera)}
        try:
            with self._lock:
                existing = self._calibrations.get(camera)
                has_intrinsics = camera in self._intrinsics
                skipped = source == "startup" and (self._startup_closed or camera in self._accepted)
            if skipped:
                # Checked under the run lock, not only in run_startup: an
                # attempt that queued behind an operator's Recalibrate must not
                # run over it. Nothing is recorded, so the operator's result
                # stays the last one.
                return {
                    "accepted": False,
                    "skipped": True,
                    "reason": "startup calibration not needed",
                    "calibration": self.status(camera),
                }
            reference = existing.reference_height_m if existing else None
            if source == "startup" and reference is None:
                return self._finish(camera, source, False, NO_REFERENCE_REASON)
            if not has_intrinsics:
                return self._finish(camera, source, False, "waiting for depth camera info")
            collection = _Collection(camera, s.frames)
            with self._lock:
                self._collection = collection
                self._calibrating[camera] = source
            try:
                collection.done.wait(s.collect_timeout_s)
            finally:
                with self._lock:
                    self._collection = None
                    self._calibrating.pop(camera, None)
            # Read after detaching: a last frame that landed between the wait
            # timing out and the detach still completed the run.
            if not collection.done.is_set():
                got = len(collection.points)
                return self._finish(camera, source, False, f"no depth frames from {camera}: {got} of {s.frames} arrived")
            fit = fit_floor(np.concatenate(collection.points), inlier_m=s.inlier_m)
            accepted, reason = validate_calibration(fit, reference, source, s.limits)
            if accepted:
                saved = self._install(camera, fit, source, reference)
                if not saved:
                    reason += " — not saved"
            return self._finish(camera, source, accepted, reason)
        finally:
            self._run_lock.release()

    def run_startup(self, active_camera) -> dict | None:
        """Startup calibrations every startup_retry_s until one is accepted or the window closes.

        active_camera() names the depth camera to calibrate, or None while no
        depth camera is delivering fresh frames. Stops early once any
        calibration of that camera, startup or operator, has been accepted in
        this process: an operator calibration during the window is the better
        one, and a startup attempt after it would only relabel it. Also stops
        for good once end_startup_window() has been called.
        """
        s = self._settings
        deadline = self._clock() + s.startup_window_s
        result = None
        while self._clock() < deadline:
            with self._lock:
                closed = self._startup_closed
            if closed:
                return result
            camera = active_camera()
            if camera is not None:
                with self._lock:
                    done = camera in self._accepted
                if done:
                    return result
                result = self.calibrate(camera, "startup")
                if result["accepted"] or result.get("skipped"):
                    return result
            # Retry a rejected calibration at the configured pace, but look
            # for a camera that is not up yet every second, so a camera that
            # comes up late is calibrated as soon as it delivers frames.
            self._sleep(s.startup_retry_s if camera is not None else min(1.0, s.startup_retry_s))
        return result

    def _install(self, camera: str, fit: FloorFit, source: str, reference: float | None) -> bool:
        new_reference = fit.plane.height_m if source == "operator" or reference is None else reference
        calibration = Calibration.from_fit(fit, new_reference, source, self._wall_clock())
        with self._lock:
            self._calibrations[camera] = calibration
            self._health[camera] = HealthMonitor(calibration.plane, self._settings.health)
            self._health_at.pop(camera, None)
            self._accepted.add(camera)
            everything = dict(self._calibrations)
        saved = self._store.save(everything)
        with self._lock:
            self._saved[camera] = saved
        self._log(
            f"FLOOR_CALIBRATION_ACCEPTED camera={camera} source={source} height_m={calibration.plane.height_m:.3f} "
            f"pitch_deg={calibration.plane.pitch_deg:.1f} roll_deg={calibration.plane.roll_deg:.1f} "
            f"reference_height_m={new_reference:.3f} saved={saved}"
        )
        return saved

    def _finish(self, camera: str, source: str, accepted: bool, reason: str) -> dict:
        with self._lock:
            self._last_result[camera] = {
                "accepted": accepted,
                "reason": reason,
                "source": source,
                "at": iso_utc(self._wall_clock()),
            }
        if not accepted:
            self._log(f"FLOOR_CALIBRATION_REJECTED camera={camera} source={source} reason={reason}")
        return {"accepted": accepted, "reason": reason, "calibration": self.status(camera)}
