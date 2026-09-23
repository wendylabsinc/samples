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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from floor_model import FloorFit, FloorPlane

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
