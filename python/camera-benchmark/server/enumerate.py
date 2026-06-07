"""Camera discovery, USB/CSI classification, and benchmark-slot assignment.

Runs in the PARENT process, so it deliberately avoids GStreamer (no ``Gst.init``
in the parent — see capture_worker.py). Discovery uses sysfs + ``v4l2-ctl`` for
USB/V4L2 devices and the libcamera ``cam`` tool for CSI cameras, mirroring how the
WendyOS agent classifies cameras (sysfs driver name; ``libcamerasrc`` for CSI,
``v4l2src`` for USB).

The benchmark always has exactly two slots — ``usb`` and ``csi``. If a real device
for a slot is absent (or libcamera/hardware is missing, or ``FORCE_SYNTHETIC=1``),
that slot falls back to a synthetic ``videotestsrc`` source so the comparison view
always shows two panels.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_USB_HINTS = ("logitech", "webcam", "uvc", "usb", "c920", "c922", "c930", "brio")
_CSI_DRIVERS = ("unicam", "rp1-cfe", "bcm2835-unicam", "tegra-video", "mxc-md", "rkisp")


# --------------------------------------------------------------------------- v4l2
def _v4l2(path: str, arg: str) -> str:
    try:
        return subprocess.check_output(
            ["v4l2-ctl", "--device", path, arg],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
    except Exception:
        return ""


def _card_name(path: str) -> str:
    for line in _v4l2(path, "--info").splitlines():
        if "Card type" in line:
            return line.split(":", 1)[1].strip()
    return Path(path).name


def _is_capture(path: str) -> bool:
    return "Video Capture" in _v4l2(path, "--all")


def _sysfs_attr(node: str, *rel: str) -> str:
    """basename of a symlink target under /sys/class/video4linux/<node>/device."""
    target = Path("/sys/class/video4linux") / node / "device"
    for r in rel:
        target = target / r
    try:
        return os.path.basename(os.path.realpath(target))
    except Exception:
        return ""


def _classify_v4l2(path: str, name: str) -> str:
    """Return 'usb' | 'csi' | 'unknown' for a /dev/videoN device."""
    node = Path(path).name
    driver = _sysfs_attr(node, "driver")
    subsystem = _sysfs_attr(node, "subsystem")
    has_of_node = (Path("/sys/class/video4linux") / node / "device" / "of_node").exists()
    lname = name.lower()

    if driver == "uvcvideo" or subsystem == "usb" or any(h in lname for h in _USB_HINTS):
        return "usb"
    if driver in _CSI_DRIVERS or has_of_node or subsystem == "platform":
        return "csi"
    return "unknown"


def list_v4l2_cameras() -> list[dict]:
    cams = []
    for path in sorted(glob.glob("/dev/video*")):
        if not _is_capture(path):
            continue
        name = _card_name(path)
        cams.append({"path": path, "name": name, "kind": _classify_v4l2(path, name)})
    return cams


# ----------------------------------------------------------------------- libcamera
# libcamera's ``cam -l`` prints one line per camera. Older builds emit
#   ``1: 'imx477' (<id>)`` and newer ones ``1: External camera 'imx477' (<id>)``;
# this matches both — quoted model name, then the id in parentheses.
_CAM_LINE = re.compile(r"^\s*\d+\s*:\s*.*?'([^']+)'\s*\((.+)\)\s*$")


def list_csi_cameras() -> list[dict]:
    """Parse ``cam -l`` (libcamera). Empty if the tool/HW is absent."""
    try:
        out = subprocess.run(
            ["cam", "-l"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception as exc:
        logger.info("libcamera `cam` not available: %s", exc)
        return []

    cams = []
    for line in out.splitlines():
        m = _CAM_LINE.match(line)
        if m:
            model, cam_id = m.group(1).strip(), m.group(2).strip()
            cams.append({"id": cam_id, "name": model})
    if cams:
        logger.info("Discovered %d CSI camera(s) via libcamera", len(cams))
    return cams


# ----------------------------------------------------------------------- slots
def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _synthetic(kind: str) -> dict:
    if kind == "usb":
        return {"kind": "usb", "label": "SYNTHETIC USB", "name": "Synthetic (videotestsrc)",
                "mode": "synthetic", "pattern": "smpte", "device": None,
                "width": 640, "height": 480, "synthetic": True}
    return {"kind": "csi", "label": "SYNTHETIC CSI", "name": "Synthetic (videotestsrc)",
            "mode": "synthetic", "pattern": "ball", "device": None,
            "width": 640, "height": 480, "synthetic": True}


def assign_slots() -> dict[str, dict]:
    """Decide the config for the ``usb`` and ``csi`` benchmark slots.

    Precedence: ``FORCE_SYNTHETIC`` > explicit env device override > auto-detect >
    synthetic fallback.
    """
    if _env_true("FORCE_SYNTHETIC"):
        logger.info("FORCE_SYNTHETIC set — both slots synthetic")
        return {"usb": _synthetic("usb"), "csi": _synthetic("csi")}

    v4l2_cams = list_v4l2_cameras()
    csi_cams = list_csi_cameras()
    logger.info("v4l2=%s csi=%s", v4l2_cams, csi_cams)

    # -- USB slot --
    usb_override = os.environ.get("CAMERA_USB_DEVICE")
    if usb_override:
        usb = {"kind": "usb", "label": "USB Webcam", "name": _card_name(usb_override),
               "mode": "v4l2", "device": usb_override, "width": 0, "height": 0,
               "synthetic": False}
    else:
        pick = next((c for c in v4l2_cams if c["kind"] == "usb"), None)
        # If nothing classified as USB but a generic capture device exists, use it.
        if pick is None:
            pick = next((c for c in v4l2_cams if c["kind"] != "csi"), None)
        if pick:
            usb = {"kind": "usb", "label": "USB Webcam", "name": pick["name"],
                   "mode": "v4l2", "device": pick["path"], "width": 0, "height": 0,
                   "synthetic": False}
        else:
            usb = _synthetic("usb")

    # -- CSI slot --
    csi_override = os.environ.get("CAMERA_CSI_ID")
    if csi_override:
        csi = {"kind": "csi", "label": "Ribbon Cam (CSI)", "name": "Ribbon Cam",
               "mode": "libcamera", "device": csi_override, "width": 1280, "height": 720,
               "synthetic": False}
    else:
        # libcamera also enumerates UVC/USB cameras; the CSI slot wants the
        # ribbon sensor, so skip anything whose id is a USB device.
        cam = next((c for c in csi_cams if "usb" not in c["id"].lower()), None)
        if cam:
            csi = {"kind": "csi", "label": "Ribbon Cam (CSI)", "name": cam["name"],
                   "mode": "libcamera", "device": cam["id"], "width": 1280, "height": 720,
                   "synthetic": False}
        else:
            csi = _synthetic("csi")

    return {"usb": usb, "csi": csi}
