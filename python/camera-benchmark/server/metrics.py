"""Metric helpers used by the parent: rolling stats, per-PID resource sampling,
and best-effort board power.

These are intentionally dependency-light (``psutil`` + stdlib). Anything that can
fail on a given board (a dead child PID, no ``vcgencmd``) degrades to ``None``
rather than raising — the benchmark must keep running.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections import deque

import psutil

logger = logging.getLogger(__name__)


class Rolling:
    """Fixed-size rolling window with p50/p95/avg summaries."""

    def __init__(self, maxlen: int = 120):
        self._d: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._d.append(value)

    def summary(self) -> dict | None:
        if not self._d:
            return None
        s = sorted(self._d)
        n = len(s)
        p = lambda q: s[min(n - 1, int(q * n))]  # noqa: E731
        return {
            "p50": round(p(0.50), 1),
            "p95": round(p(0.95), 1),
            "avg": round(sum(s) / n, 1),
        }


class ProcessSampler:
    """Samples CPU% and RSS for a single child PID via psutil.

    ``cpu_percent`` is top-style (100% == one full core). The first call seeds the
    measurement, so it is primed in ``__init__``.
    """

    def __init__(self, pid: int):
        self.pid = pid
        try:
            self.proc: psutil.Process | None = psutil.Process(pid)
            self.proc.cpu_percent(None)  # seed
        except psutil.Error:
            self.proc = None

    def sample(self) -> tuple[float | None, float | None]:
        if self.proc is None:
            return None, None
        try:
            cpu = self.proc.cpu_percent(None)
            rss = self.proc.memory_info().rss / (1024 * 1024)
            return round(cpu, 1), round(rss, 1)
        except psutil.Error:
            self.proc = None
            return None, None


# --------------------------------------------------------------------- power
_CUR_RE = re.compile(r"(\w+?)_A\s+current\(\d+\)=([\d.]+)A")
_VOLT_RE = re.compile(r"(\w+?)_V\s+volt\(\d+\)=([\d.]+)V")
_VCIO = "/dev/vcio"


def _unavailable(reason: str) -> dict:
    return {"available": False, "power_w": None, "reason": reason}


def _vcio_reason() -> str | None:
    """``None`` if ``/dev/vcio`` can be opened, else a short reason why not.

    The Pi 5 PMIC is read over the VideoCore mailbox (``/dev/vcio``, char major
    10). Inside a WendyOS container that node is exposed by the ``gpu``
    entitlement (WendyOS agent PR #970). Without it — an older agent, or the
    entitlement not declared in ``wendy.json`` — opening it fails with EPERM;
    surface that specifically rather than the misleading "Raspberry Pi 5 only".
    """
    if not os.path.exists(_VCIO):
        return "no /dev/vcio (not a Raspberry Pi 5)"
    try:
        os.close(os.open(_VCIO, os.O_RDWR))
        return None
    except PermissionError:
        return "no /dev/vcio access — needs the 'gpu' entitlement on a current WendyOS"
    except OSError as exc:
        return f"/dev/vcio open failed ({exc.strerror})"


def read_board_power() -> dict:
    """Best-effort whole-board power (watts) via ``vcgencmd pmic_read_adc`` (Pi 5).

    Sums V×I across matched PMIC rails. Returns ``{"available": False, "reason":
    ...}`` when unavailable, with ``reason`` naming the actual blocker (no
    ``/dev/vcio`` access, ``vcgencmd`` missing, or PMIC ADC unsupported) so the
    UI isn't misleading. Board-level, not per-camera.
    """
    # Check mailbox access first: if /dev/vcio can't be opened, vcgencmd can only
    # fail, so skip spawning it every sample and report the real reason.
    vcio = _vcio_reason()
    if vcio is not None:
        return _unavailable(vcio)
    if shutil.which("vcgencmd") is None:
        return _unavailable("vcgencmd not installed")
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception as exc:
        logger.debug("vcgencmd failed: %s", exc)
        return _unavailable("vcgencmd pmic_read_adc failed")

    currents = {m.group(1): float(m.group(2)) for m in _CUR_RE.finditer(out)}
    volts = {m.group(1): float(m.group(2)) for m in _VOLT_RE.finditer(out)}
    rails = currents.keys() & volts.keys()
    if not rails:
        return _unavailable("PMIC ADC unavailable on this OS")
    power = sum(currents[r] * volts[r] for r in rails)
    return {"available": True, "power_w": round(power, 2), "reason": None}
