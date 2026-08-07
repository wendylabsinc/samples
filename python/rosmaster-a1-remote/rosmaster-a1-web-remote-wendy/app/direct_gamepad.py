"""Hot-plugged evdev gamepad input for direct, on-device control.

The worker never publishes ROS messages. It feeds the web/control process's
arbiter, keeping that process the single /cmd_vel publisher whether a browser
is open or not.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


# Linux input-event-codes.h values. Keeping the small protocol surface here
# lets the pure decision layer import on development machines without evdev;
# production still uses python-evdev for enumeration and reading.
class Codes:
    EV_SYN = 0x00
    EV_KEY = 0x01
    EV_ABS = 0x03

    SYN_REPORT = 0
    SYN_DROPPED = 3

    ABS_X = 0x00
    ABS_Z = 0x02
    ABS_RZ = 0x05
    ABS_GAS = 0x09
    ABS_BRAKE = 0x0A
    ABS_HAT0X = 0x10
    ABS_HAT0Y = 0x11

    BTN_SOUTH = 0x130
    BTN_EAST = 0x131
    BTN_NORTH = 0x133
    BTN_WEST = 0x134
    BTN_TL = 0x136
    BTN_TR = 0x137
    BTN_TL2 = 0x138
    BTN_TR2 = 0x139
    BTN_SELECT = 0x13A
    BTN_START = 0x13B
    BTN_DPAD_UP = 0x220
    BTN_DPAD_DOWN = 0x221
    BTN_DPAD_LEFT = 0x222
    BTN_DPAD_RIGHT = 0x223


DIRECT_SPEED_DEFAULT = 1.50
DIRECT_SPEED_STEP = 0.05
DIRECT_SPEED_MIN = 0.0
DIRECT_SPEED_MAX = 2.0
DIRECT_STEERING_DEFAULT_PERCENT = 70
DIRECT_STEERING_STEP_PERCENT = 10
DIRECT_STEERING_MIN_PERCENT = 10
DIRECT_STEERING_MAX_PERCENT = 100
DIRECT_SCAN_INTERVAL_S = 0.5
DIRECT_HEARTBEAT_INTERVAL_S = 0.1


@dataclass(frozen=True)
class AxisRange:
    minimum: float
    maximum: float
    flat: float = 0.0
    value: float = 0.0


@dataclass
class Candidate:
    device: object
    path: str
    name: str
    uniq: str
    by_id_basenames: tuple[str, ...]
    key_codes: frozenset[int]
    axes: dict[int, AxisRange]

    @property
    def stable_id(self) -> str:
        if self.uniq:
            return self.uniq
        if self.by_id_basenames:
            return self.by_id_basenames[0]
        return ""

    def matches(self, selector: str) -> bool:
        return bool(selector) and (selector == self.uniq or selector in self.by_id_basenames)


class DroppedEvents(RuntimeError):
    pass


def clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def normalize_centered(value: float, axis: AxisRange) -> float:
    """Normalize a centred evdev axis to [-1, 1], honoring its flat zone."""

    span = axis.maximum - axis.minimum
    if not math.isfinite(span) or span <= 0:
        return 0.0
    center = axis.minimum + span / 2.0
    half = span / 2.0
    distance = float(value) - center
    flat = max(0.0, float(axis.flat))
    if abs(distance) <= flat:
        return 0.0
    usable = half - flat
    if usable <= 0:
        return 0.0
    magnitude = clamp((abs(distance) - flat) / usable, 0.0, 1.0)
    return math.copysign(magnitude, distance)


def normalize_trigger(value: float, axis: AxisRange) -> float:
    """Normalize an advertised evdev trigger range to [0, 1]."""

    span = axis.maximum - axis.minimum
    if not math.isfinite(span) or span <= 0:
        return 0.0
    distance = float(value) - axis.minimum
    flat = max(0.0, float(axis.flat))
    if distance <= flat:
        return 0.0
    usable = span - flat
    if usable <= 0:
        return 0.0
    return clamp((distance - flat) / usable, 0.0, 1.0)


def _axis_range(info: object) -> AxisRange:
    return AxisRange(
        minimum=float(getattr(info, "min", 0.0)),
        maximum=float(getattr(info, "max", 0.0)),
        flat=float(getattr(info, "flat", 0.0)),
        value=float(getattr(info, "value", 0.0)),
    )


def _capability_parts(device: object) -> tuple[frozenset[int], dict[int, AxisRange]]:
    try:
        capabilities = device.capabilities(absinfo=True)
    except TypeError:
        capabilities = device.capabilities()

    key_codes = frozenset(int(code) for code in capabilities.get(Codes.EV_KEY, []))
    axes: dict[int, AxisRange] = {}
    for entry in capabilities.get(Codes.EV_ABS, []):
        if isinstance(entry, tuple):
            code, info = entry
        else:
            code = entry
            info = device.absinfo(code)
        axes[int(code)] = _axis_range(info)
    return key_codes, axes


def trigger_axis_codes(axes: dict[int, AxisRange]) -> tuple[int, int] | None:
    """(forward, reverse) analog trigger codes for this device, or None.

    The wired xpad driver reports the triggers as ABS_RZ/ABS_Z; Bluetooth
    hid-microsoft moves them to ABS_GAS/ABS_BRAKE and reuses ABS_Z/ABS_RZ for
    the right stick, so GAS/BRAKE must win whenever both pairs are advertised
    or stick-down reads as a squeezed trigger.
    """
    if Codes.ABS_GAS in axes and Codes.ABS_BRAKE in axes:
        return (Codes.ABS_GAS, Codes.ABS_BRAKE)
    if Codes.ABS_RZ in axes and Codes.ABS_Z in axes:
        return (Codes.ABS_RZ, Codes.ABS_Z)
    return None


def compatibility_reason(key_codes: frozenset[int], axes: dict[int, AxisRange]) -> str:
    required_buttons = {Codes.BTN_SOUTH, Codes.BTN_EAST, Codes.BTN_NORTH}
    if not required_buttons.issubset(key_codes):
        return "missing_standard_action_buttons"
    if Codes.ABS_X not in axes:
        return "missing_left_steering_axis"
    digital_triggers = Codes.BTN_TL2 in key_codes and Codes.BTN_TR2 in key_codes
    if trigger_axis_codes(axes) is None and not digital_triggers:
        return "missing_forward_reverse_triggers"
    return "compatible"


def choose_candidate(candidates: list[Candidate], selector: str = "") -> tuple[Candidate | None, str]:
    if selector:
        matches = [candidate for candidate in candidates if candidate.matches(selector)]
        if len(matches) == 1:
            return matches[0], "selected_by_id"
        if len(matches) > 1:
            return None, "direct_gamepad_id_ambiguous"
        return None, "direct_gamepad_id_not_found"
    if len(candidates) == 1:
        return candidates[0], "selected"
    if len(candidates) > 1:
        return None, "multiple_compatible_gamepads"
    return None, "waiting_for_compatible_gamepad"


def by_id_basenames(path: str, directory: Path = Path("/dev/input/by-id")) -> tuple[str, ...]:
    try:
        target = os.path.realpath(path)
        matches = [entry.name for entry in directory.iterdir() if os.path.realpath(entry) == target]
    except OSError:
        return ()
    return tuple(sorted(matches))


def _load_evdev_backend():
    import evdev  # imported lazily so unit tests do not need Linux or evdev

    return evdev


class DirectGamepadWorker:
    def __init__(
        self,
        controller: object,
        *,
        backend: object | None = None,
        selector: str | None = None,
        by_id_lookup: Callable[[str], tuple[str, ...]] = by_id_basenames,
        max_steering_y: float = 0.12,
        auto_speed: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
        scan_interval_s: float = DIRECT_SCAN_INTERVAL_S,
        heartbeat_interval_s: float = DIRECT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._controller = controller
        self._backend = backend
        self._selector = (os.environ.get("DIRECT_GAMEPAD_ID", "") if selector is None else selector).strip()
        self._by_id_lookup = by_id_lookup
        self._max_steering_y = max(0.0, float(max_steering_y))
        self._clock = clock
        self._log = log
        self._scan_interval_s = max(0.01, float(scan_interval_s))
        self._heartbeat_interval_s = max(0.01, float(heartbeat_interval_s))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self._candidate: Candidate | None = None
        self._pressed: set[int] = set()
        self._motion_codes: frozenset[int] = frozenset({Codes.ABS_X})
        self._axis_values: dict[int, float] = {}
        self._hat_values = {Codes.ABS_HAT0X: 0, Codes.ABS_HAT0Y: 0}
        self._motion_dirty = False
        self._speed = DIRECT_SPEED_DEFAULT
        self._auto_speed = clamp(float(auto_speed), DIRECT_SPEED_MIN, DIRECT_SPEED_MAX)
        self._steering_percent = DIRECT_STEERING_DEFAULT_PERCENT
        self._armed = False
        self._owned = False
        self._auto_enabled = False
        self._selection_allowed = False
        self._status = {
            "worker_ok": False,
            "running": False,
            "connected": False,
            "compatible": False,
            "compatible_devices": 0,
            "id": "",
            "stable_id": "",
            "name": "",
            "path": "",
            "owned": False,
            "armed": False,
            "auto": False,
            "reason": "not_started",
            "last_event_at": 0.0,
            "reader_failures": 0,
        }

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="direct-gamepad")
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="direct-gamepad-heartbeat",
            )
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="direct-gamepad-monitor",
            )
            self._thread.start()
            self._heartbeat_thread.start()
            self._monitor_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._fail_closed("worker_shutdown", disconnect=True)
        with self._lock:
            candidate = self._candidate
        if candidate is not None:
            try:
                candidate.device.close()
            except OSError:
                pass
        for thread in (self._thread, self._heartbeat_thread, self._monitor_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=1.0)

    def snapshot(self) -> dict:
        with self._lock:
            snapshot = dict(self._status)
            snapshot.update(
                {
                    "selector": self._selector,
                    "owned": self._owned,
                    "armed": self._armed,
                    "auto": self._auto_enabled,
                    "speed": round(self._speed, 2),
                    "auto_speed": round(self._auto_speed, 2),
                    "steering_scale": round(self._steering_percent / 100.0, 2),
                    "steering_percent": self._steering_percent,
                }
            )
            last_event_at = float(snapshot.pop("last_event_at", 0.0))
        snapshot["last_event_age_s"] = round(self._clock() - last_event_at, 3) if last_event_at else None
        return snapshot

    def note_external_stop(self, reason: str = "browser_stop") -> None:
        with self._lock:
            self._armed = False
            self._owned = False
            self._auto_enabled = False
            self._motion_dirty = False
            self._status["reason"] = reason

    def apply_external_stop(self, stop: Callable[[], None], reason: str = "browser_stop") -> None:
        """Serialize a global STOP with an A-button ownership transition."""

        with self._lock:
            self._armed = False
            self._owned = False
            self._auto_enabled = False
            self._motion_dirty = False
            self._status["reason"] = reason
            # _arm uses the same worker lock while acquiring the control lock.
            # Whichever physical action reaches this lock last wins cleanly;
            # there is no gap that can leave control owned after STOP while the
            # worker believes it is unarmed.
            stop()

    def discover_candidates(self) -> list[Candidate]:
        backend = self._ensure_backend()
        candidates: list[Candidate] = []
        for path in backend.list_devices():
            device = None
            try:
                device = backend.InputDevice(path)
                key_codes, axes = _capability_parts(device)
                if compatibility_reason(key_codes, axes) != "compatible":
                    device.close()
                    continue
                candidates.append(
                    Candidate(
                        device=device,
                        path=str(getattr(device, "path", path)),
                        name=str(getattr(device, "name", "")),
                        uniq=str(getattr(device, "uniq", "") or ""),
                        by_id_basenames=self._by_id_lookup(str(getattr(device, "path", path))),
                        key_codes=key_codes,
                        axes=axes,
                    )
                )
            except Exception:  # noqa: BLE001 - one odd input node must not poison discovery
                if device is not None:
                    try:
                        device.close()
                    except Exception:  # noqa: BLE001 - best-effort cleanup of an unusable node
                        pass
        return candidates

    def scan_and_read_once(self) -> str:
        candidates = self.discover_candidates()
        selected, reason = choose_candidate(candidates, self._selector)
        with self._lock:
            self._status["compatible"] = bool(candidates)
            self._status["compatible_devices"] = len(candidates)
            if selected is None:
                self._status["connected"] = False
                self._status["reason"] = reason
        for candidate in candidates:
            if candidate is not selected:
                candidate.device.close()
        if selected is None:
            return reason
        self._read_candidate(selected)
        return reason

    def process_event(self, event: object) -> None:
        event_type = int(getattr(event, "type"))
        code = int(getattr(event, "code"))
        value = int(getattr(event, "value"))
        with self._lock:
            self._status["last_event_at"] = self._clock()

        if event_type == Codes.EV_SYN:
            if code == Codes.SYN_DROPPED:
                self._fail_closed("syn_dropped", disconnect=True)
                raise DroppedEvents("evdev reported SYN_DROPPED")
            if code == Codes.SYN_REPORT:
                self._flush_motion()
            return

        if event_type == Codes.EV_ABS:
            self._handle_abs(code, value)
        elif event_type == Codes.EV_KEY:
            self._handle_key(code, value)

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = _load_evdev_backend()
        return self._backend

    def _run(self) -> None:
        with self._lock:
            self._status.update({"running": True, "worker_ok": True, "reason": "scanning"})
        try:
            while not self._stop_event.is_set():
                try:
                    self.scan_and_read_once()
                    with self._lock:
                        self._status["worker_ok"] = True
                except Exception as exc:  # noqa: BLE001 - worker must keep scanning
                    self._fail_closed(f"scan_error:{type(exc).__name__}", disconnect=True)
                    with self._lock:
                        self._status["worker_ok"] = False
                    self._log(f"DIRECT_GAMEPAD_SCAN_FAILED {type(exc).__name__}: {exc}")
                self._stop_event.wait(self._scan_interval_s)
        finally:
            with self._lock:
                self._status["running"] = False

    def _heartbeat_loop(self) -> None:
        """Refresh a held manual command even when evdev has gone quiet.

        Gamepads emit changes, not a continuous stream of their current
        position. Without this heartbeat, holding a trigger steady eventually
        trips the control process's command watchdog even though the pad and
        reader are healthy.
        """

        while not self._stop_event.wait(self._heartbeat_interval_s):
            try:
                self._heartbeat_tick()
            except Exception as exc:  # noqa: BLE001 - motion must fail closed
                self._fail_closed("heartbeat_failure", disconnect=False)
                with self._lock:
                    self._status["worker_ok"] = False
                self._log(f"DIRECT_GAMEPAD_HEARTBEAT_FAILED {type(exc).__name__}: {exc}")

    def _heartbeat_tick(self) -> None:
        with self._lock:
            refresh = self._owned and not self._auto_enabled and self._selection_allowed
        if refresh:
            self._apply_drive()

    def _monitor_loop(self) -> None:
        """Keep fail-closed selection true after another pad is hot-plugged."""

        while not self._stop_event.wait(self._scan_interval_s):
            with self._lock:
                active = self._candidate is not None
            if not active:
                continue
            try:
                self._verify_active_selection()
            except Exception as exc:  # noqa: BLE001 - selection failures must stop motion
                self._fail_closed("selection_monitor_failure", disconnect=False)
                with self._lock:
                    self._status["worker_ok"] = False
                self._log(f"DIRECT_GAMEPAD_MONITOR_FAILED {type(exc).__name__}: {exc}")

    def _verify_active_selection(self) -> str:
        """Rescan while reading so ambiguity introduced by hot-plug is safe."""

        candidates = self.discover_candidates()
        selected, reason = choose_candidate(candidates, self._selector)
        with self._lock:
            active = self._candidate
        selected_path = selected.path if selected is not None else ""
        active_path = active.path if active is not None else ""
        for candidate in candidates:
            try:
                candidate.device.close()
            except OSError:
                pass

        valid = bool(active_path and selected_path == active_path)
        with self._lock:
            # The reader may have ended while discovery was in progress. Its
            # cleanup owns the final state in that case.
            if self._candidate is not active:
                return "reader_changed"
            self._status["compatible"] = bool(candidates)
            self._status["compatible_devices"] = len(candidates)
            self._selection_allowed = valid
            self._status["connected"] = valid
            if valid and not self._owned:
                self._status["reason"] = "connected_unarmed"
            elif not valid:
                self._status["reason"] = reason
        if not valid:
            self._fail_closed(reason, disconnect=False)
        return reason

    def _prepare_candidate(self, candidate: Candidate) -> None:
        try:
            active_keys = set(int(code) for code in candidate.device.active_keys())
        except (AttributeError, OSError):
            active_keys = set()
        with self._lock:
            self._candidate = candidate
            self._pressed = active_keys
            self._axis_values = {code: axis.value for code, axis in candidate.axes.items()}
            self._hat_values = {
                Codes.ABS_HAT0X: int(self._axis_values.get(Codes.ABS_HAT0X, 0)),
                Codes.ABS_HAT0Y: int(self._axis_values.get(Codes.ABS_HAT0Y, 0)),
            }
            self._armed = False
            self._owned = False
            self._auto_enabled = False
            self._selection_allowed = True
            self._motion_dirty = False
            self._motion_codes = frozenset({Codes.ABS_X}) | set(trigger_axis_codes(candidate.axes) or ())
            self._status.update(
                {
                    "connected": True,
                    "compatible": True,
                    "id": candidate.stable_id,
                    "stable_id": candidate.stable_id,
                    "name": candidate.name,
                    "path": candidate.path,
                    "reason": "connected_unarmed",
                }
            )

    def _read_candidate(self, candidate: Candidate) -> None:
        self._prepare_candidate(candidate)
        try:
            for event in candidate.device.read_loop():
                if self._stop_event.is_set():
                    break
                self.process_event(event)
            if not self._stop_event.is_set():
                self._fail_closed("reader_closed", disconnect=True)
        except DroppedEvents:
            pass
        except OSError as exc:
            if self._stop_event.is_set():
                self._fail_closed("worker_shutdown", disconnect=True)
            else:
                self._reader_failed("device_removed", exc)
        except Exception as exc:  # noqa: BLE001 - fail closed on decoder/read failures
            self._reader_failed("reader_failure", exc)
        finally:
            try:
                candidate.device.close()
            except OSError:
                pass
            with self._lock:
                self._candidate = None

    def _reader_failed(self, reason: str, exc: Exception) -> None:
        self._fail_closed(reason, disconnect=True)
        with self._lock:
            self._status["reader_failures"] = int(self._status["reader_failures"]) + 1
        self._log(f"DIRECT_GAMEPAD_READER_FAILED reason={reason} {type(exc).__name__}: {exc}")

    def _handle_abs(self, code: int, value: int) -> None:
        with self._lock:
            self._axis_values[code] = value
            previous_hat = self._hat_values.get(code, 0)
            if code in self._hat_values:
                self._hat_values[code] = value
            if code in self._motion_codes:
                self._motion_dirty = True

        if code == Codes.ABS_HAT0Y and value != previous_hat and value != 0:
            self._nudge_manual_speed(DIRECT_SPEED_STEP if value < 0 else -DIRECT_SPEED_STEP)
        elif code == Codes.ABS_HAT0X and value != previous_hat and value != 0:
            self._nudge_auto_speed(DIRECT_SPEED_STEP if value > 0 else -DIRECT_SPEED_STEP)

    def _handle_key(self, code: int, value: int) -> None:
        with self._lock:
            was_pressed = code in self._pressed
            if value == 0:
                self._pressed.discard(code)
            else:
                self._pressed.add(code)
            if code in {Codes.BTN_TL2, Codes.BTN_TR2}:
                self._motion_dirty = True
            rose = value == 1 and not was_pressed
        if not rose:
            return

        if code in {Codes.BTN_EAST, Codes.BTN_START}:
            self._controller_stop("direct_gamepad_stop")
        elif code == Codes.BTN_SOUTH:
            self._arm()
        elif code == Codes.BTN_NORTH:
            self._toggle_auto()
        elif code in {Codes.BTN_DPAD_UP, Codes.BTN_DPAD_DOWN}:
            self._nudge_manual_speed(DIRECT_SPEED_STEP if code == Codes.BTN_DPAD_UP else -DIRECT_SPEED_STEP)
        elif code in {Codes.BTN_DPAD_RIGHT, Codes.BTN_DPAD_LEFT}:
            self._nudge_auto_speed(DIRECT_SPEED_STEP if code == Codes.BTN_DPAD_RIGHT else -DIRECT_SPEED_STEP)
        elif code in {Codes.BTN_TR, Codes.BTN_TL}:
            self._nudge_steering(
                DIRECT_STEERING_STEP_PERCENT if code == Codes.BTN_TR else -DIRECT_STEERING_STEP_PERCENT
            )
        # BTN_WEST (X) and BTN_SELECT (View) intentionally do nothing on the
        # direct path. Their browser-only camera actions have no meaning here.

    def _arm(self) -> None:
        with self._lock:
            if not self._status["connected"] or not self._selection_allowed or self._owned:
                return
            self._controller.direct_acquire()
            self._armed = True
            self._owned = True
            self._auto_enabled = False
            self._status["reason"] = "direct_gamepad_active"
        self._apply_drive()

    def _toggle_auto(self) -> None:
        with self._lock:
            if not self._owned:
                return
            self._auto_enabled = not self._auto_enabled
            enabled = self._auto_enabled
            speed = self._auto_speed
            self._status["reason"] = "direct_gamepad_auto" if enabled else "direct_gamepad_active"
        self._controller.direct_set_auto(enabled, speed)
        if not enabled:
            self._apply_drive()

    def _nudge_manual_speed(self, delta: float) -> None:
        with self._lock:
            self._speed = round(clamp(self._speed + delta, DIRECT_SPEED_MIN, DIRECT_SPEED_MAX), 2)
            owned = self._owned and not self._auto_enabled
        if owned:
            self._apply_drive()

    def _nudge_auto_speed(self, delta: float) -> None:
        with self._lock:
            self._auto_speed = round(clamp(self._auto_speed + delta, DIRECT_SPEED_MIN, DIRECT_SPEED_MAX), 2)
            update = self._owned and self._auto_enabled
            speed = self._auto_speed
        if update:
            self._controller.direct_set_auto(True, speed)

    def _nudge_steering(self, delta_percent: int) -> None:
        with self._lock:
            self._steering_percent = int(
                clamp(
                    self._steering_percent + delta_percent,
                    DIRECT_STEERING_MIN_PERCENT,
                    DIRECT_STEERING_MAX_PERCENT,
                )
            )
            owned = self._owned and not self._auto_enabled
        if owned:
            self._apply_drive()

    def _flush_motion(self) -> None:
        with self._lock:
            dirty = self._motion_dirty
            self._motion_dirty = False
            apply = dirty and self._owned and not self._auto_enabled
        if apply:
            self._apply_drive()

    def _apply_drive(self) -> None:
        with self._lock:
            candidate = self._candidate
            if candidate is None or not self._owned or self._auto_enabled:
                return
            steer_axis = candidate.axes.get(Codes.ABS_X)
            if steer_axis is None:
                return
            steer = normalize_centered(self._axis_values.get(Codes.ABS_X, steer_axis.value), steer_axis)
            triggers = trigger_axis_codes(candidate.axes)
            if triggers is not None:
                forward_code, reverse_code = triggers
                reverse = normalize_trigger(
                    self._axis_values.get(reverse_code, candidate.axes[reverse_code].value),
                    candidate.axes[reverse_code],
                )
                forward = normalize_trigger(
                    self._axis_values.get(forward_code, candidate.axes[forward_code].value),
                    candidate.axes[forward_code],
                )
            else:
                reverse = 1.0 if Codes.BTN_TL2 in self._pressed else 0.0
                forward = 1.0 if Codes.BTN_TR2 in self._pressed else 0.0
            linear_x = (forward - reverse) * self._speed
            steering_y = -steer * self._max_steering_y * (self._steering_percent / 100.0)
        accepted = self._controller.direct_update(linear_x, steering_y)
        if accepted is False:
            self._fail_closed("ownership_lost", disconnect=False)

    def _controller_stop(self, reason: str) -> None:
        with self._lock:
            self._armed = False
            self._owned = False
            self._auto_enabled = False
            self._motion_dirty = False
            self._status["reason"] = reason
        self._controller.direct_stop(reason)

    def _fail_closed(self, reason: str, *, disconnect: bool) -> None:
        with self._lock:
            release = self._owned or self._armed or self._auto_enabled
            self._armed = False
            self._owned = False
            self._auto_enabled = False
            self._motion_dirty = False
            self._pressed.clear()
            if disconnect:
                self._status["connected"] = False
                self._selection_allowed = False
            self._status["reason"] = reason
        if release:
            self._controller.direct_release(reason)
