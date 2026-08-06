"""Fake-evdev tests for browser-free, on-device gamepad control."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "rosmaster-a1-web-remote-wendy" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import direct_gamepad as gamepad  # noqa: E402


@dataclass
class Event:
    type: int
    code: int
    value: int


class FakeControl:
    def __init__(self):
        self.calls = []

    def direct_acquire(self):
        self.calls.append(("acquire",))

    def direct_update(self, linear_x, steering_y):
        self.calls.append(("update", linear_x, steering_y))

    def direct_set_auto(self, enabled, speed):
        self.calls.append(("auto", enabled, speed))

    def direct_stop(self, reason):
        self.calls.append(("stop", reason))

    def direct_release(self, reason):
        self.calls.append(("release", reason))


class FakeDevice:
    def __init__(self, path, *, keys, axes, name="Xbox Wireless Controller", uniq="pad-uniq", events=()):
        self.path = path
        self.name = name
        self.uniq = uniq
        self._keys = list(keys)
        self._axes = dict(axes)
        self._events = events
        self.closed = False
        self._active_keys = []

    def capabilities(self, absinfo=False):
        if absinfo:
            return {
                gamepad.Codes.EV_KEY: list(self._keys),
                gamepad.Codes.EV_ABS: list(self._axes.items()),
            }
        return {
            gamepad.Codes.EV_KEY: list(self._keys),
            gamepad.Codes.EV_ABS: list(self._axes),
        }

    def absinfo(self, code):
        return self._axes[code]

    def active_keys(self):
        return list(self._active_keys)

    def read_loop(self):
        if isinstance(self._events, BaseException):
            raise self._events
        yield from self._events

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.devices = {}

    def list_devices(self):
        return list(self.devices)

    def InputDevice(self, path):
        return self.devices[path]


def absinfo(value, minimum, maximum, flat=0):
    return SimpleNamespace(value=value, min=minimum, max=maximum, flat=flat)


STANDARD_KEYS = {
    gamepad.Codes.BTN_SOUTH,
    gamepad.Codes.BTN_EAST,
    gamepad.Codes.BTN_NORTH,
    gamepad.Codes.BTN_WEST,
    gamepad.Codes.BTN_START,
    gamepad.Codes.BTN_SELECT,
    gamepad.Codes.BTN_TL,
    gamepad.Codes.BTN_TR,
}


def analog_axes():
    return {
        gamepad.Codes.ABS_X: absinfo(0, -32768, 32767, 4096),
        gamepad.Codes.ABS_Z: absinfo(0, 0, 1023, 8),
        gamepad.Codes.ABS_RZ: absinfo(0, 0, 1023, 8),
        gamepad.Codes.ABS_HAT0X: absinfo(0, -1, 1),
        gamepad.Codes.ABS_HAT0Y: absinfo(0, -1, 1),
    }


def candidate(device=None, *, uniq="pad-uniq", by_ids=("usb-xbox-event-joystick",), digital=False):
    keys = set(STANDARD_KEYS)
    axes = analog_axes()
    if digital:
        keys.update({gamepad.Codes.BTN_TL2, gamepad.Codes.BTN_TR2})
        axes.pop(gamepad.Codes.ABS_Z)
        axes.pop(gamepad.Codes.ABS_RZ)
    if device is None:
        device = FakeDevice("/dev/input/event7", keys=keys, axes=axes, uniq=uniq)
    ranges = {code: gamepad._axis_range(info) for code, info in axes.items()}
    return gamepad.Candidate(
        device=device,
        path=device.path,
        name=device.name,
        uniq=uniq,
        by_id_basenames=by_ids,
        key_codes=frozenset(keys),
        axes=ranges,
    )


class NormalizationTests(unittest.TestCase):
    def test_centered_axis_uses_advertised_min_max_and_flat(self):
        axis = gamepad.AxisRange(-32768, 32767, flat=4096)
        self.assertEqual(gamepad.normalize_centered(-1, axis), 0.0)
        self.assertEqual(gamepad.normalize_centered(3000, axis), 0.0)
        self.assertAlmostEqual(gamepad.normalize_centered(32767, axis), 1.0)
        self.assertAlmostEqual(gamepad.normalize_centered(-32768, axis), -1.0)

    def test_trigger_uses_advertised_min_max_and_flat(self):
        axis = gamepad.AxisRange(10, 1010, flat=20)
        self.assertEqual(gamepad.normalize_trigger(10, axis), 0.0)
        self.assertEqual(gamepad.normalize_trigger(30, axis), 0.0)
        self.assertAlmostEqual(gamepad.normalize_trigger(1010, axis), 1.0)
        self.assertAlmostEqual(gamepad.normalize_trigger(520, axis), 0.5, places=2)


class CapabilityAndSelectionTests(unittest.TestCase):
    def test_accepts_analog_or_digital_triggers(self):
        analog = candidate()
        self.assertEqual(gamepad.compatibility_reason(analog.key_codes, analog.axes), "compatible")
        digital = candidate(digital=True)
        self.assertEqual(gamepad.compatibility_reason(digital.key_codes, digital.axes), "compatible")

    def test_rejects_devices_without_required_gamepad_capabilities(self):
        self.assertEqual(
            gamepad.compatibility_reason(frozenset({gamepad.Codes.BTN_SOUTH}), analog_axes()),
            "missing_standard_action_buttons",
        )
        no_steer = analog_axes()
        no_steer.pop(gamepad.Codes.ABS_X)
        self.assertEqual(
            gamepad.compatibility_reason(frozenset(STANDARD_KEYS), no_steer),
            "missing_left_steering_axis",
        )

    def test_multiple_devices_fail_closed_unless_pinned(self):
        first = candidate(uniq="one", by_ids=("usb-one-event-joystick",))
        second = candidate(uniq="two", by_ids=("usb-two-event-joystick",))
        selected, reason = gamepad.choose_candidate([first, second])
        self.assertIsNone(selected)
        self.assertEqual(reason, "multiple_compatible_gamepads")

        selected, reason = gamepad.choose_candidate([first, second], "two")
        self.assertIs(selected, second)
        self.assertEqual(reason, "selected_by_id")

        selected, _ = gamepad.choose_candidate([first, second], "usb-one-event-joystick")
        self.assertIs(selected, first)

    def test_discovery_observes_hotplug_and_filters_a_keyboard(self):
        backend = FakeBackend()
        worker = gamepad.DirectGamepadWorker(FakeControl(), backend=backend, by_id_lookup=lambda _: ())
        self.assertEqual(worker.discover_candidates(), [])

        keyboard = FakeDevice("/dev/input/event1", keys={30, 31}, axes={}, name="keyboard", uniq="kbd")
        pad = FakeDevice("/dev/input/event9", keys=STANDARD_KEYS, axes=analog_axes(), uniq="hotplugged")
        backend.devices = {keyboard.path: keyboard, pad.path: pad}
        found = worker.discover_candidates()
        self.assertEqual([item.stable_id for item in found], ["hotplugged"])
        self.assertTrue(keyboard.closed)
        found[0].device.close()


class WorkerEventTests(unittest.TestCase):
    def setUp(self):
        self.control = FakeControl()
        self.now = 100.0
        self.worker = gamepad.DirectGamepadWorker(
            self.control,
            backend=FakeBackend(),
            clock=lambda: self.now,
            log=lambda _: None,
        )
        self.pad = candidate()
        self.worker._prepare_candidate(self.pad)

    def event(self, event_type, code, value):
        self.worker.process_event(Event(event_type, code, value))

    def sync(self):
        self.event(gamepad.Codes.EV_SYN, gamepad.Codes.SYN_REPORT, 0)

    def test_buttons_are_edge_triggered_and_axes_drive_after_a(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.assertEqual([call[0] for call in self.control.calls].count("acquire"), 1)
        # A key repeat cannot acquire again.
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 2)
        self.assertEqual([call[0] for call in self.control.calls].count("acquire"), 1)

        self.event(gamepad.Codes.EV_ABS, gamepad.Codes.ABS_RZ, 1023)
        self.event(gamepad.Codes.EV_ABS, gamepad.Codes.ABS_X, 32767)
        self.sync()
        update = [call for call in self.control.calls if call[0] == "update"][-1]
        self.assertAlmostEqual(update[1], 1.5)
        self.assertAlmostEqual(update[2], -0.084, places=3)

        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_EAST, 1)
        self.assertEqual(self.control.calls[-1], ("stop", "direct_gamepad_stop"))
        self.assertFalse(self.worker.snapshot()["owned"])

    def test_held_controls_are_republished_by_the_direct_heartbeat(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.event(gamepad.Codes.EV_ABS, gamepad.Codes.ABS_RZ, 1023)
        self.sync()
        self.control.calls.clear()

        # No new evdev event arrives while a trigger is held. The worker must
        # still refresh the command so the server watchdog does not interpret
        # a healthy, steady controller as a stale source.
        self.worker._heartbeat_tick()
        self.assertEqual(len(self.control.calls), 1)
        self.assertEqual(self.control.calls[0][0], "update")
        self.assertAlmostEqual(self.control.calls[0][1], 1.5)

    def test_status_exposes_stable_identity_health_and_tuning(self):
        snapshot = self.worker.snapshot()
        self.assertEqual(snapshot["stable_id"], "pad-uniq")
        self.assertEqual(snapshot["name"], "Xbox Wireless Controller")
        self.assertTrue(snapshot["connected"])
        self.assertTrue(snapshot["compatible"])
        self.assertEqual(snapshot["speed"], 1.5)
        self.assertEqual(snapshot["steering_scale"], 0.7)
        self.assertIn("worker_ok", snapshot)
        self.assertIn("last_event_age_s", snapshot)

    def test_a_second_hotplugged_pad_releases_control_until_selection_is_unique(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.assertTrue(self.worker.snapshot()["owned"])

        first_scan = FakeDevice(
            self.pad.path,
            keys=STANDARD_KEYS,
            axes=analog_axes(),
            uniq="pad-uniq",
        )
        second_scan = FakeDevice(
            "/dev/input/event8",
            keys=STANDARD_KEYS,
            axes=analog_axes(),
            uniq="second-pad",
        )
        self.worker._backend.devices = {
            first_scan.path: first_scan,
            second_scan.path: second_scan,
        }
        self.worker._verify_active_selection()

        snapshot = self.worker.snapshot()
        self.assertFalse(snapshot["owned"])
        self.assertFalse(snapshot["connected"])
        self.assertEqual(snapshot["reason"], "multiple_compatible_gamepads")
        self.assertEqual(self.control.calls[-1], ("release", "multiple_compatible_gamepads"))

        # Removing the ambiguity makes the original reader selectable again,
        # but it remains unarmed. A release and a new A edge are required.
        replacement_scan = FakeDevice(
            self.pad.path,
            keys=STANDARD_KEYS,
            axes=analog_axes(),
            uniq="pad-uniq",
        )
        self.worker._backend.devices = {replacement_scan.path: replacement_scan}
        self.worker._verify_active_selection()
        self.assertTrue(self.worker.snapshot()["connected"])
        self.assertFalse(self.worker.snapshot()["armed"])

        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 0)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.assertEqual([call[0] for call in self.control.calls].count("acquire"), 2)

    def test_y_toggles_auto_only_after_direct_acquisition(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_NORTH, 1)
        self.assertFalse(any(call[0] == "auto" for call in self.control.calls))
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_NORTH, 0)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_NORTH, 1)
        self.assertEqual(self.control.calls[-1], ("auto", True, 1.0))
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_NORTH, 0)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_NORTH, 1)
        self.assertTrue(any(call == ("auto", False, 1.0) for call in self.control.calls))

    def test_dpad_and_bumpers_use_existing_bounds_and_steps(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_DPAD_UP, 1)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_TR, 1)
        snapshot = self.worker.snapshot()
        self.assertEqual(snapshot["speed"], 1.55)
        self.assertEqual(snapshot["steering_percent"], 80)

        # Held/repeat values do not nudge again.
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_DPAD_UP, 2)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_TR, 2)
        self.assertEqual(self.worker.snapshot()["speed"], 1.55)
        self.assertEqual(self.worker.snapshot()["steering_percent"], 80)

    def test_x_and_view_are_ignored_on_direct_path(self):
        before = list(self.control.calls)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_WEST, 1)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SELECT, 1)
        self.assertEqual(self.control.calls, before)

    def test_syn_dropped_stops_and_releases_immediately(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        with self.assertRaises(gamepad.DroppedEvents):
            self.event(gamepad.Codes.EV_SYN, gamepad.Codes.SYN_DROPPED, 0)
        self.assertEqual(self.control.calls[-1], ("release", "syn_dropped"))
        self.assertFalse(self.worker.snapshot()["connected"])

    def test_disconnect_latches_off_and_reconnect_requires_a_again(self):
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.worker._fail_closed("device_removed", disconnect=True)
        self.assertEqual(self.control.calls[-1], ("release", "device_removed"))

        self.control.calls.clear()
        self.worker._prepare_candidate(self.pad)
        self.event(gamepad.Codes.EV_ABS, gamepad.Codes.ABS_RZ, 1023)
        self.sync()
        self.assertFalse(any(call[0] == "update" for call in self.control.calls))
        self.assertFalse(self.worker.snapshot()["armed"])

        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 0)
        self.event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1)
        self.assertTrue(any(call[0] == "acquire" for call in self.control.calls))

    def test_reader_failure_releases_owned_control(self):
        events = (
            Event(gamepad.Codes.EV_KEY, gamepad.Codes.BTN_SOUTH, 1),
            Event(gamepad.Codes.EV_SYN, gamepad.Codes.SYN_REPORT, 0),
        )

        class BrokenDevice(FakeDevice):
            def read_loop(inner_self):
                yield from events
                raise OSError("unplugged")

        device = BrokenDevice("/dev/input/event8", keys=STANDARD_KEYS, axes=analog_axes(), uniq="broken")
        self.worker._read_candidate(candidate(device=device, uniq="broken"))
        self.assertEqual(self.control.calls[-1], ("release", "device_removed"))
        self.assertEqual(self.worker.snapshot()["reader_failures"], 1)


if __name__ == "__main__":
    unittest.main()
