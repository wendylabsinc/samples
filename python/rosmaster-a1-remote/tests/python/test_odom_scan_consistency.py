"""Tests for scripts/odom_scan_consistency.py.

The tool reads a rosbag2 sqlite file with no ROS installed, so everything
here is synthetic: hand-encoded CDR payloads for the decoders, generated
point sets for the ICP, generated rows for the summary. numpy is a real
dependency (it is in the .venv), nothing else is.

Run: .venv/bin/python -m unittest tests.python.test_odom_scan_consistency
"""
from __future__ import annotations

import contextlib
import io
import math
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import odom_scan_consistency as osc  # noqa: E402


class CDRWriter:
    """Just enough CDR (little-endian, 4-byte encapsulation header) to
    round-trip the two message layouts the tool decodes."""

    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n: int) -> None:
        rem = (len(self.buf) - 4) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def u32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", v)

    def f32(self, v: float) -> None:
        self._align(4)
        self.buf += struct.pack("<f", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.buf += struct.pack("<d", v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.buf += raw

    def f32seq(self, values) -> None:
        self.u32(len(values))
        for v in values:
            self.f32(v)


def laser_scan_payload(sec, nsec, angle_min, angle_inc, ranges):
    w = CDRWriter()
    w.i32(sec); w.u32(nsec); w.string("laser_frame")
    w.f32(angle_min); w.f32(angle_min + angle_inc * (len(ranges) - 1)); w.f32(angle_inc)
    w.f32(0.0); w.f32(0.1); w.f32(0.03); w.f32(12.0)
    w.f32seq(ranges); w.f32seq([])
    return bytes(w.buf)


def odometry_payload(sec, nsec, x, y, yaw, vx, wz):
    w = CDRWriter()
    w.i32(sec); w.u32(nsec); w.string("odom"); w.string("base_link")
    w.f64(x); w.f64(y); w.f64(0.0)
    w.f64(0.0); w.f64(0.0); w.f64(math.sin(yaw / 2)); w.f64(math.cos(yaw / 2))
    for _ in range(36): w.f64(0.0)
    w.f64(vx); w.f64(0.0); w.f64(0.0); w.f64(0.0); w.f64(0.0); w.f64(wz)
    for _ in range(36): w.f64(0.0)
    return bytes(w.buf)


def two_walls(n=120):
    """An L of points: a wall along x at y=2 and a wall along y at x=3."""
    xs = np.linspace(-2.0, 3.0, n); ys = np.linspace(-1.0, 2.0, n)
    return np.concatenate([np.stack([xs, np.full(n, 2.0)], 1), np.stack([np.full(n, 3.0), ys], 1)])


def post_grid(step=1.0, extent=2.5):
    """A 6 x 6 grid of posts 1 m apart: every point has a unique neighbour,
    so nearest-neighbour correspondence is exact for any displacement under
    half the spacing. Straight walls, by contrast, let points slide
    tangentially and point-to-point ICP settles at a wrong fixed point."""
    axis = np.arange(-extent, extent + 1e-9, step)
    gx, gy = np.meshgrid(axis, axis)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def rot(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


class DecoderTests(unittest.TestCase):
    def test_laser_scan_decodes_stamp_angles_and_ranges(self):
        payload = laser_scan_payload(1700000000, 250_000_000, -math.pi, 0.01575, [1.0, 2.5, 0.0, 12.0])
        stamp, amin, ainc, ranges = osc.decode_laser_scan(payload)
        self.assertAlmostEqual(stamp, 1700000000.25, places=6)
        self.assertAlmostEqual(amin, -math.pi, places=6)
        self.assertAlmostEqual(ainc, 0.01575, places=6)
        np.testing.assert_allclose(ranges, [1.0, 2.5, 0.0, 12.0], rtol=1e-6)

    def test_odometry_decodes_pose_yaw_and_forward_speed(self):
        payload = odometry_payload(1700000001, 0, 3.5, -1.25, 0.7, 0.62, 0.3)
        stamp, x, y, yaw, vx = osc.decode_odometry(payload)
        self.assertEqual(stamp, 1700000001.0)
        self.assertAlmostEqual(x, 3.5); self.assertAlmostEqual(y, -1.25)
        self.assertAlmostEqual(yaw, 0.7, places=9); self.assertAlmostEqual(vx, 0.62)


class GeometryTests(unittest.TestCase):
    def test_scan_points_drops_out_of_range_beams(self):
        pts = osc.scan_points(0.0, math.pi / 2, np.array([1.0, 0.0, 9.0, 2.0]))
        np.testing.assert_allclose(pts, [[1.0, 0.0], [0.0, -2.0]], atol=1e-9)  # beam 1 too short, beam 2 too long; beam 3 points down -y

    def test_icp_recovers_a_known_rigid_transform(self):
        dst = post_grid()                                   # 36 posts, within the 30-pair minimum
        theta, t = 0.05, np.array([0.12, -0.05])            # moves a far post by ~0.31 m, under half the spacing
        src = (dst - t) @ rot(theta)                        # dst = R(theta) src + t  =>  src = R(-theta)(dst - t)
        result = osc.icp2d(src, dst)
        self.assertIsNotNone(result)
        got_theta, got_t, residual = result
        self.assertAlmostEqual(got_theta, theta, places=6)
        np.testing.assert_allclose(got_t, t, atol=1e-6)
        self.assertLess(residual, 1e-6)

    def test_icp_gives_up_without_enough_matches(self):
        self.assertIsNone(osc.icp2d(np.array([[0.0, 0.0]]), two_walls()))

    def test_body_delta_projects_onto_the_starting_heading(self):
        o0 = (0.0, 1.0, 1.0, math.pi / 2, 0.5)
        o1 = (0.5, 1.0, 1.5, math.pi / 2 + 0.1, 0.5)   # moved +0.5 along its heading (+y), turned 0.1
        dth, fwd, lat = osc.body_delta(o0, o1)
        self.assertAlmostEqual(dth, 0.1); self.assertAlmostEqual(fwd, 0.5); self.assertAlmostEqual(lat, 0.0)

    def test_body_delta_wraps_the_heading_change(self):
        o0 = (0.0, 0.0, 0.0, math.pi - 0.05, 0.0)
        o1 = (0.5, 0.0, 0.0, -math.pi + 0.05, 0.0)
        self.assertAlmostEqual(osc.body_delta(o0, o1)[0], 0.1)


class SummaryTests(unittest.TestCase):
    def rows(self, sign, ratio=1.0):
        # (t_rel, icp_dth, icp_fwd, icp_lat, odom_dth, odom_fwd, residual, span)
        return [(k * 0.5, 0.2 * ratio, sign * 0.35 * ratio, 0.02, 0.2, 0.35, 0.02, 0.5) for k in range(20)]

    def test_a_consistent_bag_is_reported_as_such(self):
        s = osc.summarise(self.rows(+1))
        self.assertEqual((s["fwd_same"], s["fwd_opposite"]), (20, 0))
        self.assertEqual((s["rot_same"], s["rot_opposite"]), (20, 0))
        self.assertAlmostEqual(s["speed_ratio"], 1.0); self.assertAlmostEqual(s["rot_ratio"], 1.0)
        self.assertEqual(s["verdicts"], ["consistent"])

    def test_a_rotated_scan_or_inverted_speed_is_called_out(self):
        s = osc.summarise(self.rows(-1))
        self.assertEqual((s["fwd_same"], s["fwd_opposite"]), (0, 20))
        self.assertIn("scan rotated 180 deg or speed sign inverted", s["verdicts"])

    def test_scale_errors_are_called_out(self):
        s = osc.summarise(self.rows(+1, ratio=0.6))
        self.assertIn("speed scale off (ratio 0.60)", s["verdicts"])
        self.assertIn("rotation scale off (ratio 0.60)", s["verdicts"])

    def test_no_windows_is_its_own_verdict(self):
        s = osc.summarise([])
        self.assertEqual(s["verdicts"], ["no moving windows"])
        self.assertEqual(
            sorted(s),
            ["fwd_opposite", "fwd_same", "lateral_m", "residual_m", "rot_opposite", "rot_ratio", "rot_same", "speed_ratio", "verdicts", "windows"],
        )
        self.assertEqual(s["windows"], 0)
        self.assertEqual((s["fwd_same"], s["fwd_opposite"], s["rot_same"], s["rot_opposite"]), (0, 0, 0, 0))
        for key in ("speed_ratio", "rot_ratio", "lateral_m", "residual_m"):
            self.assertTrue(math.isnan(s[key]), key)

    def test_partial_forward_agreement_is_called_out(self):
        # 12 of 20 windows agree in sign (60 %): below SIGN_AGREEMENT_MIN
        # (0.95) but not below the 50 % "inverted" threshold, so this is the
        # new, softer verdict, not "scan rotated 180 deg". Real-bag numbers
        # (task-3): forward 3 agree vs 200 opposite (~1.5 %) stays below 50 %
        # and keeps the "inverted" verdict; rotation 172 agree vs 6 opposite
        # (~97 %) is above 95 % and fires no rotation verdict at all.
        rows = [(k * 0.5, 0.2, 0.35 if k < 12 else -0.35, 0.02, 0.2, 0.35, 0.02, 0.5) for k in range(20)]
        s = osc.summarise(rows)
        self.assertEqual((s["fwd_same"], s["fwd_opposite"]), (12, 8))
        self.assertIn("forward sign agreement only 60 %", s["verdicts"])
        self.assertNotIn("scan rotated 180 deg or speed sign inverted", s["verdicts"])
        self.assertNotIn("consistent", s["verdicts"])

    def test_no_evidence_either_way_is_not_consistent(self):
        # fwd 0.10 m < MOVING_FWD_M (0.15), dth 0.05 rad < TURNING_RAD (0.12):
        # every window is too small to say anything either way.
        rows = [(k * 0.5, 0.05, 0.10, 0.0, 0.05, 0.10, 0.01, 0.5) for k in range(10)]
        s = osc.summarise(rows)
        self.assertIn("no forward evidence (10 windows below 0.15 m)", s["verdicts"])
        self.assertIn("no rotation evidence (10 windows below 0.12 rad)", s["verdicts"])
        self.assertNotIn("consistent", s["verdicts"])


class LagTests(unittest.TestCase):
    def test_best_lag_finds_a_shifted_gyro(self):
        # odometry yaw ramps 0.5 rad/s from t=10 to t=12; the scan "sees" the
        # same ramp 0.2 s later than the odometry stamps say.
        odom = [(t / 10, 0.0, 0.0, max(0.0, min(1.0, (t / 10 - 10.0) * 0.5)), 0.5) for t in range(0, 200)]
        rows = []
        for k in range(0, 60):
            t0 = 8.0 + k * 0.1
            icp_dth = osc.yaw_at(odom, t0 + 0.5 - 0.2) - osc.yaw_at(odom, t0 - 0.2)
            rows.append((t0, icp_dth, 0.3, 0.0, 0.0, 0.3, 0.01, 0.5))
        tau, _ = osc.best_lag(rows, odom, 0.0, [x / 20 for x in range(-10, 11)])
        self.assertAlmostEqual(tau, -0.2, places=6)

    def test_best_lag_uses_each_rows_own_span_not_a_hardcoded_half_second(self):
        # Same ramp and true offset as above, but the rows are built from a
        # 1.0 s window (icp_dth spans t0-0.2 to t0+0.8), not 0.5 s. A
        # best_lag that still assumes a fixed 0.5 s window compares against
        # the wrong odometry delta and recovers tau=0.0 with a large error
        # (confirmed against the old, unfixed best_lag before this change);
        # reading each row's own `span` field must still recover -0.2 exactly.
        odom = [(t / 10, 0.0, 0.0, max(0.0, min(1.0, (t / 10 - 10.0) * 0.5)), 0.5) for t in range(0, 200)]
        span = 1.0
        rows = []
        for k in range(0, 60):
            t0 = 8.0 + k * 0.1
            icp_dth = osc.yaw_at(odom, t0 + span - 0.2) - osc.yaw_at(odom, t0 - 0.2)
            rows.append((t0, icp_dth, 0.3, 0.0, 0.0, 0.3, 0.01, span))
        tau, err = osc.best_lag(rows, odom, 0.0, [x / 20 for x in range(-10, 11)])
        self.assertAlmostEqual(tau, -0.2, places=6)
        self.assertAlmostEqual(err, 0.0, places=6)


class MainTests(unittest.TestCase):
    """main() through a patched read_bag: argument parsing, window filtering,
    verdict composition and the exit status, without a bag on disk."""

    def synthetic_bag(self, vx_sign=1, t_shift=0.0):
        """A car driving straight for 0.6 s (forward evidence), then turning
        in place (rotation evidence): 12 scans at 10 Hz, odometry at 20 Hz.
        Each scan is the posts seen from the car, so ICP between scans
        recovers the car's own motion exactly; vx_sign=-1 records the
        odometry with the speed sign inverted, the defect the tool exists to
        catch; t_shift offsets every odometry stamp, the defect that shows up
        as a timing offset verdict."""
        posts = post_grid()

        def pose(t):
            s = t - 100.0
            if s <= 0.6:
                return 0.32 * s, 0.0, 0.0                    # straight at 0.32 m/s: forward evidence
            return 0.192, 0.0, 0.26 * (s - 0.6)              # then turning in place at 0.26 rad/s: rotation evidence

        scans = []
        for k in range(12):
            t = 100.0 + 0.1 * k
            x, y, yaw = pose(t)
            scans.append((t, (posts - np.array([x, y])) @ rot(yaw)))   # world -> car frame: R(-yaw)(p - pos)
        odom = []
        for k in range(24):
            t = 100.0 + 0.05 * k
            x, y, yaw = pose(t)
            odom.append((t + t_shift, vx_sign * x, y, yaw, vx_sign * 0.32))
        return scans, odom

    def run_main(self, argv, bag):
        with mock.patch.object(osc, "read_bag", return_value=bag):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                status = osc.main(argv)
        return status, out.getvalue()

    def test_a_consistent_bag_exits_zero(self):
        status, out = self.run_main(["fake.db3"], self.synthetic_bag())
        self.assertEqual(status, 0)
        self.assertIn("verdict: consistent", out)
        self.assertIn("scan-vs-odometry lag: +0.00 s", out)

    def test_an_inverted_speed_exits_two_with_the_rotation_verdict(self):
        status, out = self.run_main(["fake.db3"], self.synthetic_bag(vx_sign=-1))
        self.assertEqual(status, 2)
        self.assertIn("verdict: scan rotated 180 deg or speed sign inverted", out)

    def test_from_and_to_limit_the_windows(self):
        status, out = self.run_main(["fake.db3", "--from", "0.3", "--to", "0.45"], self.synthetic_bag())
        self.assertTrue(out.startswith("1 moving windows"), out.splitlines()[0])

    def test_a_timing_offset_exits_two(self):
        status, out = self.run_main(["fake.db3"], self.synthetic_bag(t_shift=0.3))
        self.assertEqual(status, 2)
        self.assertIn("timing offset", out)
        lag_line = next(line for line in out.splitlines() if "scan-vs-odometry lag" in line)
        lag = float(lag_line.split("lag:")[1].split("s")[0].strip())
        self.assertLess(abs(abs(lag) - 0.3), 0.05, lag_line)


if __name__ == "__main__":
    unittest.main()
