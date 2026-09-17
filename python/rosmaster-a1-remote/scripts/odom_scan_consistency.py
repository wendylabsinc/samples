#!/usr/bin/env python3
"""Check a drive bag's odometry against its LiDAR scans, without ROS.

For every 0.5 s window while the car moves, a 2-D ICP aligns scan k+5 onto
scan k in the laser frame. The rigid transform it finds IS the car's own
motion in its body frame (laser at base_link, identity rotation), so it is
compared directly with the odometry's body-frame displacement over the same
stamps: forward sign and magnitude, rotation sign and magnitude, and the
time offset that best aligns the two.

Written 2026-09-17 after slam_toolbox failed on a drive bag: this found the
scan rotated 180 degrees (forward sign opposite in 200 of 203 windows) and,
via the rotation ratio, the polluted gyro bias. Expected on a good bag:
sign agreement above 95 % both ways, ratios within 0.9-1.1, lag below
0.15 s (about +0.1 s is this car's normal offset: start-of-sweep stamp
plus odometry latency).

Usage:
  .venv/bin/python scripts/odom_scan_consistency.py <bag>.db3 [--from S] [--to S]

Reads the rosbag2 sqlite3 file directly (topics /scan and /odom, CDR).
Needs numpy. Exit status 0 when the verdict is "consistent", 2 otherwise.
"""
from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
import struct
import sys

import numpy as np

STEP_SCANS = 5          # scan k vs k+5: 0.5 s at the T-mini's 10 Hz
STRIDE_SCANS = 2
SCAN_HZ = 10.0          # the T-mini's scan rate; STEP_SCANS / SCAN_HZ is the window length in seconds
REJECT_M = 0.6          # nearest-neighbour pairs farther than this are ignored
RANGE_MIN_M, RANGE_MAX_M = 0.15, 8.0
MOVING_FWD_M = 0.15     # a window counts for the forward checks above this
TURNING_RAD = 0.12      # ...and for the rotation checks above this
LAG_LIMIT_S = 0.15  # the T-mini's start-of-sweep stamp plus odometry latency put a healthy bag near +0.1 s
SIGN_AGREEMENT_MIN = 0.95
Row = tuple            # (t_rel, icp_dth, icp_fwd, icp_lat, odom_dth, odom_fwd, residual, span)


class _CDR:
    """Little-endian CDR reader over a rosbag2 message blob (4-byte header)."""

    def __init__(self, data: bytes) -> None:
        self.d = data
        self.p = 4

    def _align(self, n: int) -> None:
        rem = (self.p - 4) % n
        if rem:
            self.p += n - rem

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self) -> float:
        self._align(4)
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def f64(self) -> float:
        self._align(8)
        v = struct.unpack_from("<d", self.d, self.p)[0]
        self.p += 8
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.d[self.p:self.p + n - 1].decode()
        self.p += n
        return s

    def f32seq(self) -> np.ndarray:
        n = self.u32()
        self._align(4)
        v = np.frombuffer(self.d, dtype="<f4", count=n, offset=self.p).astype(np.float64)
        self.p += 4 * n
        return v


def decode_laser_scan(data: bytes):
    c = _CDR(data)
    sec, nsec = c.i32(), c.u32()
    c.string()                                   # frame_id
    angle_min = c.f32(); c.f32(); angle_inc = c.f32()
    c.f32(); c.f32(); c.f32(); c.f32()           # time_increment, scan_time, range_min, range_max
    ranges = c.f32seq()
    return sec + nsec / 1e9, angle_min, angle_inc, ranges


def decode_odometry(data: bytes):
    c = _CDR(data)
    sec, nsec = c.i32(), c.u32()
    c.string(); c.string()                       # frame_id, child_frame_id
    x, y, _z = c.f64(), c.f64(), c.f64()
    qx, qy, qz, qw = c.f64(), c.f64(), c.f64(), c.f64()
    for _ in range(36):
        c.f64()                                  # pose covariance
    vx = c.f64()
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return sec + nsec / 1e9, x, y, yaw, vx


def scan_points(angle_min: float, angle_inc: float, ranges: np.ndarray, r_min=RANGE_MIN_M, r_max=RANGE_MAX_M) -> np.ndarray:
    angles = angle_min + angle_inc * np.arange(len(ranges))
    ok = np.isfinite(ranges) & (ranges > r_min) & (ranges < r_max)
    return np.stack([ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])], 1)


def icp2d(src: np.ndarray, dst: np.ndarray, iters: int = 40, reject: float = REJECT_M):
    """Rigid 2-D ICP: returns (theta, t, median residual) with dst ~ R(theta) src + t,
    or None when fewer than 30 point pairs survive the reject radius."""
    theta, t = 0.0, np.zeros(2)
    for _ in range(iters):
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        moved = src @ R.T + t
        d2 = ((moved[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
        nearest = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(moved)), nearest])
        keep = dist < reject
        if keep.sum() < 30:
            return None
        a, b = src[keep], dst[nearest[keep]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _S, Vt = np.linalg.svd(H)
        Rn = Vt.T @ U.T
        if np.linalg.det(Rn) < 0:
            Vt[1] *= -1
            Rn = Vt.T @ U.T
        theta_n = math.atan2(Rn[1, 0], Rn[0, 0])
        t_n = cb - ca @ Rn.T
        converged = abs(theta_n - theta) < 1e-6 and np.abs(t_n - t).max() < 1e-5
        theta, t = theta_n, t_n
        if converged:
            break
    return theta, t, float(np.median(dist[keep]))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def body_delta(o0, o1):
    """(dtheta, forward, lateral) of o1 relative to o0, in o0's heading frame."""
    _t0, x0, y0, yaw0, _v0 = o0
    _t1, x1, y1, yaw1, _v1 = o1
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(yaw0), math.sin(yaw0)
    return wrap(yaw1 - yaw0), c * dx + s * dy, -s * dx + c * dy


def read_bag(db_path: str):
    con = sqlite3.connect(db_path)
    topics = {name: tid for tid, name in con.execute("select id, name from topics")}
    for needed in ("/scan", "/odom"):
        if needed not in topics:
            raise SystemExit(f"{db_path}: no {needed} topic (has {sorted(topics)})")
    scans = []
    for (data,) in con.execute("select data from messages where topic_id=? order by timestamp", (topics["/scan"],)):
        t, amin, ainc, ranges = decode_laser_scan(data)
        scans.append((t, scan_points(amin, ainc, ranges)))
    odom = [decode_odometry(data) for (data,) in con.execute("select data from messages where topic_id=? order by timestamp", (topics["/odom"],))]
    scans.sort(key=lambda s: s[0])
    odom.sort(key=lambda o: o[0])
    return scans, odom


def _odom_at(odom, t):
    stamps = [o[0] for o in odom]
    i = min(bisect.bisect_left(stamps, t), len(odom) - 1)
    return odom[i]


def _stamps(odom):
    return [o[0] for o in odom]


def _yaw_interp(odom, stamps, t: float) -> float:
    """Odometry yaw at t, linearly interpolated (wrap-aware) between the samples
    around it; clamped to the first/last sample outside the recording."""
    i = bisect.bisect_left(stamps, t)
    if i <= 0:
        return odom[0][3]
    if i >= len(odom):
        return odom[-1][3]
    t0, t1 = stamps[i - 1], stamps[i]
    y0, y1 = odom[i - 1][3], odom[i][3]
    f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return y0 + f * wrap(y1 - y0)


def yaw_at(odom, t: float) -> float:
    return _yaw_interp(odom, _stamps(odom), t)


def pair_windows(scans, odom, step=STEP_SCANS, stride=STRIDE_SCANS, t_from=None, t_to=None):
    rows = []
    t_bag0 = scans[0][0] if scans else 0.0
    for k in range(0, len(scans) - step, stride):
        t0, p0 = scans[k]
        t1, p1 = scans[k + step]
        t_rel = t0 - t_bag0
        if t_from is not None and t_rel < t_from:
            continue
        if t_to is not None and t_rel > t_to:
            continue
        o0, o1 = _odom_at(odom, t0), _odom_at(odom, t1)
        odom_dth, odom_fwd, _lat = body_delta(o0, o1)
        if abs(o0[4]) < 0.25 and abs(o1[4]) < 0.25 and abs(odom_dth) < 0.05:
            continue                                # the car is not moving
        result = icp2d(p1, p0)
        if result is None:
            continue
        theta, t, residual = result
        rows.append((t_rel, theta, float(t[0]), float(t[1]), odom_dth, odom_fwd, residual, t1 - t0))
    return rows


def best_lag(rows, odom, t_bag0, taus):
    """Shift the odometry window by tau and return the tau minimising the summed
    |icp rotation - odometry rotation|. Scan content corresponds to the
    odometry state at stamp + tau: a scan stamped later than what it saw (a
    delayed sensor pipeline) gives a negative tau; the T-mini's start-of-sweep
    stamp gives about +0.1 s on this car (mid-sweep plus the odometry's own
    latency). Ties go to the smallest |tau|."""
    stamps = _stamps(odom)
    best = None
    for tau in taus:
        err = 0.0
        for t_rel, icp_dth, *_rest, span in rows:
            ts = t_bag0 + t_rel + tau
            err += abs(icp_dth - wrap(_yaw_interp(odom, stamps, ts + span) - _yaw_interp(odom, stamps, ts)))
        if best is None or err < best[1] - 1e-9 or (abs(err - best[1]) <= 1e-9 and abs(tau) < abs(best[0])):
            best = (tau, err)
    return best


def summarise(rows) -> dict:
    if not rows:
        return {
            "windows": 0,
            "fwd_same": 0, "fwd_opposite": 0, "speed_ratio": float("nan"),
            "rot_same": 0, "rot_opposite": 0, "rot_ratio": float("nan"),
            "lateral_m": float("nan"), "residual_m": float("nan"),
            "verdicts": ["no moving windows"],
        }
    fwd = [(r[2], r[5]) for r in rows if abs(r[5]) > MOVING_FWD_M]
    rot = [(r[1], r[4]) for r in rows if abs(r[4]) > TURNING_RAD]
    fwd_same = sum(1 for a, b in fwd if a * b > 0)
    rot_same = sum(1 for a, b in rot if a * b > 0)
    speed_ratio = statistics.median(abs(a) / abs(b) for a, b in fwd) if fwd else float("nan")
    rot_ratio = statistics.median(a / b for a, b in rot) if rot else float("nan")
    verdicts = []
    if not fwd:
        verdicts.append(f"no forward evidence ({len(rows)} windows below {MOVING_FWD_M} m)")
    if not rot:
        verdicts.append(f"no rotation evidence ({len(rows)} windows below {TURNING_RAD} rad)")
    if fwd and fwd_same < 0.5 * len(fwd):
        verdicts.append("scan rotated 180 deg or speed sign inverted")
    elif fwd and fwd_same < SIGN_AGREEMENT_MIN * len(fwd):
        verdicts.append(f"forward sign agreement only {round(100 * fwd_same / len(fwd))} %")
    if rot and rot_same < 0.5 * len(rot):
        verdicts.append("scan mirrored or gyro sign inverted")
    elif rot and rot_same < SIGN_AGREEMENT_MIN * len(rot):
        verdicts.append(f"rotation sign agreement only {round(100 * rot_same / len(rot))} %")
    if fwd and not (0.9 <= speed_ratio <= 1.1):
        verdicts.append(f"speed scale off (ratio {speed_ratio:.2f})")
    if rot and not (0.9 <= rot_ratio <= 1.1):
        verdicts.append(f"rotation scale off (ratio {rot_ratio:.2f})")
    return {
        "windows": len(rows),
        "fwd_same": fwd_same, "fwd_opposite": len(fwd) - fwd_same, "speed_ratio": speed_ratio,
        "rot_same": rot_same, "rot_opposite": len(rot) - rot_same, "rot_ratio": rot_ratio,
        "lateral_m": statistics.median(abs(r[3]) for r in rows),
        "residual_m": statistics.median(r[6] for r in rows),
        "verdicts": verdicts or ["consistent"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="rosbag2 .db3 file with /scan and /odom")
    parser.add_argument("--from", dest="t_from", type=float, default=None, help="bag-relative start, s")
    parser.add_argument("--to", dest="t_to", type=float, default=None, help="bag-relative end, s")
    args = parser.parse_args(argv)
    scans, odom = read_bag(args.bag)
    rows = pair_windows(scans, odom, t_from=args.t_from, t_to=args.t_to)
    s = summarise(rows)
    print(f"{s['windows']} moving windows of {STEP_SCANS / SCAN_HZ:.1f} s")
    if s["windows"]:
        print(f"forward:  ICP agrees with odometry {s['fwd_same']}, opposite {s['fwd_opposite']}; |ICP|/|odom| median {s['speed_ratio']:.2f}")
        print(f"rotation: ICP agrees with gyro {s['rot_same']}, opposite {s['rot_opposite']}; ICP/gyro median {s['rot_ratio']:.2f}")
        print(f"lateral slip median {s['lateral_m']:.3f} m; ICP match residual median {s['residual_m']:.3f} m")
        tau, _err = best_lag(rows, odom, scans[0][0], [x / 50 for x in range(-25, 26)])
        print(f"scan-vs-odometry lag: {tau:+.2f} s (scan content corresponds to odometry at stamp+lag)")
        if abs(tau) > LAG_LIMIT_S:
            s["verdicts"] = [v for v in s["verdicts"] if v != "consistent"] + [f"timing offset {tau:+.2f} s"]
    print("verdict:", "; ".join(s["verdicts"]))
    return 0 if s["verdicts"] == ["consistent"] else 2


if __name__ == "__main__":
    sys.exit(main())
