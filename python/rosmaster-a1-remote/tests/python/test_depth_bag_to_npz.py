"""Tests for scripts/depth_bag_to_npz.py.

Synthetic throughout, like test_odom_scan_consistency.py: hand-encoded CDR
Image and CameraInfo payloads in a hand-built rosbag2 sqlite3 file, no ROS.

Run: .venv/bin/python -m unittest tests.python.test_depth_bag_to_npz
"""
from __future__ import annotations

import contextlib
import io
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import depth_bag_to_npz as tool  # noqa: E402


class CDRWriter:
    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n: int) -> None:
        rem = (len(self.buf) - 4) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def u8(self, v: int) -> None:
        self.buf += bytes([v])

    def u32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.buf += struct.pack("<d", v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.buf += raw

    def header(self, sec: int, nsec: int, frame_id: str = "camera_depth_optical_frame") -> None:
        self.i32(sec)
        self.u32(nsec)
        self.string(frame_id)


def image_blob(sec: int, depth: np.ndarray) -> bytes:
    w = CDRWriter()
    w.header(sec, 500_000_000)
    height, width = depth.shape
    w.u32(height)
    w.u32(width)
    w.string("16UC1")
    w.u8(0)
    w.u32(width * 2)
    raw = depth.astype("<u2").tobytes()
    w.u32(len(raw))
    w.buf += raw
    return bytes(w.buf)


def info_blob(k: list[float], width: int = 4, height: int = 3) -> bytes:
    w = CDRWriter()
    w.header(1, 0)
    w.u32(height)
    w.u32(width)
    w.string("plumb_bob")
    w.u32(5)
    for _ in range(5):
        w.f64(0.0)
    for v in k:
        w.f64(v)
    for v in [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0.0] * 12:
        w.f64(float(v))
    for _ in range(6):
        w.u32(0)
    w.u8(0)
    return bytes(w.buf)


K = [385.196, 0.0, 321.163, 0.0, 385.196, 234.056, 0.0, 0.0, 1.0]


def write_bag(path: Path, frames: int = 10) -> list[np.ndarray]:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    con.execute("INSERT INTO topics VALUES (1, ?, 'sensor_msgs/msg/Image', 'cdr', '')", (tool.IMAGE_TOPIC,))
    con.execute("INSERT INTO topics VALUES (2, ?, 'sensor_msgs/msg/CameraInfo', 'cdr', '')", (tool.INFO_TOPIC,))
    depths = []
    for i in range(frames):
        depth = np.full((3, 4), 1000 + i, dtype=np.uint16)
        depths.append(depth)
        stamp_ns = 10_000_000_000 + i * 100_000_000
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)", (stamp_ns, image_blob(10 + i, depth)))
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (2, ?, ?)", (stamp_ns, info_blob(K)))
    con.commit()
    con.close()
    return depths


class DecodeTests(unittest.TestCase):
    def test_an_image_decodes_to_its_millimetres(self):
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
        stamp, decoded = tool.decode_image(image_blob(7, depth))
        self.assertAlmostEqual(stamp, 7.5)
        np.testing.assert_array_equal(decoded, depth)

    def test_camera_info_decodes_to_its_k(self):
        width, height, k = tool.decode_camera_info(info_blob(K, width=640, height=480))
        self.assertEqual((width, height), (640, 480))
        np.testing.assert_allclose(k, K)


class ExtractTests(unittest.TestCase):
    def test_frames_are_spread_evenly_over_the_window_and_saved(self):
        tmp = Path(tempfile.mkdtemp())
        bag, out = tmp / "bag.db3", tmp / "fixture.npz"
        depths = write_bag(bag)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main([str(bag), str(out), "--frames", "3", "--from", "0.2", "--to", "0.6"]), 0)
        fixture = np.load(out)
        self.assertEqual(fixture["depth"].shape, (3, 3, 4))
        self.assertEqual([int(frame[0, 0]) for frame in fixture["depth"]], [int(depths[i][0, 0]) for i in (2, 4, 6)])
        np.testing.assert_allclose(fixture["k"], K)
        self.assertEqual(int(fixture["width"]), 4)

    def test_too_few_frames_in_the_window_is_an_error(self):
        tmp = Path(tempfile.mkdtemp())
        bag = tmp / "bag.db3"
        write_bag(bag, frames=2)
        with self.assertRaises(SystemExit):
            tool.extract(str(bag), 5, None, None, tool.IMAGE_TOPIC, tool.INFO_TOPIC)


if __name__ == "__main__":
    unittest.main()
