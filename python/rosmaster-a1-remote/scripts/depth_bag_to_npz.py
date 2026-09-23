#!/usr/bin/env python3
"""Cut a few depth frames and their camera_info out of a bag, for a test fixture.

The floor model's real-data tests (tests/python/test_floor_fixtures.py)
replay frames the car's own depth camera recorded. A bag is far too big to
commit, so this keeps a handful of frames, evenly spaced over a window, plus
the intrinsics, in one compressed .npz:

  depth   (N, H, W) uint16, millimetres, as published
  k       (9,) float64, the camera_info K matrix
  width, height
  stamps  (N,) float64, seconds

Usage:
  .venv/bin/python scripts/depth_bag_to_npz.py <bag>.db3 <out>.npz [--frames 5] [--from S] [--to S]
      [--image-topic /camera/camera/depth/image_rect_raw] [--info-topic /camera/camera/depth/camera_info]

--from and --to are seconds from the start of the bag. Reads the rosbag2
sqlite3 file directly (CDR), no ROS; needs numpy.
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys

import numpy as np

IMAGE_TOPIC = "/camera/camera/depth/image_rect_raw"
INFO_TOPIC = "/camera/camera/depth/camera_info"


class _CDR:
    """Little-endian CDR reader over a rosbag2 message blob (4-byte header)."""

    def __init__(self, data: bytes) -> None:
        self.d = data
        self.p = 4

    def _align(self, n: int) -> None:
        rem = (self.p - 4) % n
        if rem:
            self.p += n - rem

    def u8(self) -> int:
        v = self.d[self.p]
        self.p += 1
        return v

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

    def string(self) -> str:
        n = self.u32()
        s = self.d[self.p : self.p + n - 1].decode()
        self.p += n
        return s

    def f64s(self, n: int) -> np.ndarray:
        self._align(8)
        v = np.frombuffer(self.d, dtype="<f8", count=n, offset=self.p).copy()
        self.p += 8 * n
        return v

    def u8seq(self) -> bytes:
        n = self.u32()
        v = self.d[self.p : self.p + n]
        self.p += n
        return v


def _header(c: _CDR) -> float:
    sec, nsec = c.i32(), c.u32()
    c.string()  # frame_id
    return sec + nsec / 1e9


def decode_image(data: bytes) -> tuple[float, np.ndarray]:
    """(stamp, depth) for a 16UC1 sensor_msgs/Image."""
    c = _CDR(data)
    stamp = _header(c)
    height, width = c.u32(), c.u32()
    encoding = c.string()
    bigendian = c.u8()
    step = c.u32()
    raw = c.u8seq()
    if encoding.lower() not in {"16uc1", "mono16"}:
        raise ValueError(f"expected 16UC1 depth, got {encoding}")
    dtype = ">u2" if bigendian else "<u2"
    depth = np.frombuffer(raw, dtype=dtype).reshape((height, step // 2))[:, :width]
    return stamp, depth.astype(np.uint16)


def decode_camera_info(data: bytes) -> tuple[int, int, np.ndarray]:
    """(width, height, K) for a sensor_msgs/CameraInfo."""
    c = _CDR(data)
    _header(c)
    height, width = c.u32(), c.u32()
    c.string()  # distortion_model
    c.f64s(c.u32())  # d
    return width, height, c.f64s(9)


def extract(db_path: str, frames: int, start_s: float | None, end_s: float | None, image_topic: str, info_topic: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        topics = {name: topic_id for topic_id, name in con.execute("SELECT id, name FROM topics")}
        missing = [t for t in (image_topic, info_topic) if t not in topics]
        if missing:
            raise SystemExit(f"topics not in the bag: {', '.join(missing)} (have {', '.join(sorted(topics))})")
        info_row = con.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1", (topics[info_topic],)
        ).fetchone()
        rows = con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topics[image_topic],)
        ).fetchall()
    finally:
        con.close()
    if info_row is None or not rows:
        raise SystemExit("the bag has no camera_info or no depth frames")
    first = rows[0][0]
    window = [
        data
        for timestamp, data in rows
        if (start_s is None or (timestamp - first) / 1e9 >= start_s) and (end_s is None or (timestamp - first) / 1e9 <= end_s)
    ]
    if len(window) < frames:
        raise SystemExit(f"only {len(window)} depth frames in the window, wanted {frames}")
    picks = np.linspace(0, len(window) - 1, frames).round().astype(int)
    decoded = [decode_image(window[i]) for i in picks]
    width, height, k = decode_camera_info(info_row[0])
    return {
        "depth": np.stack([depth for _, depth in decoded]),
        "k": k,
        "width": np.int32(width),
        "height": np.int32(height),
        "stamps": np.array([stamp for stamp, _ in decoded]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--from", dest="start_s", type=float, default=None)
    parser.add_argument("--to", dest="end_s", type=float, default=None)
    parser.add_argument("--image-topic", default=IMAGE_TOPIC)
    parser.add_argument("--info-topic", default=INFO_TOPIC)
    args = parser.parse_args(argv)
    fixture = extract(args.bag, args.frames, args.start_s, args.end_s, args.image_topic, args.info_topic)
    np.savez_compressed(args.out, **fixture)
    depth = fixture["depth"]
    print(f"wrote {args.out}: {depth.shape[0]} frames {depth.shape[2]}x{depth.shape[1]}, fx={fixture['k'][0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
