#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import queue
import socket
import ssl
import threading
import time
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image as RosImage
from sensor_msgs.msg import Imu, JointState, LaserScan, MagneticField, PointCloud2
from std_msgs.msg import Float32, String


PORT = int(os.environ.get("PORT", "8091"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
# How long a drive command stays in force with no follow up before the car is
# zeroed. This is a deadman, not a throttle refresh: the operator asked for the
# car to hold its last command and keep going rather than restuttering every
# time a packet is late, which on this Wi-Fi link happens constantly. 0.5 s was
# short enough that ordinary jitter kept cutting the throttle. Longer means the
# car keeps rolling for up to this long if the tab dies mid throttle, so the
# page still sends an explicit zero on neutral stick, hard stop, pad disconnect
# and page hide. This is the backstop for when none of those get through.
HTTPS_PORT = int(os.environ.get("HTTPS_PORT", "8443"))
CMD_TIMEOUT_S = float(os.environ.get("CMD_TIMEOUT_S", "3.0"))
DRIVE_MAX_AGE_MS = float(os.environ.get("DRIVE_MAX_AGE_MS", "400"))
DRIVE_REJECT_LOG_INTERVAL_S = float(os.environ.get("DRIVE_REJECT_LOG_INTERVAL_S", "1.0"))
PUBLISH_HZ = float(os.environ.get("PUBLISH_HZ", "20"))
# count_publishers/count_subscribers are unbounded rclpy graph queries. This
# is deliberately far slower than PUBLISH_HZ: the cache only needs to be
# fresh enough for a status panel and a readiness check, and sampling it on
# its own thread rather than in the /cmd_vel publish tick is the whole point
# (see _sample_graph_counts_loop) -- a slow tick here costs a stale reading,
# never a stalled command.
GRAPH_COUNT_SAMPLE_INTERVAL_S = float(os.environ.get("GRAPH_COUNT_SAMPLE_INTERVAL_S", "0.5"))
MAX_LINEAR_X = float(os.environ.get("MAX_LINEAR_X", "1000.00"))
MAX_STEERING_Y = float(os.environ.get("MAX_STEERING_Y", "0.12"))
MAX_ANGULAR_Z = float(os.environ.get("MAX_ANGULAR_Z", "1.0"))
AUTO_SPEED = float(os.environ.get("AUTO_SPEED", "1.00"))
AUTO_STOP_DISTANCE = float(os.environ.get("AUTO_STOP_DISTANCE", "0.35"))
AUTO_AVOID_DISTANCE = float(os.environ.get("AUTO_AVOID_DISTANCE", "0.85"))
AUTO_MAX_STEERING = float(os.environ.get("AUTO_MAX_STEERING", "0.12"))
AUTO_CLEAR_DISTANCE = float(os.environ.get("AUTO_CLEAR_DISTANCE", "1.60"))
AUTO_BRAKE_S = float(os.environ.get("AUTO_BRAKE_S", "0.20"))
AUTO_TURN_OUT_S = float(os.environ.get("AUTO_TURN_OUT_S", "1.45"))
# The blind reverse budget ===================================================
#
# RESIDUAL RISK, and it is not removed by anything below. This car has no rear
# sensor. The LiDAR sweeps roughly plus or minus 105 degrees across the front
# and the depth camera faces forward, so everything behind the rear bumper is
# space no sensor on this vehicle has ever observed. It is an Ackermann
# chassis and cannot turn in place to go and look. Every metre of reverse is
# therefore a metre driven blind, and the only real fix is a rear sensor.
# Until one is fitted these numbers do not make reversing safe, they only make
# the unsafe part small and finite.
#
# AUTO_REVERSE_BUDGET_M is 0.25 m because the planner already models this car
# as a disc of radius AUTO_GAP_SAFETY_RADIUS_M, 0.32 m, when it inflates
# obstacles. A total reverse shorter than that radius keeps the back of the car
# inside the footprint it occupied when it braked, which is ground the forward
# sensors watched on the way in. Past that the car is somewhere it has never
# looked.
#
# AUTO_REVERSE_BUDGET_S is 1.50 s, which at the reverse speed below covers
# 0.24 m, so on level ground the clock runs out first and the distance bound is
# there for the case the clock cannot catch: a slope, where the car rolls
# faster than it was asked to.
#
# AUTO_REVERSE_SPEED_MAX is 0.16 m/s, the cap the previous code already had.
# What it did not have was a ceiling: it took the larger of that and 45 percent
# of the cruise speed, and with the default 1.0 m/s cruise that meant reversing
# blind at 0.45 m/s. The reverse no longer scales with the cruise speed at all.
#
# All three are budgets for one stuck episode, shared across every attempt
# within it, and every one of them can only be lowered by the environment. A
# safety bound an environment variable can undo is not a safety bound.
AUTO_REVERSE_SPEED_MAX = max(0.0, min(float(os.environ.get("AUTO_REVERSE_SPEED_MAX", "0.16")), 0.16))
AUTO_REVERSE_BUDGET_S = max(0.0, min(float(os.environ.get("AUTO_REVERSE_BUDGET_S", "1.50")), 1.50))
AUTO_REVERSE_BUDGET_M = max(0.0, min(float(os.environ.get("AUTO_REVERSE_BUDGET_M", "0.25")), 0.25))
AUTO_CRUISE_CLEAR_MARGIN_M = float(os.environ.get("AUTO_CRUISE_CLEAR_MARGIN_M", "0.35"))
AUTO_GAP_FOV_DEG = float(os.environ.get("AUTO_GAP_FOV_DEG", "105"))
AUTO_GAP_SAFETY_RADIUS_M = float(os.environ.get("AUTO_GAP_SAFETY_RADIUS_M", "0.32"))
AUTO_GAP_DISPARITY_M = float(os.environ.get("AUTO_GAP_DISPARITY_M", "0.55"))
AUTO_GAP_MIN_WIDTH_DEG = float(os.environ.get("AUTO_GAP_MIN_WIDTH_DEG", "12"))
LIDAR_STALE_S = float(os.environ.get("LIDAR_STALE_S", "0.75"))
SENSOR_STALE_S = float(os.environ.get("SENSOR_STALE_S", "2.0"))
HP60C_DEPTH_TOPIC = os.environ.get("HP60C_DEPTH_TOPIC", "/ascamera_hp60c/camera_publisher/depth0/image_raw")
HP60C_RGB_TOPIC = os.environ.get("HP60C_RGB_TOPIC", "/ascamera_hp60c/camera_publisher/rgb0/image")
HP60C_POINTS_TOPIC = os.environ.get("HP60C_POINTS_TOPIC", "/ascamera_hp60c/camera_publisher/depth0/points")
HP60C_POINTS_ENABLED = os.environ.get("HP60C_POINTS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
HP60C_STALE_S = float(os.environ.get("HP60C_STALE_S", "1.0"))
HP60C_DEPTH_VALID_MIN_RATIO = float(os.environ.get("HP60C_DEPTH_VALID_MIN_RATIO", "0.005"))
HP60C_OBSTACLE_X_MIN = float(os.environ.get("HP60C_OBSTACLE_X_MIN", "0.22"))
HP60C_OBSTACLE_X_MAX = float(os.environ.get("HP60C_OBSTACLE_X_MAX", "0.78"))
HP60C_OBSTACLE_Y_MIN = float(os.environ.get("HP60C_OBSTACLE_Y_MIN", "0.06"))
HP60C_OBSTACLE_Y_MAX = float(os.environ.get("HP60C_OBSTACLE_Y_MAX", "0.55"))
HP60C_FLOOR_Y_MIN = float(os.environ.get("HP60C_FLOOR_Y_MIN", "0.68"))
HP60C_RED_DISTANCE_M = float(os.environ.get("HP60C_RED_DISTANCE_M", "0.45"))
HP60C_RED_MIN_PIXELS = int(os.environ.get("HP60C_RED_MIN_PIXELS", "32"))
# The Intel RealSense D435i, fitted in place of the HP60C. It publishes the
# four views the operator asked for as four separate topics. Topic names vary
# between driver versions and namespaces, so every one of them is overridable
# and must be checked against the live car before this is trusted.
REALSENSE_DEPTH_TOPIC = os.environ.get("REALSENSE_DEPTH_TOPIC", "/camera/camera/depth/image_rect_raw")
REALSENSE_INFRA1_TOPIC = os.environ.get("REALSENSE_INFRA1_TOPIC", "/camera/camera/infra1/image_rect_raw")
REALSENSE_INFRA2_TOPIC = os.environ.get("REALSENSE_INFRA2_TOPIC", "/camera/camera/infra2/image_rect_raw")
REALSENSE_COLOR_TOPIC = os.environ.get("REALSENSE_COLOR_TOPIC", "/camera/camera/color/image_raw")
REALSENSE_STALE_S = float(os.environ.get("REALSENSE_STALE_S", "1.0"))
REALSENSE_TOPICS = {
    "depth": REALSENSE_DEPTH_TOPIC,
    "infra1": REALSENSE_INFRA1_TOPIC,
    "infra2": REALSENSE_INFRA2_TOPIC,
    "color": REALSENSE_COLOR_TOPIC,
}
# How long an MJPEG response waits for a new frame before it closes and hands
# its request thread back. Every open stream owns a thread for as long as it
# runs, and a stream whose camera is silent writes nothing, so it never
# discovers that the browser has gone: without a bound it loops until the
# process dies, and each reconnect leaves another thread behind. The page
# reopens a closed stream on its own, staggered, so a live feed that stutters
# costs one reconnect rather than a wedged thread.
CAMERA_STREAM_IDLE_TIMEOUT_S = float(os.environ.get("CAMERA_STREAM_IDLE_TIMEOUT_S", "10.0"))

# Ceiling on how often a preview is drawn and JPEG encoded, per feed.
#
# Encoding happens on the executor thread that also ticks the /cmd_vel publish
# timer. With four tiles open the RealSense delivers about sixty frames a second
# between them, and encoding every one saturated that thread on the Jetson: the
# command timer starved, the base bridge stopped hearing from us, its watchdog
# zeroed the motors, and the car stopped responding to the operator. That is
# what this limit exists to prevent, and it is why the number is low.
#
# Six a second is still a usable view for driving, and the depth statistics the
# planner reads are computed on every frame regardless: only the drawing and
# encoding are rationed, never the sensing.
PREVIEW_MAX_FPS = float(os.environ.get("PREVIEW_MAX_FPS", "6.0"))
PREVIEW_MIN_INTERVAL_S = 1.0 / PREVIEW_MAX_FPS if PREVIEW_MAX_FPS > 0 else 0.0
# No depth feed is named here. Which depth camera this car carries is decided
# at runtime, and naming one in a static list is exactly the bug that left Auto
# Nav refusing to engage on a car whose HP60C had been replaced by a RealSense.
# sensors_snapshot adds the depth row for whichever camera is fitted.
DEFAULT_REQUIRED_SENSORS = (
    "lidar,imu,magnetometer,joint_states,velocity_feedback,voltage,"
    "base_firmware,base_probe_status,lidar_probe_status"
)
REQUIRED_SENSOR_NAMES = {
    name.strip()
    for name in os.environ.get("REQUIRED_SENSOR_NAMES", DEFAULT_REQUIRED_SENSORS).split(",")
    if name.strip()
}
HP60C_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
# Best effort, unlike the HP60C above, for two reasons. A best effort
# subscriber matches a publisher of either reliability, and realsense2_camera
# publishes its image topics best effort on some versions, so a reliable
# subscriber can sit there matching nothing while ros2 topic list shows the
# topic present, which is a miserable thing to debug on a live car. And a
# preview frame is worth nothing once it is late, so retransmitting one only
# delays the next.
REALSENSE_QOS = qos_profile_sensor_data


def clamp(value: float, limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def clamp_range(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def floor_finite(value: object, low: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(low, parsed)


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def zero_command(source: str = "zero") -> dict:
    return {
        "enabled": False,
        "linear_x": 0.0,
        "steering_y": 0.0,
        "angular_z": 0.0,
        "updated_at": 0.0,
        "source": source,
    }


def finite_percentile(values: list[float], percentile: float) -> float | None:
    finite = [value for value in values if math.isfinite(value) and value > 0.02]
    if not finite:
        return None
    return float(np.percentile(np.array(finite, dtype=np.float32), percentile))


def finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def depth_zone_stats(depth: np.ndarray, finite: np.ndarray, close_distance_m: float) -> dict:
    valid_pixels = int(finite.sum())
    close = finite & (depth <= close_distance_m)
    stats = {
        "valid_ratio": round(float(finite.mean()), 3) if finite.size else 0.0,
        "valid_pixels": valid_pixels,
        "close_ratio": round(float(close.mean()), 3) if close.size else 0.0,
        "close_pixels": int(close.sum()),
        "near_m": None,
        "p20_m": None,
    }
    if valid_pixels:
        values = depth[finite]
        stats["near_m"] = round(float(np.percentile(values, 5)), 3)
        stats["p20_m"] = round(float(np.percentile(values, 20)), 3)
    return stats


# The camera feed registry ===================================================
#
# The page renders one gallery tile per entry the server reports, so this list
# is the whole answer to "which cameras does this car have". Nothing about the
# tiles is hardcoded in the client: a feed the server does not report produces
# no tile, and a feed it reports appears with no change to the page.
#
# Six feeds are defined here and no car carries all six. The HP60C computes
# depth on the module and its SDK has no media type for the stereo pair, so it
# can only ever publish depth and rgb: for AS_SDK_CAM_MODEL_HP60C the vendor
# driver (rosmaster-a1-hp60c-wendy/src/ascamera/src/CameraPublisher.cpp, around
# lines 215 to 230) creates depth image, depth camera_info, rgb image, rgb
# camera_info and depth points, and the infrared publishers exist only for the
# NUWA and VEGA models. The RealSense D435i fitted in its place publishes
# depth, both infrared views and colour as four separate topics, which is the
# four views the operator asked for. Both cameras stay defined so either can
# be fitted; which of them is actually on the car is decided at runtime by
# camera_feeds below, not here.
#
# `camera` names the snapshot a feed's health is read from and `stream` the key
# inside it, so staleness has one source of truth rather than a second one
# invented here.
CAMERA_FEEDS = (
    {"id": "hp60c_depth", "label": "Depth", "path": "/stream_hp60c_depth.mjpg", "camera": "hp60c", "stream": "depth"},
    {"id": "hp60c_rgb", "label": "RGB", "path": "/stream_hp60c_rgb.mjpg", "camera": "hp60c", "stream": "rgb"},
    # Registry order is gallery order, and the gallery is a two by two block, so
    # this reads top left, top right, bottom left, bottom right. The operator
    # asked for RGB first, then depth, then the stereo pair: the human readable
    # view leads, its derived depth sits beside it, and the two raw infrared
    # views that produced that depth sit together underneath, left before right
    # so they match the physical imagers.
    {
        "id": "realsense_color",
        "label": "RGB",
        "path": "/stream_realsense_color.mjpg",
        "camera": "realsense",
        "stream": "color",
    },
    {
        "id": "realsense_depth",
        "label": "Depth",
        "path": "/stream_realsense_depth.mjpg",
        "camera": "realsense",
        "stream": "depth",
    },
    {
        "id": "realsense_infra1",
        "label": "Left Stereo",
        "path": "/stream_realsense_infra1.mjpg",
        "camera": "realsense",
        "stream": "infra1",
    },
    {
        "id": "realsense_infra2",
        "label": "Right Stereo",
        "path": "/stream_realsense_infra2.mjpg",
        "camera": "realsense",
        "stream": "infra2",
    },
)

CAMERA_STREAM_PATHS = {feed["path"]: (feed["camera"], feed["stream"]) for feed in CAMERA_FEEDS}

# The depth cameras autonomy is allowed to plan on ==========================
#
# The planner wants a distance in metres and the zone statistics derived from
# it. It does not care which module produced them, and it must not: an HP60C
# was fitted here, a RealSense D435i is fitted now, in the same position, so
# the floor line and the obstacle box are unchanged and either camera's frame
# goes through the same decoder. Naming one camera in the planner is what made
# Auto Nav refuse to engage after the swap.
#
# `feed_id` points at the registry entry above, so "which camera is fitted" is
# still answered in exactly one place, camera_feeds, and this is not a second
# answer to it. `stale_s` is each camera's own freshness bound, because the two
# drivers publish at different rates and each already has one.
#
# Order is the tie break when a car somehow carries both: the HP60C is first
# because it is the camera this planner was written and tuned against.
DEPTH_SOURCES = (
    {"camera": "hp60c", "feed_id": "hp60c_depth", "stale_s": HP60C_STALE_S},
    {"camera": "realsense", "feed_id": "realsense_depth", "stale_s": REALSENSE_STALE_S},
)


def feed_is_fitted(stream: dict | None, publishers: object) -> bool:
    """Whether the hardware behind one feed exists, which is not whether it is well.

    Two pieces of evidence, and either is enough. A frame has arrived on the
    topic at some point in this process's life, which is proof the camera was
    there. Or something is publishing on the topic right now, which is proof it
    is there even though it has not produced an image yet.

    The frame half is deliberately sticky and never expires. It is what keeps a
    driver restart, which drops the publisher count to zero for several
    seconds, from emptying the gallery and refilling it.
    """
    frames = (stream or {}).get("frames")
    try:
        seen = int(frames or 0) > 0
    except (TypeError, ValueError):
        seen = False
    try:
        publishing = int(publishers or 0) > 0
    except (TypeError, ValueError):
        publishing = False
    return seen or publishing


def camera_feeds(hp60c: dict | None = None, realsense: dict | None = None) -> list[dict]:
    """One registry entry per camera feed this car actually has.

    The page renders one tile per entry, so this list answers two different
    questions at once and they must not be run together:

        is this camera fitted?      membership, decided here
        is this camera happy?       the ok and age_s fields on each entry

    Membership follows the hardware. Six feeds are defined, two on the HP60C
    and four on the RealSense, and listing all six would mean that whichever
    camera is absent contributes tiles that can never fill. A tile that can
    never fill is worse than no tile at all: the operator cannot tell it apart
    from one that is merely late, so they spend the drive waiting on an image
    that was never coming. An empty gallery on a car with no camera is not a
    good look either, but it is at least true, and the sensors panel says why.

    A feed is listed if its topic has ever delivered a frame, or if something
    is publishing on it right now. See feed_is_fitted. The publisher half makes
    the gallery appear as the camera comes up rather than only once it works.
    The frame half is sticky for the life of the process, and that is what
    holds the tiles still while a feed is unwell.

    So the distinction the rule draws is between "this camera is not fitted",
    which removes the tile, and "this camera is fitted and currently unhappy",
    which keeps the tile and reports it as stale or waiting. Health never
    removes anything: a feed that has gone quiet stays listed with ok false and
    an age_s the operator can watch climb, and a feed with a publisher but no
    frames yet reports a null age_s, which the page shows as waiting rather
    than as stale.

    One consequence is worth stating plainly. Because the frame half never
    expires, swapping the camera on a running car leaves the old camera's tiles
    behind until the app restarts. A deploy restarts it, and a deploy is what
    happens when the hardware changes, so this buys steady tiles for a cost
    that is not paid in practice. The alternative, expiring a feed after some
    period of silence, would put back exactly the vanishing tile this avoids,
    and would do it at the worst moment: while the driver is already down.
    """
    snapshots = {"hp60c": hp60c or {}, "realsense": realsense or {}}
    feeds = []
    for spec in CAMERA_FEEDS:
        snapshot = snapshots.get(spec["camera"]) or {}
        stream = snapshot.get(spec["stream"]) or {}
        publishers = (snapshot.get("publishers") or {}).get(spec["stream"], 0)
        if not feed_is_fitted(stream, publishers):
            continue
        feeds.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "path": spec["path"],
                "ok": bool(stream.get("ok")),
                "age_s": stream.get("age_s"),
            }
        )
    return feeds


def select_depth_source(hp60c: dict | None = None, realsense: dict | None = None, now: float | None = None) -> dict | None:
    """The depth camera autonomy should plan on, or None if the car has none.

    Membership is not decided here. A depth camera counts as fitted when its
    depth feed is listed by camera_feeds, which is the same evidence the
    gallery renders from: a frame has arrived on the topic at some point, or
    something is publishing on it right now. Asking that question twice, in two
    places, with two answers, is how the planner and the gallery would come to
    disagree about what hardware the car has.

    Freshness is decided here, from `updated_at` and the source's own
    `stale_s`, rather than read off the snapshot's `ok` field. The planner is
    handed the locked stream metadata on its own thread, where `ok` has not
    been recomputed, and a stale True there would be a stale True that puts the
    car in motion.

    Returns the camera name, its freshness bound, the age of its last frame,
    whether that is fresh, and the depth statistics themselves. A fresh source
    wins over a stale one when both cameras are fitted; failing that the first
    fitted one is returned, so the readiness reason can still name the camera
    the operator is waiting on rather than saying nothing.
    """
    now = time.monotonic() if now is None else now
    fitted = {feed["id"] for feed in camera_feeds(hp60c, realsense)}
    snapshots = {"hp60c": hp60c or {}, "realsense": realsense or {}}
    candidates = []
    for spec in DEPTH_SOURCES:
        if spec["feed_id"] not in fitted:
            continue
        stream = (snapshots.get(spec["camera"]) or {}).get("depth") or {}
        updated_at = finite_float(stream.get("updated_at"), 0.0)
        age = now - updated_at if updated_at else None
        frames = int(finite_float(stream.get("frames"), 0.0))
        candidates.append(
            {
                "camera": spec["camera"],
                "feed_id": spec["feed_id"],
                "stale_s": spec["stale_s"],
                "age_s": round(age, 3) if age is not None else None,
                "fresh": age is not None and age < spec["stale_s"] and frames > 0,
                "depth": stream,
            }
        )
    if not candidates:
        return None
    for candidate in candidates:
        if candidate["fresh"]:
            return candidate
    return candidates[0]


def encode_jpeg(frame: PILImage.Image, quality: int) -> bytes | None:
    buf = BytesIO()
    frame.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# A handler thread must never block on stdout. CommandFreshness used to log a
# rejected command with print(flush=True) while holding its lock; when the
# container's stdout pipe stalled (log-collector backpressure), that call
# never returned, the lock was never released, and every later drive/status
# request wedged in check() waiting for it. log_line() only ever
# put_nowait()s onto this bounded queue, drained by the daemon thread below,
# so a stalled sink can stall the writer, never the caller. A full queue
# drops the line and counts the drop; the next line that gets through is
# prefixed with a marker so the gap is visible in the log instead of silent.
_LOG_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=256)
_LOG_STATE_LOCK = threading.Lock()
_LOG_DROPPED = 0


def _log_output(line: str) -> None:
    """The writer thread's sink. A module attribute, read fresh per line, so
    a test can swap it for something that blocks without touching print."""
    print(line, flush=True)


def log_line(msg: str) -> None:
    global _LOG_DROPPED
    with _LOG_STATE_LOCK:
        dropped = _LOG_DROPPED
        line = f"LOG_DROPPED n={dropped} {msg}" if dropped else msg
        try:
            _LOG_QUEUE.put_nowait(line)
        except queue.Full:
            _LOG_DROPPED = dropped + 1
            return
        _LOG_DROPPED = 0


def _log_writer() -> None:
    while True:
        _log_output(_LOG_QUEUE.get())


threading.Thread(target=_log_writer, daemon=True, name="log-writer").start()


class RosmasterControl(Node):
    def __init__(self) -> None:
        super().__init__("rosmaster_web_remote")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        self._scan_subscription = self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self._imu_subscription = self.create_subscription(Imu, "/imu/data_raw", self._on_imu, 10)
        self._mag_subscription = self.create_subscription(MagneticField, "/imu/mag", self._on_mag, 10)
        self._joint_subscription = self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self._velocity_subscription = self.create_subscription(Twist, "/vel_raw", self._on_velocity_feedback, 10)
        self._voltage_subscription = self.create_subscription(Float32, "/voltage", self._on_voltage, 10)
        self._edition_subscription = self.create_subscription(String, "/edition", self._on_edition, 10)
        self._probe_camera_subscription = self.create_subscription(
            CompressedImage,
            "/sensor_probe/video0/image/compressed",
            self._on_probe_camera,
            10,
        )
        self._probe_status_subscription = self.create_subscription(String, "/sensor_probe/status", self._on_probe_status, 10)
        self._base_bridge_status_subscription = self.create_subscription(
            String,
            "/base_bridge/status",
            self._on_base_bridge_status,
            10,
        )
        self._lidar_probe_status_subscription = self.create_subscription(
            String,
            "/lidar_sensor_probe/status",
            self._on_lidar_probe_status,
            10,
        )
        self._hp60c_depth_subscription = self.create_subscription(RosImage, HP60C_DEPTH_TOPIC, self._on_hp60c_depth, HP60C_QOS)
        self._hp60c_rgb_subscription = self.create_subscription(RosImage, HP60C_RGB_TOPIC, self._on_hp60c_rgb, HP60C_QOS)
        self._hp60c_points_subscription = (
            self.create_subscription(PointCloud2, HP60C_POINTS_TOPIC, self._on_hp60c_points, HP60C_QOS)
            if HP60C_POINTS_ENABLED
            else None
        )
        self._realsense_depth_subscription = self.create_subscription(
            RosImage, REALSENSE_DEPTH_TOPIC, self._on_realsense_depth, REALSENSE_QOS
        )
        self._realsense_infra1_subscription = self.create_subscription(
            RosImage, REALSENSE_INFRA1_TOPIC, self._on_realsense_infra1, REALSENSE_QOS
        )
        self._realsense_infra2_subscription = self.create_subscription(
            RosImage, REALSENSE_INFRA2_TOPIC, self._on_realsense_infra2, REALSENSE_QOS
        )
        self._realsense_color_subscription = self.create_subscription(
            RosImage, REALSENSE_COLOR_TOPIC, self._on_realsense_color, REALSENSE_QOS
        )
        self._lock = threading.Lock()
        # count_publishers/count_subscribers are unbounded rclpy graph
        # queries; the request path used to call them straight from the
        # handler thread, and a stalled rmw layer hung /api/drive and
        # /api/status directly. Sampled instead on their own thread (see
        # _sample_graph_counts_loop, started below) and cached here so a
        # handler thread only ever reads a dict under self._lock. Zeros until
        # the first sample match "nothing discovered yet", the same meaning
        # an unfitted camera or absent base driver already has.
        self._graph_publisher_topics = (
            HP60C_DEPTH_TOPIC,
            HP60C_RGB_TOPIC,
            HP60C_POINTS_TOPIC,
            *REALSENSE_TOPICS.values(),
        )
        self._graph_publishers: dict[str, int] = {topic: 0 for topic in self._graph_publisher_topics}
        self._graph_cmd_vel_subscribers = 0
        # Consecutive failed ticks in _sample_graph_counts_tick, so the loop
        # can log once when it starts failing and once when it recovers
        # rather than once per tick -- the same throttling instinct
        # CommandFreshness's DRIVE_REJECTED log applies to a bad link's
        # rejection bursts, applied here to a bad rmw layer's failure bursts.
        self._graph_sample_failures = 0
        self._command = zero_command()
        self._last_published = zero_command()
        self._publish_count = 0
        self._last_publish_at = 0.0
        self._scan = self._empty_scan()
        self._hp60c = self._empty_hp60c()
        self._realsense = self._empty_realsense()
        # How many MJPEG responses are currently reading each feed, keyed
        # "camera:stream". See open_camera_viewer.
        self._viewers: dict[str, int] = {}
        # Last time each feed drew a preview, keyed "camera:stream".
        self._preview_last: dict[str, float] = {}
        self._sensors = self._empty_sensors()
        self._auto = {
            "enabled": False,
            "speed": AUTO_SPEED,
            "stop_distance": AUTO_STOP_DISTANCE,
            "avoid_distance": AUTO_AVOID_DISTANCE,
            "clear_distance": AUTO_CLEAR_DISTANCE,
            "updated_at": 0.0,
        }
        self._auto_decision = {
            "action": "disabled",
            "linear_x": 0.0,
            "steering_y": 0.0,
            "reason": "auto disabled",
        }
        self._auto_state = self._new_auto_state()
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
        threading.Thread(target=self._sample_graph_counts_loop, daemon=True, name="graph-count-sampler").start()

    def update(self, payload: dict) -> dict:
        enabled = bool(payload.get("enabled", False))
        cmd = {
            "enabled": enabled,
            "linear_x": finite_float(payload.get("linear_x", 0.0)) if enabled else 0.0,
            "steering_y": clamp(float(payload.get("steering_y", 0.0)), MAX_STEERING_Y) if enabled else 0.0,
            "angular_z": clamp(float(payload.get("angular_z", 0.0)), MAX_ANGULAR_Z) if enabled else 0.0,
            "updated_at": time.monotonic(),
            "source": "web",
        }
        with self._lock:
            if enabled:
                self._auto["enabled"] = False
            self._command = cmd
        return cmd

    def set_auto(self, payload: dict) -> dict:
        enabled = bool(payload.get("enabled", False))
        speed = floor_finite(payload.get("speed", AUTO_SPEED), 0.0, AUTO_SPEED)
        stop_distance = clamp_range(float(payload.get("stop_distance", AUTO_STOP_DISTANCE)), 0.20, 1.50)
        avoid_distance = clamp_range(float(payload.get("avoid_distance", AUTO_AVOID_DISTANCE)), stop_distance + 0.10, 2.50)
        clear_distance = clamp_range(
            float(payload.get("clear_distance", AUTO_CLEAR_DISTANCE)),
            avoid_distance + 0.10,
            3.50,
        )
        with self._lock:
            self._auto = {
                "enabled": enabled,
                "speed": speed,
                "stop_distance": stop_distance,
                "avoid_distance": avoid_distance,
                "clear_distance": clear_distance,
                "updated_at": time.monotonic(),
            }
            self._auto_state = self._new_auto_state()
            if enabled:
                self._command = zero_command("auto")
                self._command["updated_at"] = time.monotonic()
        if not enabled:
            self._publish_zero_burst()
        return self.auto_snapshot()

    def stop(self) -> None:
        with self._lock:
            self._auto["enabled"] = False
            self._command = zero_command("stop")
            self._command["updated_at"] = time.monotonic()
            self._auto_decision = {
                "action": "stopped",
                "linear_x": 0.0,
                "steering_y": 0.0,
                "reason": "stop requested",
            }
            self._auto_state = self._new_auto_state()
        self._publish_zero_burst()

    def start(self) -> dict:
        command = {
            "enabled": True,
            "linear_x": 0.0,
            "steering_y": 0.0,
            "angular_z": 0.0,
            "updated_at": time.monotonic(),
            "source": "start",
        }
        with self._lock:
            self._auto["enabled"] = False
            self._command = command
            self._auto_decision = {
                "action": "manual_armed",
                "linear_x": 0.0,
                "steering_y": 0.0,
                "reason": "start requested",
            }
            self._auto_state = self._new_auto_state()
        return command

    def _cached_publisher_count(self, topic: str) -> int:
        """A request thread's only way to ask about the ROS graph.

        count_publishers itself is a request thread's business only through
        this accessor: it just reads _sample_graph_counts's last result under
        self._lock, never the graph. See _sample_graph_counts for why.
        """
        with self._lock:
            return self._graph_publishers.get(topic, 0)

    def _cached_cmd_vel_subscribers(self) -> int:
        with self._lock:
            return self._graph_cmd_vel_subscribers

    def _sample_graph_counts(self) -> None:
        """Query the ROS graph and refresh the cache. One call, one round trip.

        count_publishers/count_subscribers hit rmw with no timeout of their
        own. They used to be called inline from snapshot, _auto_ready,
        _planner_depth_source, hp60c_snapshot and realsense_snapshot, all
        reachable from POST /api/drive and GET /api/status; a stalled rmw
        layer hung both straight from the handler thread. The queries happen
        here, outside self._lock, so a slow graph call delays only whoever
        called this, not every handler blocked on the lock behind it; only
        the already-known result is stored under the lock, for
        _cached_publisher_count and _cached_cmd_vel_subscribers to read.
        Kept as its own method, separate from the loop below, so a test can
        prime the cache deterministically without waiting on a timer.
        """
        publishers = {topic: self.count_publishers(topic) for topic in self._graph_publisher_topics}
        cmd_vel_subscribers = self.count_subscribers("/cmd_vel")
        with self._lock:
            self._graph_publishers = publishers
            self._graph_cmd_vel_subscribers = cmd_vel_subscribers

    def _sample_graph_counts_tick(self) -> None:
        """One sampler tick: query, store, and survive whatever the graph does.

        Split out from _sample_graph_counts_loop so a test can drive exactly
        one tick deterministically -- including one that raises -- without
        racing the real thread or its sleep interval.

        A raising count_publishers/count_subscribers is the same rmw failure
        this whole file is being hardened against, just surfacing as an
        exception instead of a hang. An uncaught one here would kill this
        daemon thread silently (a bare exception in a thread target is only
        ever reported to threading.excepthook, which nothing here monitors),
        freezing the cache at its last values forever with no visible sign
        anything had gone wrong. Caught instead, so the thread keeps ticking
        and the cache recovers on its own the moment the graph does. Logged
        once when it starts failing and once when it recovers with a count in
        between, not once per tick, so a stuck rmw layer does not bury the
        log the way DRIVE_REJECTED's throttling exists to prevent for a bad
        link's rejection bursts.
        """
        try:
            self._sample_graph_counts()
        except Exception as exc:  # noqa: BLE001 - this thread must never die
            self._graph_sample_failures += 1
            if self._graph_sample_failures == 1:
                log_line(f"GRAPH_COUNT_SAMPLE_FAILED {type(exc).__name__}: {exc}")
        else:
            if self._graph_sample_failures:
                log_line(f"GRAPH_COUNT_SAMPLE_RECOVERED after {self._graph_sample_failures} failed tick(s)")
            self._graph_sample_failures = 0

    def _sample_graph_counts_loop(self) -> None:
        """Refresh the graph-count cache on its own thread, forever.

        This used to run at the top of _publish(), the callback the 20 Hz
        /cmd_vel timer ticks -- the single executor thread that also runs
        every subscription callback. PREVIEW_MAX_FPS above documents what
        happens when that thread is made to wait: a saturated executor once
        starved the command timer, the base bridge stopped hearing from us,
        its watchdog zeroed the motors, and the car stopped answering the
        operator. count_publishers/count_subscribers are exactly the
        unbounded rmw calls this whole file is being hardened against, so
        calling them from the executor would have reproduced that failure
        from a new cause. A dedicated daemon thread means a stalled or
        raising graph query only leaves the cache stale -- it can no longer
        touch the publish tick, a sensor callback, or a handler thread, and
        (see _sample_graph_counts_tick) it can no longer take this thread
        down either.
        """
        while True:
            self._sample_graph_counts_tick()
            time.sleep(GRAPH_COUNT_SAMPLE_INTERVAL_S)

    def snapshot(self) -> dict:
        with self._lock:
            command = dict(self._command)
            last_published = dict(self._last_published)
        stale_s = time.monotonic() - command["updated_at"] if command["updated_at"] else None
        return {
            "command": command,
            "last_published": last_published,
            "publish_count": self._publish_count,
            "last_publish_age_s": round(time.monotonic() - self._last_publish_at, 3) if self._last_publish_at else None,
            "stale_s": round(stale_s, 3) if stale_s is not None else None,
            "cmd_vel_subscribers": self._cached_cmd_vel_subscribers(),
            "limits": {
                "max_linear_x": MAX_LINEAR_X,
                "max_steering_y": MAX_STEERING_Y,
                "max_angular_z": MAX_ANGULAR_Z,
                "cmd_timeout_s": CMD_TIMEOUT_S,
                "lidar_stale_s": LIDAR_STALE_S,
                "sensor_stale_s": SENSOR_STALE_S,
            },
        }

    def auto_snapshot(self) -> dict:
        source = self._planner_depth_source()
        with self._lock:
            auto = dict(self._auto)
            decision = dict(self._auto_decision)
            state = dict(self._auto_state)
            scan = dict(self._scan)
            feedback = self._motion_feedback_locked()
        ready = self._auto_ready()
        _, preview = self._compute_auto_command(scan, auto, source, state=state, update_state=False, feedback=feedback)
        return {
            "enabled": auto["enabled"],
            "speed": round(auto["speed"], 3),
            "stop_distance": round(auto["stop_distance"], 3),
            "avoid_distance": round(auto["avoid_distance"], 3),
            "clear_distance": round(auto["clear_distance"], 3),
            "ready": ready["ready"],
            "ready_reason": ready["reason"],
            "decision": decision,
            "preview": preview,
            "state": state,
        }

    def navigation_snapshot(self) -> dict:
        ready = self._auto_ready()
        source = self.depth_source()
        camera = source["camera"] if source else None
        depth_ok = bool(source and source["fresh"])
        return {
            "mode": "lidar_farthest_corridor_with_depth_object_veto" if depth_ok else "lidar_farthest_corridor",
            "primary_sensor": "ydlidar_farthest_corridor",
            "depth_source": camera,
            "depth_ok": depth_ok,
            "camera_role": f"{camera}_upper_roi_object_veto" if depth_ok else "waiting_for_depth_ros_topics",
            "ready": ready["ready"],
            "reason": ready["reason"],
        }

    def depth_source(self) -> dict | None:
        """Which depth camera this car has, for anything answering an HTTP request.

        Full snapshots, so publisher counts count as evidence of a fitted
        camera and a driver that has come up but not yet produced a frame is
        still named in the readiness reason. The counts themselves are cache
        reads (see _sample_graph_counts), so this is cheap enough to call from
        a request thread; the planner still has its own leaner version below,
        because the image bookkeeping this does is more than it needs.
        """
        return select_depth_source(self.hp60c_snapshot(), self.realsense_snapshot())

    def _planner_depth_source(self) -> dict | None:
        """The same question, asked on the thread that publishes /cmd_vel.

        Same registry, same rule, and the two depth topics' cached publisher
        counts filled in so the answer matches the one above, with none of
        the image bookkeeping: reading the other four feeds' publishers would
        buy the planner nothing, since it only ever plans on depth.
        """
        with self._lock:
            hp60c = self._hp60c_meta_locked()
            realsense = self._realsense_meta_locked()
        hp60c["publishers"] = {"depth": self._cached_publisher_count(HP60C_DEPTH_TOPIC)}
        realsense["publishers"] = {"depth": self._cached_publisher_count(REALSENSE_DEPTH_TOPIC)}
        return select_depth_source(hp60c, realsense)

    def hp60c_snapshot(self) -> dict:
        with self._lock:
            hp60c = self._hp60c_meta_locked()
        now = time.monotonic()
        for key in ["depth", "rgb", "points"]:
            age = now - hp60c[key]["updated_at"] if hp60c[key]["updated_at"] else None
            hp60c[key]["age_s"] = round(age, 3) if age is not None else None
            hp60c[key]["ok"] = age is not None and age < HP60C_STALE_S and hp60c[key]["frames"] > 0
        hp60c["topics"] = {
            "depth": HP60C_DEPTH_TOPIC,
            "rgb": HP60C_RGB_TOPIC,
            "points": HP60C_POINTS_TOPIC,
        }
        hp60c["points_enabled"] = HP60C_POINTS_ENABLED
        hp60c["publishers"] = {
            "depth": self._cached_publisher_count(HP60C_DEPTH_TOPIC),
            "rgb": self._cached_publisher_count(HP60C_RGB_TOPIC),
            "points": self._cached_publisher_count(HP60C_POINTS_TOPIC),
        }
        hp60c["usable_for_navigation"] = bool(
            hp60c["depth"]["ok"]
            and hp60c["depth"].get("obstacle_p20_m") is not None
            and hp60c["depth"].get("obstacle_valid_ratio", 0.0) >= HP60C_DEPTH_VALID_MIN_RATIO
        )
        return hp60c

    def realsense_snapshot(self) -> dict:
        """The same shape hp60c_snapshot returns, for the D435i's four streams.

        `frames` and `publishers` are what camera_feeds reads to decide whether
        this camera is on the car at all, so both are reported per stream
        rather than for the camera as a whole. A D435i with a dead infrared
        cable is still a D435i, and its other three tiles should stay.
        """
        with self._lock:
            realsense = self._realsense_meta_locked()
            viewers = {name: self._viewers.get(f"realsense:{name}", 0) for name in REALSENSE_TOPICS}
        now = time.monotonic()
        for name in REALSENSE_TOPICS:
            stream = realsense[name]
            age = now - stream["updated_at"] if stream["updated_at"] else None
            stream["age_s"] = round(age, 3) if age is not None else None
            stream["ok"] = age is not None and age < REALSENSE_STALE_S and stream["frames"] > 0
        realsense["topics"] = dict(REALSENSE_TOPICS)
        realsense["publishers"] = {name: self._cached_publisher_count(topic) for name, topic in REALSENSE_TOPICS.items()}
        realsense["viewers"] = viewers
        realsense["stale_s"] = REALSENSE_STALE_S
        return realsense

    def open_camera_viewer(self, camera: str, stream: str) -> None:
        """Register an MJPEG response as reading this feed.

        Decoding and JPEG encoding a preview happens in the ROS callback, on
        the same executor thread that ticks the /cmd_vel publish timer, so
        every millisecond spent on a preview is a millisecond the drive command
        is not going out. Four RealSense streams at 15 Hz is four times that
        work for as long as the driver runs, most of it usually for nobody: a
        gallery tile is only open while the page is. Counting viewers lets the
        callback skip the expensive half when no one is watching, and it is a
        count rather than a flag because the same feed can be open in two tabs.
        """
        key = f"{camera}:{stream}"
        with self._lock:
            self._viewers[key] = self._viewers.get(key, 0) + 1

    def close_camera_viewer(self, camera: str, stream: str) -> None:
        key = f"{camera}:{stream}"
        with self._lock:
            remaining = max(0, self._viewers.get(key, 0) - 1)
            self._viewers[key] = remaining
            if remaining == 0:
                # Drop the cached frame with the last viewer. Kept, it would be
                # the first thing the next viewer saw, a picture of whatever
                # was in front of the car minutes ago presented as live.
                slot = f"{stream}_jpeg"
                if camera == "realsense" and slot in self._realsense:
                    self._realsense[slot] = None

    def _has_viewer_locked(self, camera: str, stream: str) -> bool:
        return self._viewers.get(f"{camera}:{stream}", 0) > 0

    def _preview_due_locked(self, key: str) -> bool:
        """True when this feed is allowed to draw another preview.

        Call with the lock held. Rations encoding per feed rather than in total,
        so one busy tile cannot starve the others, and stamps the time on the way
        through so a caller that is told yes has already claimed its slot.
        """
        if PREVIEW_MIN_INTERVAL_S <= 0:
            return True
        now = time.monotonic()
        if now - self._preview_last.get(key, 0.0) < PREVIEW_MIN_INTERVAL_S:
            return False
        self._preview_last[key] = now
        return True

    def camera_frame(self, camera: str, stream: str) -> bytes | None:
        with self._lock:
            if camera == "realsense":
                return self._realsense.get(f"{stream}_jpeg")
            if stream == "rgb":
                return self._hp60c["rgb_jpeg"]
            return self._hp60c["depth_jpeg"]

    def sensors_snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            sensors = json.loads(json.dumps(self._sensors))
        depth_sensor = self._required_depth_sensor()
        if depth_sensor not in sensors["items"]:
            # A RealSense that is fitted but has not yet delivered a frame has
            # no row of its own, because the RealSense rows are registered on
            # first frame so that a car without one does not carry four red
            # rows for ever. It still has to appear as a missing requirement:
            # this is the sensor autonomy is waiting on.
            sensors["items"][depth_sensor] = {"updated_at": 0.0, "age_s": None, "frames": 0, "ok": False, "data": {}}
        for sensor in sensors["items"].values():
            age = now - sensor["updated_at"] if sensor["updated_at"] else None
            sensor["age_s"] = round(age, 3) if age is not None else None
            sensor["ok"] = age is not None and age < SENSOR_STALE_S and sensor["frames"] > 0
        required_names = REQUIRED_SENSOR_NAMES | {depth_sensor}
        required = sorted(name for name in required_names if name in sensors["items"])
        missing = sorted(name for name in required if not sensors["items"][name]["ok"])
        sensors["required"] = required
        sensors["optional"] = sorted(name for name in sensors["items"] if name not in required_names)
        sensors["missing"] = missing
        sensors["ok"] = not missing
        return sensors

    def _required_depth_sensor(self) -> str:
        """The sensor row that stands for "this car can see forward in depth".

        Which row that is follows the hardware, so a RealSense car is judged on
        realsense_depth and an HP60C car on hp60c_depth. A car with no depth
        camera at all is judged on the first entry in DEPTH_SOURCES, which will
        read as missing, and that is the point: dropping the requirement
        instead would turn the sensors panel green on a car that cannot self
        drive at all.
        """
        source = select_depth_source(self.hp60c_snapshot(), self.realsense_snapshot())
        return f"{source['camera']}_depth" if source else DEPTH_SOURCES[0]["feed_id"]

    def lidar_snapshot(self) -> dict:
        with self._lock:
            scan = json.loads(json.dumps(self._scan))
        scan.pop("gap_samples", None)
        age = time.monotonic() - scan["updated_at"] if scan["updated_at"] else None
        scan["age_s"] = round(age, 3) if age is not None else None
        scan["ok"] = age is not None and age < LIDAR_STALE_S and scan["finite_ranges"] > 0
        return scan

    def _auto_ready(self) -> dict:
        """Whether autonomy may engage, and if not, the one thing it is waiting on.

        The three conditions are unchanged: fresh LiDAR, fresh depth, and a
        /cmd_vel subscriber to receive the commands. What changed is that the
        depth condition no longer names a camera model. It asks whichever depth
        camera is fitted, and the reason string says which one that is, so
        "waiting for fresh depth" is answerable by an operator looking at the
        car rather than at the source.
        """
        scan = self.lidar_snapshot()
        if not scan["ok"]:
            return {"ready": False, "reason": "waiting for fresh lidar"}
        if self._cached_cmd_vel_subscribers() < 1:
            return {"ready": False, "reason": "waiting for base driver"}
        source = self.depth_source()
        if source is None:
            return {"ready": False, "reason": "waiting for a depth camera"}
        if not source["fresh"]:
            return {"ready": False, "reason": f"waiting for fresh {source['camera']} depth frames"}
        front = scan.get("sectors", {}).get("front", {})
        if front.get("near_m") is None or front.get("count", 0) < 5:
            return {"ready": False, "reason": "front lidar sector sparse"}
        return {"ready": True, "reason": "ready"}

    def _on_scan(self, msg: LaserScan) -> None:
        front: list[float] = []
        front_left: list[float] = []
        front_right: list[float] = []
        left: list[float] = []
        right: list[float] = []
        rear: list[float] = []
        gap_samples: list[tuple[float, float]] = []
        finite_count = 0
        min_m: float | None = None
        points: list[dict] = []
        stride = max(1, len(msg.ranges) // 96)

        for idx, raw in enumerate(msg.ranges):
            value = float(raw)
            if not math.isfinite(value) or value <= max(0.02, msg.range_min):
                continue
            if msg.range_max > 0 and value > msg.range_max:
                continue
            finite_count += 1
            min_m = value if min_m is None else min(min_m, value)
            angle = msg.angle_min + idx * msg.angle_increment
            deg = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
            if -35 <= deg <= 35:
                front.append(value)
            if 8 < deg <= 65:
                front_left.append(value)
            elif -65 <= deg < -8:
                front_right.append(value)
            if 35 < deg <= 125:
                left.append(value)
            elif -125 <= deg < -35:
                right.append(value)
            elif deg >= 155 or deg <= -155:
                rear.append(value)
            if abs(deg) <= AUTO_GAP_FOV_DEG:
                gap_samples.append((deg, value))
            if idx % stride == 0:
                points.append({"a": round(deg, 1), "r": round(min(value, 3.5), 3)})

        sectors = {
            "front": self._sector(front),
            "front_left": self._sector(front_left),
            "front_right": self._sector(front_right),
            "left": self._sector(left),
            "right": self._sector(right),
            "rear": self._sector(rear),
        }
        with self._lock:
            self._scan = {
                "updated_at": time.monotonic(),
                "frame_id": msg.header.frame_id,
                "ranges": len(msg.ranges),
                "finite_ranges": finite_count,
                "min_m": round(min_m, 3) if min_m is not None else None,
                "sectors": sectors,
                "points": points,
                "gap_samples": gap_samples,
            }
            self._record_sensor_locked(
                "lidar",
                {
                    "frame_id": msg.header.frame_id,
                    "ranges": len(msg.ranges),
                    "finite_ranges": finite_count,
                    "min_m": round(min_m, 3) if min_m is not None else None,
                },
            )

    def _on_imu(self, msg: Imu) -> None:
        with self._lock:
            self._record_sensor_locked(
                "imu",
                {
                    "frame_id": msg.header.frame_id,
                    "linear_acceleration": [
                        round(float(msg.linear_acceleration.x), 4),
                        round(float(msg.linear_acceleration.y), 4),
                        round(float(msg.linear_acceleration.z), 4),
                    ],
                    "angular_velocity": [
                        round(float(msg.angular_velocity.x), 4),
                        round(float(msg.angular_velocity.y), 4),
                        round(float(msg.angular_velocity.z), 4),
                    ],
                },
            )

    def _on_mag(self, msg: MagneticField) -> None:
        with self._lock:
            self._record_sensor_locked(
                "magnetometer",
                {
                    "frame_id": msg.header.frame_id,
                    "magnetic_field": [
                        round(float(msg.magnetic_field.x), 6),
                        round(float(msg.magnetic_field.y), 6),
                        round(float(msg.magnetic_field.z), 6),
                    ],
                },
            )

    def _on_joint_states(self, msg: JointState) -> None:
        with self._lock:
            self._record_sensor_locked(
                "joint_states",
                {
                    "names": list(msg.name),
                    "positions": [round(float(value), 4) for value in msg.position],
                    "velocities": [round(float(value), 4) for value in msg.velocity],
                },
            )

    def _on_velocity_feedback(self, msg: Twist) -> None:
        with self._lock:
            self._record_sensor_locked(
                "velocity_feedback",
                {
                    "linear": [
                        round(float(msg.linear.x), 4),
                        round(float(msg.linear.y), 4),
                        round(float(msg.linear.z), 4),
                    ],
                    "angular": [
                        round(float(msg.angular.x), 4),
                        round(float(msg.angular.y), 4),
                        round(float(msg.angular.z), 4),
                    ],
                },
            )

    def _on_voltage(self, msg: Float32) -> None:
        with self._lock:
            self._record_sensor_locked("voltage", {"volts": round(float(msg.data), 3)})

    def _on_edition(self, msg: String) -> None:
        with self._lock:
            self._record_sensor_locked("base_firmware", {"version": msg.data})

    def _on_probe_camera(self, msg: CompressedImage) -> None:
        with self._lock:
            self._record_sensor_locked(
                "probe_camera",
                {
                    "frame_id": msg.header.frame_id,
                    "format": msg.format,
                    "bytes": len(msg.data),
                },
            )

    def _on_probe_status(self, msg: String) -> None:
        with self._lock:
            self._record_sensor_locked("base_probe_status", {"bytes": len(msg.data)})

    def _on_base_bridge_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {"raw": msg.data}
        with self._lock:
            self._record_sensor_locked("base_bridge_status", payload)

    def _on_lidar_probe_status(self, msg: String) -> None:
        with self._lock:
            self._record_sensor_locked("lidar_probe_status", {"bytes": len(msg.data)})

    def _on_hp60c_depth(self, msg: RosImage) -> None:
        frame, stats = self._depth_image_to_preview(msg)
        if frame is None:
            return
        self._annotate_ros_frame(frame, "HP60C depth", msg.encoding, stats)
        encoded = self._encode_frame(frame, 78)
        if encoded is None:
            return
        with self._lock:
            self._hp60c["depth_jpeg"] = encoded
            self._hp60c["depth"].update(
                {
                    "updated_at": time.monotonic(),
                    "frames": self._hp60c["depth"]["frames"] + 1,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                    **stats,
                }
            )
            self._record_sensor_locked(
                "hp60c_depth",
                {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                    "valid_ratio": stats.get("valid_ratio", 0.0),
                    "above_floor_close_pixels": stats.get("above_floor_close_pixels", 0),
                },
            )

    def _on_hp60c_rgb(self, msg: RosImage) -> None:
        frame = self._rgb_image_to_preview(msg)
        if frame is None:
            return
        stats = {"valid_ratio": 1.0}
        self._annotate_ros_frame(frame, "HP60C rgb", msg.encoding, stats)
        encoded = self._encode_frame(frame, 78)
        if encoded is None:
            return
        with self._lock:
            self._hp60c["rgb_jpeg"] = encoded
            self._hp60c["rgb"].update(
                {
                    "updated_at": time.monotonic(),
                    "frames": self._hp60c["rgb"]["frames"] + 1,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                }
            )
            self._record_sensor_locked(
                "hp60c_rgb",
                {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                },
            )

    def _on_hp60c_points(self, msg: PointCloud2) -> None:
        with self._lock:
            self._hp60c["points"].update(
                {
                    "updated_at": time.monotonic(),
                    "frames": self._hp60c["points"]["frames"] + 1,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "point_step": int(msg.point_step),
                    "row_step": int(msg.row_step),
                    "frame_id": msg.header.frame_id,
                }
            )
            self._record_sensor_locked(
                "hp60c_points",
                {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "point_step": int(msg.point_step),
                    "row_step": int(msg.row_step),
                    "frame_id": msg.header.frame_id,
                },
            )

    def _on_realsense_depth(self, msg: RosImage) -> None:
        self._record_realsense_frame("depth", "RealSense depth", msg, depth=True)

    def _on_realsense_infra1(self, msg: RosImage) -> None:
        self._record_realsense_frame("infra1", "RealSense infra left", msg)

    def _on_realsense_infra2(self, msg: RosImage) -> None:
        self._record_realsense_frame("infra2", "RealSense infra right", msg)

    def _on_realsense_color(self, msg: RosImage) -> None:
        self._record_realsense_frame("color", "RealSense colour", msg)

    def _record_realsense_frame(self, stream: str, title: str, msg: RosImage, depth: bool = False) -> None:
        """One path for all four D435i streams, splitting cheap work from dear.

        Freshness and geometry are recorded whatever happens: those are what
        camera_feeds and the diagnostics read, and a feed that only exists
        while someone is looking at it would be a camera that disappears from
        the status page when the gallery is closed.

        Encoding a JPEG is the dear half, and it is skipped when no MJPEG
        response is reading this feed.

        The depth stream is the exception, and only in part. Its zone
        statistics are what the planner vetoes on when this is the camera
        fitted, so they are computed for every depth frame, watched or not: a
        car whose autonomy worked only while a browser tab was open would be a
        far worse trap than a wasted percent of a CPU. The picture built from
        those statistics is still only drawn for a viewer.

        16UC1 depth in millimetres goes through the existing depth decoder,
        which already divides by 1000 and colorizes. mono8 infrared and rgb8 or
        bgr8 colour go through the existing rgb decoder, which already handles
        all three. No new decoder exists here, and none should.
        """
        with self._lock:
            # Two gates, not one. Nobody watching means no encode at all, and a
            # watcher still only gets PREVIEW_MAX_FPS of them: four tiles at the
            # camera's full rate saturated this thread and cost us the car.
            wanted = self._has_viewer_locked("realsense", stream) and self._preview_due_locked(
                f"realsense:{stream}"
            )

        encoded = None
        stats: dict = {}
        # Contained on purpose. This runs on the executor thread that also
        # ticks the /cmd_vel publish timer, and an exception escaping a
        # subscription callback takes that thread down with it: the car would
        # stop being sent commands because a preview could not be drawn. The
        # frame formats this camera really emits are not verified against the
        # car yet, so a surprise here costs the tile and, for depth, the
        # statistics, which the planner reads as absent and refuses to drive
        # on. That is the failure direction we want.
        try:
            frame = None
            if depth:
                depth_m, finite, geometry, stats = self._depth_image_to_stats(msg)
                if depth_m is None:
                    # An undecodable frame blanks the statistics instead of
                    # leaving the last good ones behind a freshly stamped
                    # updated_at. Left alone they would read to the planner as
                    # distances measured a moment ago, when they were measured
                    # from the last frame this code could make sense of.
                    stats = self._empty_depth_stats()
                elif wanted:
                    frame = self._depth_preview_image(msg, depth_m, finite, geometry)
            elif wanted:
                frame = self._rgb_image_to_preview(msg)
            if frame is not None:
                self._annotate_ros_frame(frame, title, msg.encoding, stats)
                encoded = self._encode_frame(frame, 78)
        except Exception as exc:  # noqa: BLE001 - a preview is never worth the executor
            log_line(
                f"REALSENSE_FRAME_FAILED stream={stream} encoding={msg.encoding} "
                f"{type(exc).__name__}: {exc}"
            )

        meta = {
            "updated_at": time.monotonic(),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": msg.encoding,
            "frame_id": msg.header.frame_id,
        }
        with self._lock:
            self._realsense[stream].update({**meta, **stats, "frames": self._realsense[stream]["frames"] + 1})
            if encoded is not None:
                self._realsense[f"{stream}_jpeg"] = encoded
            # Registered on first frame rather than up front, so a car with no
            # RealSense does not carry four rows that are red for ever. The
            # HP60C entries are pre-registered instead, because hp60c_depth is
            # a required sensor and its absence has to be visible.
            self._record_sensor_locked(
                f"realsense_{stream}",
                {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                    "topic": REALSENSE_TOPICS[stream],
                    **(
                        {
                            "valid_ratio": stats.get("valid_ratio", 0.0),
                            "above_floor_close_pixels": stats.get("above_floor_close_pixels", 0),
                        }
                        if depth
                        else {}
                    ),
                },
            )

    def _depth_image_to_preview(self, msg: RosImage) -> tuple[PILImage.Image | None, dict]:
        """Statistics and a colorized preview, for callers that want both."""
        depth_m, finite, geometry, stats = self._depth_image_to_stats(msg)
        if depth_m is None:
            return None, stats
        return self._depth_preview_image(msg, depth_m, finite, geometry), stats

    def _depth_image_to_stats(self, msg: RosImage):
        """The half of the depth pipeline autonomy needs, split from the half it does not.

        The zone statistics are what the planner vetoes on, so they are
        computed for every depth frame whether or not a browser has the tile
        open. Colorizing, drawing the region boxes and encoding a JPEG is the
        expensive half and buys nothing when nobody is watching, so it lives in
        _depth_preview_image and is called only when someone is. Both run on
        the ROS executor thread, which is also the thread that publishes
        /cmd_vel, so the split is the difference between paying for a picture
        nobody sees at fifteen frames a second and not.

        Returns the metric depth array, the mask of usable pixels, the region
        geometry the preview draws, and the statistics.
        """
        encoding = msg.encoding.lower()
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None, None, {}, {}
        try:
            if encoding in {"16uc1", "mono16"}:
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.step // 2))[:, : msg.width].astype(np.float32)
                depth_m = depth / 1000.0
            elif encoding in {"32fc1"}:
                depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.step // 4))[:, : msg.width]
            else:
                return None, None, {}, {"encoding": msg.encoding}
        except ValueError:
            return None, None, {}, {"encoding": msg.encoding}

        finite = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 8.0)
        x0 = int(msg.width * clamp_range(HP60C_OBSTACLE_X_MIN, 0.0, 0.98))
        x1 = int(msg.width * clamp_range(HP60C_OBSTACLE_X_MAX, 0.02, 1.0))
        y0 = int(msg.height * clamp_range(HP60C_OBSTACLE_Y_MIN, 0.0, 0.98))
        y1 = int(msg.height * clamp_range(HP60C_OBSTACLE_Y_MAX, 0.02, 1.0))
        if x1 <= x0:
            x0, x1 = int(msg.width * 0.22), int(msg.width * 0.78)
        if y1 <= y0:
            y0, y1 = int(msg.height * 0.06), int(msg.height * 0.55)
        floor_y0 = int(msg.height * clamp_range(HP60C_FLOOR_Y_MIN, 0.0, 0.98))
        obstacle = depth_m[y0:y1, x0:x1]
        floor = depth_m[floor_y0:, :]
        above_floor = depth_m[y0:floor_y0, :]
        left_side = depth_m[y0:floor_y0, :x0]
        right_side = depth_m[y0:floor_y0, x1:]
        obstacle_finite = np.isfinite(obstacle) & (obstacle > 0.05) & (obstacle < 8.0)
        floor_finite = np.isfinite(floor) & (floor > 0.05) & (floor < 8.0)
        above_floor_finite = np.isfinite(above_floor) & (above_floor > 0.05) & (above_floor < 8.0)
        left_side_finite = np.isfinite(left_side) & (left_side > 0.05) & (left_side < 8.0)
        right_side_finite = np.isfinite(right_side) & (right_side > 0.05) & (right_side < 8.0)
        above_floor_stats = depth_zone_stats(above_floor, above_floor_finite, HP60C_RED_DISTANCE_M)
        left_side_stats = depth_zone_stats(left_side, left_side_finite, HP60C_RED_DISTANCE_M)
        right_side_stats = depth_zone_stats(right_side, right_side_finite, HP60C_RED_DISTANCE_M)
        stats = {
            "valid_ratio": round(float(finite.mean()), 3) if finite.size else 0.0,
            "obstacle_valid_ratio": round(float(obstacle_finite.mean()), 3) if obstacle_finite.size else 0.0,
            "obstacle_valid_pixels": int(obstacle_finite.sum()),
            "obstacle_near_m": None,
            "obstacle_p20_m": None,
            "floor_valid_ratio": round(float(floor_finite.mean()), 3) if floor_finite.size else 0.0,
            "floor_p20_m": None,
            "center_valid_ratio": round(float(obstacle_finite.mean()), 3) if obstacle_finite.size else 0.0,
            "center_valid_pixels": int(obstacle_finite.sum()),
            "center_near_m": None,
            "center_p20_m": None,
            "min_m": None,
            "max_m": None,
            "red_distance_m": HP60C_RED_DISTANCE_M,
            "red_min_pixels": HP60C_RED_MIN_PIXELS,
            "above_floor_near_m": above_floor_stats["near_m"],
            "above_floor_p20_m": above_floor_stats["p20_m"],
            "above_floor_valid_ratio": above_floor_stats["valid_ratio"],
            "above_floor_valid_pixels": above_floor_stats["valid_pixels"],
            "above_floor_close_ratio": above_floor_stats["close_ratio"],
            "above_floor_close_pixels": above_floor_stats["close_pixels"],
            "left_side_near_m": left_side_stats["near_m"],
            "left_side_p20_m": left_side_stats["p20_m"],
            "left_side_valid_ratio": left_side_stats["valid_ratio"],
            "left_side_valid_pixels": left_side_stats["valid_pixels"],
            "left_side_close_ratio": left_side_stats["close_ratio"],
            "left_side_close_pixels": left_side_stats["close_pixels"],
            "right_side_near_m": right_side_stats["near_m"],
            "right_side_p20_m": right_side_stats["p20_m"],
            "right_side_valid_ratio": right_side_stats["valid_ratio"],
            "right_side_valid_pixels": right_side_stats["valid_pixels"],
            "right_side_close_ratio": right_side_stats["close_ratio"],
            "right_side_close_pixels": right_side_stats["close_pixels"],
            "obstacle_roi": {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
            },
            "floor_roi": {
                "y0": floor_y0,
            },
        }
        if finite.any():
            valid_depth = depth_m[finite]
            stats["min_m"] = round(float(np.percentile(valid_depth, 2)), 3)
            stats["max_m"] = round(float(np.percentile(valid_depth, 98)), 3)
        if obstacle_finite.any():
            obstacle_depth = obstacle[obstacle_finite]
            stats["obstacle_near_m"] = round(float(np.percentile(obstacle_depth, 5)), 3)
            stats["obstacle_p20_m"] = round(float(np.percentile(obstacle_depth, 20)), 3)
            stats["center_near_m"] = stats["obstacle_near_m"]
            stats["center_p20_m"] = stats["obstacle_p20_m"]
        if floor_finite.any():
            floor_depth = floor[floor_finite]
            stats["floor_p20_m"] = round(float(np.percentile(floor_depth, 20)), 3)

        return depth_m, finite, {"x0": x0, "x1": x1, "y0": y0, "floor_y0": floor_y0, "y1": y1}, stats

    def _depth_preview_image(self, msg: RosImage, depth_m, finite, geometry: dict) -> PILImage.Image:
        """The expensive half: colorize the frame and draw the regions on it."""
        x0, x1, y0, y1, floor_y0 = geometry["x0"], geometry["x1"], geometry["y0"], geometry["y1"], geometry["floor_y0"]
        clipped = np.where(finite, np.clip(depth_m, 0.2, 4.0), 0.0)
        normalized = np.zeros_like(clipped, dtype=np.uint8)
        normalized[finite] = np.uint8(255 - ((clipped[finite] - 0.2) / 3.8 * 255).clip(0, 255))
        preview = self._colorize_depth(normalized)
        preview[~finite] = (8, 10, 9)
        image = PILImage.fromarray(preview, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(255, 245, 0), width=2)
        draw.rectangle((0, y0, max(0, x0 - 1), max(y0, floor_y0 - 1)), outline=(245, 120, 80), width=1)
        draw.rectangle((x1, y0, msg.width - 1, max(y0, floor_y0 - 1)), outline=(245, 120, 80), width=1)
        draw.line((0, floor_y0, msg.width - 1, floor_y0), fill=(75, 210, 90), width=2)
        return image

    def _colorize_depth(self, normalized: np.ndarray) -> np.ndarray:
        n = normalized.astype(np.float32) / 255.0
        red = np.clip(1.8 * n, 0.0, 1.0)
        green = np.clip(1.8 - np.abs((n * 2.0) - 1.0) * 1.8, 0.0, 1.0)
        blue = np.clip(1.8 * (1.0 - n), 0.0, 1.0)
        return (np.dstack((red, green, blue)).astype(np.float32).clip(0.0, 1.0) * 255).astype(np.uint8)

    def _rgb_image_to_preview(self, msg: RosImage) -> PILImage.Image | None:
        encoding = msg.encoding.lower()
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None
        try:
            if encoding == "bgr8":
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step // 3, 3))[:, : msg.width]
                return PILImage.fromarray(bgr[:, :, ::-1].copy(), mode="RGB")
            if encoding == "rgb8":
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step // 3, 3))[:, : msg.width]
                return PILImage.fromarray(rgb.copy(), mode="RGB")
            if encoding in {"mono8", "8uc1"}:
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))[:, : msg.width]
                return PILImage.fromarray(gray.copy(), mode="L").convert("RGB")
        except ValueError:
            return None
        return None

    def _annotate_ros_frame(self, frame: PILImage.Image, name: str, encoding: str, stats: dict) -> None:
        label = f"{name} {encoding} {time.strftime('%H:%M:%S')}"
        near = stats.get("obstacle_p20_m")
        if near is not None:
            label += f" obstacle {near:.2f}m"
        draw = ImageDraw.Draw(frame)
        draw.rectangle((8, 8, min(frame.width - 8, 560), 42), fill=(0, 0, 0))
        draw.text((16, 20), label, fill=(235, 244, 238))

    def _encode_frame(self, frame: PILImage.Image, quality: int) -> bytes | None:
        return encode_jpeg(frame, quality)

    def _sector(self, values: list[float]) -> dict:
        p20 = finite_percentile(values, 20)
        p50 = finite_percentile(values, 50)
        return {
            "count": len(values),
            "near_m": round(p20, 3) if p20 is not None else None,
            "median_m": round(p50, 3) if p50 is not None else None,
        }

    def _publish(self) -> None:
        with self._lock:
            auto_enabled = bool(self._auto["enabled"])
            command = dict(self._command)
        if auto_enabled:
            msg, published = self._auto_command()
        elif not command["enabled"] or time.monotonic() - command["updated_at"] > CMD_TIMEOUT_S:
            msg = Twist()
            published = zero_command("watchdog")
        else:
            msg = Twist()
            msg.linear.x = command["linear_x"]
            msg.linear.y = command["steering_y"]
            msg.angular.z = command["angular_z"]
            published = command
        self.publisher.publish(msg)
        self._publish_count += 1
        self._last_publish_at = time.monotonic()
        with self._lock:
            self._last_published = published

    def _auto_command(self) -> tuple[Twist, dict]:
        source = self._planner_depth_source()
        with self._lock:
            scan = dict(self._scan)
            auto = dict(self._auto)
            state = dict(self._auto_state)
            feedback = self._motion_feedback_locked()
        msg, decision = self._compute_auto_command(scan, auto, source, state=state, update_state=True, feedback=feedback)
        published = {
            "enabled": True,
            "linear_x": round(msg.linear.x, 4),
            "steering_y": round(msg.linear.y, 4),
            "angular_z": 0.0,
            "updated_at": time.monotonic(),
            "source": "auto",
        }
        decision["linear_x"] = published["linear_x"]
        decision["steering_y"] = published["steering_y"]
        with self._lock:
            self._auto_decision = decision
        return msg, published

    def _compute_auto_command(
        self,
        scan: dict,
        auto: dict,
        depth_source: dict | None = None,
        state: dict | None = None,
        update_state: bool = False,
        feedback: dict | None = None,
    ) -> tuple[Twist, dict]:
        now = time.monotonic()
        state = dict(state or self._new_auto_state())
        age = time.monotonic() - scan["updated_at"] if scan["updated_at"] else None
        sectors = scan.get("sectors", {})
        front = sectors.get("front", {})
        front_left = sectors.get("front_left", {})
        front_right = sectors.get("front_right", {})
        left = sectors.get("left", {})
        right = sectors.get("right", {})
        front_near = finite_or_none(front.get("near_m"))
        front_left_near = finite_or_none(front_left.get("near_m"))
        front_right_near = finite_or_none(front_right.get("near_m"))
        left_near = finite_or_none(left.get("near_m"))
        right_near = finite_or_none(right.get("near_m"))

        msg = Twist()
        reason = ""
        action = "stop"
        # Whichever depth camera is fitted, already chosen for us. The planner
        # reads distances and zone statistics and never asks which module
        # produced them; all it keeps hold of is the camera's name, so that
        # every reason string it writes says what it is waiting on.
        source = depth_source or {}
        depth_camera = source.get("camera")
        depth_stale_s = finite_float(source.get("stale_s"), HP60C_STALE_S)
        depth = source.get("depth") or {}
        depth_age = time.monotonic() - depth["updated_at"] if depth.get("updated_at") else None
        depth_near = finite_or_none(depth.get("obstacle_p20_m"))
        depth_above_near = finite_or_none(depth.get("above_floor_near_m"))
        depth_left_near = finite_or_none(depth.get("left_side_p20_m"))
        depth_right_near = finite_or_none(depth.get("right_side_p20_m"))
        depth_valid_ratio = float(depth.get("obstacle_valid_ratio", 0.0) or 0.0)
        depth_above_valid_ratio = float(depth.get("above_floor_valid_ratio", 0.0) or 0.0)
        depth_above_close_pixels = int(depth.get("above_floor_close_pixels", 0) or 0)
        depth_left_close_pixels = int(depth.get("left_side_close_pixels", 0) or 0)
        depth_right_close_pixels = int(depth.get("right_side_close_pixels", 0) or 0)
        depth_ok = (
            depth_camera is not None
            and depth_age is not None
            and depth_age < depth_stale_s
            and (depth_near is not None or depth_above_near is not None)
            and max(depth_valid_ratio, depth_above_valid_ratio) >= HP60C_DEPTH_VALID_MIN_RATIO
        )
        corridor = None
        if age is None or age > LIDAR_STALE_S or not front_near:
            reason = "waiting for lidar"
        elif self._cached_cmd_vel_subscribers() < 1:
            reason = "waiting for base driver"
        elif depth_camera is None:
            reason = "waiting for a depth camera"
            action = "wait_for_depth_camera"
        elif not depth_ok:
            reason = f"waiting for {depth_camera} depth frames"
            action = "wait_for_depth_frames"
        else:
            corridor = self._best_lidar_corridor(scan, auto)
            depth_side = self._depth_side_steering(depth_left_near, depth_right_near, depth_left_close_pixels, depth_right_close_pixels)
            depth_stop = (
                depth_ok
                and depth_above_near is not None
                and depth_above_near <= auto["stop_distance"]
                and depth_above_close_pixels >= HP60C_RED_MIN_PIXELS
            )
            depth_avoid = depth_ok and depth_near is not None and depth_near <= auto["avoid_distance"]
            lidar_stop = front_near <= auto["stop_distance"]
            lidar_avoid = front_near <= auto["avoid_distance"] or corridor["near_m"] <= auto["avoid_distance"]
            hazard = depth_stop or lidar_stop
            clear_for_cruise = self._clear_for_cruise(front_near, depth_near, depth_above_near, corridor, auto)
            escape_direction = self._escape_direction(corridor, depth_side, depth_left_close_pixels, depth_right_close_pixels)

            # Blind travel is charged to the episode before anything decides
            # what to do next, so a reverse that has used up the budget is
            # already known to have used it up on the tick it happens.
            if state["name"] == "reverse_escape":
                state = self._charge_reverse_budget(state, now, auto, feedback)
            reverse_spent = self._reverse_budget_spent(state)

            if state["name"] == "cruise" and hazard:
                state = self._enter_auto_state(
                    "brake",
                    now,
                    AUTO_BRAKE_S,
                    escape_direction,
                    min(int(state.get("attempts", 0)) + 1, 6),
                    "depth camera sees an object above the floor line" if depth_stop else "lidar front too close",
                    state,
                )
            elif state["name"] == "brake" and now >= state["until"]:
                if reverse_spent:
                    # There is nowhere left to go. Forward is the hazard that
                    # braked us and backward is unobserved space this episode
                    # has already spent its whole allowance on.
                    state = self._enter_auto_state(
                        "blocked",
                        now,
                        0.0,
                        state["direction"],
                        int(state.get("attempts", 1)),
                        "boxed in and the blind reverse budget is spent; stopped for the operator",
                        state,
                    )
                else:
                    state = self._enter_auto_state(
                        "reverse_escape",
                        now,
                        self._reverse_seconds_left(state),
                        state["direction"],
                        int(state.get("attempts", 1)),
                        "short reverse to create turning room",
                        state,
                    )
            elif state["name"] == "reverse_escape" and (reverse_spent or now >= state["until"]):
                if hazard and not clear_for_cruise:
                    state = self._enter_auto_state(
                        "blocked",
                        now,
                        0.0,
                        state["direction"],
                        min(int(state.get("attempts", 1)) + 1, 6),
                        "still boxed in and the blind reverse budget is spent; stopped for the operator",
                        state,
                    )
                else:
                    # Forward again as soon as the reverse has bought enough
                    # room to point somewhere else. Everything from here is
                    # driven into space the LiDAR and the depth camera can see.
                    state = self._enter_auto_state(
                        "turn_out",
                        now,
                        AUTO_TURN_OUT_S,
                        escape_direction,
                        int(state.get("attempts", 1)),
                        "forward arc toward open corridor",
                        state,
                    )
            elif state["name"] == "turn_out" and hazard and now - state["entered_at"] > 0.90:
                state = self._enter_auto_state(
                    "brake",
                    now,
                    AUTO_BRAKE_S,
                    escape_direction,
                    min(int(state.get("attempts", 1)) + 1, 6),
                    "blocked during turn-out",
                    state,
                )
            elif state["name"] == "turn_out" and now >= state["until"] and clear_for_cruise:
                state = self._enter_auto_state("cruise", now, 0.0, escape_direction, 0, "clear corridor reacquired", state)
            elif state["name"] == "blocked" and clear_for_cruise:
                # Whatever it was has moved, or somebody moved the car. The
                # corridor ahead is open and observed, so the episode is over
                # and the next one starts with its budget back.
                state = self._enter_auto_state("cruise", now, 0.0, escape_direction, 0, "clear corridor reacquired", state)
            elif state["name"] not in {"cruise", "brake", "reverse_escape", "turn_out", "blocked"}:
                state = self._enter_auto_state("cruise", now, 0.0, escape_direction, 0, "reset auto state", state)

            if state["name"] == "brake":
                reason = state["reason"]
                action = f"brake_{state['direction']}"
            elif state["name"] == "blocked":
                # Zero, and it stays zero. Auto mode is left engaged rather
                # than switched off from under the operator: the car is
                # stationary, the readout says why, and taking the stick hands
                # control straight back because a manual drive command turns
                # auto off on its way through.
                reason = state["reason"]
                action = "blocked_needs_operator"
            elif state["name"] == "reverse_escape":
                reason = state["reason"]
                action = f"reverse_escape_{state['direction']}"
                msg.linear.x = -self._reverse_command_speed(auto)
                reverse_steering = -AUTO_MAX_STEERING if state["direction"] == "left" else AUTO_MAX_STEERING
                msg.linear.y = reverse_steering
            elif state["name"] == "turn_out":
                reason = state["reason"]
                action = f"turn_out_{state['direction']}"
                msg.linear.x = max(auto["speed"] * 0.32, min(0.16, auto["speed"]))
                msg.linear.y = AUTO_MAX_STEERING if state["direction"] == "left" else -AUTO_MAX_STEERING
            elif depth_avoid:
                reason = "depth camera object avoid"
                action = f"avoid_{depth_side['direction']}"
                msg.linear.x = auto["speed"] * 0.25
                msg.linear.y = depth_side["steering"] or corridor["avoid_steering"]
            elif lidar_avoid:
                reason = "lidar corridor narrow"
                action = f"navigate_{corridor['direction']}"
                msg.linear.x = auto["speed"] * 0.35
                msg.linear.y = corridor["steering"]
            else:
                clearance = clamp_range(
                    (corridor["near_m"] - auto["avoid_distance"]) / max(0.01, auto["clear_distance"] - auto["avoid_distance"]),
                    0.45,
                    1.0,
                )
                action = f"navigate_{corridor['direction']}"
                msg.linear.x = auto["speed"] * clearance
                msg.linear.y = corridor["steering"]
                reason = "lidar farthest corridor"
                if state["name"] != "cruise":
                    state = self._enter_auto_state("cruise", now, 0.0, escape_direction, 0, "clear corridor reacquired")

        if update_state:
            with self._lock:
                self._auto_state = dict(state)

        decision = {
            "action": action,
            "linear_x": round(msg.linear.x, 4),
            "steering_y": round(msg.linear.y, 4),
            "reason": reason,
            "front_m": front_near,
            "front_left_m": front_left_near,
            "front_right_m": front_right_near,
            "left_m": left_near,
            "right_m": right_near,
            "lidar_direction": corridor["direction"] if corridor else None,
            "lidar_direction_m": corridor["score_m"] if corridor else None,
            "lidar_left_m": corridor["left_score_m"] if corridor else None,
            "lidar_front_m": corridor["front_score_m"] if corridor else None,
            "lidar_right_m": corridor["right_score_m"] if corridor else None,
            "lidar_planner": corridor.get("planner") if corridor else None,
            "lidar_gap_target_deg": corridor.get("gap_target_deg") if corridor else None,
            "lidar_gap_width_deg": corridor.get("gap_width_deg") if corridor else None,
            "lidar_gap_count": corridor.get("gap_count") if corridor else None,
            # Named for what they are rather than for the camera that used to
            # produce them. A field called hp60c_obstacle_m carrying a RealSense
            # measurement is a readout that lies, and a readout that lies is
            # worse than no readout: depth_source says which camera it came
            # from and the rest say what was measured.
            "depth_source": depth_camera,
            "depth_center_m": depth_near,
            "depth_center_valid_ratio": round(depth_valid_ratio, 3),
            "depth_obstacle_m": depth_near,
            "depth_obstacle_valid_ratio": round(depth_valid_ratio, 3),
            "depth_above_floor_m": depth_above_near,
            "depth_above_floor_valid_ratio": round(depth_above_valid_ratio, 3),
            "depth_above_floor_close_pixels": depth_above_close_pixels,
            "depth_left_side_m": depth_left_near,
            "depth_left_side_close_pixels": depth_left_close_pixels,
            "depth_right_side_m": depth_right_near,
            "depth_right_side_close_pixels": depth_right_close_pixels,
            "depth_ok": depth_ok,
            "lidar_age_s": round(age, 3) if age is not None else None,
            "depth_age_s": round(depth_age, 3) if depth_age is not None else None,
            "auto_state": state["name"],
            "auto_state_direction": state["direction"],
            "auto_state_remaining_s": round(max(0.0, state["until"] - now), 2),
            "auto_escape_attempts": int(state.get("attempts", 0)),
            # The blind reverse allowance for this stuck episode, and what is
            # left of it. Worth reporting because it is the one budget on this
            # car that an operator cannot see being spent by watching the car.
            "auto_reverse_used_s": round(finite_float(state.get("reverse_used_s"), 0.0), 2),
            "auto_reverse_used_m": round(finite_float(state.get("reverse_used_m"), 0.0), 3),
            "auto_reverse_budget_s": AUTO_REVERSE_BUDGET_S,
            "auto_reverse_budget_m": AUTO_REVERSE_BUDGET_M,
        }
        return msg, decision

    def _new_auto_state(self) -> dict:
        return {
            "name": "cruise",
            "direction": "left",
            "entered_at": time.monotonic(),
            "until": 0.0,
            "attempts": 0,
            "reason": "ready",
            "reverse_used_s": 0.0,
            "reverse_used_m": 0.0,
            "reverse_tick_at": 0.0,
        }

    def _enter_auto_state(
        self,
        name: str,
        now: float,
        duration: float,
        direction: str,
        attempts: int,
        reason: str,
        previous: dict | None = None,
    ) -> dict:
        """Move to the next state, carrying the episode's blind reverse budget with it.

        One stuck episode owns one budget and every state inside it spends from
        the same one, which is the whole point: a per attempt allowance is not
        an allowance when the machine can start another attempt. Only cruise
        clears it, because only a reacquired corridor means the car has driven
        forward through sensed space and knows where it is again.
        """
        carried = previous or {}
        spent_s = 0.0 if name == "cruise" else finite_float(carried.get("reverse_used_s"), 0.0)
        spent_m = 0.0 if name == "cruise" else finite_float(carried.get("reverse_used_m"), 0.0)
        return {
            "name": name,
            "direction": "left" if direction == "left" else "right",
            "entered_at": now,
            "until": now + max(0.0, duration),
            "attempts": attempts,
            "reason": reason,
            "reverse_used_s": spent_s,
            "reverse_used_m": spent_m,
            "reverse_tick_at": now,
        }

    def _reverse_command_speed(self, auto: dict) -> float:
        """How fast to reverse: slowly, and never faster for a faster cruise.

        This used to be max(cruise * 0.45, min(0.16, cruise)), so the default
        1.0 m/s cruise reversed blind at 0.45 m/s. There is no version of "the
        car is stuck against something it cannot see past" that is improved by
        going backwards faster.
        """
        return min(AUTO_REVERSE_SPEED_MAX, max(0.0, finite_float(auto.get("speed"), 0.0)))

    def _reverse_speed_estimate(self, auto: dict, feedback: dict | None, now: float) -> float:
        """How fast to assume the car really went, for charging the distance budget.

        The larger of what was asked for and what the base reports, because
        both can be wrong and only one direction of error is safe.

        The commanded speed is a good upper bound on the flat: the on car sweep
        measured 0.25 commanded as 0.227 travelled and 0.50 as 0.480, never
        faster than asked. It is not an upper bound on a slope.

        The measured speed catches the slope, but the board derives it from the
        wheel encoders and one of this chassis's four channels reads a constant
        zero, so it can under report, and an under reporting speedometer must
        never be able to lengthen a blind run. Taking the larger of the two
        means neither failure buys extra travel. Stale feedback is ignored
        entirely and the commanded speed stands alone.
        """
        commanded = self._reverse_command_speed(auto)
        measured = finite_or_none((feedback or {}).get("speed_mps"))
        updated_at = finite_float((feedback or {}).get("updated_at"), 0.0)
        if measured is not None and updated_at > 0.0 and (now - updated_at) < SENSOR_STALE_S:
            return max(commanded, abs(measured))
        return commanded

    def _charge_reverse_budget(self, state: dict, now: float, auto: dict, feedback: dict | None) -> dict:
        """Charge the time and distance this tick of reverse cost the episode."""
        last = finite_float(state.get("reverse_tick_at"), 0.0)
        # The whole gap is charged, however long it is. A gap means this loop
        # stalled, and a stalled loop does not stop the car: the last command
        # published stands until the base's own watchdog gives up on it, so the
        # car went on reversing through every second of it. Charging less than
        # the real elapsed time would be a blind run that got longer precisely
        # because the software lost track of it.
        elapsed = max(0.0, now - last) if last else 0.0
        charged = dict(state)
        charged["reverse_tick_at"] = now
        charged["reverse_used_s"] = finite_float(state.get("reverse_used_s"), 0.0) + elapsed
        charged["reverse_used_m"] = finite_float(state.get("reverse_used_m"), 0.0) + elapsed * self._reverse_speed_estimate(
            auto, feedback, now
        )
        return charged

    def _reverse_budget_spent(self, state: dict) -> bool:
        return (
            finite_float(state.get("reverse_used_s"), 0.0) >= AUTO_REVERSE_BUDGET_S
            or finite_float(state.get("reverse_used_m"), 0.0) >= AUTO_REVERSE_BUDGET_M
        )

    def _reverse_seconds_left(self, state: dict) -> float:
        return max(0.0, AUTO_REVERSE_BUDGET_S - finite_float(state.get("reverse_used_s"), 0.0))

    def _escape_direction(self, corridor: dict, depth_side: dict, left_close_pixels: int, right_close_pixels: int) -> str:
        if depth_side.get("direction") in {"left", "right"}:
            return depth_side["direction"]
        if left_close_pixels >= HP60C_RED_MIN_PIXELS and right_close_pixels < HP60C_RED_MIN_PIXELS:
            return "right"
        if right_close_pixels >= HP60C_RED_MIN_PIXELS and left_close_pixels < HP60C_RED_MIN_PIXELS:
            return "left"
        if corridor.get("avoid_direction") in {"left", "right"}:
            return corridor["avoid_direction"]
        return "left" if corridor.get("left_score_m", 0.0) >= corridor.get("right_score_m", 0.0) else "right"

    def _clear_for_cruise(
        self,
        front_near: float | None,
        depth_near: float | None,
        depth_above_near: float | None,
        corridor: dict,
        auto: dict,
    ) -> bool:
        required = auto["avoid_distance"] + AUTO_CRUISE_CLEAR_MARGIN_M
        if front_near is None or front_near < required:
            return False
        if corridor["near_m"] < required:
            return False
        if depth_above_near is not None and depth_above_near < auto["stop_distance"] + AUTO_CRUISE_CLEAR_MARGIN_M:
            return False
        if depth_near is not None and depth_near < auto["stop_distance"] + AUTO_CRUISE_CLEAR_MARGIN_M:
            return False
        return True

    def _depth_side_steering(
        self,
        left_near: float | None,
        right_near: float | None,
        left_close_pixels: int,
        right_close_pixels: int,
    ) -> dict:
        if left_near is None and right_near is None:
            return {"direction": "clearer_side_unknown", "steering": 0.0}
        if left_near is None:
            return {"direction": "left", "steering": AUTO_MAX_STEERING * 0.55}
        if right_near is None:
            return {"direction": "right", "steering": -AUTO_MAX_STEERING * 0.55}
        if left_close_pixels >= HP60C_RED_MIN_PIXELS and right_close_pixels < HP60C_RED_MIN_PIXELS:
            return {"direction": "right", "steering": -AUTO_MAX_STEERING * 0.55}
        if right_close_pixels >= HP60C_RED_MIN_PIXELS and left_close_pixels < HP60C_RED_MIN_PIXELS:
            return {"direction": "left", "steering": AUTO_MAX_STEERING * 0.55}
        if left_near < right_near:
            return {"direction": "right", "steering": -AUTO_MAX_STEERING * 0.45}
        if right_near < left_near:
            return {"direction": "left", "steering": AUTO_MAX_STEERING * 0.45}
        return {"direction": "center", "steering": 0.0}

    def _best_lidar_corridor(self, scan: dict, auto: dict) -> dict:
        gap = self._follow_gap_corridor(scan, auto)
        if gap is not None:
            return gap

        sectors = scan.get("sectors", {})
        front = sectors.get("front", {})
        front_left = sectors.get("front_left", {})
        front_right = sectors.get("front_right", {})
        left = sectors.get("left", {})
        right = sectors.get("right", {})

        front_near = finite_or_none(front.get("near_m"))
        front_median = finite_or_none(front.get("median_m")) or front_near or 0.0
        front_left_near = finite_or_none(front_left.get("near_m"))
        front_left_median = finite_or_none(front_left.get("median_m")) or front_left_near or 0.0
        front_right_near = finite_or_none(front_right.get("near_m"))
        front_right_median = finite_or_none(front_right.get("median_m")) or front_right_near or 0.0
        left_near = finite_or_none(left.get("near_m"))
        left_median = finite_or_none(left.get("median_m")) or left_near or 0.0
        right_near = finite_or_none(right.get("near_m"))
        right_median = finite_or_none(right.get("median_m")) or right_near or 0.0

        front_score = self._corridor_score(front_median, front_near, auto)
        left_raw = max(front_left_median, left_median * 0.85)
        right_raw = max(front_right_median, right_median * 0.85)
        left_score = self._corridor_score(left_raw, self._min_present(front_left_near, left_near), auto)
        right_score = self._corridor_score(right_raw, self._min_present(front_right_near, right_near), auto)

        scores = {
            "left": left_score,
            "front": front_score,
            "right": right_score,
        }
        direction = max(scores, key=scores.get)
        if direction == "left":
            steering = AUTO_MAX_STEERING
            near = self._min_present(front_left_near, left_near) or left_score
        elif direction == "right":
            steering = -AUTO_MAX_STEERING
            near = self._min_present(front_right_near, right_near) or right_score
        else:
            steering = clamp((left_score - right_score) * 0.035, AUTO_MAX_STEERING * 0.55)
            near = front_near or front_score

        avoid_direction = "left" if left_score >= right_score else "right"
        avoid_steering = AUTO_MAX_STEERING if avoid_direction == "left" else -AUTO_MAX_STEERING
        return {
            "direction": direction,
            "steering": steering,
            "near_m": max(0.0, float(near or 0.0)),
            "score_m": round(float(scores[direction]), 3),
            "left_score_m": round(float(left_score), 3),
            "front_score_m": round(float(front_score), 3),
            "right_score_m": round(float(right_score), 3),
            "avoid_direction": avoid_direction,
            "avoid_steering": avoid_steering,
            "planner": "sector_fallback",
        }

    def _follow_gap_corridor(self, scan: dict, auto: dict) -> dict | None:
        samples = [
            (float(angle), float(distance))
            for angle, distance in scan.get("gap_samples", [])
            if math.isfinite(float(angle)) and math.isfinite(float(distance)) and float(distance) > 0.02
        ]
        if len(samples) < 8:
            return None
        samples.sort(key=lambda item: item[0])
        angles = [item[0] for item in samples]
        raw_ranges = [min(float(item[1]), auto["clear_distance"]) for item in samples]
        blocked = [False] * len(samples)
        safety_radius = max(0.10, AUTO_GAP_SAFETY_RADIUS_M)

        for index, distance in enumerate(raw_ranges):
            if distance <= auto["stop_distance"]:
                blocked[index] = True
            if distance <= auto["avoid_distance"]:
                bubble_deg = math.degrees(math.atan2(safety_radius, max(distance, 0.08)))
                self._block_angle_window(blocked, angles, angles[index], bubble_deg)

        for index in range(len(raw_ranges) - 1):
            left_distance = raw_ranges[index]
            right_distance = raw_ranges[index + 1]
            if abs(right_distance - left_distance) < AUTO_GAP_DISPARITY_M:
                continue
            close_index = index if left_distance < right_distance else index + 1
            bubble_deg = math.degrees(math.atan2(safety_radius * 0.75, max(raw_ranges[close_index], 0.08)))
            self._block_angle_window(blocked, angles, angles[close_index], bubble_deg)

        gaps: list[dict] = []
        start: int | None = None
        for index, is_blocked in enumerate(blocked + [True]):
            if index < len(blocked) and not is_blocked and start is None:
                start = index
            elif (index == len(blocked) or is_blocked) and start is not None:
                end = index - 1
                width_deg = angles[end] - angles[start]
                if width_deg >= AUTO_GAP_MIN_WIDTH_DEG:
                    gap_ranges = raw_ranges[start : end + 1]
                    best_offset = max(range(start, end + 1), key=lambda idx: raw_ranges[idx] - abs(angles[idx]) * 0.006)
                    target_angle = angles[best_offset]
                    near = min(gap_ranges)
                    score = float(np.percentile(np.array(gap_ranges, dtype=np.float32), 55)) + width_deg * 0.01 - abs(target_angle) * 0.006
                    gaps.append(
                        {
                            "start": start,
                            "end": end,
                            "width_deg": width_deg,
                            "target_angle": target_angle,
                            "near_m": near,
                            "score": score,
                        }
                    )
                start = None
        if not gaps:
            return None

        best = max(gaps, key=lambda gap: gap["score"])
        target_angle = best["target_angle"]
        direction = "front"
        if target_angle > 10:
            direction = "left"
        elif target_angle < -10:
            direction = "right"
        steering = clamp(target_angle / max(1.0, AUTO_GAP_FOV_DEG) * AUTO_MAX_STEERING * 1.85, AUTO_MAX_STEERING)
        left_scores = [gap["score"] for gap in gaps if gap["target_angle"] > 4]
        right_scores = [gap["score"] for gap in gaps if gap["target_angle"] < -4]
        left_score = max(left_scores) if left_scores else 0.0
        right_score = max(right_scores) if right_scores else 0.0
        avoid_direction = "left" if left_score >= right_score else "right"
        return {
            "direction": direction,
            "steering": steering,
            "near_m": round(float(best["near_m"]), 3),
            "score_m": round(float(best["score"]), 3),
            "left_score_m": round(float(left_score), 3),
            "front_score_m": round(float(max((gap["score"] for gap in gaps if abs(gap["target_angle"]) <= 10), default=0.0)), 3),
            "right_score_m": round(float(right_score), 3),
            "avoid_direction": avoid_direction,
            "avoid_steering": AUTO_MAX_STEERING if avoid_direction == "left" else -AUTO_MAX_STEERING,
            "planner": "follow_gap",
            "gap_target_deg": round(float(target_angle), 1),
            "gap_width_deg": round(float(best["width_deg"]), 1),
            "gap_count": len(gaps),
        }

    def _block_angle_window(self, blocked: list[bool], angles: list[float], center_deg: float, radius_deg: float) -> None:
        low = center_deg - radius_deg
        high = center_deg + radius_deg
        for index, angle in enumerate(angles):
            if low <= angle <= high:
                blocked[index] = True

    def _corridor_score(self, clearance: float, near: float | None, auto: dict) -> float:
        score = max(0.0, float(clearance or 0.0))
        if near is None:
            return score * 0.35
        if near <= auto["stop_distance"]:
            return score * 0.05
        if near <= auto["avoid_distance"]:
            penalty = clamp_range(
                (near - auto["stop_distance"]) / max(0.01, auto["avoid_distance"] - auto["stop_distance"]),
                0.15,
                0.60,
            )
            return score * penalty
        return score

    def _min_present(self, *values: float | None) -> float | None:
        present = [value for value in values if value is not None and math.isfinite(value)]
        return min(present) if present else None

    def _best_escape_steering(
        self,
        front_left: float | None,
        front_right: float | None,
        left: float | None,
        right: float | None,
    ) -> float:
        left_score = max(front_left or 0.0, left or 0.0)
        right_score = max(front_right or 0.0, right or 0.0)
        return AUTO_MAX_STEERING if left_score >= right_score else -AUTO_MAX_STEERING

    def _side_balance(
        self,
        front_left: float | None,
        front_right: float | None,
        left: float | None,
        right: float | None,
    ) -> float:
        left_score = min(value for value in [front_left or 3.5, left or 3.5])
        right_score = min(value for value in [front_right or 3.5, right or 3.5])
        return clamp((left_score - right_score) * 0.055, AUTO_MAX_STEERING * 0.75)

    def _publish_zero_burst(self) -> None:
        for _ in range(3):
            self.publisher.publish(Twist())
            time.sleep(0.025)

    def _empty_scan(self) -> dict:
        return {
            "updated_at": 0.0,
            "frame_id": "",
            "ranges": 0,
            "finite_ranges": 0,
            "min_m": None,
            "sectors": {
                "front": {"count": 0, "near_m": None, "median_m": None},
                "front_left": {"count": 0, "near_m": None, "median_m": None},
                "front_right": {"count": 0, "near_m": None, "median_m": None},
                "left": {"count": 0, "near_m": None, "median_m": None},
                "right": {"count": 0, "near_m": None, "median_m": None},
                "rear": {"count": 0, "near_m": None, "median_m": None},
            },
            "points": [],
            "gap_samples": [],
            "gap": {},
        }

    @staticmethod
    def _empty_depth_stats() -> dict:
        """The zone statistics a depth stream reports before it has reported any.

        Shared by both cameras. The planner reads these keys off whichever
        depth source is fitted, so a source that has never produced a frame
        presents the same shape with nothing in it rather than a dict missing
        the keys, and every distance in it is None rather than a comfortable
        default that would read as open space.
        """
        return {
            "valid_ratio": 0.0,
            "obstacle_valid_ratio": 0.0,
            "obstacle_valid_pixels": 0,
            "obstacle_near_m": None,
            "obstacle_p20_m": None,
            "floor_valid_ratio": 0.0,
            "floor_p20_m": None,
            "center_valid_ratio": 0.0,
            "center_near_m": None,
            "center_p20_m": None,
            "center_valid_pixels": 0,
            "min_m": None,
            "max_m": None,
            "red_distance_m": HP60C_RED_DISTANCE_M,
            "red_min_pixels": HP60C_RED_MIN_PIXELS,
            "above_floor_near_m": None,
            "above_floor_p20_m": None,
            "above_floor_valid_ratio": 0.0,
            "above_floor_valid_pixels": 0,
            "above_floor_close_ratio": 0.0,
            "above_floor_close_pixels": 0,
            "left_side_near_m": None,
            "left_side_p20_m": None,
            "left_side_valid_ratio": 0.0,
            "left_side_valid_pixels": 0,
            "left_side_close_ratio": 0.0,
            "left_side_close_pixels": 0,
            "right_side_near_m": None,
            "right_side_p20_m": None,
            "right_side_valid_ratio": 0.0,
            "right_side_valid_pixels": 0,
            "right_side_close_ratio": 0.0,
            "right_side_close_pixels": 0,
            "obstacle_roi": {},
            "floor_roi": {},
        }

    @staticmethod
    def _empty_camera_stream() -> dict:
        return {
            "updated_at": 0.0,
            "frames": 0,
            "width": 0,
            "height": 0,
            "encoding": "",
            "frame_id": "",
            "ok": False,
            "age_s": None,
        }

    def _empty_hp60c(self) -> dict:
        empty_stream = self._empty_camera_stream()
        return {
            "depth": {**empty_stream, **self._empty_depth_stats()},
            "rgb": dict(empty_stream),
            "points": {
                **empty_stream,
                "point_step": 0,
                "row_step": 0,
            },
            "depth_jpeg": None,
            "rgb_jpeg": None,
        }

    def _empty_realsense(self) -> dict:
        realsense = {name: self._empty_camera_stream() for name in REALSENSE_TOPICS}
        # The depth stream carries the same statistics block the HP60C's does,
        # because either camera can be the one the planner reads.
        realsense["depth"].update(self._empty_depth_stats())
        realsense.update({f"{name}_jpeg": None for name in REALSENSE_TOPICS})
        return realsense

    def _empty_sensors(self) -> dict:
        names = [
            "lidar",
            "imu",
            "magnetometer",
            "joint_states",
            "velocity_feedback",
            "voltage",
            "base_firmware",
            "probe_camera",
            "base_probe_status",
            "base_bridge_status",
            "lidar_probe_status",
            "hp60c_depth",
            "hp60c_rgb",
        ]
        if HP60C_POINTS_ENABLED:
            names.append("hp60c_points")
        return {
            "stale_s": SENSOR_STALE_S,
            "items": {
                name: {
                    "updated_at": 0.0,
                    "age_s": None,
                    "frames": 0,
                    "ok": False,
                    "data": {},
                }
                for name in names
            },
        }

    def _record_sensor_locked(self, name: str, data: dict) -> None:
        if name not in self._sensors["items"]:
            self._sensors["items"][name] = {
                "updated_at": 0.0,
                "age_s": None,
                "frames": 0,
                "ok": False,
                "data": {},
            }
        sensor = self._sensors["items"][name]
        sensor["updated_at"] = time.monotonic()
        sensor["frames"] += 1
        sensor["data"] = data

    def _motion_feedback_locked(self) -> dict:
        """How fast the base says it is actually going, from /vel_raw.

        The board derives this from the wheel encoders, and one of the four
        channels on this chassis reads a constant zero, so the number is a
        measurement with a known bias towards under reporting. Everything that
        uses it has to treat it that way: see _reverse_travel_estimate_m.
        """
        sensor = self._sensors["items"].get("velocity_feedback") or {}
        linear = (sensor.get("data") or {}).get("linear") or []
        speed = finite_or_none(linear[0]) if linear else None
        return {
            "speed_mps": abs(speed) if speed is not None else None,
            "updated_at": finite_float(sensor.get("updated_at"), 0.0),
        }

    def _hp60c_meta_locked(self) -> dict:
        return {
            "depth": dict(self._hp60c["depth"]),
            "rgb": dict(self._hp60c["rgb"]),
            "points": dict(self._hp60c["points"]),
        }

    def _realsense_meta_locked(self) -> dict:
        return {name: dict(self._realsense[name]) for name in REALSENSE_TOPICS}


rclpy.init()
control = RosmasterControl()
gamepad_lock = threading.Lock()
gamepad_state = {
    "ok": False,
    "updated_at": 0.0,
    "age_s": None,
    "enabled": False,
    "armed": False,
    "auto": False,
    "index": None,
    "id": "",
    "mapping": "",
    "buttons": 0,
    "axes": [],
    "pressed": [],
}


def update_gamepad_state(payload: dict) -> dict:
    now = time.monotonic()
    pressed = payload.get("pressed", [])
    if not isinstance(pressed, list):
        pressed = []
    axes = payload.get("axes", [])
    if not isinstance(axes, list):
        axes = []
    clean = {
        "ok": True,
        "updated_at": now,
        "age_s": 0.0,
        "enabled": bool(payload.get("enabled", False)),
        "armed": bool(payload.get("armed", False)),
        "auto": bool(payload.get("auto", False)),
        "index": payload.get("index"),
        "id": str(payload.get("id", ""))[:160],
        "mapping": str(payload.get("mapping", ""))[:40],
        "buttons": int(payload.get("buttons", 0) or 0),
        "axes": [round(float(value), 3) for value in axes[:8] if isinstance(value, (int, float))],
        "pressed": [
            {
                "index": int(item.get("index", -1)),
                "name": str(item.get("name", ""))[:24],
                "value": round(float(item.get("value", 0.0) or 0.0), 3),
            }
            for item in pressed[:8]
            if isinstance(item, dict)
        ],
    }
    with gamepad_lock:
        gamepad_state.update(clean)
        snapshot = dict(gamepad_state)
    return snapshot


def gamepad_snapshot() -> dict:
    with gamepad_lock:
        snapshot = dict(gamepad_state)
    if snapshot["updated_at"]:
        snapshot["age_s"] = round(time.monotonic() - snapshot["updated_at"], 3)
    return snapshot


def spin_ros() -> None:
    rclpy.spin(control)


class CommandFreshness:
    """Decides whether a drive command still reflects what the operator wants.

    The operator reported the car continuing to move after they released the
    stick. On a link whose drive POSTs have a 1212 ms 90th percentile, a burst
    of throttle commands can still be in flight when the zero that follows them
    lands, and whichever arrives last wins. Two checks stop that, and neither
    needs the browser and the car to agree on what time it is:

      seq      a counter the page increments once per command, restarted on
               each page load and tagged with a client_id so a reload is not
               mistaken for a reordering. Anything not strictly newer than the
               newest already seen is dropped, so a late throttle cannot
               overwrite the release that overtook it.

      age_ms   how long the page had gone without reading its inputs when it
               issued the request, measured as the difference between two
               readings of the browser's own monotonic clock. A difference
               carries the same meaning on both machines, which an absolute
               wall clock would not: nothing here assumes the two agree.

    A payload carrying neither field is applied unchanged. An older page must
    not be locked out of driving the car by a check it has never heard of.

    A rejected command is not merely ignored: the caller applies it as a zero,
    because a command we cannot trust and a command to stop are the same thing
    when the ambiguity is which way the car should err.
    """

    def __init__(self, max_age_ms: float = DRIVE_MAX_AGE_MS, log=None) -> None:
        self._lock = threading.Lock()
        self._max_age_ms = float(max_age_ms)
        self._log = log if log is not None else log_line
        self._client = None
        self._last_seq: float | None = None
        self._rejected = {"stale": 0, "out_of_order": 0, "total": 0}
        self._last_reason = ""
        self._last_rejected_at = 0.0
        self._last_log_at: float | None = None
        self._since_last_log = 0

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def check(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            return {"fresh": True, "reason": "unversioned"}
        seq = self._number(payload.get("seq"))
        age = self._number(payload.get("age_ms"))
        if seq is None and age is None:
            return {"fresh": True, "reason": "unversioned"}

        client = str(payload.get("client_id", ""))[:64]
        # Rejection carries a log call, and a rejection made every later
        # drive/status request block behind this lock while that call sat
        # waiting on a stalled stdout pipe. rejection is (verdict, line) so
        # the write can happen after `with` releases the lock below; only
        # the counters and the throttling decision need to happen inside it.
        rejection = None
        with self._lock:
            if client != self._client:
                self._client = client
                self._last_seq = None
            if seq is not None and self._last_seq is not None and seq <= self._last_seq:
                rejection = self._reject_locked("out_of_order", "out of order", seq, age)
            else:
                if seq is not None:
                    # Recorded before the age check on purpose. A stale command
                    # is still the newest thing the page has produced, so
                    # accepting an older sibling behind it would put back the
                    # very throttle we are refusing.
                    self._last_seq = seq
                if age is not None and age > self._max_age_ms:
                    rejection = self._reject_locked("stale", "stale", seq, age)
        if rejection is None:
            return {"fresh": True, "reason": "fresh"}
        verdict, line = rejection
        if line is not None:
            self._log(line)
        return verdict

    def _reject_locked(self, counter: str, reason: str, seq: float | None, age: float | None) -> tuple[dict, str | None]:
        self._rejected[counter] += 1
        self._rejected["total"] += 1
        self._last_reason = reason
        self._last_rejected_at = time.monotonic()
        self._since_last_log += 1
        # Throttled, because a bad link rejects in bursts and one line per
        # dropped command would bury everything else in the device log.
        # None rather than 0.0, so the first rejection after start up is
        # logged rather than compared against a monotonic clock that may not
        # have reached the interval yet.
        now = time.monotonic()
        line = None
        if self._last_log_at is None or now - self._last_log_at >= DRIVE_REJECT_LOG_INTERVAL_S:
            self._last_log_at = now
            dropped = self._since_last_log
            self._since_last_log = 0
            line = (
                f"DRIVE_REJECTED reason={reason} seq={seq} age_ms={age} "
                f"max_age_ms={self._max_age_ms} in_last_burst={dropped} total={self._rejected['total']}"
            )
        return {"fresh": False, "reason": reason}, line

    def snapshot(self) -> dict:
        with self._lock:
            age = time.monotonic() - self._last_rejected_at if self._last_rejected_at else None
            return {
                "max_age_ms": self._max_age_ms,
                "rejected": dict(self._rejected),
                "last_reason": self._last_reason,
                "last_rejected_age_s": round(age, 3) if age is not None else None,
                "client_id": self._client,
                "last_seq": self._last_seq,
            }


command_freshness = CommandFreshness()


class Handler(BaseHTTPRequestHandler):
    server_version = "RosmasterWebRemote/0.3"
    # HTTP/1.0 closed the socket after every response, so each of the eight
    # commands a second opened a fresh TCP connection. On a link that averages
    # 319 ms round trip, most of the command budget went on handshakes.
    # _send_json and _send_file already send Content-Length and are safe on a
    # persistent connection. The two MJPEG responses cannot be, and say so for
    # themselves.
    protocol_version = "HTTP/1.1"

    # Persistent connections need an idle timeout or they leak threads.
    # ThreadingHTTPServer dedicates a thread per connection, and with keep-alive
    # that thread sits in readline() waiting for a request that may never come.
    # A browser opens several connections per origin and reconnects the camera
    # stream periodically, so without this the server accumulates blocked
    # threads until it stops answering: the port still accepts TCP while every
    # request hangs, which is exactly how this failed on the car. Ten seconds is
    # far longer than the 200 ms command heartbeat and the 750 ms status poll,
    # so a live client is never disconnected mid-use.
    timeout = 10

    def handle_one_request(self) -> None:
        # A timed out keep-alive connection is normal housekeeping, not an
        # error. Close it quietly rather than logging a traceback per idle
        # browser tab.
        try:
            super().handle_one_request()
        except (TimeoutError, socket.timeout):
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        log_line("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            hp60c = control.hp60c_snapshot()
            realsense = control.realsense_snapshot()
            self._send_json(
                {
                    "ok": True,
                    "cameras": camera_feeds(hp60c, realsense),
                    "control": control.snapshot(),
                    "lidar": control.lidar_snapshot(),
                    "hp60c": hp60c,
                    "realsense": realsense,
                    "sensors": control.sensors_snapshot(),
                    "auto": control.auto_snapshot(),
                    "navigation": control.navigation_snapshot(),
                    "gamepad": gamepad_snapshot(),
                    "commands": command_freshness.snapshot(),
                }
            )
        elif parsed.path == "/api/gamepad":
            self._send_json({"ok": True, "gamepad": gamepad_snapshot()})
        elif parsed.path in CAMERA_STREAM_PATHS:
            # Routed from the registry rather than a hand written branch per
            # feed, so a feed added to CAMERA_FEEDS is served as well as
            # advertised. Every path is routed whether or not the camera behind
            # it is fitted: an absent camera is answered with a stream that
            # sends nothing and closes, which is a great deal easier to read
            # than a 404 on a page that has just been told the feed exists.
            self._stream_camera(*CAMERA_STREAM_PATHS[parsed.path])
        elif parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/")
            self._send_file(STATIC_DIR / rel, self._content_type(rel))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # The body is read once, up front, whatever the route does with it. On
        # a persistent connection a body left unread becomes the first bytes of
        # the next request, and /api/stop is POSTed with a body no stop path
        # has ever looked at.
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"ok": False, "error": f"invalid json: {exc}"}, status=400)
            return

        if parsed.path == "/api/drive":
            verdict = command_freshness.check(payload)
            # A command we cannot trust is applied as a zero rather than
            # discarded. Discarding it would leave whatever the car was already
            # doing in place, which is the wrong way to resolve the doubt.
            cmd = control.update(payload if verdict["fresh"] else {"enabled": False})
            self._send_json({
                "ok": True,
                "command": cmd,
                "rejected": not verdict["fresh"],
                "reason": verdict["reason"],
                "control": control.snapshot(),
                "auto": control.auto_snapshot(),
            })
        elif parsed.path == "/api/auto":
            auto = control.set_auto(payload)
            self._send_json({"ok": True, "auto": auto, "control": control.snapshot()})
        elif parsed.path == "/api/stop":
            control.stop()
            self._send_json({"ok": True, "control": control.snapshot(), "auto": control.auto_snapshot()})
        elif parsed.path == "/api/start":
            command = control.start()
            self._send_json({"ok": True, "command": command, "control": control.snapshot(), "auto": control.auto_snapshot()})
        elif parsed.path == "/api/gamepad":
            self._send_json({"ok": True, "gamepad": update_gamepad_state(payload)})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Never let a browser hold on to the page or its scripts. With no cache
        # directive at all, browsers apply their own heuristic freshness and
        # happily keep serving a stale index.html and app.js after a redeploy.
        # That cost a whole round of field testing: the operator kept driving an
        # old build, reporting bugs that had already been fixed, while the car
        # was serving the new one. The files are a few tens of kilobytes over a
        # LAN, so there is nothing to gain by caching them.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream_camera(self, camera: str, stream: str) -> None:
        """Serve one feed as MJPEG for as long as it has something to send.

        This response owns a request thread for its whole life, and the gallery
        opens one per tile. Four tiles is double what this server has carried,
        and the two rules that keep that from becoming a thread leak are here.

        The first is the idle bound. A stream only finds out that the browser
        has gone by failing to write to it, so a feed with no new frames used
        to loop and sleep forever, holding a thread that no longer had a client
        on the other end, and every reconnect added another. Now a run of
        CAMERA_STREAM_IDLE_TIMEOUT_S with nothing new to send ends the response
        and the thread with it. The page reopens closed streams on its own,
        staggered per tile, so a feed that comes back is picked up again.

        The second is that a frame is encoded once, in the ROS callback, and
        every viewer of that feed writes out the same bytes. Two operators
        watching the same tile cost two threads and no extra work on the car.
        """
        # No Content-Length, because the body never ends. Under HTTP/1.1 that
        # leaves the browser waiting for a length it will never be told, so
        # this response opts out of the persistent connection explicitly. It
        # also means the thread is released at the end of this method rather
        # than parked in readline waiting for a request that will not come.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        control.open_camera_viewer(camera, stream)
        try:
            last_frame = None
            last_sent_at = time.monotonic()
            while True:
                jpg = control.camera_frame(camera, stream)
                if jpg is None or jpg == last_frame:
                    if time.monotonic() - last_sent_at > CAMERA_STREAM_IDLE_TIMEOUT_S:
                        return
                    time.sleep(0.1 if jpg is None else 0.03)
                    continue
                last_frame = jpg
                last_sent_at = time.monotonic()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
        finally:
            control.close_camera_viewer(camera, stream)

    def _content_type(self, rel: str) -> str:
        if rel.endswith(".css"):
            return "text/css; charset=utf-8"
        if rel.endswith(".js"):
            return "application/javascript; charset=utf-8"
        return "application/octet-stream"


def serve_https() -> None:
    """Serve the same app over TLS, so the browser will expose the gamepad.

    Browsers only hand the Gamepad API to secure contexts. Plain HTTP to the
    car's LAN address is not one, so a perfectly good controller is invisible
    to the page and the remote looks broken. localhost is also a secure origin,
    which is why a TCP forward works, but that puts a manual step between the
    operator and driving. Serving TLS directly means the car is drivable from
    any machine on the network with no forwarding.

    The certificate is self signed, so the browser shows an interstitial once
    per machine. Clicking through leaves the origin a secure context, which is
    all the Gamepad API asks for. Absent a cert this quietly does nothing and
    plain HTTP carries on unaffected.
    """
    cert = os.environ.get("TLS_CERT", "/tmp/webremote-cert.pem")
    key = os.environ.get("TLS_KEY", "/tmp/webremote-key.pem")
    if not (os.path.exists(cert) and os.path.exists(key)):
        print(f"WEB_REMOTE_TLS_SKIPPED cert={cert} missing", flush=True)
        return
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server = ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), Handler)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"WEB_REMOTE_READY_TLS port={HTTPS_PORT}", flush=True)
        server.serve_forever()
    except Exception as exc:  # noqa: BLE001 - TLS is a bonus, never fatal
        print(f"WEB_REMOTE_TLS_FAILED {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    threading.Thread(target=spin_ros, daemon=True).start()
    threading.Thread(target=serve_https, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"WEB_REMOTE_READY port={PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        control.stop()
        control.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
