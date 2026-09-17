#!/usr/bin/env python3
"""Dead-reckoning odometry for the Rosmaster A1.

Integrates the firmware's forward speed (/vel_raw linear.x) with the IMU's
yaw rate (/imu/data_raw angular_velocity.z) into /odom and the
odom -> base_link transform. /vel_raw's angular.z is meaningless on the
Ackermann A1 (Yahboom's own driver marks it invalid) and linear.y is the
steer angle, so yaw comes from the gyro alone.

Good enough for slam_toolbox to scan-match against; no sensor fusion. See
docs/superpowers/specs/2026-09-17-odometry-node-design.md.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

# Anything past these is a decode error, not a manoeuvre: the A1 tops out
# around 1 m/s and its gyro at ±8.7 rad/s (500 °/s).
MAX_SPEED_MPS = 5.0
MAX_YAW_RATE_RAD_S = 20.0

# Fixed, diagonal, honest-enough covariances: modest confidence on the planar
# states we actually observe, none at all on z, roll and pitch.
_OBSERVED = 0.05
_UNOBSERVED = 1e3


def _diagonal(values) -> list:
    cov = [0.0] * 36
    for index, value in enumerate(values):
        cov[index * 7] = value
    return cov


POSE_COVARIANCE = _diagonal([_OBSERVED, _OBSERVED, _UNOBSERVED, _UNOBSERVED, _UNOBSERVED, _OBSERVED])
TWIST_COVARIANCE = _diagonal([_OBSERVED, _OBSERVED, _UNOBSERVED, _UNOBSERVED, _UNOBSERVED, _OBSERVED])


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation of `yaw` about z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def odometry_message(pose: Pose, frame: str, child_frame: str, stamp) -> Odometry:
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.child_frame_id = child_frame
    msg.pose.pose.position.x = pose.x
    msg.pose.pose.position.y = pose.y
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_quaternion(pose.yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.pose.covariance = list(POSE_COVARIANCE)
    msg.twist.twist.linear.x = pose.vx
    msg.twist.twist.angular.z = pose.yaw_rate
    msg.twist.covariance = list(TWIST_COVARIANCE)
    return msg


def transform_message(pose: Pose, frame: str, child_frame: str, stamp) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = pose.x
    tf.transform.translation.y = pose.y
    tf.transform.translation.z = 0.0
    qx, qy, qz, qw = yaw_quaternion(pose.yaw)
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    vx: float
    yaw_rate: float
    at: float


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


class DeadReckoner:
    """Planar unicycle integration on each velocity frame.

    Pure Python, no ROS: the clock is injected so tests own time.
    """

    def __init__(
        self,
        *,
        clock=time.monotonic,
        max_dt_s: float = 0.25,
        imu_stale_s: float = 0.5,
        bias_still_s: float = 2.0,
        still_speed_mps: float = 0.01,
    ) -> None:
        self._clock = clock
        self.max_dt_s = max_dt_s
        self.imu_stale_s = imu_stale_s
        self.bias_still_s = bias_still_s
        self.still_speed_mps = still_speed_mps
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.bias: float | None = None
        self.dropped = 0
        self.frames = 0
        self.imu_stale = False
        self._gyro: float | None = None
        self._gyro_at: float | None = None
        self._last_vel_at: float | None = None
        self._still_since: float | None = None
        self._still_sum = 0.0
        self._still_count = 0

    @property
    def state(self) -> str:
        if self._last_vel_at is None:
            return "waiting_for_vel_raw"
        return "tracking" if self.bias is not None else "calibrating_gyro"

    def imu(self, yaw_rate: float) -> None:
        if not math.isfinite(yaw_rate) or abs(yaw_rate) > MAX_YAW_RATE_RAD_S:
            self.dropped += 1
            return
        self._gyro = yaw_rate
        self._gyro_at = self._clock()
        if self._still_since is not None:
            self._still_sum += yaw_rate
            self._still_count += 1

    def velocity(self, vx: float) -> Pose | None:
        if not math.isfinite(vx) or abs(vx) > MAX_SPEED_MPS:
            self.dropped += 1
            return None
        now = self._clock()
        dt = 0.0 if self._last_vel_at is None else min(now - self._last_vel_at, self.max_dt_s)
        self._last_vel_at = now
        self.frames += 1
        still = abs(vx) < self.still_speed_mps
        self._update_bias(now, still)
        w = self._yaw_rate(now)
        yaw_rate = 0.0 if still else w
        if dt > 0.0:
            yaw_mid = self.yaw + yaw_rate * dt / 2.0
            self.x += vx * math.cos(yaw_mid) * dt
            self.y += vx * math.sin(yaw_mid) * dt
            self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        return Pose(self.x, self.y, self.yaw, vx, yaw_rate, now)

    def _update_bias(self, now: float, still: bool) -> None:
        """Adopt the mean gyro reading over a full still window as the bias.

        The window restarts on motion and after every adoption, so each
        estimate comes from fresh samples; later windows blend 20 % in so a
        single odd window cannot swing the bias.
        """
        if not still:
            self._still_since = None
            self._still_sum = 0.0
            self._still_count = 0
            return
        if self._still_since is None:
            self._still_since = now
            self._still_sum = 0.0
            self._still_count = 0
            return
        if now - self._still_since >= self.bias_still_s and self._still_count > 0:
            mean = self._still_sum / self._still_count
            self.bias = mean if self.bias is None else 0.8 * self.bias + 0.2 * mean
            self._still_since = now
            self._still_sum = 0.0
            self._still_count = 0

    def _yaw_rate(self, now: float) -> float:
        """Bias-corrected gyro, or 0 when there is no bias yet or the IMU is
        stale: better to integrate a straight line than stale spin."""
        stale = self._gyro_at is None or now - self._gyro_at > self.imu_stale_s
        self.imu_stale = stale
        if stale or self.bias is None:
            return 0.0
        return self._gyro - self.bias
