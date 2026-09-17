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

    def imu(self, yaw_rate: float) -> None:
        self._gyro = yaw_rate
        self._gyro_at = self._clock()

    def velocity(self, vx: float) -> Pose | None:
        now = self._clock()
        dt = 0.0 if self._last_vel_at is None else min(now - self._last_vel_at, self.max_dt_s)
        self._last_vel_at = now
        self.frames += 1
        yaw_rate = 0.0
        if dt > 0.0:
            yaw_mid = self.yaw + yaw_rate * dt / 2.0
            self.x += vx * math.cos(yaw_mid) * dt
            self.y += vx * math.sin(yaw_mid) * dt
            self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        return Pose(self.x, self.y, self.yaw, vx, yaw_rate, now)
