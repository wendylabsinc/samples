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
        self._still_since: float | None = None
        self._still_sum = 0.0
        self._still_count = 0

    @property
    def state(self) -> str:
        if self._last_vel_at is None:
            return "waiting_for_vel_raw"
        return "tracking" if self.bias is not None else "calibrating_gyro"

    def imu(self, yaw_rate: float) -> None:
        self._gyro = yaw_rate
        self._gyro_at = self._clock()
        if self._still_since is not None:
            self._still_sum += yaw_rate
            self._still_count += 1

    def velocity(self, vx: float) -> Pose | None:
        now = self._clock()
        dt = 0.0 if self._last_vel_at is None else min(now - self._last_vel_at, self.max_dt_s)
        self._last_vel_at = now
        self.frames += 1
        still = abs(vx) < self.still_speed_mps
        self._update_bias(now, still)
        yaw_rate = 0.0 if still else self._yaw_rate(now)
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
