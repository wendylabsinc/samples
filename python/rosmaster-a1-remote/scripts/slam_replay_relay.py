#!/usr/bin/env python3
"""Rebuild odom -> base_link for a bag recorded before the odometry fixes.

Subscribes to the bag's /odom (remapped to /odom_bag by the harness) for the
forward speed and stamps, and to /imu/data_raw for the yaw rate, integrates
a planar unicycle with a fixed gyro bias, and republishes /odom plus the
transform. RELAY_VX_SIGN flips the speed; RELAY_GZ_BIAS is subtracted from
the gyro. Yaw is not integrated while |vx| < 0.01 m/s, as the odometry node
does. Used by scripts/slam_offline_check.sh with RELAY=1.
"""
from __future__ import annotations

import math
import os

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

VX_SIGN = float(os.environ.get("RELAY_VX_SIGN", "1"))
GZ_BIAS = float(os.environ.get("RELAY_GZ_BIAS", "0.0002"))


class Relay(Node):
    def __init__(self) -> None:
        super().__init__("slam_replay_relay")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.broadcaster = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(Imu, "/imu/data_raw", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom_bag", self.on_odom, 50)
        self.gz = 0.0
        self.x = self.y = self.yaw = 0.0
        self.last_t: float | None = None
        self.count = 0

    def on_imu(self, msg) -> None:
        self.gz = msg.angular_velocity.z - GZ_BIAS

    def on_odom(self, msg) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        vx = VX_SIGN * msg.twist.twist.linear.x
        if self.last_t is not None:
            dt = min(max(t - self.last_t, 0.0), 0.25)
            w = 0.0 if abs(vx) < 0.01 else self.gz
            mid = self.yaw + 0.5 * w * dt
            self.x += vx * math.cos(mid) * dt
            self.y += vx * math.sin(mid) * dt
            self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        self.last_t = t
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist.twist.linear.x = vx
        out.twist.twist.angular.z = self.gz
        self.pub.publish(out)
        self.count += 1
        if self.count % 1000 == 0:
            print(f"relay {self.count} msgs pose {self.x:.2f} {self.y:.2f} {self.yaw:.2f}", flush=True)


def main() -> None:
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
