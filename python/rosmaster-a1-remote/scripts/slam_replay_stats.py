#!/usr/bin/env python3
"""Record what the slam service publishes during a replay and judge it.

Runs inside the service image for `seconds` (argv[1]), then writes
/out/stats.json (counts, max map->odom correction, final poses, the last
/slam/status) and /out/traj.json (slam and odometry trajectories for the
renderer). Used by scripts/slam_offline_inner.sh.
"""
from __future__ import annotations

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("slam_replay_stats")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self.on_pose, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        self.create_subscription(String, "/slam/status", self.on_status, 10)
        self.maps = 0
        self.map_info = None
        self.poses: list = []
        self.odoms: list = []
        self.map_odom: list = []
        self.last_status = None
        self.last_live_status = None   # the last status that still saw scans: what the verdict judges

    def on_map(self, m) -> None:
        self.maps += 1
        self.map_info = {"width": m.info.width, "height": m.info.height, "resolution": m.info.resolution, "origin": [m.info.origin.position.x, m.info.origin.position.y], "occupied": m.data.count(100), "free": m.data.count(0), "unknown": m.data.count(-1)}

    def on_pose(self, p) -> None:
        self.poses.append((p.header.stamp.sec + p.header.stamp.nanosec / 1e9, p.pose.pose.position.x, p.pose.pose.position.y, yaw_of(p.pose.pose.orientation)))

    def on_odom(self, o) -> None:
        self.odoms.append((o.header.stamp.sec + o.header.stamp.nanosec / 1e9, o.pose.pose.position.x, o.pose.pose.position.y, yaw_of(o.pose.pose.orientation)))

    def on_tf(self, msg) -> None:
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                self.map_odom.append((t.header.stamp.sec + t.header.stamp.nanosec / 1e9, t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)))

    def on_status(self, msg) -> None:
        self.last_status = json.loads(msg.data)
        age = self.last_status.get("scan_age_s")
        if age is not None and age < 2.0:
            self.last_live_status = self.last_status


def main() -> None:
    rclpy.init()
    node = Recorder()
    deadline = time.monotonic() + float(sys.argv[1])
    while time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
    stats = {
        "maps": node.maps, "map": node.map_info, "poses": len(node.poses), "odoms": len(node.odoms), "map_odom_tfs": len(node.map_odom),
        "slam_final_pose": list(node.poses[-1][1:]) if node.poses else None,
        "odom_final_pose": list(node.odoms[-1][1:]) if node.odoms else None,
        "map_odom_max_abs_yaw": max((abs(m[3]) for m in node.map_odom), default=None),
        "map_odom_max_xy": max((math.hypot(m[1], m[2]) for m in node.map_odom), default=None),
        "map_odom_last": list(node.map_odom[-1][1:]) if node.map_odom else None,
        "last_status": node.last_status,
        "last_live_status": node.last_live_status,
    }
    with open("/out/stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    with open("/out/traj.json", "w") as f:
        json.dump({"slam": node.poses, "odom": node.odoms[::10], "map_odom": node.map_odom[::5]}, f)
    print(json.dumps(stats, indent=1))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
