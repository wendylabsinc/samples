#!/usr/bin/env python3
import json
import math
import os
import struct
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, MagneticField
from std_msgs.msg import Float32, String

from Rosmaster_Lib import Rosmaster


# The motor library documents this chassis class as v_x=[-1.8, 1.8] and the
# board clamps anything beyond it, so a larger number here is a promise the
# firmware ignores. Measured on the car, commands of 0.75 and 1.00 both produced
# about 0.72 m/s, so the drivetrain saturates well below even this. Kept as an
# environment variable because the ceiling should be findable in one place, not
# because raising it achieves anything.
MAX_VX = float(os.environ.get("ROSMASTER_MAX_VX", "1.8"))

def finite(value, fallback=0.0):
    try:
        value = float(value)
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


class RosmasterBaseBridge(Node):
    HEAD = 0xFF
    DEVICE_ID = 0xFC
    COMPLEMENT = 257 - DEVICE_ID
    FUNC_AUTO_REPORT = 0x01
    FUNC_REPORT_SPEED = 0x0A
    FUNC_REPORT_MPU_RAW = 0x0B
    FUNC_REPORT_ENCODER = 0x0D
    FUNC_REPORT_ICM_RAW = 0x0E
    FUNC_REQUEST_DATA = 0x50
    FUNC_VERSION = 0x51

    def __init__(self):
        super().__init__("rosmaster_base_bridge")
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.bot = None
        self.port = None
        self.connected = False
        self.last_error = None
        self.last_frame_time = 0.0
        self.frame_counts = {}
        self.version = None
        self.last_command = Twist()
        self.last_command_time = 0.0

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.status_pub = self.create_publisher(String, "/base_bridge/status", 10)
        self.edition_pub = self.create_publisher(String, "/edition", 10)
        self.voltage_pub = self.create_publisher(Float32, "/voltage", 10)
        self.vel_pub = self.create_publisher(Twist, "/vel_raw", 10)
        self.imu_pub = self.create_publisher(Imu, "/imu/data_raw", 10)
        self.mag_pub = self.create_publisher(MagneticField, "/imu/mag", 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.create_timer(1.0, self.publish_status)
        self.create_timer(5.0, self.request_version)
        threading.Thread(target=self.connect_loop, daemon=True).start()

    def candidate_ports(self):
        candidates = [
            os.environ.get("ROSMASTER_SERIAL_PORT", ""),
            "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
            "/dev/myserial",
            "/dev/ttyUSB1",
            "/dev/ttyUSB2",
        ]
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen and os.path.exists(candidate):
                seen.add(candidate)
                yield candidate

    def connect_loop(self):
        while rclpy.ok():
            if self.connected:
                time.sleep(1.0)
                continue
            for port in self.candidate_ports():
                try:
                    self.get_logger().info(f"opening Rosmaster serial port {port}")
                    bot = Rosmaster(com=port, debug=False)
                    with self.write_lock:
                        bot.set_car_type(9)
                        bot.set_auto_report_state(True)
                    try:
                        bot.ser.reset_input_buffer()
                    except Exception:
                        pass
                    with self.lock:
                        self.bot = bot
                        self.port = port
                        self.connected = True
                        self.last_error = None
                        self.last_frame_time = 0.0
                        self.frame_counts = {}
                        self.version = None
                    threading.Thread(target=self.read_loop, args=(bot, port), daemon=True).start()
                    self.get_logger().info(f"Rosmaster serial connected on {port}")
                    break
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"{port}: {exc}"
                    self.get_logger().warning(f"Rosmaster serial open failed on {port}: {exc}")
            time.sleep(2.0)

    def on_cmd_vel(self, msg):
        with self.lock:
            self.last_command = msg
            self.last_command_time = time.time()
            bot = self.bot if self.connected else None
        if bot is None:
            return
        # This clamp, not the speed slider, is the throttle ceiling. Raising it
        # was tried and reverted: see MAX_VX above for why a bigger number here
        # buys nothing.
        vx = max(-MAX_VX, min(MAX_VX, finite(msg.linear.x)))
        steering = max(-0.45, min(0.45, finite(msg.linear.y)))
        angular = max(-3.0, min(3.0, finite(msg.angular.z)))
        try:
            with self.write_lock:
                bot.set_car_motion(vx, steering, angular)
            # Log what actually went to the board, and what the board reports
            # back, whenever a nonzero command is in flight. Without this a
            # motionless car is indistinguishable from a car that never
            # received the command, which is exactly the ambiguity that cost
            # us an afternoon. Rate limited so idle zero traffic stays quiet.
            if vx or steering or angular:
                now = time.time()
                if now - getattr(self, "_last_cmd_log", 0.0) >= 0.5:
                    self._last_cmd_log = now
                    try:
                        measured = bot.get_motion_data()
                    except Exception as exc:  # noqa: BLE001
                        measured = f"unavailable: {exc}"
                    self.get_logger().info(
                        "CMD_WRITE " + json.dumps({
                            "sent": {"vx": vx, "steering": steering, "angular": angular},
                            "measured": measured,
                        }, default=str)
                    )
        except Exception as exc:
            with self.lock:
                self.connected = False
                self.last_error = f"cmd_vel write failed: {exc}"
            self.get_logger().warning(f"cmd_vel write failed: {exc}")

    def publish_status(self):
        with self.lock:
            version = self.version
            payload = {
                "connected": self.connected,
                "port": self.port,
                "last_error": self.last_error,
                "frame_counts": dict(self.frame_counts),
                "last_frame_age_s": round(time.time() - self.last_frame_time, 3)
                if self.last_frame_time
                else None,
                "version": version,
                "last_command_age_s": round(time.time() - self.last_command_time, 3)
                if self.last_command_time
                else None,
            }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)
        if version is not None:
            self.publish_edition(version)

    def request_version(self):
        with self.lock:
            bot = self.bot if self.connected else None
            has_version = self.version is not None
        if bot is None:
            return
        if not has_version:
            self.write_packet([self.FUNC_REQUEST_DATA, self.FUNC_VERSION, 0])

    def write_packet(self, payload):
        with self.lock:
            bot = self.bot if self.connected else None
        if bot is None:
            return
        cmd = [self.HEAD, self.DEVICE_ID, len(payload) + 2, *payload]
        cmd.append(sum(cmd, self.COMPLEMENT) & 0xFF)
        with self.write_lock:
            bot.ser.write(bytes(cmd))
            bot.ser.flush()

    def read_loop(self, bot, port):
        while rclpy.ok():
            with self.lock:
                if bot is not self.bot or not self.connected:
                    return
            try:
                first = bot.ser.read(1)
                if not first:
                    continue
                if first[0] != self.HEAD:
                    continue
                second = bot.ser.read(1)
                if not second or second[0] != self.DEVICE_ID - 1:
                    continue
                header = bot.ser.read(2)
                if len(header) != 2:
                    continue
                ext_len, ext_type = header
                data_len = ext_len - 2
                if data_len <= 0 or data_len > 64:
                    continue
                ext_data = bot.ser.read(data_len)
                if len(ext_data) != data_len:
                    continue
                payload = ext_data[:-1]
                checksum = ext_data[-1]
                if ((ext_len + ext_type + sum(payload)) & 0xFF) != checksum:
                    continue
                with self.lock:
                    self.last_frame_time = time.time()
                    key = f"0x{ext_type:02x}"
                    self.frame_counts[key] = self.frame_counts.get(key, 0) + 1
                self.parse_frame(ext_type, payload)
            except Exception as exc:
                with self.lock:
                    if bot is self.bot:
                        self.connected = False
                        self.last_error = f"serial read failed on {port}: {exc}"
                self.get_logger().warning(f"serial read failed on {port}: {exc}")
                return

    def parse_frame(self, ext_type, data):
        stamp = self.get_clock().now().to_msg()
        if ext_type == self.FUNC_REPORT_SPEED and len(data) >= 7:
            vx = struct.unpack("h", data[0:2])[0] / 1000.0
            vy = struct.unpack("h", data[2:4])[0] / 1000.0
            vz = struct.unpack("h", data[4:6])[0] / 1000.0
            volts = data[6] / 10.0
            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.angular.z = vz
            self.vel_pub.publish(twist)
            voltage = Float32()
            voltage.data = volts
            self.voltage_pub.publish(voltage)
        elif ext_type in (self.FUNC_REPORT_MPU_RAW, self.FUNC_REPORT_ICM_RAW) and len(data) >= 18:
            if ext_type == self.FUNC_REPORT_MPU_RAW:
                gyro_ratio = 1 / 3754.9
                accel_ratio = 1 / 1671.84
                mag_ratio = 1.0
                gx = struct.unpack("h", data[0:2])[0] * gyro_ratio
                gy = struct.unpack("h", data[2:4])[0] * -gyro_ratio
                gz = struct.unpack("h", data[4:6])[0] * -gyro_ratio
            else:
                gyro_ratio = 1 / 1000.0
                accel_ratio = 1 / 1000.0
                mag_ratio = 1 / 1000.0
                gx = struct.unpack("h", data[0:2])[0] * gyro_ratio
                gy = struct.unpack("h", data[2:4])[0] * gyro_ratio
                gz = struct.unpack("h", data[4:6])[0] * gyro_ratio
            ax = struct.unpack("h", data[6:8])[0] * accel_ratio
            ay = struct.unpack("h", data[8:10])[0] * accel_ratio
            az = struct.unpack("h", data[10:12])[0] * accel_ratio
            mx = struct.unpack("h", data[12:14])[0] * mag_ratio
            my = struct.unpack("h", data[14:16])[0] * mag_ratio
            mz = struct.unpack("h", data[16:18])[0] * mag_ratio
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = ax
            imu.linear_acceleration.y = ay
            imu.linear_acceleration.z = az
            imu.angular_velocity.x = gx
            imu.angular_velocity.y = gy
            imu.angular_velocity.z = gz
            self.imu_pub.publish(imu)
            mag = MagneticField()
            mag.header.stamp = stamp
            mag.header.frame_id = "imu_link"
            mag.magnetic_field.x = mx
            mag.magnetic_field.y = my
            mag.magnetic_field.z = mz
            self.mag_pub.publish(mag)
        elif ext_type == self.FUNC_REPORT_ENCODER and len(data) >= 16:
            encoders = [struct.unpack("i", data[idx : idx + 4])[0] for idx in range(0, 16, 4)]
            joints = JointState()
            joints.header.stamp = stamp
            joints.header.frame_id = "joint_states"
            joints.name = ["encoder_m1", "encoder_m2", "encoder_m3", "encoder_m4"]
            joints.position = [float(value) for value in encoders]
            self.joint_pub.publish(joints)
        elif ext_type == self.FUNC_VERSION and len(data) >= 2:
            version = f"{data[0]}.{data[1]}"
            with self.lock:
                self.version = version
            self.publish_edition(version)

    def publish_edition(self, version):
        edition = String()
        edition.data = version
        self.edition_pub.publish(edition)

def main():
    rclpy.init()
    node = RosmasterBaseBridge()
    try:
        rclpy.spin(node)
    finally:
        with node.lock:
            bot = node.bot
        if bot is not None:
            try:
                bot.set_car_motion(0, 0, 0)
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
