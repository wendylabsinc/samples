#!/usr/bin/env python3
import json
import math
import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu, JointState, LaserScan, MagneticField
from std_msgs.msg import Float32, String


def emit(event, **fields):
    payload = {"event": event, "time": round(time.time(), 3)}
    payload.update(fields)
    print("SENSOR_PROBE " + json.dumps(payload, sort_keys=True), flush=True)


def finite(values):
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


class SensorProbe(Node):
    def __init__(self):
        super().__init__("sensor_probe")
        self.samples = {}
        self.states = {}
        self.lidar_failures = []
        self.base_failures = []
        self.camera_publishers = {
            "/dev/video0": (
                "/sensor_probe/video0/image/compressed",
                self.create_publisher(CompressedImage, "/sensor_probe/video0/image/compressed", 1),
            ),
            "/dev/video1": (
                "/sensor_probe/video1/image/compressed",
                self.create_publisher(CompressedImage, "/sensor_probe/video1/image/compressed", 1),
            ),
        }
        self.status_publisher = self.create_publisher(String, "/sensor_probe/status", 10)
        self.imu_publisher = self.create_publisher(Imu, "/imu/data_raw", 10)
        self.mag_publisher = self.create_publisher(MagneticField, "/imu/mag", 10)
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.vel_publisher = self.create_publisher(Twist, "/vel_raw", 10)
        self.voltage_publisher = self.create_publisher(Float32, "/voltage", 10)

        self.create_subscription(Imu, "/imu/data_raw", self.on_imu, 10)
        self.create_subscription(MagneticField, "/imu/mag", self.on_mag, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint, 10)
        self.create_subscription(Twist, "/vel_raw", self.on_vel, 10)
        self.create_subscription(Float32, "/voltage", self.on_voltage, 10)
        self.create_subscription(String, "/edition", self.on_edition, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

        self.create_timer(2.0, self.publish_status)
        if os.environ.get("PROBE_BASE_SERIAL") == "1":
            threading.Thread(target=self.base_serial_loop, daemon=True).start()
        threading.Thread(target=self.camera_loop, daemon=True).start()
        threading.Thread(target=self.audio_loop, daemon=True).start()
        if os.environ.get("PROBE_RAW_LIDAR") == "1":
            threading.Thread(target=self.lidar_probe, daemon=True).start()
        else:
            self.record_state("lidar", "waiting_for_scan", {"topic": "/scan"})

    def record_once(self, key, data):
        reported = self.samples.get(key, {}).get("reported", False)
        self.samples[key] = {"last": time.time(), "data": data, "reported": reported}
        self.states.pop(key, None)
        if not reported:
            self.samples[key]["reported"] = True
            emit("sample", sensor=key, data=data)

    def record_state(self, key, state, data):
        self.states[key] = {"last": time.time(), "state": state, "data": data}

    def on_imu(self, msg):
        self.record_once(
            "imu",
            {
                "linear_acceleration": [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ],
                "angular_velocity": [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ],
            },
        )

    def on_mag(self, msg):
        self.record_once(
            "magnetometer",
            {
                "magnetic_field": [
                    msg.magnetic_field.x,
                    msg.magnetic_field.y,
                    msg.magnetic_field.z,
                ]
            },
        )

    def on_joint(self, msg):
        self.record_once(
            "joint_states",
            {"names": list(msg.name), "position": list(msg.position)},
        )

    def on_vel(self, msg):
        self.record_once(
            "velocity_feedback",
            {
                "linear": [msg.linear.x, msg.linear.y, msg.linear.z],
                "angular": [msg.angular.x, msg.angular.y, msg.angular.z],
            },
        )

    def on_voltage(self, msg):
        self.record_once("voltage", {"volts": msg.data})

    def on_edition(self, msg):
        self.record_once("base_firmware", {"version": msg.data})

    def on_scan(self, msg):
        ranges = finite(msg.ranges)
        data = {
            "frame_id": msg.header.frame_id,
            "ranges": len(msg.ranges),
            "finite_ranges": len(ranges),
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
        }
        if ranges:
            data["min_m"] = round(min(ranges), 3)
            data["max_m"] = round(max(ranges), 3)
        self.record_once("lidar", data)

    def publish_status(self):
        status = {
            key: {
                "state": "sample",
                "age_sec": round(time.time() - sample["last"], 2),
                "data": sample["data"],
            }
            for key, sample in sorted(self.samples.items())
        }
        for key, state in sorted(self.states.items()):
            if key in status:
                continue
            status[key] = {
                "state": state["state"],
                "age_sec": round(time.time() - state["last"], 2),
                "data": state["data"],
            }
        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(msg)
        emit("status", sensors=sorted(status.keys()))

    def camera_loop(self):
        while rclpy.ok():
            for device, (topic, publisher) in self.camera_publishers.items():
                self.capture_camera(device, topic, publisher)
            time.sleep(2.0)

    def capture_camera(self, device, topic, publisher):
        if not os.path.exists(device):
            return
        out_path = Path(tempfile.gettempdir()) / f"sensor_probe_{Path(device).name}.mjpg"
        commands = [
            [
                "v4l2-ctl",
                "--device",
                device,
                "--set-fmt-video=width=640,height=480,pixelformat=MJPG",
                "--stream-mmap=3",
                "--stream-count=1",
                f"--stream-to={out_path}",
            ],
            [
                "v4l2-ctl",
                "--device",
                device,
                "--stream-mmap=3",
                "--stream-count=1",
                f"--stream-to={out_path}",
            ],
        ]
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                emit("camera_error", device=device, error=str(exc))
                continue
            if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                continue
            data = out_path.read_bytes()
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = Path(device).name
            msg.format = "jpeg" if data.startswith(b"\xff\xd8") else "raw"
            msg.data = data
            publisher.publish(msg)
            self.record_once(
                f"camera:{device}",
                {"bytes": len(data), "format": msg.format, "topic": topic},
            )
            return
        emit("camera_no_frame", device=device)

    def audio_loop(self):
        while rclpy.ok():
            self.capture_audio()
            time.sleep(10.0)

    def capture_audio(self):
        out_path = Path(tempfile.gettempdir()) / "sensor_probe_audio.wav"
        devices = ["default", "plughw:2,0", "hw:2,0", "plughw:0,0", "plughw:1,0"]
        failures = []
        for device in devices:
            if out_path.exists():
                out_path.unlink()
            command = [
                "arecord",
                "-q",
                "-D",
                device,
                "-d",
                "1",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                str(out_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=4,
                )
            except Exception as exc:
                failures.append({"device": device, "error": str(exc)})
                continue
            if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44:
                self.record_once(
                    "audio",
                    {"bytes": out_path.stat().st_size, "format": "wav", "device": device},
                )
                return
            failures.append({"device": device, "stderr": result.stderr.strip()[-180:]})
        emit("audio_no_sample", failures=failures)

    def lidar_probe(self):
        candidates = [
            "/dev/ttyUSB0",
            "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        ]
        bauds = [460800, 256000, 230400, 115200]
        while rclpy.ok():
            self.lidar_failures = []
            for device in candidates:
                if not os.path.exists(device):
                    continue
                for baud in bauds:
                    result = self.try_rplidar(device, baud)
                    if result:
                        self.record_once("lidar", result)
                        return
            data = {"tried": candidates, "bauds": bauds, "failures": self.lidar_failures[-8:]}
            self.record_state("lidar", "no_sample", data)
            emit("lidar_no_sample", **data)
            time.sleep(10.0)

    def try_rplidar(self, device, baud):
        try:
            with serial.Serial(device, baudrate=baud, timeout=0.2) as ser:
                ser.reset_input_buffer()
                ser.write(b"\xA5\x50")
                ser.flush()
                descriptor = ser.read(7)
                if len(descriptor) != 7 or descriptor[:2] != b"\xA5\x5A":
                    return None
                info = ser.read(20)
                if len(info) != 20:
                    return None
                ser.write(b"\xA5\x20")
                ser.flush()
                scan_descriptor = ser.read(7)
                raw = ser.read(1500)
                ser.write(b"\xA5\x25")
                distances = self.parse_legacy_scan(raw)
                payload = {
                    "device": device,
                    "baud": baud,
                    "model": info[0],
                    "firmware": f"{info[2]}.{info[1]}",
                    "hardware": info[3],
                    "serial_hex": info[4:].hex(),
                    "raw_scan_bytes": len(raw),
                    "scan_descriptor_hex": scan_descriptor.hex(),
                    "valid_ranges": len(distances),
                }
                if distances:
                    payload["min_m"] = round(min(distances), 3)
                    payload["max_m"] = round(max(distances), 3)
                return payload
        except Exception as exc:
            failure = {"device": device, "baud": baud, "error": str(exc)}
            self.lidar_failures.append(failure)
            self.record_state("lidar", "error", failure)
            emit("lidar_error", **failure)
            return None

    def parse_legacy_scan(self, raw):
        distances = []
        for idx in range(0, len(raw) - 4, 5):
            b0, b1, b2, b3, b4 = raw[idx : idx + 5]
            start = b0 & 0x01
            inverse_start = (b0 >> 1) & 0x01
            check = b1 & 0x01
            if start == inverse_start or check != 1:
                continue
            distance_m = ((b3 | (b4 << 8)) / 4.0) / 1000.0
            if distance_m > 0:
                distances.append(distance_m)
        return distances

    def base_serial_loop(self):
        candidates = [
            "/dev/ttyUSB1",
            "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
            "/dev/ttyUSB2",
        ]
        while rclpy.ok():
            self.base_failures = []
            for device in candidates:
                if not os.path.exists(device):
                    continue
                try:
                    self.record_state("base_serial", "opening", {"device": device})
                    with serial.Serial(device, baudrate=115200, timeout=1.0) as ser:
                        ser.reset_input_buffer()
                        self.enable_base_auto_report(ser)
                        self.record_once("base_serial", {"device": device, "baud": 115200})
                        self.read_base_frames(ser)
                except Exception as exc:
                    failure = {"device": device, "error": str(exc)}
                    self.base_failures.append(failure)
                    self.record_state("base_serial", "error", failure)
                    emit("base_serial_error", **failure)
            self.record_state("base_serial", "no_port", {"tried": candidates, "failures": self.base_failures[-8:]})
            time.sleep(5.0)

    def enable_base_auto_report(self, ser):
        cmd = [0xFF, 0xFC, 0x05, 0x01, 0x01, 0x00]
        cmd.append((sum(cmd) + 0x05) & 0xFF)
        ser.write(bytes(cmd))
        ser.flush()
        emit("base_auto_report_enabled")

    def read_base_frames(self, ser):
        while rclpy.ok():
            first = ser.read(1)
            if not first:
                continue
            if first[0] != 0xFF:
                continue
            second = ser.read(1)
            if not second or second[0] != 0xFB:
                continue
            header = ser.read(2)
            if len(header) != 2:
                continue
            ext_len, ext_type = header
            data_len = ext_len - 2
            if data_len <= 0 or data_len > 64:
                continue
            ext_data = ser.read(data_len)
            if len(ext_data) != data_len:
                continue
            payload = ext_data[:-1]
            checksum = ext_data[-1]
            if ((ext_len + ext_type + sum(payload)) & 0xFF) != checksum:
                continue
            self.parse_base_frame(ext_type, payload)

    def parse_base_frame(self, ext_type, data):
        if ext_type == 0x0A and len(data) >= 7:
            vx = struct.unpack("h", data[0:2])[0] / 1000.0
            vy = struct.unpack("h", data[2:4])[0] / 1000.0
            vz = struct.unpack("h", data[4:6])[0] / 1000.0
            volts = data[6] / 10.0
            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.angular.z = vz
            self.vel_publisher.publish(twist)
            voltage = Float32()
            voltage.data = volts
            self.voltage_publisher.publish(voltage)
            self.record_once("velocity_feedback", {"linear": [vx, vy, 0.0], "angular": [0.0, 0.0, vz]})
            self.record_once("voltage", {"volts": volts})
        elif ext_type in (0x0B, 0x0E) and len(data) >= 18:
            if ext_type == 0x0B:
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
            now = self.get_clock().now().to_msg()
            imu = Imu()
            imu.header.stamp = now
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = ax
            imu.linear_acceleration.y = ay
            imu.linear_acceleration.z = az
            imu.angular_velocity.x = gx
            imu.angular_velocity.y = gy
            imu.angular_velocity.z = gz
            self.imu_publisher.publish(imu)
            mag = MagneticField()
            mag.header.stamp = now
            mag.header.frame_id = "imu_link"
            mag.magnetic_field.x = mx
            mag.magnetic_field.y = my
            mag.magnetic_field.z = mz
            self.mag_publisher.publish(mag)
            self.record_once("imu", {"linear_acceleration": [ax, ay, az], "angular_velocity": [gx, gy, gz]})
            self.record_once("magnetometer", {"magnetic_field": [mx, my, mz]})
        elif ext_type == 0x0D and len(data) >= 16:
            encoders = [struct.unpack("i", data[idx : idx + 4])[0] for idx in range(0, 16, 4)]
            joints = JointState()
            joints.header.stamp = self.get_clock().now().to_msg()
            joints.header.frame_id = "joint_states"
            joints.name = ["encoder_m1", "encoder_m2", "encoder_m3", "encoder_m4"]
            joints.position = [float(value) for value in encoders]
            self.joint_publisher.publish(joints)
            self.record_once("joint_states", {"names": joints.name, "position": joints.position})


def main():
    rclpy.init()
    node = SensorProbe()
    emit("started", note="sensor-only; no command topics are published")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
