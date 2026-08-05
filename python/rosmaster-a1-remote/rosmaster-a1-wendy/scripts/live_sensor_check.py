#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


REQUIRED_SENSORS = {
    "audio",
    "camera",
    "lidar",
    "imu",
    "magnetometer",
    "joint_states",
    "velocity_feedback",
    "voltage",
}

COMMAND_TOPICS = {"/cmd_vel", "/Servo", "/Buzzer"}


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class Evidence:
    samples: dict[str, Any] = field(default_factory=dict)
    status_seen: set[str] = field(default_factory=set)
    verifier_summary: dict[str, Any] | None = None
    parse_errors: list[str] = field(default_factory=list)

    def add_sample(self, sensor: str, data: Any) -> None:
        normalized = normalize_sensor(sensor)
        if normalized in REQUIRED_SENSORS and normalized not in self.samples:
            self.samples[normalized] = data

    def add_status_sensor(self, sensor: str) -> None:
        normalized = normalize_sensor(sensor)
        if normalized in REQUIRED_SENSORS:
            self.status_seen.add(normalized)

    @property
    def present(self) -> set[str]:
        present = set(self.samples) | set(self.status_seen)
        if self.verifier_summary:
            present.update(self.verifier_summary.get("samples", {}).keys())
        return {normalize_sensor(sensor) for sensor in present}


def normalize_sensor(sensor: str) -> str:
    if sensor.startswith("camera:"):
        return "camera"
    if sensor in {"base_probe_status", "lidar_probe_status", "base_firmware"}:
        return sensor
    return sensor


def run(command: list[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command,
            None,
            exc.stdout or "",
            exc.stderr or "",
            timed_out=True,
        )
    except FileNotFoundError as exc:
        # macOS-only tools (networksetup, ipconfig) are unconditionally invoked by
        # ethernet_devices()/local_network_state() below; on Linux they don't exist,
        # and without this the missing binary would traceback the whole script
        # instead of surfacing as a normal failed-command result.
        return CommandResult(command, None, "", str(exc))


def stream_for(command: list[str], seconds: float) -> CommandResult:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + seconds
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    try:
        while time.time() < deadline and process.poll() is None:
            time.sleep(0.2)
        if process.poll() is None:
            process.terminate()
            try:
                out, err = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                out, err = process.communicate(timeout=3)
        else:
            out, err = process.communicate(timeout=3)
        stdout_chunks.append(out or "")
        stderr_chunks.append(err or "")
    finally:
        if process.poll() is None:
            process.kill()
    return CommandResult(command, process.returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def parse_json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_message(line: str) -> str:
    parsed = parse_json_or_none(line)
    if isinstance(parsed, dict):
        for key in ("message", "msg", "log", "body", "line"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return line


def parse_probe_payload(message: str) -> dict[str, Any] | None:
    marker = "SENSOR_PROBE "
    if marker not in message:
        return None
    payload = message.split(marker, 1)[1].strip()
    return parse_json_or_none(payload)


def parse_verify_sample(message: str) -> dict[str, Any] | None:
    marker = "VERIFY_SAMPLE "
    if marker not in message:
        return None
    payload = message.split(marker, 1)[1].strip()
    return parse_json_or_none(payload)


def parse_verify_summary(message: str) -> dict[str, Any] | None:
    marker = "VERIFY_SUMMARY "
    if marker not in message:
        return None
    payload = message.split(marker, 1)[1].strip()
    return parse_json_or_none(payload)


def parse_evidence(text: str) -> Evidence:
    evidence = Evidence()
    for line in text.splitlines():
        message = extract_message(line)

        probe = parse_probe_payload(message)
        if isinstance(probe, dict):
            event = probe.get("event")
            if event == "sample" and isinstance(probe.get("sensor"), str):
                evidence.add_sample(probe["sensor"], probe.get("data", {}))
            elif event == "status":
                for sensor in probe.get("sensors", []):
                    if isinstance(sensor, str):
                        evidence.add_status_sensor(sensor)
            continue

        sample = parse_verify_sample(message)
        if isinstance(sample, dict) and isinstance(sample.get("sensor"), str):
            evidence.add_sample(sample["sensor"], sample.get("data", {}))
            continue

        summary = parse_verify_summary(message)
        if isinstance(summary, dict):
            evidence.verifier_summary = summary
            for sensor, data in summary.get("samples", {}).items():
                evidence.add_sample(sensor, data)
            continue
    return evidence


def topic_records(topics_payload: Any) -> list[dict[str, Any]]:
    if isinstance(topics_payload, list):
        return [item for item in topics_payload if isinstance(item, dict)]
    if isinstance(topics_payload, dict):
        for key in ("topics", "items", "data"):
            value = topics_payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def topic_name(record: dict[str, Any]) -> str | None:
    for key in ("name", "topic", "topicName"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def publisher_count(record: dict[str, Any]) -> int | None:
    for key in ("publishers", "publisherCount", "publisher_count"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return len(value)
    return None


def command_topic_publishers(topics_payload: Any) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for record in topic_records(topics_payload):
        name = topic_name(record)
        if name in COMMAND_TOPICS:
            counts[name] = publisher_count(record)
    return counts


def print_command_result(name: str, result: CommandResult) -> None:
    status = "timeout" if result.timed_out else str(result.returncode)
    print(f"{name}: exit={status}", file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip()[-500:], file=sys.stderr)


def ethernet_devices() -> list[str]:
    result = run(["networksetup", "-listallhardwareports"], timeout=5)
    devices: list[str] = []
    current_port = ""
    for line in result.stdout.splitlines():
        if line.startswith("Hardware Port: "):
            current_port = line.split(": ", 1)[1]
        elif line.startswith("Device: ") and "Ethernet" in current_port:
            devices.append(line.split(": ", 1)[1])
    return devices


def local_network_state(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    route = run(["route", "-n", "get", device], timeout=5)
    state["route"] = route.stdout.strip() or route.stderr.strip()
    interfaces: dict[str, Any] = {}
    for interface in ethernet_devices():
        ifconfig = run(["ifconfig", interface], timeout=5)
        ipaddr = run(["ipconfig", "getifaddr", interface], timeout=5)
        interfaces[interface] = {
            "ip": ipaddr.stdout.strip(),
            "ifconfig": ifconfig.stdout.strip(),
        }
    state["ethernet_interfaces"] = interfaces
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensor-only Rosmaster A1 live verification helper.")
    parser.add_argument("--device", default="192.168.2.8")
    parser.add_argument("--log-seconds", type=float, default=12.0)
    parser.add_argument("--logs-file", help="Parse an existing log file instead of streaming device logs.")
    args = parser.parse_args()

    combined_logs = ""
    command_counts: dict[str, int | None] = {}
    reachability: dict[str, Any] = {}

    if args.logs_file:
        with open(args.logs_file, "r", encoding="utf-8") as handle:
            combined_logs = handle.read()
    else:
        reachability["local_network"] = local_network_state(args.device)
        ping = run(["ping", "-c", "3", args.device], timeout=15)
        reachability["ping_exit"] = ping.returncode
        print_command_result("ping", ping)
        if ping.returncode != 0:
            print(json.dumps({"ok": False, "error": "device_unreachable", "reachability": reachability}, sort_keys=True))
            return 2

        discover = run(["wendy", "discover", "--json"], timeout=20)
        reachability["discover_exit"] = discover.returncode
        print_command_result("discover", discover)

        topics = run(["wendy", "--json", "device", "ros2", "topics", "--all", "--device", args.device], timeout=25)
        print_command_result("topics", topics)
        topics_payload = parse_json_or_none(topics.stdout)
        command_counts = command_topic_publishers(topics_payload)

        for service in ("base", "lidar"):
            logs = stream_for(
                [
                    "wendy", "--json", "device", "logs",
                    "--app", "rosmaster-a1", "--service", service,
                    "--tail", "100", "--device", args.device,
                ],
                seconds=args.log_seconds,
            )
            print_command_result(f"logs:{service}", logs)
            combined_logs += "\n" + logs.stdout

    evidence = parse_evidence(combined_logs)
    present = evidence.present
    missing = sorted(REQUIRED_SENSORS - present)
    unsafe_publishers = {
        topic: count
        for topic, count in command_counts.items()
        if count not in (0, None)
    }

    summary = {
        "ok": not missing and not unsafe_publishers,
        "present": sorted(present & REQUIRED_SENSORS),
        "missing": missing,
        "samples": evidence.samples,
        "status_seen": sorted(evidence.status_seen),
        "command_topic_publishers": command_counts,
        "unsafe_command_publishers": unsafe_publishers,
        "reachability": reachability,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
