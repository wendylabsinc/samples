"""Guards rosmaster-a1-slam-wendy/app/slam_params.yaml.

No PyYAML in the .venv and none needed: the file is a flat `key: value`
mapping under `slam_toolbox: ros__parameters:`. The stock slam_toolbox
config uses base_footprint and interactive mode; both would silently break
this car (no such frame; a Qt-less container), so the keys are pinned here.

Run: .venv/bin/python -m unittest tests.python.test_slam_params
"""
from __future__ import annotations

import unittest
from pathlib import Path

PARAMS = Path(__file__).resolve().parents[2] / "rosmaster-a1-slam-wendy" / "app" / "slam_params.yaml"


def load_flat(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.endswith(":"):
            continue
        key, _, value = stripped.partition(":")
        values[key.strip()] = value.strip()
    return values


class SlamParamsTests(unittest.TestCase):
    def setUp(self):
        self.params = load_flat(PARAMS)
        self.lines = PARAMS.read_text().splitlines()

    def test_the_file_addresses_the_slam_toolbox_node(self):
        self.assertEqual(self.lines[0].strip(), "slam_toolbox:")
        self.assertEqual(self.lines[1].strip(), "ros__parameters:")

    def test_frames_and_topic(self):
        self.assertEqual(self.params["odom_frame"], "odom")
        self.assertEqual(self.params["map_frame"], "map")
        self.assertEqual(self.params["base_frame"], "base_link")
        self.assertEqual(self.params["scan_topic"], "/scan")
        self.assertNotIn("base_footprint", PARAMS.read_text())

    def test_mode_rates_and_ranges(self):
        p = self.params
        self.assertEqual(p["mode"], "mapping")
        self.assertEqual(p["use_map_saver"], "true")
        self.assertEqual(p["transform_publish_period"], "0.05")
        self.assertEqual(p["map_update_interval"], "1.0")
        self.assertEqual(p["resolution"], "0.05")
        self.assertEqual(p["max_laser_range"], "12.0")
        self.assertEqual(p["minimum_time_interval"], "0.2")
        self.assertEqual(p["transform_timeout"], "0.2")
        self.assertEqual(p["minimum_travel_distance"], "0.2")
        self.assertEqual(p["minimum_travel_heading"], "0.2")
        self.assertEqual(p["do_loop_closing"], "true")

    def test_no_interactive_mode_and_no_sim_time_in_the_file(self):
        self.assertEqual(self.params["enable_interactive_mode"], "false")
        self.assertNotIn("use_sim_time", self.params, "sim time is an entrypoint argument (SLAM_USE_SIM_TIME), never baked in")
