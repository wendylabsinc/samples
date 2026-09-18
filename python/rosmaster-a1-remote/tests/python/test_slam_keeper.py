"""Tests for rosmaster-a1-slam-wendy/app/slam_keeper.py.

Same stub arrangement as test_odometry.py: tests/stubs stands in for rclpy,
the message packages and slam_toolbox's services, so the module imports with
no ROS installed. SessionStore is exercised on a temporary directory,
KeeperState with an injected clock; the node tests feed SimpleNamespace
messages to the callbacks and read the stub publishers.

Run: .venv/bin/python -m unittest tests.python.test_slam_keeper
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-slam-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import slam_keeper  # noqa: E402  (import must follow the sys.path setup above)


class ImportTests(unittest.TestCase):
    def test_the_module_imports_against_the_stubs(self):
        self.assertTrue(hasattr(slam_keeper, "SlamKeeper"))
        self.assertEqual(slam_keeper.ODOM_RESET_EXIT_STATUS, 75)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "maps"
        self.store = slam_keeper.SessionStore(self.root, keep=3)
        self.t0 = 1_800_000_000.0  # some wall-clock second

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_creates_a_named_directory_session_json_and_latest(self):
        session = self.store.start(self.t0)
        self.assertEqual(session.name, slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0)))
        self.assertTrue(session.dir.is_dir())
        meta = json.loads((session.dir / "session.json").read_text())
        self.assertEqual(meta["name"], session.name)
        self.assertEqual(meta["started_at"], self.t0)
        self.assertEqual(meta["saves"], 0)
        self.assertEqual(os.readlink(self.root / "latest"), session.name)
        self.assertEqual(self.store.latest_dir(), session.dir)

    def test_rotation_keeps_the_newest_sessions(self):
        for k in range(5):
            self.store.start(self.t0 + 60 * k)
        self.assertEqual(len(self.store.sessions()), 3)
        self.assertEqual(self.store.sessions()[-1], self.store.latest_dir().name)
        self.assertFalse((self.root / slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0))).exists())

    def test_a_second_start_in_the_same_second_gets_a_distinct_name(self):
        a = self.store.start(self.t0)
        b = self.store.start(self.t0)
        self.assertNotEqual(a.name, b.name)
        self.assertTrue(b.dir.is_dir())

    def test_attach_when_latest_is_younger_than_the_slam_node(self):
        old = self.store.start(self.t0)
        attached = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 - 1)
        self.assertEqual(attached.name, old.name)
        self.assertEqual(len(self.store.sessions()), 1)

    def test_start_fresh_when_the_slam_node_is_younger_than_latest(self):
        old = self.store.start(self.t0)
        fresh = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 + 2)
        self.assertNotEqual(fresh.name, old.name)
        self.assertEqual(len(self.store.sessions()), 2)

    def test_attach_falls_back_when_latest_session_json_is_malformed(self):
        self.store.start(self.t0)
        for content in ("null", "[1, 2]", '{"started_at": null}', "{not json"):
            latest = self.store.latest_dir()                   # the session attach_or_start will read
            (latest / "session.json").write_text(content)
            fresh = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 - 1)
            self.assertNotEqual(fresh.dir, latest, content)    # never attaches to the malformed one
            self.assertTrue((fresh.dir / "session.json").is_file())

    def test_start_fresh_when_nothing_is_known(self):
        self.assertIsNone(self.store.latest_dir())
        fresh = self.store.attach_or_start(self.t0, node_started_at=None)
        self.assertTrue(fresh.dir.is_dir())

    def test_staging_and_commit_move_files_into_place(self):
        session = self.store.start(self.t0)
        base = self.store.staging_base(session)
        self.assertTrue(base.endswith("/.saving/map"))
        for ext in ("posegraph", "data"):
            Path(f"{base}.{ext}").write_text(ext)
        moved = self.store.commit_save(session, "graph")
        self.assertEqual([p.name for p in moved], ["map.posegraph", "map.data"])
        self.assertEqual((session.dir / "map.data").read_text(), "data")
        self.assertFalse(Path(f"{base}.posegraph").exists())
        for ext in ("pgm", "yaml"):
            Path(f"{base}.{ext}").write_text(ext)
        self.assertEqual([p.name for p in self.store.commit_save(session, "grid")], ["map.pgm", "map.yaml"])

    def test_commit_reports_missing_outputs_without_raising(self):
        session = self.store.start(self.t0)
        self.store.staging_base(session)
        self.assertEqual(self.store.commit_save(session, "grid"), [])

    def test_session_json_updates_are_merged_and_atomic(self):
        session = self.store.start(self.t0)
        meta = self.store.update_session_json(session, saves=3, last_pose={"x": 1.0, "y": 2.0, "yaw": 0.5})
        self.assertEqual(meta["saves"], 3)
        self.assertEqual(self.store.read_session_json(session)["last_pose"]["y"], 2.0)
        self.assertEqual(self.store.read_session_json(session)["started_at"], self.t0)
        self.assertEqual(sorted(p.name for p in session.dir.iterdir()), ["session.json"])


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t


def keeper_state(**overrides):
    clock = FakeClock()
    cfg = slam_keeper.KeeperConfig(**overrides)
    return slam_keeper.KeeperState(cfg, clock=clock), clock


class KeeperConfigTests(unittest.TestCase):
    def test_defaults_match_the_spec(self):
        cfg = slam_keeper.KeeperConfig()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/maps", 30.0, 5, ""))
        self.assertEqual((cfg.trajectory_min_step_m, cfg.trajectory_max_poses), (0.05, 5000))
        self.assertEqual((cfg.odom_jump_m, cfg.odom_jump_rad, cfg.down_s, cfg.save_timeout_s), (1.0, 1.0, 10.0, 20.0))

    def test_from_env_reads_knobs_and_ignores_garbage(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {"SLAM_MAPS_DIR": "/tmp/m", "SLAM_AUTOSAVE_S": "5", "SLAM_KEEP_SESSIONS": "2", "SLAM_MAP_FILE": "/maps/x/map", "SLAM_ODOM_JUMP_M": "abc", "SLAM_DOWN_S": ""}, clear=True):
            cfg = slam_keeper.KeeperConfig.from_env()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/tmp/m", 5.0, 2, "/maps/x/map"))
        self.assertEqual((cfg.odom_jump_m, cfg.down_s), (1.0, 10.0))


class KeeperStateTests(unittest.TestCase):
    def test_state_progresses_from_waiting_to_mapping(self):
        state, clock = keeper_state()
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")
        state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")

    def test_scan_and_odom_go_stale_after_two_seconds(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.0, 0.0, 0.0)
        clock.t += 2.5
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")

    def test_slam_down_when_the_map_odom_transform_stops(self):
        state, clock = keeper_state(down_s=10.0)
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, 0.0, 0.0)
        clock.t += 9.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_slam_down_when_no_transform_ever_arrives_within_a_minute(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        clock.t += 59.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_trajectory_decimates_and_caps(self):
        state, clock = keeper_state(trajectory_min_step_m=0.05, trajectory_max_poses=3)
        self.assertTrue(state.on_pose(0.0, 0.0, 0.0, "s0"))
        self.assertFalse(state.on_pose(0.02, 0.0, 0.1, "s1"))     # moved 2 cm: not a new trajectory point
        self.assertTrue(state.on_pose(0.06, 0.0, 0.1, "s2"))
        self.assertTrue(state.on_pose(0.12, 0.0, 0.1, "s3"))
        self.assertTrue(state.on_pose(0.18, 0.0, 0.1, "s4"))
        self.assertEqual([p[3] for p in state.trajectory], ["s2", "s3", "s4"])  # capped at 3, oldest dropped
        self.assertEqual(state.poses_since_save, 5)

    def test_autosave_waits_for_the_interval_and_a_new_pose(self):
        state, clock = keeper_state(autosave_s=30.0)
        self.assertFalse(state.wants_save(), "nothing mapped yet")
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertTrue(state.wants_save(), "first save as soon as there is a pose")
        state.save_started()
        self.assertFalse(state.wants_save(), "one in flight")
        state.save_finished(True, "/maps/x/map.pgm")
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (1, 0, 0))
        clock.t += 31.0
        self.assertFalse(state.wants_save(), "no new pose since the last save")
        state.on_pose(1.0, 0.0, 0.0, "s1")
        self.assertTrue(state.wants_save())

    def test_autosave_can_be_disabled(self):
        state, clock = keeper_state(autosave_s=0.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertFalse(state.wants_save())

    def test_a_save_that_never_completes_expires_as_an_error(self):
        state, clock = keeper_state(save_timeout_s=20.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started()
        clock.t += 19.0
        self.assertFalse(state.expire_save())
        clock.t += 2.0
        self.assertTrue(state.expire_save())
        self.assertEqual((state.saves, state.save_errors), (0, 1))
        self.assertTrue(state.wants_save(), "the pose is still unsaved and nothing is in flight")

    def test_a_failed_save_is_counted_and_the_pose_stays_unsaved(self):
        state, clock = keeper_state()
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started(); state.save_finished(False, None)
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (0, 1, 1))

    def test_note_save_unavailable_reports_a_failure_without_counting_it(self):
        state, clock = keeper_state()
        state.note_save_unavailable("maps volume is read-only")
        status = state.status(None)
        self.assertEqual(status["last_save"], {"age_s": 0.0, "ok": False, "path": None, "reason": "maps volume is read-only"})
        self.assertEqual((state.saves, state.save_errors), (0, 0), "no save was attempted, so nothing is counted")

    def test_an_odometry_jump_is_a_reset(self):
        state, clock = keeper_state(odom_jump_m=1.0, odom_jump_rad=1.0)
        self.assertFalse(state.on_odom(0.0, 0.0, 0.0))
        self.assertFalse(state.on_odom(0.5, 0.0, 0.2))
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2), "2.5 m between consecutive messages")
        self.assertEqual(state.odom_resets, 1)
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2 + 2.0), "2 rad between consecutive messages")
        self.assertEqual(state.odom_resets, 2)

    def test_status_has_every_key_and_reflects_the_inputs(self):
        state, clock = keeper_state()
        empty = state.status(None)
        self.assertEqual(sorted(empty), ["last_save", "map", "map_odom", "map_odom_age_s", "odom_resets", "odom_tf_age_s", "pose", "save_errors", "saves", "scan_age_s", "session", "state", "trajectory_poses"])
        self.assertEqual(empty["state"], "waiting_for_scan")
        self.assertIsNone(empty["scan_age_s"]); self.assertIsNone(empty["map"]); self.assertIsNone(empty["session"])
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, -0.2, 0.05)
        state.on_map(200, 100, 0.05, 30, 500, 19470); state.on_pose(1.0, 2.0, 0.5, "s0")
        state.save_started(); state.save_finished(True, "/maps/s/map.pgm")
        clock.t += 1.5
        status = state.status({"name": "s", "started_at": 1.0, "dir": "/maps/s"})
        self.assertEqual(status["state"], "mapping")
        self.assertAlmostEqual(status["scan_age_s"], 1.5, places=3)
        self.assertEqual(status["map"]["width"], 200); self.assertEqual(status["map"]["unknown"], 19470)
        self.assertAlmostEqual(status["map"]["age_s"], 1.5, places=3)
        self.assertEqual(status["pose"]["x"], 1.0); self.assertAlmostEqual(status["pose"]["age_s"], 1.5, places=3)
        self.assertEqual(status["map_odom"], {"x": 0.1, "y": -0.2, "yaw": 0.05})
        self.assertEqual(status["last_save"]["ok"], True); self.assertEqual(status["last_save"]["path"], "/maps/s/map.pgm")
        self.assertEqual((status["saves"], status["save_errors"], status["trajectory_poses"], status["odom_resets"]), (1, 0, 1, 0))
        self.assertEqual(status["session"]["name"], "s")
        json.dumps(status, sort_keys=True)  # must be JSON-serialisable as is


class FakeFuture:
    """Stands in for an rclpy Future: records the done-callback and lets a
    test resolve it independently of the other request's future, either
    with a result or by raising, so it can drive the two calls out of
    order and check each combination RosMapSaver has to handle."""

    def __init__(self):
        self._callback = None
        self._result = None
        self._exc = None

    def add_done_callback(self, callback):
        self._callback = callback

    def resolve(self, *, result=None, exc=None):
        self._result, self._exc = result, exc
        self._callback(self)

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class FakeServiceClient:
    def __init__(self):
        self.futures: list = []

    def call_async(self, request):
        future = FakeFuture()
        self.futures.append(future)
        return future


class FakeSaverNode:
    """No rclpy.node.Node at all: just enough of `create_client` for
    RosMapSaver to keep the two clients it calls by name."""

    def __init__(self):
        self.clients: dict = {}

    def create_client(self, _srv_type, name):
        client = FakeServiceClient()
        self.clients[name] = client
        return client


def save_result(code):
    return types.SimpleNamespace(result=code)


class RosMapSaverTests(unittest.TestCase):
    def setUp(self):
        self.node = FakeSaverNode()
        self.saver = slam_keeper.RosMapSaver(self.node)
        self.calls: list = []
        self.saver.save("/tmp/session/.saving/map", self.calls.append)

    def _futures(self):
        graph = self.node.clients["/slam_toolbox/serialize_map"].futures[-1]
        grid = self.node.clients["/slam_toolbox/save_map"].futures[-1]
        return graph, grid

    def test_both_services_ok_calls_done_true_once(self):
        graph, grid = self._futures()
        graph.resolve(result=save_result(0))
        self.assertEqual(self.calls, [], "only one of the two services has answered so far")
        grid.resolve(result=save_result(0))
        self.assertEqual(self.calls, [True])

    def test_a_failing_result_code_calls_done_false(self):
        graph, grid = self._futures()
        graph.resolve(result=save_result(0))
        grid.resolve(result=save_result(1))    # a non-zero result code is a failure
        self.assertEqual(self.calls, [False])

    def test_a_raising_future_calls_done_false_instead_of_raising(self):
        graph, grid = self._futures()
        graph.resolve(exc=RuntimeError("service unavailable"))
        grid.resolve(result=save_result(0))
        self.assertEqual(self.calls, [False])

    def test_done_fires_exactly_once_per_attempt(self):
        graph, grid = self._futures()
        graph.resolve(result=save_result(0))
        grid.resolve(result=save_result(0))
        self.assertEqual(len(self.calls), 1)


class FakeSaver:
    def __init__(self):
        self.calls: list = []

    def save(self, base, done):
        self.calls.append((base, done))

    def complete(self, index, ok, files=("posegraph", "data", "pgm", "yaml")):
        base, done = self.calls[index]
        if ok:
            for ext in files:
                Path(f"{base}.{ext}").write_text(ext)
        done(ok)


def stamp(sec=1, nanosec=0):
    return types.SimpleNamespace(sec=sec, nanosec=nanosec)


def scan_msg():
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="laser_frame"), ranges=[1.0] * 400)


def odom_msg(x, y, yaw):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="odom"), pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=q)))


def tf_msg(pairs):
    transforms = []
    for parent, child, x, y, yaw in pairs:
        q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
        transforms.append(types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id=parent), child_frame_id=child, transform=types.SimpleNamespace(translation=types.SimpleNamespace(x=x, y=y, z=0.0), rotation=q)))
    return types.SimpleNamespace(transforms=transforms)


def map_msg(width, height, data):
    info = types.SimpleNamespace(resolution=0.05, width=width, height=height, origin=None)
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="map"), info=info, data=data)


def pose_msg(x, y, yaw, sec=1):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(sec), frame_id="map"), pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=q)))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.cfg = slam_keeper.KeeperConfig(maps_dir=self.tmp.name, autosave_s=30.0, keep_sessions=2, trajectory_min_step_m=0.05)
        self.store = slam_keeper.SessionStore(Path(self.tmp.name), keep=2)
        self.saver = FakeSaver()
        self.node = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=self.saver, clock=self.clock, wall_clock=lambda: 1_800_000_000.0)

    def tearDown(self):
        self.tmp.cleanup()

    def last_status(self):
        return json.loads(self.node.status_pub.messages[-1].data)

    def test_construction_starts_a_session_and_the_first_tick_publishes_status(self):
        self.assertTrue(self.node.session.dir.is_dir())
        self.node.tick()
        status = self.last_status()
        self.assertEqual(status["state"], "waiting_for_scan")
        self.assertEqual(status["session"]["name"], self.node.session.name)
        self.assertEqual(status["session"]["dir"], str(self.node.session.dir))

    def test_inputs_drive_the_state_and_the_map_counts(self):
        self.node.on_scan(scan_msg())
        self.node.on_odom(odom_msg(0.0, 0.0, 0.0))
        self.node.on_tf(tf_msg([("odom", "base_link", 0.0, 0.0, 0.0), ("map", "odom", 0.5, -0.25, 0.1)]))
        self.node.on_map(map_msg(3, 2, [-1, -1, 0, 0, 100, 100]))
        self.node.tick()
        status = self.last_status()
        self.assertEqual(status["state"], "mapping")
        self.assertEqual(status["map"], {"width": 3, "height": 2, "resolution": 0.05, "occupied": 2, "free": 2, "unknown": 2, "age_s": 0.0})
        self.assertEqual(status["map_odom"]["x"], 0.5)
        self.assertAlmostEqual(status["map_odom"]["yaw"], 0.1, places=9)

    def test_poses_build_a_latched_path_in_the_map_frame(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0, sec=1))
        self.node.on_pose(pose_msg(0.02, 0.0, 0.0, sec=2))   # too close: no new path point, no new message
        self.node.on_pose(pose_msg(0.5, 0.0, 0.3, sec=3))
        self.assertEqual(len(self.node.trajectory_pub.messages), 2)
        path = self.node.trajectory_pub.messages[-1]
        self.assertEqual(path.header.frame_id, "map")
        self.assertEqual(path.header.stamp.sec, 3)
        self.assertEqual([p.pose.position.x for p in path.poses], [0.0, 0.5])
        self.assertAlmostEqual(path.poses[-1].pose.orientation.z, math.sin(0.15), places=9)
        self.assertEqual(path.poses[-1].header.stamp.sec, 3)
        qos = self.node.trajectory_pub.args[2]
        self.assertEqual(qos.durability, slam_keeper.DurabilityPolicy.TRANSIENT_LOCAL)

    def test_autosave_stages_commits_and_records(self):
        self.node.on_scan(scan_msg())
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 1)
        self.assertEqual(self.saver.calls[0][0], str(self.node.session.dir / ".saving" / "map"))
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 1, "one save in flight at a time")
        self.saver.complete(0, ok=True)
        self.assertTrue((self.node.session.dir / "map.pgm").is_file())
        self.assertTrue((self.node.session.dir / "map.posegraph").is_file())
        meta = self.store.read_session_json(self.node.session)
        self.assertEqual(meta["saves"], 1)
        self.assertEqual(meta["scans"], 1)
        self.assertEqual(meta["last_pose"], {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.node.tick()
        self.assertEqual(self.last_status()["last_save"]["ok"], True)
        self.assertEqual(self.last_status()["last_save"]["path"], str(self.node.session.dir / "map.pgm"))

    def test_a_failed_save_is_counted_and_retried_next_interval(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.saver.complete(0, ok=False)
        self.node.tick()
        self.assertEqual(self.last_status()["save_errors"], 1)
        self.assertEqual(len(self.saver.calls), 1)
        self.clock.t += 31.0
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 2)

    def test_an_odometry_jump_requests_the_exit_status_and_flushes_session_json(self):
        self.node.on_odom(odom_msg(0.0, 0.0, 0.0))
        self.assertIsNone(self.node.exit_status)
        self.node.on_odom(odom_msg(5.0, 0.0, 0.0))
        self.assertEqual(self.node.exit_status, slam_keeper.ODOM_RESET_EXIT_STATUS)
        self.assertEqual(self.store.read_session_json(self.node.session)["odom_resets"], 1)

    def test_a_restarted_keeper_attaches_to_a_young_session(self):
        first = self.node.session
        second = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_100.0, node_started_at=1_799_999_999.0)
        self.assertEqual(second.session.name, first.name)
        third = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_200.0, node_started_at=1_800_000_150.0)
        self.assertNotEqual(third.session.name, first.name)

    def test_a_save_that_answers_after_its_timeout_is_ignored(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.clock.t += 21.0
        self.node.tick()                       # expired: counted as an error
        self.saver.complete(0, ok=True)        # the late answer
        self.node.tick()
        status = self.last_status()
        self.assertEqual((status["saves"], status["save_errors"]), (0, 1))
        self.assertFalse((self.node.session.dir / "map.pgm").exists(), "nothing is committed from a late answer")

    def test_a_late_answer_from_an_expired_attempt_is_ignored_after_a_retry(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()                        # starts the first save attempt
        self.clock.t += 21.0
        self.node.tick()                        # expires: counted as an error, no retry this tick
        self.node.tick()                        # retries: a second attempt starts
        self.assertEqual(len(self.saver.calls), 2)
        self.saver.complete(0, ok=True)         # the first attempt's late answer arrives while the second is in flight
        self.node.tick()
        status = self.last_status()
        self.assertEqual((status["saves"], status["save_errors"]), (0, 1), "a stale answer must not be credited to the new attempt")
        self.assertFalse((self.node.session.dir / "map.pgm").exists())
        self.saver.complete(1, ok=True)         # the real answer for the attempt actually in flight
        self.node.tick()
        status = self.last_status()
        self.assertEqual((status["saves"], status["save_errors"]), (1, 1))
        self.assertTrue((self.node.session.dir / "map.pgm").is_file())

    def test_a_partial_save_commits_nothing_and_clears_the_staged_leftovers(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.saver.complete(0, ok=True, files=("pgm", "yaml"))   # only the grid half completed
        self.node.tick()
        status = self.last_status()
        self.assertEqual((status["saves"], status["save_errors"]), (0, 1))
        self.assertFalse((self.node.session.dir / "map.pgm").exists(), "nothing is committed from a partial save")
        self.assertFalse((self.node.session.dir / "map.yaml").exists())
        staging = self.node.session.dir / ".saving"
        self.assertEqual(list(staging.iterdir()), [], "the partial staged files must not leak into the next attempt")

    def test_an_unwritable_volume_keeps_status_and_trajectory_alive(self):
        blocker = Path(self.tmp.name) / "blocked"
        blocker.write_text("not a directory")
        store = slam_keeper.SessionStore(blocker / "maps", keep=2)   # mkdir under a file: OSError
        node = slam_keeper.SlamKeeper(cfg=self.cfg, store=store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_000.0)
        self.assertIsNone(node.session)
        node.on_pose(pose_msg(0.0, 0.0, 0.0))
        node.tick()
        status = json.loads(node.status_pub.messages[-1].data)
        self.assertIsNone(status["session"])
        self.assertEqual(status["last_save"]["ok"], False)
        self.assertEqual(status["save_errors"], 0, "an unwritable volume at startup is not a save attempt")
        self.assertEqual(status["trajectory_poses"], 1)
        self.assertEqual(len(node.trajectory_pub.messages), 1)

    def test_subscriptions_and_publishers_use_the_spec_topics(self):
        topics = sorted(sub.args[1] for sub in self.node.subs)
        self.assertEqual(topics, ["/map", "/odom", "/pose", "/scan", "/tf"])
        self.assertEqual(self.node.status_pub.args[1], "/slam/status")
        self.assertEqual(self.node.trajectory_pub.args[1], "/slam/trajectory")


class MainTests(unittest.TestCase):
    def test_main_does_not_crash_when_the_maps_volume_is_unwritable_at_startup(self):
        from unittest import mock

        tmp = tempfile.TemporaryDirectory()
        try:
            blocker = Path(tmp.name) / "blocked"
            blocker.write_text("not a directory")
            env = {"SLAM_MAPS_DIR": str(blocker / "maps"), "SLAM_KEEP_SESSIONS": "2"}
            # rclpy.ok() patched False so the spin loop never runs: exit_status
            # would otherwise never become non-None and the loop would hang.
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(slam_keeper.rclpy, "ok", return_value=False):
                exit_status = slam_keeper.main()
            self.assertEqual(exit_status, 0)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
