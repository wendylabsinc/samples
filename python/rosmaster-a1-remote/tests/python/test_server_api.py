"""Regression tests for the Rosmaster A1 web remote HTTP API.

These tests exercise rosmaster-a1-web-remote-wendy/app/server.py off the
robot. server.py imports rclpy and the ROS message packages unconditionally
and starts a ROS node at module scope, so it cannot be imported on a Mac
without help. tests/stubs provides fake versions of those packages on
sys.path; numpy and Pillow are real installed dependencies (see
tests/README.md for the venv setup) because server.py's pure functions for
depth colorization and JPEG encoding are real code we want to exercise, not
something to fake.

The tests start a real ThreadingHTTPServer on an ephemeral port and talk to
it over HTTP with http.client, the same way a browser or the Xbox controller
web page would. server.Handler reads the module global `control` directly on
every request, so a test can substitute a scripted fake by assigning
server.control before making a request, and must restore the original
afterward.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-web-remote-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import server  # noqa: E402  (import must follow the sys.path setup above)


class ServerTestCase(unittest.TestCase):
    """Boots one real HTTP server for the whole test class."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self._orig_control = server.control
        self._reset_control()

    def tearDown(self):
        server.control = self._orig_control
        self._reset_control()

    @staticmethod
    def _reset_control():
        """Bring the shared control singleton back to a known baseline.

        server.control is a module scope singleton, so state written by one
        test class is visible to the next. DriveEndpointTests leaves
        linear_x 5000.0 and enabled True behind, and the suite passed only
        because unittest happens to run the classes in an order where the one
        class that reset anything ran after it. Reset here instead, so no test
        depends on where it sits in the alphabet.

        Camera state is reset for the same reason. Frame counts never decay by
        design, so one test feeding a stream would otherwise leave that feed
        looking fitted, and registry membership is decided on exactly that.
        """
        server.control.stop()
        server.control.publisher.messages.clear()
        server.control._realsense = server.control._empty_realsense()
        server.control._viewers.clear()

    def _connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)

    def _post_json(self, path: str, payload: dict | None):
        conn = self._connection()
        if payload is None:
            body = b""
        else:
            body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data.decode("utf-8"))

    def _post_raw(self, path: str, raw: bytes):
        conn = self._connection()
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        try:
            response = conn.getresponse()
            data = response.read()
            return response.status, data
        except http.client.RemoteDisconnected:
            # The handler has no try/except around json.loads, so a bad
            # request body kills the request thread before any response is
            # written and the connection is simply closed. This is the real,
            # if unfriendly, current behavior.
            return None, b""
        finally:
            conn.close()

    def _get(self, path: str):
        conn = self._connection()
        conn.request("GET", path)
        response = conn.getresponse()
        data = response.read()
        headers = dict(response.getheaders())
        conn.close()
        return response.status, data, headers


class DriveEndpointTests(ServerTestCase):
    def test_normal_command_is_accepted_and_reaches_control(self):
        status, body = self._post_json(
            "/api/drive",
            {"enabled": True, "linear_x": 0.4, "steering_y": 0.05, "angular_z": 0.2},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["command"]["enabled"], True)
        self.assertEqual(body["command"]["linear_x"], 0.4)
        self.assertEqual(body["command"]["steering_y"], 0.05)
        self.assertEqual(body["command"]["angular_z"], 0.2)
        # It really reached the shared control object, not just the response body.
        self.assertEqual(server.control.snapshot()["command"]["linear_x"], 0.4)

    def test_steering_y_is_clamped_to_max_steering_y(self):
        status, body = self._post_json(
            "/api/drive",
            {"enabled": True, "linear_x": 0.0, "steering_y": 5.0, "angular_z": 0.0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["steering_y"], server.MAX_STEERING_Y)

        status, body = self._post_json(
            "/api/drive",
            {"enabled": True, "linear_x": 0.0, "steering_y": -5.0, "angular_z": 0.0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["steering_y"], -server.MAX_STEERING_Y)

    def test_angular_z_is_clamped_to_max_angular_z(self):
        status, body = self._post_json(
            "/api/drive",
            {"enabled": True, "linear_x": 0.0, "steering_y": 0.0, "angular_z": 50.0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["angular_z"], server.MAX_ANGULAR_Z)

    def test_linear_x_is_not_range_clamped_only_non_finite_is_sanitized(self):
        # server.py defines MAX_LINEAR_X and reports it in /api/status limits,
        # but RosmasterControl.update() never calls clamp() on linear_x (see
        # server.py line 230), so out-of-range linear_x passes straight
        # through. This test documents that real, if surprising, behavior
        # rather than the tighter clamping one might expect by analogy with
        # steering_y and angular_z.
        status, body = self._post_json(
            "/api/drive",
            {"enabled": True, "linear_x": 5000.0, "steering_y": 0.0, "angular_z": 0.0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["linear_x"], 5000.0)

        # Non-finite values are still replaced with the 0.0 default.
        status, raw = self._post_raw(
            "/api/drive",
            b'{"enabled": true, "linear_x": NaN, "steering_y": 0.0, "angular_z": 0.0}',
        )
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(body["command"]["linear_x"], 0.0)

    def test_malformed_json_does_not_crash_the_server(self):
        status, data = self._post_raw("/api/drive", b"{not valid json")
        # A body that will not parse is answered with a 400 rather than left
        # to kill the request thread. On a persistent connection a dropped
        # thread also drops the socket the browser was about to reuse.
        self.assertTrue(status is None or status >= 400)
        # The server itself (a ThreadingHTTPServer) must still be alive and
        # answer the next, well-formed request.
        status, body = self._post_json("/api/drive", {"enabled": False})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])


class StopEndpointTests(ServerTestCase):
    def test_stop_reaches_hard_stop_and_disables_auto(self):
        self._post_json("/api/drive", {"enabled": True, "linear_x": 0.6, "steering_y": 0.1, "angular_z": 0.5})
        self._post_json("/api/auto", {"enabled": True})
        self.assertTrue(server.control.auto_snapshot()["enabled"])

        status, body = self._post_json("/api/stop", None)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["control"]["command"]["enabled"], False)
        self.assertEqual(body["control"]["command"]["linear_x"], 0.0)
        self.assertEqual(body["control"]["command"]["steering_y"], 0.0)
        self.assertEqual(body["control"]["command"]["angular_z"], 0.0)
        self.assertEqual(body["control"]["command"]["source"], "stop")
        self.assertFalse(body["auto"]["enabled"])
        self.assertFalse(server.control.auto_snapshot()["enabled"])


class PublishTwistTests(ServerTestCase):
    """Exercises RosmasterControl._publish(), the code that actually builds
    the geometry_msgs Twist sent to /cmd_vel (server.py lines 839-858).

    The stub rclpy.node.Node.create_timer() returns an inert placeholder
    (tests/stubs/rclpy/node.py) and never calls _publish() on its own, so
    these tests call it directly, the same narrow entry point a real timer
    tick would use in production.
    """

    def test_publish_maps_command_fields_onto_twist(self):
        server.control.update(
            {"enabled": True, "linear_x": 0.4, "steering_y": 0.05, "angular_z": 0.2}
        )
        server.control._publish()

        msg = server.control.publisher.messages[-1]
        # linear.y carries the Ackermann steering angle, not lateral
        # velocity; angular.z carries the turn rate. See server.py lines
        # 850-852.
        self.assertEqual(msg.linear.x, 0.4)
        self.assertEqual(msg.linear.y, 0.05)
        self.assertEqual(msg.angular.z, 0.2)

    def test_publish_falls_back_to_zero_twist_after_command_timeout(self):
        # Make the command look CMD_TIMEOUT_S seconds stale by faking the
        # clock only while the command is recorded, not by sleeping the
        # wall clock. _publish() reads the real clock afterward, so the gap
        # between the two is deterministic.
        stale_at = time.monotonic() - server.CMD_TIMEOUT_S - 1.0
        with mock.patch("server.time.monotonic", return_value=stale_at):
            server.control.update(
                {"enabled": True, "linear_x": 0.4, "steering_y": 0.05, "angular_z": 0.2}
            )

        server.control._publish()

        msg = server.control.publisher.messages[-1]
        self.assertEqual(msg.linear.x, 0.0)
        self.assertEqual(msg.linear.y, 0.0)
        self.assertEqual(msg.angular.z, 0.0)


class HandlerReadsGlobalsTests(ServerTestCase):
    """Proves Handler reads server.control fresh on each request, so a later
    task can substitute a scripted fake without touching server.py."""

    class _FakeControl:
        def __init__(self):
            self.calls = []

        def update(self, payload):
            self.calls.append(("update", dict(payload)))
            return {
                "enabled": True,
                "linear_x": 1.0,
                "steering_y": 0.0,
                "angular_z": 0.0,
                "updated_at": 0.0,
                "source": "fake",
            }

        def snapshot(self):
            return {"command": {"source": "fake"}}

        def auto_snapshot(self):
            return {"enabled": False}

        def stop(self):
            self.calls.append(("stop",))

    def test_drive_and_stop_are_dispatched_to_the_substituted_control(self):
        fake = self._FakeControl()
        server.control = fake

        status, body = self._post_json("/api/drive", {"enabled": True, "linear_x": 1.0})
        self.assertEqual(status, 200)
        self.assertEqual(body["command"]["source"], "fake")
        self.assertEqual(fake.calls[-1], ("update", {"enabled": True, "linear_x": 1.0}))

        status, body = self._post_json("/api/stop", None)
        self.assertEqual(status, 200)
        self.assertEqual(fake.calls[-1], ("stop",))


class GamepadEndpointTests(ServerTestCase):
    def test_gamepad_round_trip_sanitizes_axes_buttons_and_ids(self):
        payload = {
            "enabled": True,
            "armed": True,
            "auto": False,
            "index": 0,
            "id": "x" * 200,
            "mapping": "standard-gamepad-mapping-string-way-past-forty-chars",
            "buttons": 7,
            "axes": [0.1, 0.2, "junk", 0.3, None, 0.4, 0.5, 0.6, 0.7, 0.8],
            "pressed": [
                {"index": 0, "name": "A", "value": 1.0},
                {"index": 1, "name": "B", "value": 0.0},
                "not-a-dict",
                {"index": 2, "name": "X", "value": 1.0},
                {"index": 3, "name": "Y", "value": 1.0},
                {"index": 4, "name": "LB", "value": 1.0},
                {"index": 5, "name": "RB", "value": 1.0},
                {"index": 6, "name": "LT", "value": 1.0},
                {"index": 7, "name": "RT", "value": 1.0},
                {"index": 8, "name": "extra_should_be_capped", "value": 1.0},
            ],
        }
        status, body = self._post_json("/api/gamepad", payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        status, body, _ = self._get("/api/gamepad")
        self.assertEqual(status, 200)
        gamepad = json.loads(body.decode("utf-8"))["gamepad"]

        self.assertTrue(gamepad["ok"])
        self.assertEqual(len(gamepad["id"]), 160)
        self.assertEqual(gamepad["mapping"], "standard-gamepad-mapping-string-way-past-f"[:40])
        self.assertEqual(gamepad["buttons"], 7)

        # axes[:8] is sliced first, then non-numeric junk in that slice is
        # dropped, so "junk" and None (both inside the first 8) disappear,
        # and the 9th/10th values (0.7, 0.8) never make it in at all.
        self.assertEqual(gamepad["axes"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # pressed[:8] is sliced first (dropping the 9th item), then the
        # non-dict "not-a-dict" entry inside that slice is filtered out.
        self.assertEqual(len(gamepad["pressed"]), 7)
        names = [item["name"] for item in gamepad["pressed"]]
        self.assertEqual(names, ["A", "B", "X", "Y", "LB", "RB", "LT"])

    def test_gamepad_rejects_non_numeric_junk_entirely(self):
        payload = {"enabled": True, "axes": ["a", "b", None, True], "pressed": ["nope", 1, 2.0]}
        self._post_json("/api/gamepad", payload)
        status, body, _ = self._get("/api/gamepad")
        gamepad = json.loads(body.decode("utf-8"))["gamepad"]
        # bool is technically an int subclass, so isinstance(True, (int, float))
        # is True; True survives as 1.0. The strings and None do not.
        self.assertEqual(gamepad["axes"], [1.0])
        self.assertEqual(gamepad["pressed"], [])


class StatusEndpointTests(ServerTestCase):
    def test_status_returns_expected_top_level_keys(self):
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        for key in (
            "cameras",
            "control",
            "lidar",
            "hp60c",
            "sensors",
            "auto",
            "navigation",
            "gamepad",
            "commands",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["control"]["limits"]["max_steering_y"], server.MAX_STEERING_Y)
        self.assertEqual(payload["control"]["limits"]["max_angular_z"], server.MAX_ANGULAR_Z)
        self.assertEqual(payload["control"]["limits"]["max_linear_x"], server.MAX_LINEAR_X)

    def test_status_reports_the_command_enabled_flag_the_stop_confirmation_reads(self):
        """control.command.enabled is load bearing for the browser.

        refreshStatus in static/app.js confirms an unconfirmed manual stop
        with Boolean(status.control.command.enabled). If control.command were
        renamed or the flag dropped, that expression would read undefined,
        Boolean(undefined) is False, and every pending stop would confirm
        itself instantly and silently: the page would show a calm stopped car
        on no evidence at all. Nothing else pins this shape, so this does.
        """
        self._post_json("/api/drive", {"enabled": True, "linear_x": 0.3})
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        command = json.loads(body.decode("utf-8"))["control"]["command"]
        self.assertIn("enabled", command)
        self.assertIs(command["enabled"], True)

        self._post_json("/api/stop", None)
        status, body, _ = self._get("/api/status")
        command = json.loads(body.decode("utf-8"))["control"]["command"]
        self.assertIs(command["enabled"], False)


class StaticFileTests(ServerTestCase):
    def test_both_front_end_scripts_are_served_with_javascript_content_type(self):
        for name in ("gamepad.js", "app.js"):
            with self.subTest(script=name):
                status, body, headers = self._get(f"/static/{name}")
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], "application/javascript; charset=utf-8")
                on_disk = (APP_DIR / "static" / name).read_bytes()
                self.assertEqual(body, on_disk)
                self.assertGreater(len(body), 0)

    def test_index_html_loads_both_scripts(self):
        # app.js used to be an inline script in index.html and is a served file
        # now, so the page is dead in the browser if either tag or either file
        # goes missing. Nothing else would notice.
        status, body, _ = self._get("/")
        self.assertEqual(status, 200)
        html = body.decode("utf-8")
        self.assertIn('src="/static/gamepad.js"', html)
        self.assertIn('src="/static/app.js"', html)
        self.assertLess(
            html.index('src="/static/gamepad.js"'),
            html.index('src="/static/app.js"'),
            "app.js calls into gamepad.js at load time, so gamepad.js must come first",
        )


class KeepAliveTests(ServerTestCase):
    """Every command used to open a new TCP connection.

    Handler never set protocol_version, so Python answered HTTP/1.0 and closed
    the socket after each response. At eight commands a second on a link whose
    round trip averages 319 ms, most of the budget went on handshakes.
    """

    def test_the_handler_speaks_http_1_1(self):
        self.assertEqual(server.Handler.protocol_version, "HTTP/1.1")

    def test_two_requests_share_one_connection(self):
        conn = self._connection()
        try:
            for expected in (0.11, 0.22):
                body = json.dumps({"enabled": True, "linear_x": expected}).encode("utf-8")
                conn.request("POST", "/api/drive", body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                self.assertEqual(response.version, 11)
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["command"]["linear_x"], expected)
                self.assertFalse(response.will_close, "the connection must stay open for the next command")
        finally:
            conn.close()

    def test_a_post_whose_body_the_handler_ignores_does_not_poison_the_next_request(self):
        """/api/stop is POSTed with a body the stop path never reads.

        Under HTTP/1.0 that did not matter, because the socket closed after the
        response. On a persistent connection those unread bytes become the
        first bytes of the next request, and the browser's next drive command
        is parsed as garbage. The handler has to drain the body whatever the
        route does with it.
        """
        conn = self._connection()
        try:
            conn.request("POST", "/api/stop", body=b'{"padding": "not read by the stop path"}',
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()

            body = json.dumps({"enabled": True, "linear_x": 0.33}).encode("utf-8")
            conn.request("POST", "/api/drive", body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["command"]["linear_x"], 0.33)
        finally:
            conn.close()

    def test_a_404_carries_a_content_length_so_the_connection_can_be_reused(self):
        conn = self._connection()
        try:
            conn.request("GET", "/no/such/path")
            response = conn.getresponse()
            self.assertEqual(response.status, 404)
            self.assertIsNotNone(response.getheader("Content-Length"))
            self.assertEqual(len(response.read()), int(response.getheader("Content-Length")))
        finally:
            conn.close()

    def test_malformed_json_is_answered_rather_than_dropping_the_connection(self):
        status, raw = self._post_raw("/api/drive", b"{not valid json")
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(raw.decode("utf-8"))["ok"])


class MjpegStreamTests(ServerTestCase):
    """The MJPEG responses send no Content-Length, because the body never ends.
    Under HTTP/1.1 that is a promise the server cannot keep, so each must say
    Connection: close or the browser waits forever for a body length it will
    never be told. The gallery opens one of these per tile, so a response that
    holds its connection open costs a thread per tile per reconnect."""

    class _FakeControl:
        def __init__(self, frame=b"\xff\xd8jpegbytes\xff\xd9"):
            self.frame = frame
            self.viewers = []

        def camera_frame(self, camera, stream):
            return self.frame

        def open_camera_viewer(self, camera, stream):
            self.viewers.append(("open", camera, stream))

        def close_camera_viewer(self, camera, stream):
            self.viewers.append(("close", camera, stream))

    def _stream_headers(self, path):
        conn = self._connection()
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return response.status, response.getheader("Connection"), response.getheader("Content-Type")
        finally:
            conn.close()

    def test_every_gallery_stream_closes_its_connection(self):
        server.control = self._FakeControl()
        for feed in server.CAMERA_FEEDS:
            with self.subTest(feed=feed["id"]):
                status, connection, content_type = self._stream_headers(feed["path"])
                self.assertEqual(status, 200)
                self.assertEqual((connection or "").lower(), "close")
                self.assertTrue(content_type.startswith("multipart/x-mixed-replace"))


class CameraFeedRegistryTests(ServerTestCase):
    """The gallery renders one tile per entry in this registry.

    Six feeds are defined and no car carries all six: the HP60C publishes
    depth and rgb, the RealSense D435i publishes depth, both infrared views
    and colour. Membership therefore has to follow the hardware that is
    actually fitted, because a tile for a camera that is not there can never
    fill, and the operator cannot tell a tile that will never fill apart from
    one that is merely late.

    Membership is separate from health. A feed that has produced frames and
    then gone quiet stays listed and reports not ok, because a driver restart
    drops a stream for a few seconds and a tile that vanishes and returns reads
    as a broken page rather than a hiccup.
    """

    HP60C_STREAMS = ("depth", "rgb")
    REALSENSE_STREAMS = ("depth", "infra1", "infra2", "color")

    @classmethod
    def _snapshot(cls, streams, live=(), publishers=(), **overrides):
        """A camera snapshot in the shape hp60c_snapshot/realsense_snapshot return.

        `live` names the streams that have delivered frames and are fresh.
        `publishers` names the streams that have a publisher on the topic right
        now, which is how a driver that has just started but not yet produced
        an image announces itself.
        """
        base = {"ok": False, "age_s": None, "frames": 0}
        snapshot = {
            name: {**base, **({"ok": True, "age_s": 0.06, "frames": 12} if name in live else {})}
            for name in streams
        }
        for name, value in overrides.items():
            snapshot[name] = {**snapshot[name], **value}
        snapshot["publishers"] = {name: (1 if name in publishers else 0) for name in streams}
        return snapshot

    @classmethod
    def _hp60c(cls, live=(), publishers=(), **overrides):
        return cls._snapshot(cls.HP60C_STREAMS, live, publishers, **overrides)

    @classmethod
    def _realsense(cls, live=(), publishers=(), **overrides):
        return cls._snapshot(cls.REALSENSE_STREAMS, live, publishers, **overrides)

    @staticmethod
    def _ids(hp60c=None, realsense=None):
        return [feed["id"] for feed in server.camera_feeds(hp60c, realsense)]

    @staticmethod
    def _feeds_by_id(hp60c=None, realsense=None):
        return {feed["id"]: feed for feed in server.camera_feeds(hp60c, realsense)}

    def test_every_feed_carries_an_id_label_path_health_and_age(self):
        feeds = server.camera_feeds(self._hp60c(live=("depth", "rgb")))
        self.assertTrue(feeds, "the registry must not be empty on hardware that has cameras")
        for feed in feeds:
            self.assertEqual(
                sorted(feed),
                ["age_s", "id", "label", "ok", "path"],
                "the client renders straight from these keys",
            )
            self.assertTrue(feed["label"], "a tile with no label tells the operator nothing")

    def test_an_hp60c_car_lists_two_feeds(self):
        self.assertEqual(
            self._ids(self._hp60c(live=("depth", "rgb")), self._realsense()),
            ["hp60c_depth", "hp60c_rgb"],
        )

    def test_a_realsense_car_lists_four_feeds(self):
        self.assertEqual(
            self._ids(self._hp60c(), self._realsense(live=("depth", "infra1", "infra2", "color"))),
            ["realsense_color", "realsense_depth", "realsense_infra1", "realsense_infra2"],
        )

    def test_a_car_with_both_cameras_lists_all_six(self):
        self.assertEqual(
            self._ids(
                self._hp60c(live=("depth", "rgb")),
                self._realsense(live=("depth", "infra1", "infra2", "color")),
            ),
            [
                "hp60c_depth",
                "hp60c_rgb",
                "realsense_color",
                "realsense_depth",
                "realsense_infra1",
                "realsense_infra2",
            ],
        )

    def test_a_car_with_no_camera_lists_nothing(self):
        self.assertEqual(self._ids(self._hp60c(), self._realsense()), [])

    def test_the_four_realsense_feeds_say_what_the_operator_is_looking_at(self):
        feeds = self._feeds_by_id(None, self._realsense(live=("depth", "infra1", "infra2", "color")))
        self.assertEqual(feeds["realsense_depth"]["label"], "Depth")
        self.assertEqual(feeds["realsense_infra1"]["label"], "Left Stereo")
        self.assertEqual(feeds["realsense_infra2"]["label"], "Right Stereo")
        self.assertEqual(feeds["realsense_color"]["label"], "RGB")

    def test_a_feed_that_has_gone_quiet_stays_listed_and_reports_its_age(self):
        """Fitted and unhappy is not the same as absent, and only one of them removes a tile."""
        feeds = self._feeds_by_id(
            None,
            self._realsense(
                live=("depth", "infra1", "infra2"),
                color={"ok": False, "age_s": 5.2, "frames": 91},
            ),
        )
        self.assertIn("realsense_color", feeds, "an intermittent feed must go stale, not disappear")
        self.assertFalse(feeds["realsense_color"]["ok"])
        self.assertEqual(feeds["realsense_color"]["age_s"], 5.2)

    def test_a_feed_never_seen_and_with_no_publisher_is_absent(self):
        feeds = self._feeds_by_id(None, self._realsense(live=("depth",)))
        self.assertEqual(sorted(feeds), ["realsense_depth"])

    def test_a_publisher_with_no_frames_yet_is_listed_and_reports_waiting(self):
        """A driver that has started but not yet produced an image is fitted hardware.

        Listing it here is what makes the gallery appear as the camera comes up
        rather than only once it works, and a null age_s is how the page tells
        waiting apart from stale.
        """
        feeds = self._feeds_by_id(None, self._realsense(publishers=("depth", "infra1", "infra2", "color")))
        self.assertEqual(len(feeds), 4)
        self.assertFalse(feeds["realsense_depth"]["ok"])
        self.assertIsNone(
            feeds["realsense_depth"]["age_s"],
            "never seen and gone stale are different states and the tile shows them differently",
        )

    def test_a_frame_seen_once_keeps_the_tile_through_a_driver_restart(self):
        """The publisher count drops to zero while the driver comes back up."""
        feeds = self._feeds_by_id(
            None,
            self._realsense(depth={"ok": False, "age_s": 3.4, "frames": 400}),
        )
        self.assertEqual(sorted(feeds), ["realsense_depth"])

    def test_a_missing_snapshot_does_not_raise(self):
        for empty in (None, {}, {"depth": None}, {"depth": None, "publishers": None}):
            with self.subTest(snapshot=empty):
                self.assertEqual(server.camera_feeds(empty, empty), [])

    def test_status_carries_the_registry(self):
        """The stub ROS node reports no publishers and no frames, so this car has no camera."""
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload["cameras"], [])
        self.assertIn("realsense", payload, "the diagnostics need the topics and publisher counts")

    def test_every_advertised_path_is_a_route_the_server_answers(self):
        """A registry entry the server cannot stream is a black tile by another name."""
        server.control = MjpegStreamTests._FakeControl()
        feeds = server.camera_feeds(
            self._hp60c(live=("depth", "rgb")),
            self._realsense(live=("depth", "infra1", "infra2", "color")),
        )
        self.assertEqual(len(feeds), 6)
        for feed in feeds:
            with self.subTest(feed=feed["id"]):
                conn = self._connection()
                try:
                    conn.request("GET", feed["path"])
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.getheader("Content-Type").startswith("multipart/x-mixed-replace"))
                finally:
                    conn.close()


class RealSenseSubscriptionTests(ServerTestCase):
    """The four D435i streams, decoded by the same code as the HP60C ones."""

    class _FakeHeader:
        frame_id = "camera_link"

    @staticmethod
    def _image(encoding, width=64, height=48):
        msg = server.RosImage()
        msg.width = width
        msg.height = height
        msg.encoding = encoding
        msg.header = RealSenseSubscriptionTests._FakeHeader()
        if encoding == "16uc1":
            msg.step = width * 2
            msg.data = (900).to_bytes(2, "little") * (width * height)
        elif encoding == "mono8":
            msg.step = width
            msg.data = bytes([128]) * (width * height)
        else:
            msg.step = width * 3
            msg.data = bytes([10, 120, 250]) * (width * height)
        return msg

    def test_the_default_topics_are_the_ones_the_driver_launches(self):
        self.assertEqual(server.REALSENSE_DEPTH_TOPIC, "/camera/camera/depth/image_rect_raw")
        self.assertEqual(server.REALSENSE_INFRA1_TOPIC, "/camera/camera/infra1/image_rect_raw")
        self.assertEqual(server.REALSENSE_INFRA2_TOPIC, "/camera/camera/infra2/image_rect_raw")
        self.assertEqual(server.REALSENSE_COLOR_TOPIC, "/camera/camera/color/image_raw")

    def test_each_stream_records_its_own_freshness(self):
        control = server.control
        control.open_camera_viewer("realsense", "depth")
        control.open_camera_viewer("realsense", "infra1")
        try:
            control._on_realsense_depth(self._image("16uc1"))
            control._on_realsense_infra1(self._image("mono8"))
            snapshot = control.realsense_snapshot()
        finally:
            control.close_camera_viewer("realsense", "depth")
            control.close_camera_viewer("realsense", "infra1")

        self.assertTrue(snapshot["depth"]["ok"])
        self.assertTrue(snapshot["infra1"]["ok"])
        self.assertFalse(snapshot["infra2"]["ok"], "a stream with no frames must not borrow another one's health")
        self.assertIsNone(snapshot["infra2"]["age_s"])
        self.assertEqual(snapshot["depth"]["encoding"], "16uc1")
        self.assertEqual(snapshot["infra1"]["encoding"], "mono8")

    def test_a_watched_stream_produces_a_jpeg_for_its_route(self):
        control = server.control
        for stream, encoding, callback in (
            ("depth", "16uc1", "_on_realsense_depth"),
            ("infra1", "mono8", "_on_realsense_infra1"),
            ("infra2", "mono8", "_on_realsense_infra2"),
            ("color", "rgb8", "_on_realsense_color"),
        ):
            with self.subTest(stream=stream):
                control.open_camera_viewer("realsense", stream)
                try:
                    getattr(control, callback)(self._image(encoding))
                    frame = control.camera_frame("realsense", stream)
                finally:
                    control.close_camera_viewer("realsense", stream)
                self.assertIsNotNone(frame)
                self.assertTrue(frame.startswith(b"\xff\xd8"), "the gallery is fed JPEG, not raw pixels")

    def test_an_unwatched_stream_is_tracked_but_not_encoded(self):
        """Four cameras at fifteen frames a second share one ROS executor thread.

        The executor is the same thread that ticks the /cmd_vel publish timer,
        so work done in an image callback is time the drive command is not
        being sent. Encoding a preview nobody has open buys nothing and costs
        that, so an unwatched feed records its freshness and stops there.
        """
        control = server.control
        control._on_realsense_color(self._image("rgb8"))
        self.assertIsNone(control.camera_frame("realsense", "color"))
        self.assertTrue(control.realsense_snapshot()["color"]["ok"], "freshness is tracked whether or not anyone is watching")

    def test_the_last_viewer_leaving_drops_the_cached_frame(self):
        """A frame held over from the last viewer would be replayed as if it were live."""
        control = server.control
        control.open_camera_viewer("realsense", "color")
        control._on_realsense_color(self._image("rgb8"))
        self.assertIsNotNone(control.camera_frame("realsense", "color"))
        control.close_camera_viewer("realsense", "color")
        self.assertIsNone(control.camera_frame("realsense", "color"))


class CameraStreamThreadTests(ServerTestCase):
    """Each open MJPEG response holds a request thread for as long as it runs.

    Four tiles is double what this server has carried, and the loop used to
    have no way out when no frame ever arrived: it slept and looped without
    writing anything, so it never noticed the browser had gone and never gave
    the thread back. Every reconnect of a feed in that state left another
    thread behind, which is how the server ran out of them.
    """

    class _SilentControl:
        def camera_frame(self, camera, stream):
            return None

        def open_camera_viewer(self, camera, stream):
            return None

        def close_camera_viewer(self, camera, stream):
            return None

    def test_a_stream_with_no_frames_gives_its_thread_back(self):
        server.control = self._SilentControl()
        original = server.CAMERA_STREAM_IDLE_TIMEOUT_S
        server.CAMERA_STREAM_IDLE_TIMEOUT_S = 0.3
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            started = time.monotonic()
            try:
                conn.request("GET", "/stream_realsense_infra1.mjpg")
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                body = response.read()
            finally:
                conn.close()
        finally:
            server.CAMERA_STREAM_IDLE_TIMEOUT_S = original
        self.assertEqual(body, b"", "a feed with nothing to send sends nothing and closes")
        self.assertLess(
            time.monotonic() - started,
            4.0,
            "the response must end on its own rather than hold the thread forever",
        )

    def test_a_stream_reports_its_viewer_to_the_control(self):
        fake = MjpegStreamTests._FakeControl()
        server.control = fake
        original = server.CAMERA_STREAM_IDLE_TIMEOUT_S
        server.CAMERA_STREAM_IDLE_TIMEOUT_S = 0.3
        try:
            conn = self._connection()
            try:
                conn.request("GET", "/stream_realsense_color.mjpg")
                conn.getresponse()
            finally:
                conn.close()
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and ("close", "realsense", "color") not in fake.viewers:
                time.sleep(0.05)
        finally:
            server.CAMERA_STREAM_IDLE_TIMEOUT_S = original
        self.assertIn(("open", "realsense", "color"), fake.viewers)
        self.assertIn(
            ("close", "realsense", "color"),
            fake.viewers,
            "a viewer that is not released leaves the feed encoding for nobody",
        )


class RawV4LRemovalTests(ServerTestCase):
    """CameraWorker could never produce a frame: start() set an error and served
    a placeholder, and every capture method returned None or False. It was a
    hundred lines of code that could not work, in an app about to be published
    as a public sample."""

    def test_the_raw_stream_endpoint_is_gone(self):
        status, _, _ = self._get("/stream.mjpg")
        self.assertEqual(status, 404)

    def test_the_camera_device_endpoint_is_gone(self):
        status, _ = self._post_raw("/api/camera", b'{"device": "/dev/video0"}')
        self.assertEqual(status, 404)

    def test_the_worker_and_its_globals_are_gone(self):
        for name in ("CameraWorker", "camera", "CAMERA_DEVICE", "RAW_CAMERA_ENABLED"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(server, name))

    def test_status_stays_well_formed_without_the_dead_camera_key(self):
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertNotIn("camera", payload)
        for key in ("cameras", "control", "lidar", "hp60c", "sensors", "auto", "navigation", "gamepad", "commands"):
            self.assertIn(key, payload)


class CommandFreshnessTests(unittest.TestCase):
    """The dangerous one: a command that arrives after the operator has moved
    on and puts the car back into motion.

    Two checks, and neither needs the browser and the car to agree on what time
    it is. `seq` is a counter the page increments per command, so the car can
    drop anything that arrives behind something newer. `age_ms` is a difference
    between two readings of the browser's own monotonic clock, never an
    absolute time, so it means the same thing on both machines.
    """

    def setUp(self):
        self.freshness = server.CommandFreshness(max_age_ms=400.0)

    def test_a_fresh_command_is_applied(self):
        verdict = self.freshness.check({"client_id": "abc", "seq": 1, "age_ms": 12, "linear_x": 0.4})
        self.assertTrue(verdict["fresh"])

    def test_an_over_age_command_is_rejected(self):
        self.freshness.check({"client_id": "abc", "seq": 1, "age_ms": 5})
        verdict = self.freshness.check({"client_id": "abc", "seq": 2, "age_ms": 900})
        self.assertFalse(verdict["fresh"])
        self.assertEqual(verdict["reason"], "stale")

    def test_the_age_threshold_is_inclusive_of_the_boundary(self):
        self.assertTrue(self.freshness.check({"client_id": "a", "seq": 1, "age_ms": 400})["fresh"])
        self.assertFalse(self.freshness.check({"client_id": "a", "seq": 2, "age_ms": 401})["fresh"])

    def test_an_out_of_order_sequence_is_ignored(self):
        self.assertTrue(self.freshness.check({"client_id": "abc", "seq": 7, "age_ms": 5})["fresh"])
        verdict = self.freshness.check({"client_id": "abc", "seq": 6, "age_ms": 5})
        self.assertFalse(verdict["fresh"])
        self.assertEqual(verdict["reason"], "out of order")

    def test_a_repeated_sequence_number_is_ignored(self):
        self.freshness.check({"client_id": "abc", "seq": 7, "age_ms": 5})
        self.assertFalse(self.freshness.check({"client_id": "abc", "seq": 7, "age_ms": 5})["fresh"])

    def test_a_rejected_stale_command_still_advances_the_sequence(self):
        """Otherwise the reordered burst it came from could still land.

        The stale command is the newest thing the page has produced. Dropping
        it and then accepting an older sibling that arrives a moment later
        would put back exactly the throttle we refused.
        """
        self.freshness.check({"client_id": "abc", "seq": 1, "age_ms": 5})
        self.assertFalse(self.freshness.check({"client_id": "abc", "seq": 5, "age_ms": 900})["fresh"])
        self.assertFalse(self.freshness.check({"client_id": "abc", "seq": 4, "age_ms": 5})["fresh"])
        self.assertTrue(self.freshness.check({"client_id": "abc", "seq": 6, "age_ms": 5})["fresh"])

    def test_a_payload_with_no_timestamp_still_works(self):
        """An older page must not be locked out of driving the car."""
        for _ in range(3):
            self.assertTrue(self.freshness.check({"enabled": True, "linear_x": 0.4})["fresh"])

    def test_a_page_reload_restarts_the_sequence_without_locking_itself_out(self):
        self.freshness.check({"client_id": "first", "seq": 90, "age_ms": 5})
        verdict = self.freshness.check({"client_id": "second", "seq": 1, "age_ms": 5})
        self.assertTrue(verdict["fresh"], "a reloaded page starts counting again and must still be able to drive")

    def test_rejections_are_counted_for_the_status_payload(self):
        self.freshness.check({"client_id": "a", "seq": 1, "age_ms": 5})
        self.freshness.check({"client_id": "a", "seq": 2, "age_ms": 900})
        self.freshness.check({"client_id": "a", "seq": 1, "age_ms": 5})
        snapshot = self.freshness.snapshot()
        self.assertEqual(snapshot["rejected"]["stale"], 1)
        self.assertEqual(snapshot["rejected"]["out_of_order"], 1)
        self.assertEqual(snapshot["rejected"]["total"], 2)
        self.assertEqual(snapshot["max_age_ms"], 400.0)

    def test_rejection_logging_is_throttled(self):
        logged = []
        freshness = server.CommandFreshness(max_age_ms=400.0, log=logged.append)
        for seq in range(2, 200):
            freshness.check({"client_id": "a", "seq": seq, "age_ms": 900})
        self.assertGreaterEqual(len(logged), 1, "a rejection the operator cannot see in the logs is a rejection they cannot debug")
        self.assertLessEqual(len(logged), 3, "198 rejections in one burst must not become 198 log lines")


class DriveFreshnessEndpointTests(ServerTestCase):
    def setUp(self):
        super().setUp()
        self._orig_freshness = server.command_freshness
        server.command_freshness = server.CommandFreshness(max_age_ms=400.0)

    def tearDown(self):
        server.command_freshness = self._orig_freshness
        super().tearDown()

    def test_a_stale_drive_command_reaches_the_car_as_a_zero(self):
        self._post_json("/api/drive", {"client_id": "a", "seq": 1, "age_ms": 5, "enabled": True, "linear_x": 0.5})
        self.assertEqual(server.control.snapshot()["command"]["linear_x"], 0.5)

        status, body = self._post_json(
            "/api/drive",
            {"client_id": "a", "seq": 2, "age_ms": 1500, "enabled": True, "linear_x": 0.5},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["rejected"])
        self.assertEqual(body["reason"], "stale")
        self.assertEqual(body["command"]["enabled"], False)
        self.assertEqual(body["command"]["linear_x"], 0.0)
        self.assertEqual(server.control.snapshot()["command"]["linear_x"], 0.0)

    def test_an_out_of_order_drive_command_cannot_resurrect_an_old_throttle(self):
        self._post_json("/api/drive", {"client_id": "a", "seq": 9, "age_ms": 5, "enabled": True, "linear_x": 0.6})
        # The zero the operator sent on releasing the stick.
        self._post_json("/api/drive", {"client_id": "a", "seq": 10, "age_ms": 5, "enabled": False})
        self.assertEqual(server.control.snapshot()["command"]["linear_x"], 0.0)

        # The throttle that was still in flight when they let go.
        status, body = self._post_json(
            "/api/drive",
            {"client_id": "a", "seq": 9, "age_ms": 5, "enabled": True, "linear_x": 0.6},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["rejected"])
        self.assertEqual(server.control.snapshot()["command"]["linear_x"], 0.0)
        self.assertEqual(server.control.snapshot()["command"]["enabled"], False)

    def test_status_surfaces_the_rejection_count(self):
        self._post_json("/api/drive", {"client_id": "a", "seq": 1, "age_ms": 5, "enabled": True, "linear_x": 0.2})
        self._post_json("/api/drive", {"client_id": "a", "seq": 2, "age_ms": 5000, "enabled": True, "linear_x": 0.2})
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        commands = json.loads(body.decode("utf-8"))["commands"]
        self.assertEqual(commands["rejected"]["stale"], 1)
        self.assertEqual(commands["rejected"]["total"], 1)


class VoiceRemovalTests(ServerTestCase):
    """The voice app is not deployed and will not be. voice_snapshot() polled
    its status URL on every /api/status request with a 250 ms timeout, adding
    that much to a path the browser hits every 750 ms."""

    def test_the_voice_endpoint_is_gone(self):
        status, _, _ = self._get("/api/voice")
        self.assertEqual(status, 404)

    def test_status_no_longer_carries_a_voice_key(self):
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        self.assertNotIn("voice", json.loads(body.decode("utf-8")))

    def test_the_server_no_longer_polls_the_voice_endpoint(self):
        # The poll itself is gone, not merely unused: voice_snapshot() opened a
        # urllib request with a 250 ms timeout on every status request, and the
        # browser polls status every 750 ms.
        self.assertFalse(hasattr(server, "voice_snapshot"))
        self.assertFalse(hasattr(server, "VOICE_STATUS_URL"))
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("the status path must not poll voice")):
            status, _, _ = self._get("/api/status")
        self.assertEqual(status, 200)


class DepthSourceTests(ServerTestCase):
    """Autonomy plans on whichever depth camera is fitted, not on a named model.

    The HP60C this planner was written against has been removed and an Intel
    RealSense D435i sits in its place. Auto Nav refused to engage afterwards,
    because the readiness check asked for HP60C depth by name. Refusing was the
    right direction to fail in, but the car has a depth camera and it should be
    able to use it.
    """

    @staticmethod
    def _stream(age_s=None, frames=0, **stats):
        """One depth stream, as old as you say and with the stats you give it."""
        stream = {"updated_at": (time.monotonic() - age_s) if age_s is not None else 0.0, "frames": frames}
        stream.update(stats)
        return stream

    @classmethod
    def _hp60c(cls, depth=None, publishers=()):
        return {
            "depth": depth if depth is not None else cls._stream(),
            "rgb": cls._stream(),
            "publishers": {name: 1 for name in publishers},
        }

    @classmethod
    def _realsense(cls, depth=None, publishers=()):
        return {
            "depth": depth if depth is not None else cls._stream(),
            "infra1": cls._stream(),
            "infra2": cls._stream(),
            "color": cls._stream(),
            "publishers": {name: 1 for name in publishers},
        }

    @classmethod
    def _usable_depth(cls, age_s=0.05):
        """A depth frame the planner would be willing to drive on."""
        return cls._stream(
            age_s=age_s,
            frames=40,
            obstacle_p20_m=1.4,
            above_floor_near_m=1.6,
            obstacle_valid_ratio=0.6,
            above_floor_valid_ratio=0.6,
        )

    def _ready(self, hp60c, realsense, subscribers=1, lidar_ok=True):
        """_auto_ready with the LiDAR and the base driver taken as given.

        Those two conditions have their own tests. These are about the third.
        """
        scan = {
            "ok": lidar_ok,
            "sectors": {"front": {"near_m": 1.30, "count": 40}},
        }
        control = server.control
        with mock.patch.object(control, "lidar_snapshot", return_value=scan), mock.patch.object(
            control, "count_subscribers", return_value=subscribers
        ), mock.patch.object(control, "hp60c_snapshot", return_value=hp60c), mock.patch.object(
            control, "realsense_snapshot", return_value=realsense
        ):
            return control._auto_ready()

    def test_a_realsense_car_with_no_hp60c_can_engage_autonomy(self):
        ready = self._ready(self._hp60c(), self._realsense(depth=self._usable_depth()))
        self.assertTrue(ready["ready"], ready["reason"])

    def test_an_hp60c_car_with_no_realsense_can_still_engage_autonomy(self):
        """The camera that was here first did not stop working when it was replaced."""
        ready = self._ready(self._hp60c(depth=self._usable_depth()), self._realsense())
        self.assertTrue(ready["ready"], ready["reason"])

    def test_a_car_with_no_depth_camera_at_all_is_not_ready(self):
        ready = self._ready(self._hp60c(), self._realsense())
        self.assertFalse(ready["ready"])
        self.assertIn("depth camera", ready["reason"])

    def test_stale_depth_refuses_whichever_camera_is_fitted(self):
        for camera, snapshots in (
            ("hp60c", (self._hp60c(depth=self._usable_depth(age_s=9.0)), self._realsense())),
            ("realsense", (self._hp60c(), self._realsense(depth=self._usable_depth(age_s=9.0)))),
        ):
            with self.subTest(camera=camera):
                ready = self._ready(*snapshots)
                self.assertFalse(ready["ready"], "stale depth must never let autonomy engage")
                self.assertIn(camera, ready["reason"])
                self.assertIn("fresh", ready["reason"])

    def test_the_reason_names_the_camera_the_operator_is_waiting_on(self):
        """"Waiting for fresh depth" is unanswerable without knowing which camera."""
        realsense = self._ready(self._hp60c(), self._realsense(publishers=("depth",)))
        self.assertIn("realsense", realsense["reason"])
        self.assertNotIn("hp60c", realsense["reason"])
        hp60c = self._ready(self._hp60c(publishers=("depth",)), self._realsense())
        self.assertIn("hp60c", hp60c["reason"])
        self.assertNotIn("realsense", hp60c["reason"])

    def test_a_camera_that_is_up_but_silent_is_named_rather_than_ignored(self):
        """A publisher and no frames yet is fitted hardware, and not ready."""
        ready = self._ready(self._hp60c(), self._realsense(publishers=("depth",)))
        self.assertFalse(ready["ready"])
        self.assertIn("realsense", ready["reason"])

    def test_readiness_still_requires_lidar_and_a_base_driver(self):
        """A working depth camera does not excuse either of the other two."""
        fitted = (self._hp60c(), self._realsense(depth=self._usable_depth()))
        self.assertFalse(self._ready(*fitted, lidar_ok=False)["ready"])
        self.assertFalse(self._ready(*fitted, subscribers=0)["ready"])

    def test_membership_comes_from_the_camera_registry_and_not_a_second_list(self):
        """Fitted means what the gallery means by fitted, on the same evidence."""
        for label, hp60c, realsense in (
            ("hp60c", self._hp60c(depth=self._usable_depth()), self._realsense()),
            ("realsense", self._hp60c(), self._realsense(depth=self._usable_depth())),
            ("neither", self._hp60c(), self._realsense()),
        ):
            with self.subTest(fitted=label):
                listed = {feed["id"] for feed in server.camera_feeds(hp60c, realsense)}
                source = server.select_depth_source(hp60c, realsense)
                if source is None:
                    self.assertNotIn("hp60c_depth", listed)
                    self.assertNotIn("realsense_depth", listed)
                else:
                    self.assertIn(source["feed_id"], listed)

    def test_a_fresh_camera_wins_over_a_stale_one_when_both_are_fitted(self):
        source = server.select_depth_source(
            self._hp60c(depth=self._usable_depth(age_s=9.0)),
            self._realsense(depth=self._usable_depth()),
        )
        self.assertEqual(source["camera"], "realsense")
        self.assertTrue(source["fresh"])

    def test_each_camera_is_judged_against_its_own_freshness_bound(self):
        source = server.select_depth_source(None, self._realsense(depth=self._usable_depth()))
        self.assertEqual(source["stale_s"], server.REALSENSE_STALE_S)
        source = server.select_depth_source(self._hp60c(depth=self._usable_depth()), None)
        self.assertEqual(source["stale_s"], server.HP60C_STALE_S)

    def test_the_planner_reports_which_camera_it_planned_on(self):
        control = server.control
        scan = {"updated_at": time.monotonic(), "sectors": {"front": {"near_m": 1.4, "count": 40}}, "gap_samples": []}
        auto = {"speed": 0.4, "stop_distance": 0.35, "avoid_distance": 0.85, "clear_distance": 1.6}
        for camera, hp60c, realsense in (
            ("hp60c", self._hp60c(depth=self._usable_depth()), self._realsense()),
            ("realsense", self._hp60c(), self._realsense(depth=self._usable_depth())),
        ):
            with self.subTest(camera=camera):
                source = server.select_depth_source(hp60c, realsense)
                with mock.patch.object(control, "count_subscribers", return_value=1):
                    _, decision = control._compute_auto_command(scan, auto, source, update_state=False)
                self.assertEqual(decision["depth_source"], camera)
                self.assertTrue(decision["depth_ok"], decision["reason"])

    def test_the_planner_refuses_and_says_so_when_no_camera_is_fitted(self):
        control = server.control
        scan = {"updated_at": time.monotonic(), "sectors": {"front": {"near_m": 1.4, "count": 40}}, "gap_samples": []}
        auto = {"speed": 0.4, "stop_distance": 0.35, "avoid_distance": 0.85, "clear_distance": 1.6}
        with mock.patch.object(control, "count_subscribers", return_value=1):
            msg, decision = control._compute_auto_command(scan, auto, None, update_state=False)
        self.assertEqual(msg.linear.x, 0.0)
        self.assertIsNone(decision["depth_source"])
        self.assertIn("depth camera", decision["reason"])

    def test_no_readout_claims_a_camera_it_did_not_read(self):
        """A field named for the wrong camera is worse than no field at all."""
        control = server.control
        scan = {"updated_at": time.monotonic(), "sectors": {"front": {"near_m": 1.4, "count": 40}}, "gap_samples": []}
        auto = {"speed": 0.4, "stop_distance": 0.35, "avoid_distance": 0.85, "clear_distance": 1.6}
        source = server.select_depth_source(None, self._realsense(depth=self._usable_depth()))
        with mock.patch.object(control, "count_subscribers", return_value=1):
            _, decision = control._compute_auto_command(scan, auto, source, update_state=False)
        self.assertFalse(
            [key for key in decision if key.startswith("hp60c")],
            "the decision came from a RealSense and must not be labelled hp60c",
        )

    def test_the_required_depth_sensor_follows_the_fitted_camera(self):
        control = server.control
        for expected, hp60c, realsense in (
            ("hp60c_depth", self._hp60c(depth=self._usable_depth()), self._realsense()),
            ("realsense_depth", self._hp60c(), self._realsense(depth=self._usable_depth())),
            ("hp60c_depth", self._hp60c(), self._realsense()),
        ):
            with self.subTest(expected=expected):
                with mock.patch.object(control, "hp60c_snapshot", return_value=hp60c), mock.patch.object(
                    control, "realsense_snapshot", return_value=realsense
                ):
                    sensors = control.sensors_snapshot()
                self.assertIn(expected, sensors["required"])

    def test_a_realsense_car_is_not_marked_healthy_for_missing_an_hp60c(self):
        control = server.control
        with mock.patch.object(control, "hp60c_snapshot", return_value=self._hp60c()), mock.patch.object(
            control, "realsense_snapshot", return_value=self._realsense(depth=self._usable_depth())
        ):
            sensors = control.sensors_snapshot()
        self.assertNotIn("hp60c_depth", sensors["required"])
        self.assertNotIn("hp60c_depth", sensors["missing"])

    def test_a_car_with_no_depth_camera_still_reports_a_missing_depth_sensor(self):
        """Dropping the requirement would turn the panel green on a car that cannot self drive."""
        control = server.control
        with mock.patch.object(control, "hp60c_snapshot", return_value=self._hp60c()), mock.patch.object(
            control, "realsense_snapshot", return_value=self._realsense()
        ):
            sensors = control.sensors_snapshot()
        self.assertTrue([name for name in sensors["missing"] if name.endswith("_depth")])
        self.assertFalse(sensors["ok"])

    def test_the_static_required_list_no_longer_names_a_camera(self):
        self.assertFalse(
            [name for name in server.REQUIRED_SENSOR_NAMES if name.endswith("_depth")],
            "which depth camera is required is a runtime question",
        )

    def test_realsense_depth_statistics_are_produced_with_nobody_watching(self):
        """Autonomy that only worked while a browser tab was open would be a trap.

        The preview is skipped for an unwatched feed, and should be. The zone
        statistics behind it are what the planner vetoes on, so they are not.
        """
        control = server.control
        control._on_realsense_depth(RealSenseSubscriptionTests._image("16uc1"))
        self.assertIsNone(control.camera_frame("realsense", "depth"), "an unwatched tile still costs no JPEG")
        snapshot = control.realsense_snapshot()
        self.assertIsNotNone(snapshot["depth"]["obstacle_p20_m"])
        self.assertGreater(snapshot["depth"]["valid_ratio"], 0.0)
        source = server.select_depth_source(control.hp60c_snapshot(), snapshot)
        self.assertEqual(source["camera"], "realsense")
        self.assertTrue(source["fresh"])

    def test_an_undecodable_depth_frame_does_not_leave_stale_distances_behind(self):
        """A fresh timestamp over the last frame's distances is a readout that lies."""
        control = server.control
        control._on_realsense_depth(RealSenseSubscriptionTests._image("16uc1"))
        self.assertIsNotNone(control.realsense_snapshot()["depth"]["obstacle_p20_m"])
        control._on_realsense_depth(RealSenseSubscriptionTests._image("yuyv"))
        snapshot = control.realsense_snapshot()
        self.assertTrue(snapshot["depth"]["ok"], "frames are still arriving, so the feed is not stale")
        self.assertIsNone(
            snapshot["depth"]["obstacle_p20_m"],
            "a frame this code cannot read measures nothing, and must report nothing",
        )
        source = server.select_depth_source(control.hp60c_snapshot(), snapshot)
        scan = {"updated_at": time.monotonic(), "sectors": {"front": {"near_m": 1.4, "count": 40}}, "gap_samples": []}
        auto = {"speed": 0.4, "stop_distance": 0.35, "avoid_distance": 0.85, "clear_distance": 1.6}
        with mock.patch.object(control, "count_subscribers", return_value=1):
            msg, decision = control._compute_auto_command(scan, auto, source, update_state=False)
        self.assertFalse(decision["depth_ok"])
        self.assertEqual(msg.linear.x, 0.0)

    def test_an_untouched_realsense_reports_the_same_empty_shape_as_the_hp60c(self):
        """The planner reads the same keys off either camera, present or not."""
        control = server.control
        realsense = control._empty_realsense()["depth"]
        hp60c = control._empty_hp60c()["depth"]
        self.assertEqual(sorted(hp60c), sorted(realsense))
        self.assertIsNone(realsense["obstacle_p20_m"])


class ReverseEscapeBoundTests(unittest.TestCase):
    """The recovery manoeuvre drives the car into the one place it cannot see.

    Reported from driving the real car: "Lidar doesn't scan behind, so it got
    stuck running away from something." The LiDAR covers roughly plus or minus
    105 degrees at the front and the depth camera faces forward too, so nothing
    on this chassis observes what is behind it, and an Ackermann car cannot
    turn in place to go and look first. Reversing is therefore the single
    manoeuvre guaranteed to move the car through unobserved space, and the
    escape counters used to make it reverse further the longer it struggled:
    less information bought more movement, which is backwards.

    Every bound here is on a whole stuck episode rather than on one attempt.
    A per attempt bound is not a bound at all when the state machine is free to
    make another attempt.
    """

    # The bar, in the units an operator cares about. These are deliberately
    # written as literals rather than read off the server's own constants: a
    # test that asks the code what its limit is cannot notice the limit moving.
    BLIND_REVERSE_CEILING_M = 0.30
    BLIND_REVERSE_CEILING_S = 2.00

    AUTO = {"speed": 1.00, "stop_distance": 0.35, "avoid_distance": 0.85, "clear_distance": 1.60}
    BOXED_IN_M = 0.20
    CLEAR_M = 3.00

    def setUp(self):
        self.control = server.control
        self.clock = 10_000.0
        self.control._auto_state = self.control._new_auto_state()

    def tearDown(self):
        self.control._auto_state = self.control._new_auto_state()

    @contextlib.contextmanager
    def _driving(self, measured_speed=None):
        """Run the planner on a clock this test owns, with a base driver listening.

        The state machine is written in wall clock seconds, so the only way to
        watch a two second manoeuvre without waiting two seconds is to hand it
        a clock. Nothing else in this test class touches the network.
        """
        self.measured_speed = measured_speed
        with mock.patch.object(server.time, "monotonic", side_effect=lambda: self.clock), mock.patch.object(
            self.control, "count_subscribers", return_value=1
        ):
            yield

    def _scan(self, front_m):
        return {
            "updated_at": self.clock,
            "sectors": {"front": {"near_m": front_m, "count": 40, "median_m": front_m}},
            "gap_samples": [],
        }

    def _depth_source(self):
        return {
            "camera": "realsense",
            "feed_id": "realsense_depth",
            "stale_s": server.REALSENSE_STALE_S,
            "age_s": 0.05,
            "fresh": True,
            "depth": {
                "updated_at": self.clock - 0.05,
                "frames": 40,
                # Clear ahead as far as the camera is concerned. The LiDAR is
                # what boxes the car in here, so the depth veto stays out of it.
                "obstacle_p20_m": 1.40,
                "above_floor_near_m": 1.60,
                "above_floor_close_pixels": 0,
                "obstacle_valid_ratio": 0.60,
                "above_floor_valid_ratio": 0.60,
            },
        }

    def _feedback(self):
        if self.measured_speed is None:
            return {"speed_mps": None, "updated_at": 0.0}
        return {"speed_mps": self.measured_speed, "updated_at": self.clock}

    def _drive(self, seconds, front_m, dt=0.05):
        """Tick the planner at 20 Hz and record every command it produced."""
        samples = []
        for _ in range(int(round(seconds / dt))):
            self.clock += dt
            msg, decision = self.control._compute_auto_command(
                self._scan(front_m),
                dict(self.AUTO),
                self._depth_source(),
                state=dict(self.control._auto_state),
                update_state=True,
                feedback=self._feedback(),
            )
            samples.append({"dt": dt, "linear_x": msg.linear.x, "state": decision["auto_state"], "action": decision["action"], "reason": decision["reason"]})
        return samples

    @staticmethod
    def _reversing(samples):
        return [sample for sample in samples if sample["linear_x"] < 0.0]

    @classmethod
    def _blind_seconds(cls, samples):
        return sum(sample["dt"] for sample in cls._reversing(samples))

    @classmethod
    def _blind_metres(cls, samples):
        # The commanded speed is the upper bound on the real one: the on car
        # sweep measured 0.25 commanded as 0.227 travelled, 0.50 as 0.480, and
        # never faster than asked.
        return sum(abs(sample["linear_x"]) * sample["dt"] for sample in cls._reversing(samples))

    def test_one_reverse_is_bounded_in_both_time_and_distance(self):
        with self._driving():
            samples = self._drive(6.0, self.BOXED_IN_M)
        self.assertTrue(self._reversing(samples), "the car should still try to back out of a dead end")
        self.assertLessEqual(self._blind_seconds(samples), self.BLIND_REVERSE_CEILING_S)
        self.assertLessEqual(self._blind_metres(samples), self.BLIND_REVERSE_CEILING_M)

    def test_being_more_stuck_never_buys_a_longer_reverse(self):
        """Half a minute boxed in, which used to be attempt after growing attempt."""
        with self._driving():
            samples = self._drive(30.0, self.BOXED_IN_M)
        self.assertLessEqual(
            self._blind_metres(samples),
            self.BLIND_REVERSE_CEILING_M,
            "repeated attempts must share one budget, not each get their own",
        )
        self.assertLessEqual(self._blind_seconds(samples), self.BLIND_REVERSE_CEILING_S)

    def test_the_reverse_does_not_grow_from_one_attempt_to_the_next(self):
        with self._driving():
            samples = self._drive(30.0, self.BOXED_IN_M)
        runs = []
        for sample in samples:
            if sample["linear_x"] < 0.0:
                if not runs or runs[-1] is None:
                    runs.append(0.0)
                runs[-1] += sample["dt"]
            elif runs and runs[-1] is not None:
                runs.append(None)
        runs = [run for run in runs if run is not None]
        for earlier, later in zip(runs, runs[1:]):
            self.assertLessEqual(round(later, 3), round(earlier, 3), f"reverse runs grew: {runs}")

    def test_a_spent_budget_stops_the_car_instead_of_reversing_further(self):
        with self._driving():
            samples = self._drive(30.0, self.BOXED_IN_M)
        tail = samples[-40:]
        self.assertTrue(all(sample["linear_x"] == 0.0 for sample in tail), "a car with no budget left must sit still")
        self.assertTrue(
            all("blocked" in sample["action"] for sample in tail),
            f"the operator has to be told why it stopped, got {tail[-1]['action']}",
        )

    def test_the_reverse_speed_is_low_and_ignores_the_cruise_speed(self):
        """Cruise speed defaults to 1.0 m/s, and the reverse used to scale with it."""
        with self._driving():
            samples = self._drive(6.0, self.BOXED_IN_M)
        fastest = max(abs(sample["linear_x"]) for sample in self._reversing(samples))
        self.assertLessEqual(fastest, 0.16, "reversing blind is not the moment to hurry")
        self.assertLess(fastest, self.AUTO["speed"] * 0.45, "the reverse must not scale with the cruise speed")

    def test_a_car_rolling_faster_than_commanded_ends_its_reverse_sooner(self):
        """The distance bound is what catches a slope the time bound cannot."""
        with self._driving():
            slow = self._blind_seconds(self._drive(30.0, self.BOXED_IN_M))
        self.setUp()
        with self._driving(measured_speed=0.60):
            fast = self._blind_seconds(self._drive(30.0, self.BOXED_IN_M))
        self.assertLess(fast, slow, "measured travel has to be able to end the reverse before the clock does")

    def test_a_reacquired_corridor_earns_a_fresh_budget(self):
        """A budget spent is spent for that episode, not for the rest of the drive."""
        with self._driving():
            first = self._drive(30.0, self.BOXED_IN_M)
            self.assertTrue(self._reversing(first))
            self._drive(4.0, self.CLEAR_M)
            self.assertEqual(self.control._auto_state["name"], "cruise", "a clear corridor ends the episode")
            second = self._drive(6.0, self.BOXED_IN_M)
        self.assertTrue(self._reversing(second), "a car that got itself clear may try again")
        self.assertLessEqual(self._blind_metres(second), self.BLIND_REVERSE_CEILING_M)

    def test_a_stalled_control_loop_does_not_buy_extra_blind_travel(self):
        """A gap in the loop is not a gap in the car's motion.

        Nothing stops when this loop stops. The last command published stands
        until the base gives up on it, so a tick that arrives late has to be
        charged for every second of the gap, not for one tick's worth.

        The gap here is set so that only the distance charge can end the
        reverse: the clock has not run out and the state's own deadline has not
        passed, so a version that under charges the stall keeps reversing.
        """
        with self._driving(measured_speed=0.20):
            self._drive(0.35, self.BOXED_IN_M)
            self.assertEqual(self.control._auto_state["name"], "reverse_escape")
            self.clock += 1.20
            self._drive(0.05, self.BOXED_IN_M)
        state = self.control._auto_state
        self.assertLess(state["reverse_used_s"], server.AUTO_REVERSE_BUDGET_S, "the time budget must not be what ended this")
        self.assertGreaterEqual(state["reverse_used_m"], server.AUTO_REVERSE_BUDGET_M)
        self.assertEqual(state["name"], "blocked", "a second of unwatched reverse is still a second of reverse")

    def test_nothing_extends_a_reverse_for_being_stuck_any_more(self):
        self.assertFalse(
            hasattr(server, "AUTO_STUCK_EXTRA_REVERSE_S"),
            "the constant that made a more stuck car reverse further is the bug",
        )

    def test_the_configured_budget_respects_the_ceiling(self):
        """An environment variable must not be able to undo a safety bound."""
        self.assertLessEqual(server.AUTO_REVERSE_BUDGET_S, self.BLIND_REVERSE_CEILING_S)
        self.assertLessEqual(server.AUTO_REVERSE_BUDGET_M, self.BLIND_REVERSE_CEILING_M)
        self.assertLessEqual(server.AUTO_REVERSE_SPEED_MAX * server.AUTO_REVERSE_BUDGET_S, server.AUTO_REVERSE_BUDGET_M)

    def test_the_budget_is_reported_so_an_operator_can_watch_it_drain(self):
        with self._driving():
            samples = self._drive(2.0, self.BOXED_IN_M)
        self.assertTrue(self._reversing(samples))
        state = self.control._auto_state
        self.assertIn("reverse_used_s", state)
        self.assertIn("reverse_used_m", state)
        self.assertGreater(state["reverse_used_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
