from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from collie_adapter import BorderCollieAdapter, VoiceIntent, interpret_command


class _Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.requests.append((self.path, payload))
        body = json.dumps({"status": "accepted"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


class InterpretCommandTests(unittest.TestCase):
    def test_accepts_only_the_supported_fruit_phrases(self) -> None:
        self.assertEqual(
            interpret_command("Go to the pear"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )
        self.assertEqual(
            interpret_command("go to apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )
        self.assertIsNone(interpret_command("go forward"))
        self.assertIsNone(interpret_command("go to banana"))

    def test_accepts_natural_find_requests_and_the_pear_homophone(self) -> None:
        self.assertEqual(
            interpret_command("can you find a pair for me uh"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )
        self.assertEqual(
            interpret_command("please locate the red apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )

    def test_does_not_trigger_on_a_casual_or_ambiguous_fruit_mention(self) -> None:
        self.assertIsNone(interpret_command("I like pears"))
        self.assertIsNone(interpret_command("find an apple and a pear"))

    def test_accepts_a_narrow_stop_vocabulary(self) -> None:
        self.assertEqual(interpret_command("stop the demo"), VoiceIntent("stop_demo"))


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.adapter = BorderCollieAdapter(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_starts_disarmed(self) -> None:
        result = self.adapter.dispatch("go to pear")
        self.assertIn("disarmed", result["error"])
        self.assertEqual(_Handler.requests, [])

    def test_armed_activation_uses_the_existing_demo_api(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch("go to pear")
        self.assertEqual(
            _Handler.requests,
            [("/api/run", {"target_fruit": "pear", "activation_source": "voice"})],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")


if __name__ == "__main__":
    unittest.main()
