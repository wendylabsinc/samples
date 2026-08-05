"""Small, explicit voice-command surface for the Border Collie demo."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VoiceIntent:
    action: str
    target_fruit: Optional[str] = None


def interpret_command(text: str) -> Optional[VoiceIntent]:
    """Interpret only the deliberately supported stage phrases."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    match = re.fullmatch(r"go to (?:the )?(pear|apple)", normalized)
    if match:
        return VoiceIntent(action="activate_demo", target_fruit=match.group(1))
    if normalized in {"stop", "stop demo", "stop the demo"}:
        return VoiceIntent(action="stop_demo")
    return None


class BorderCollieAdapter:
    """Dispatch allowlisted intents through the demo's existing HTTP safety gates."""

    def __init__(self, base_url: str, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> None:
        with self._lock:
            self._armed = True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False

    def dispatch(self, text: str) -> dict:
        intent = interpret_command(text)
        if intent is None:
            return {"calls": [], "error": "unsupported voice command"}
        if not self.armed:
            return {
                "calls": [],
                "error": "voice actions are disarmed; review the transcript and arm them in the UI",
            }

        if intent.action == "activate_demo":
            payload = {"target_fruit": intent.target_fruit, "activation_source": "voice"}
            response = self._post("/api/run", payload)
            return {
                "calls": [{
                    "tool": "activate_demo",
                    "args": payload,
                    "result": _response_summary(response),
                }]
            }

        response = self._post("/api/stop", {})
        return {
            "calls": [{
                "tool": "request_stop",
                "args": {},
                "result": _response_summary(response),
            }]
        }

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Border Collie rejected the command ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Border Collie API is unavailable: {exc.reason}") from exc
        if not body:
            return {}
        return json.loads(body)


def _response_summary(response: dict) -> str:
    for key in ("message", "status", "state", "run_id"):
        value = response.get(key)
        if value is not None:
            return str(value)
    return "accepted"
