"""Minimal Streamable HTTP MCP client.

Adapted from walter-monitor's `monitor/mcpclient.py` so this demo speaks to the
studio's real MCP servers (wendystudio-lights, wendystudio-essentials) exactly
the way Walter does, rather than inventing a second dialect. Deliberately a
close copy: the transport details (protocol header, session id, SSE-or-JSON
response bodies) are the part that must match.

Only initialize, tools/list and tools/call are implemented - the whole of what
driving tools needs.

Those servers listen on the device itself (host networking, 127.0.0.1:3000 and
:3001) and do not answer from other machines, so an app that wants them runs on
the same device.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


def _parse_body(body: str) -> dict:
    """Accept both a plain JSON response and a server-sent-event stream."""
    stripped = body.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        return json.loads(stripped)
    for line in stripped.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                return json.loads(data)
    raise MCPError(f"could not parse MCP response: {stripped[:200]}")


class MCPClient:
    def __init__(self, url: str, timeout: float = 20.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session: str | None = None
        self._id = 0

    # -- transport ----------------------------------------------------------

    def _post(self, method: str, params: dict) -> dict:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self._session = session
        except urllib.error.HTTPError as exc:
            detail = exc.read(400).decode("utf-8", errors="replace")
            raise MCPError(f"MCP server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MCPError(f"MCP server unreachable at {self.url}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            # urllib only wraps CONNECTION failures in URLError; a timeout while
            # reading comes out bare. This class promises callers MCPError only.
            raise MCPError(f"MCP request to {self.url} failed: {exc}") from exc

        parsed = _parse_body(body)
        if "error" in parsed:
            raise MCPError(f"{method} failed: {parsed['error']}")
        return parsed.get("result", {})

    # -- protocol -----------------------------------------------------------

    def initialize(self) -> dict:
        result = self._post(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "wendy-voice-demo", "version": "1.0.0"},
            },
        )
        # Fire and forget; a 202 with no body is normal.
        try:
            request = urllib.request.Request(
                self.url,
                data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": PROTOCOL_VERSION,
                    **({"Mcp-Session-Id": self._session} if self._session else {}),
                },
                method="POST",
            )
            urllib.request.urlopen(request, timeout=self.timeout).close()
        except Exception:
            pass
        return result

    def list_tools(self) -> list[dict]:
        tools = self._post("tools/list", {}).get("tools")
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict) -> Any:
        return self._post("tools/call", {"name": name, "arguments": arguments})


class MultiMCP:
    """Several MCP servers behind one tool namespace.

    Mirrors how Walter aggregates wendystudio-lights and wendystudio-essentials:
    tools are collected from every server that answers, and a call is routed to
    whichever server declared that tool. A server that is down costs only its own
    tools, never the whole set.
    """

    def __init__(self, urls: list[str], timeout: float = 20.0) -> None:
        self.clients = [MCPClient(url, timeout=timeout) for url in urls]
        self._owner: dict[str, MCPClient] = {}
        self.tools: list[dict] = []
        self.errors: list[str] = []

    def refresh(self) -> list[dict]:
        self._owner.clear()
        self.tools = []
        self.errors = []
        for client in self.clients:
            try:
                client.initialize()
                for tool in client.list_tools():
                    name = tool.get("name")
                    if not name or name in self._owner:
                        continue
                    self._owner[name] = client
                    self.tools.append(tool)
            except Exception as exc:
                self.errors.append(f"{client.url}: {exc}")
        return self.tools

    def call_tool(self, name: str, arguments: dict) -> Any:
        client = self._owner.get(name)
        if client is None:
            raise MCPError(f"no MCP server provides tool {name!r}")
        return client.call_tool(name, arguments)

    def to_ollama_tools(self) -> list[dict]:
        """The tool list in the function-calling shape Ollama expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description") or "",
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
            if t.get("name")
        ]
