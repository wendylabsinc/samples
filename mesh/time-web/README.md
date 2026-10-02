# time-web

"What is the current time?" as an auto-refreshing HTML page on the
conventional `_http._tcp` service — published through **avahi-compat
(`dns_sd.h`)**, not raw sockets. This is the reference app for the `avahi`
entitlement path.

- `main.go` — HTTP route (stdlib) + `DNSServiceRegister` via cgo, with
  one registration attempt, confirmed by a single-owner callback pump
- `avahi_shim.c` — C trampoline (Go functions can't be C callbacks directly)
- `Dockerfile` — production (B2): slim image, client lib only; the **agent**
  provides the per-app avahi-daemon and filtered bus (see `wendy.json`)
- `Dockerfile.dev` — same app plus container-local dbus + avahi-daemon for
  testing without an agent (entrypoint requires daemon readiness, then starts the app)
- `wendy.json` — **mesh** network mode (own netns, *not* host) + published
  TCP port 8081 + `{ "type": "avahi" }`

Reachability and acceptance:

- Mesh peers: use `time-browser` to discover and fetch the page.
- Operator PC on the host LAN: optional `http://<device-lan-ip>:8081/` check;
  candidate reachability has not been verified.
- LAN mDNS browsers (`dns-sd -B _http._tcp` on the PC): will **not** list
  it — the platform does not reflect mesh records onto the physical LAN.

```bash
# logic test without an agent (dev image bundles a local daemon):
docker build -f Dockerfile.dev -t wendy-time-web-dev .
docker network create wendydemo
docker run -d --name web --network wendydemo -p 127.0.0.1:8081:8081 wendy-time-web-dev
curl http://127.0.0.1:8081/
```

Device deployment needs the B2 agent (branch `golden/mesh-avahi-b2`,
unmerged) and matching CLI. Follow [the tour](../MESH-TOUR.md) for exact
deployment and HTML observer commands.

The platform assigns the address and prepares private Avahi before the app
entrypoint executes. This app reads IPv4 once and registers once. It waits
only for DNS-SD's asynchronous registration callback; unavailable networking
or transport fails immediately.

For a run-specific service alias, set TIME_SERVICE_HOSTNAME. Without this
override, the app uses the agent-supplied WENDY_DEVICE_HOSTNAME.
