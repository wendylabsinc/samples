# SLAM viewer panel and bridge for the Rosmaster A1 web remote — design

Date: 2026-09-18. Status: **approved by Ethan** (designed section by section
on 2026-09-18; every section was confirmed before this file was written).
Linear: WDY-1637 (bridge: expose map, pose, trajectory and scan to the web
viewer), WDY-1638 (viewer UI, delivered as an embedded panel; Ethan confirmed with
the initiative's creator on 2026-09-18 that a panel satisfies it), WDY-1639 (served from the robot, met by construction).
Branch `slam-viewer-panel`, stacked on `slam-service` (Samples PR #28), which
carries the `slam` service and the `/slam/*` contract this builds on.

## Goal

Show the live SLAM map in the browser page that already drives the car: the
occupancy grid, the robot's pose in it, its trajectory and the current LiDAR
scan, updating while the car drives, with every "why is it blank" state
written on the panel itself. Acceptance bar: with the five services running,
an operator opens the remote, sees the room map with the robot marker
moving through it and the trajectory trailing behind, can pan and zoom to
inspect it, and when slam_toolbox restarts or the slam service is stopped the
panel says so within a few seconds and recovers on its own when it returns.

## Decisions that shaped this

- **Embedded panel, not a standalone app.** WDY-1638's June wording asks for
  a Vite React Three Fiber app; Ethan confirmed with the initiative's creator
  on 2026-09-18 that a panel in the existing remote page satisfies the
  initiative. The design still keeps a later conversion cheap: the
  bridge is a module with no dependency on the drive code, the panel's client
  code is a module with a pure layer and a thin render layer, and the HTTP
  API is documented as the contract. A standalone viewer consumes the same
  routes unchanged; a second Wendy app reuses `slam_bridge.py` behind a small
  HTTP shell, because its container subscribes to the same DDS topics over
  host networking exactly as the `slam` service does.
- **Polling, not a websocket.** The 2026-08 "car freezes" post-mortem traced
  four causes, every one a long-lived browser connection (TLS handshake on
  the accept thread, the six-per-origin socket budget, Safari pinning MJPEG
  sockets). The page now has zero long-lived connections and this design
  keeps it that way: every SLAM response is finite, and the panel's whole
  fetch chain runs behind one in-flight guard so the panel costs exactly one
  socket. A 4 Hz polled pose is indistinguishable from a 20 Hz pushed one on
  a map at driving speed.
- **2D canvas, not WebGL.** No dependencies, no build step, the same pattern
  as the existing LiDAR canvas, and testable in the `node:vm` harness. A
  later React Three Fiber rewrite replaces only the render layer.
- **Pose from `/tf`, not `/pose`.** slam_toolbox publishes `/pose` once per
  processed scan (every 0.2 m or 0.2 rad of travel), so at rest and between
  scans there is nothing. Composing `map -> odom` (slam, 20 Hz) with
  `odom -> base_link` (odometry, 24 Hz) gives a pose that moves with the
  wheels and is corrected whenever slam_toolbox corrects.

## Non-goals

Stated so they are not reintroduced by accident:

- Websocket or server-sent-events transport.
- A standalone Vite/React Three Fiber app, a sixth service or a second app
  (the conversion path is described at the end, not built).
- Driving the car from the map, waypoints, goals.
- A save-map button (`/slam_toolbox/save_map` works from the CLI, see the
  README).
- Localization mode, scan filtering, map editing.
- Touch pinch zoom (wheel and buttons only; the remote is used on laptops).
- Running the web container inside the offline bag harness
  (`scripts/slam_offline_check.sh`); it would add only DDS plumbing the five
  services already exercise daily.
- 3D rendering; the map is a plane.

## Part 1 — the bridge (`rosmaster-a1-web-remote-wendy/app/slam_bridge.py`)

A class `SlamBridge(node, clock=time.monotonic)` that registers its
subscriptions on whatever rclpy node it is given and exposes snapshot
methods for the HTTP layer. In the web service the node is the existing
`RosmasterControl` instance, so no new DDS participant is created (the
participant-index pressure noted in the slam spec does not grow). In a
standalone service it would be that service's own node. The module imports
rclpy QoS types, the message packages, numpy and Pillow, and nothing from
`server.py` or `direct_gamepad.py`.

`server.py` constructs `slam_bridge = SlamBridge(control)` at module scope
next to `control`, and the three new routes in `Handler.do_GET` call it. The
API tests substitute `server.slam_bridge` the way they substitute
`server.control`.

### Subscriptions and what each keeps

All writes happen on the ROS spin thread under one `threading.Lock`; HTTP
threads copy out under the same lock. Ages are computed from receipt times
(`clock()` at callback), never from message header stamps, matching how the
rest of the server measures freshness.

| Topic | Type | QoS | Kept |
|---|---|---|---|
| `/tf` | `tf2_msgs/TFMessage` | depth 100 | `map -> odom` (x, y, yaw, received_at) and `odom -> base_link` (x, y, yaw, received_at); other transforms ignored |
| `/map` | `nav_msgs/OccupancyGrid` | reliable, transient local, depth 1 | PNG bytes, version, width, height, resolution, origin x/y/yaw, received_at |
| `/scan` | `sensor_msgs/LaserScan` | sensor data | at most 360 Cartesian points in `base_link`, received_at |
| `/slam/status` | `std_msgs/String` (JSON) | depth 10 | the parsed object verbatim, received_at |
| `/slam/trajectory` | `nav_msgs/Path` | reliable, transient local, depth 1 | epoch, append-only point list for the epoch, received_at |

The bridge subscribes to `/scan` itself rather than reading the drive code's
downsample, so the module stays independent; a second subscription on the
same node is one more callback at ~10 Hz and no new participant.

### Pose composition

On every `/tf` message that carries either transform, recompute
`map -> base_link`:

```
x   = x_mo + cos(yaw_mo) * x_ob - sin(yaw_mo) * y_ob
y   = y_mo + sin(yaw_mo) * x_ob + cos(yaw_mo) * y_ob
yaw = wrap(yaw_mo + yaw_ob)            # wrap to (-pi, pi]
```

Yaw from a quaternion is `atan2(2(wz + xy), 1 - 2(y² + z²))`, reimplemented
here (not imported from the keeper, which is another service). The pose is
`None` until both transforms have arrived at least once. `pose.age_s` is the
age of the most recent recomposition; the staleness of each input is
reported separately through `reason` (see "State derivation"), so a pose
that is coasting on odometry while `map -> odom` is stale is still drawn but
labelled.

### Map encoding

On each `/map`:

1. Reject and log (once per received grid, keeping the previous map) if
   `len(data) != width * height` or either side exceeds
   `SLAM_MAP_MAX_SIDE = 4096`.
2. `cells = np.asarray(msg.data, dtype=np.int8).reshape(height, width)`.
3. Palette index per cell: `-1` (or any negative) → 0 unknown; `0..49` → 1
   free; `50..100` → 2 occupied.
4. Flip rows (`[::-1]`) so image row 0 is the grid's highest-y row (north
   up); build a Pillow `P`-mode image, attach the three-colour palette, save
   PNG. Colours: unknown `#101513` (the page's panel background), free
   `#253029`, occupied `#dfe6e2`.
5. Store bytes, `version += 1`, metadata, `received_at`.

Cost: a 400 × 400 grid encodes in single-digit milliseconds; slam_toolbox
republishes about once a second while scans arrive. No viewer lease is
needed (unlike camera frames, which encode at 30 Hz).

### Scan

Keep every return that is finite and inside `(max(0.02, range_min),
range_max]`, then take every k-th so that at most `SLAM_SCAN_MAX_POINTS =
360` remain. Convert to Cartesian in the laser frame, which is `base_link`
because the lidar service publishes `laser_frame` as a static identity (README,
"SLAM topics"). Round to centimetres. Stored as a flat list
`[x0, y0, x1, y1, ...]`.

### Trajectory epochs

The keeper republishes the whole `nav_msgs/Path` on each new pose: append-only
in normal operation, trimmed to the newest 5000, and restarted when a new
session opens (odometry reset, slam_toolbox restart). The bridge turns that
into an append-only list per **epoch**, so indices are stable and fetches can
be incremental:

- First message ever: `epoch = 1`, list = its poses.
- Later message: search it from the end for the stored last point (exact
  float equality on raw x and y, which are the keeper's own doubles
  republished). Found at index k → append `poses[k+1:]`. Not found (a reset,
  or an empty path after a non-empty one) → `epoch += 1`, list = its poses.
- If the list would exceed `SLAM_TRAJECTORY_MAX_POINTS = 20000` (1 km at
  5 cm spacing), `epoch += 1` and the list restarts from the message's own
  poses. Logged.
- The list is stored as `(x, y)` rounded to centimetres; the raw, unrounded
  last point is kept separately and is what the match above compares against.
  `epoch = 0, count = 0` before any message.

An epoch change is logged with the old and new counts.

### State derivation

Computed at snapshot time, in this order:

1. No `/slam/status` ever, or its age > `SLAM_STATUS_STALE_S = 3.0` →
   `slam_unreachable`, reason `"no /slam/status yet"` or
   `"no /slam/status for 4.2 s"`. This is the slam service not deployed,
   crashed or unreachable, distinct from the keeper saying `slam_down`.
2. Otherwise take the keeper's `state`. `mapping` with no grid received yet
   becomes `waiting_for_map`. Known states pass through: `slam_down`,
   `waiting_for_scan`, `waiting_for_odom_tf`, `mapping`. An unknown string
   passes through unchanged (forward compatible with keeper changes).
3. `reason` (independent of state) names the stalest input over its
   threshold, or `null`: `map -> odom` and `odom -> base_link` over
   `SLAM_TF_STALE_S = 2.0`, scan over `SLAM_SCAN_STALE_S = 2.0`, map over
   `SLAM_MAP_STALE_S = 10.0` while the state is `mapping`. Format
   `"map -> odom 4.1 s old"`.

The bridge logs a line when `state` changes (`slam_bridge: mapping ->
slam_unreachable (no /slam/status for 3.2 s)`), never per poll, and does not
log reason-only changes.

### Threading and lifetime

Callbacks run on the single `rclpy.spin(control)` thread, as every other
subscription in the service does. The PNG bytes and point lists handed to
HTTP threads are immutable objects replaced wholesale under the lock, so a
response never observes a half-written map. No background threads, no
timers.

## Part 2 — the HTTP contract

Three GET routes on the existing HTTP (8091) and HTTPS (8443) servers. Every
response is finite, sends `Content-Length`, `Cache-Control: no-store`, and
serialises JSON with sorted keys like `_send_json`. POST to any of them is
404 like every other unknown route. Metres everywhere, rounded to
centimetres; radians for yaw. Request lines are logged like every other
route's.

### `GET /api/slam`

One JSON object, about 4 KB with a full scan, polled by the panel at 4 Hz.

```json
{
  "ok": true,
  "bridge": {"state": "mapping", "reason": null},
  "slam": { ...the keeper's /slam/status object verbatim... },
  "slam_age_s": 0.4,
  "pose": {"x": 1.23, "y": -0.45, "yaw": 1.52, "age_s": 0.03},
  "map_odom": {"x": 0.81, "y": 0.12, "yaw": 0.27, "age_s": 0.05},
  "map": {"version": 41, "width": 248, "height": 142, "resolution": 0.05,
          "origin": {"x": -6.2, "y": -3.55, "yaw": 0.0}, "age_s": 0.7},
  "scan": {"age_s": 0.08, "points": [0.52, 0.01, 0.53, 0.03]},
  "trajectory": {"epoch": 2, "count": 412}
}
```

- `bridge.state` is one of `slam_unreachable`, `waiting_for_scan`,
  `waiting_for_odom_tf`, `waiting_for_map`, `mapping`, `slam_down` (or an
  unknown keeper string passed through). `bridge.reason` is a string or null.
- `slam` is null and `slam_age_s` null before the first `/slam/status`.
- `pose`, `map_odom`, `map`, `scan` are each null until their inputs exist.
- `map.version` is the only field that should trigger a PNG fetch.
- `scan.points` is a flat `[x, y, ...]` list in `base_link`.
- `trajectory.epoch` is 0 and `count` 0 before the first path.

### `GET /api/slam/map.png`

The cached PNG. Headers: `Content-Type: image/png`, `ETag: "<version>"`,
`X-Map-Version`, `X-Map-Width`, `X-Map-Height`, `X-Map-Resolution`,
`X-Map-Origin-X`, `X-Map-Origin-Y`, `X-Map-Origin-Yaw`. The client places
the image from **these** headers, never from an earlier snapshot, so image
and metadata can never disagree. `If-None-Match` equal to the current ETag →
304 with no body. No map yet → 404, which the panel treats as "waiting for
map", not as a failure.

### `GET /api/slam/trajectory?epoch=E&from=N`

```json
{"epoch": 2, "from": 400, "total": 412, "points": [1.20, -0.41, 1.25, -0.40]}
```

- `epoch` matches the server's → `points = list[from:]`, `from` clamped to
  `[0, total]`.
- `epoch` missing, malformed or different → the whole list from 0 under the
  current epoch, so a client that has fallen behind or seen a reset
  resynchronises in one request.
- Before any path: `{"epoch": 0, "from": 0, "total": 0, "points": []}`.

## Part 3 — the panel

### Placement and markup

A new `<section class="panel">` "Map" in the main column's `.stack`,
directly beneath the camera gallery and above the Controller panel, so it
gets the wide column rather than the 360 px drive sidebar.

- Title row: `Map`, a state pill `#slamState` (text such as `mapping`,
  `waiting for map`, `SLAM service not running`), and `#slamStats`
  (`3 saves, 412 poses`).
- Body: `<canvas id="slamCanvas" class="lidar-canvas">` (same full-width,
  1.6 aspect, dark style as the LiDAR canvas), a toolbar with a Follow
  checkbox `#slamFollow` (checked by default), a Reset view button
  `#slamReset`, zoom buttons `#slamZoomIn` and `#slamZoomOut`, and a Hide
  toggle `#slamHide`; and one
  readout line `#slamReadout`: `x 1.23 m  y -0.45 m  heading 87°  map 12.4 ×
  7.1 m`, with the bridge `reason` appended in amber when present.
- Loaded by a third script tag, `/static/slam.js`, before `app.js`, in the
  same plain-script style as `gamepad.js` (no modules, no build step). The
  Dockerfile already copies `app/static/`.

### `slam.js` layers

**Pure layer** (no DOM, no fetch, no globals from the page), every function
callable from `node --test`:

- `slamReduce(model, event)`: folds `{type: "snapshot", body}`,
  `{type: "map", meta, image}`, `{type: "trajectory", body}`,
  `{type: "failure"}`, `{type: "hidden", hidden}`, `{type: "drag", dx, dy}`,
  `{type: "wheel", factor, atX, atY}`, `{type: "follow", on}`,
  `{type: "reset"}` into a new view model.
- `slamView(model, width, height)`: the world-to-canvas transform
  (`px = (wx - cx) * scale + W/2`, `py = H/2 - (wy - cy) * scale`, so y is
  up), and its inverse for zoom-about-cursor.
- `slamPlan(model)`: which fetches the next cycle needs: always the
  snapshot; the PNG only if the snapshot's `map` is non-null and its
  `version` differs from the image the model holds (a null `map` keeps the
  old image, see "Edge cases"); the trajectory only if the snapshot's epoch or count differs from
  the list the model holds.
- `slamMerge(list, reply)`: same epoch → append `reply.points` from
  `reply.from` (truncating first if the local list is longer); different
  epoch → replace.
- `slamOverlay(model)`: state → `{title, detail}` or null.

**DOM layer**: `drawSlam(ctx, model, W, H)`, pointer and wheel handlers that
dispatch reducer events, `refreshSlam()` and its scheduler. `app.js` only
calls `startSlamPanel(els)` once at load.

### Drawing

North-up. In order: background `#080a09`; the map PNG via `drawImage`
inside a transform that translates to the origin, rotates by origin yaw and
flips y, so the image's `width * resolution` metres line up with world
metres; the trajectory as a 1.5 px `#f0b429` polyline; the scan as 2 px
`#58c897` squares transformed by the pose; the robot as a filled triangle
0.30 m long by 0.20 m wide in world units with a minimum of 10 px, `#eef2ef`
fill; a scale bar bottom-left, 1 m when the scale is at least 40 px/m,
otherwise 5 m, with a label.

Follow keeps `(cx, cy)` on the pose on every snapshot. Default scale fits the
map bounds with 10 % margin, clamped to `[20, 400]` px/m; before the first
map, 60 px/m centred on the pose or the origin.

When `state` is anything but `mapping` the map, trajectory and robot are drawn
at 50 % alpha and the overlay title is drawn centred in bold with the detail
beneath it. The panel never blanks: a slam restart keeps showing the last
map until the new one arrives.

Redraw happens once per completed poll cycle and once per interaction event,
not on an animation-frame loop.

### Interaction

- Drag pans and sets Follow off (the checkbox reflects it).
- Wheel zooms by 1.15 per notch about the cursor; the +/- buttons zoom about
  the centre.
- Follow checkbox turns following back on and recentres on the next snapshot.
- Reset view fits the map bounds (or a 5 m box around the pose if there is no
  map) and sets Follow off.
- Hide collapses the body, stops polling, and shows `hidden` in the pill;
  unhiding polls immediately.

### Polling

`refreshSlam()` every `SLAM_POLL_MS = 250` behind a single in-flight guard
that covers the whole chain, run sequentially:

1. `GET /api/slam` (timeout `FETCH_TIMEOUT_MS`, the page's existing 4 s via
   `AbortController`).
2. If `slamPlan` says so, `GET /api/slam/map.png` with `If-None-Match`;
   decode with `createImageBitmap(blob)` when available, else an `Image` on
   an object URL (revoked when replaced). 304 keeps the current image; 404
   clears it and is not a failure.
3. If `slamPlan` says so, `GET /api/slam/trajectory?epoch=E&from=N`, merged
   with `slamMerge`.
4. One redraw.

Never two requests in flight, so the panel holds exactly one browser socket.
Polling pauses while `document.hidden` or the panel is hidden, and a
`visibilitychange` back to visible polls immediately.

Any network error, timeout, non-2xx (other than the PNG 304 and 404), image
decode error or JSON parse error aborts the cycle and counts one failure; a completed cycle resets the
counter. `SLAM_UNREACHABLE_FAILURES = 3` consecutive failures show "Car not
reachable" on the canvas. **These failures do not feed the page's control
circuit breaker** (`noteControlFailure`): a slow map fetch must never blank
the camera tiles or interfere with the drive path.

## Failure handling and logging

**Panel wording** for `bridge.state`, drawn over the dimmed last map:

| state | title | detail |
|---|---|---|
| client: 3 failed polls | Car not reachable | last error text |
| `slam_unreachable` | SLAM service not running | `bridge.reason` |
| `slam_down` | slam_toolbox restarting | `bridge.reason` |
| `waiting_for_scan` | Waiting for LiDAR | |
| `waiting_for_odom_tf` | Waiting for odometry | |
| `waiting_for_map` | Waiting for first map | |
| `mapping` | (no overlay) | `reason` in the readout line if present |
| unknown string | the string itself | `bridge.reason` |

**Server logs**: bridge state transitions with reason (once per change);
rejected grids (once per received grid); trajectory epoch changes with old
and new counts. Request lines log as for every route. Nothing logs per poll
beyond the request line.

**Edge cases decided**: map origin yaw honoured in the transform (slam_toolbox
publishes 0, the code does not assume it); a trajectory reply whose epoch
differs from the client's replaces the list wholesale; the scan is not drawn
while the pose is null; a snapshot arriving with `map: null` after a map
existed (a new session before its first grid) keeps drawing the old image
dimmed under the `waiting_for_map` overlay; the PNG metadata used for
placement always comes from the PNG response's headers.

## Build, deploy, docs, tracking

- **Dockerfile** (web): add `ros-humble-nav-msgs` and `ros-humble-tf2-msgs`
  (the base image installs nav-msgs explicitly and the slam image gets both
  through slam_toolbox; the web image today has neither) and
  `COPY app/slam_bridge.py /app/slam_bridge.py`.
- **Manifest**: unchanged. The web service already has host networking and
  the ROS 2 framework; the panel is served from the same origin on 8443, so
  WDY-1639's "served from the robot, reachable on the LAN" holds by
  construction at `https://<car>.local:8443`.
- **Deploy**: `scripts/deploy_car.sh <car>.local:50052 web`.
- **README**: a "SLAM viewer" subsection after "SLAM topics" holding the
  three routes, their fields, headers and error meanings (WDY-1637's
  documented schema); one line each in "What it does", the `web` row of the
  services table, and "Driving it" for the Map panel controls.
- **Linear**: WDY-1637 → In Progress at the start of implementation, Done
  when deployed and documented. WDY-1638 → In Progress with implementation, Done when
  the panel is live-validated, with a comment stating what shipped (embedded
  2D canvas panel, scope confirmed with the initiative's creator) and the
  conversion path below. WDY-1639 → a comment with URL and
  port.
- **Budget**: one extra browser socket; ~16 KB/s of snapshots, under 50 KB per
  map at ≤ 1 Hz, trajectory deltas negligible; one PNG encode per grid on the
  car.

## Testing

Same three suites as today, written test-first, and tests that run the code
rather than match its text (see `tests/README.md` for why).

**Python, `tests/python/test_slam_bridge.py`**, the bridge with the existing
stubs (`nav_msgs`, `tf2_msgs`, `geometry_msgs`, `sensor_msgs`, `std_msgs`,
`rclpy`) and an injected clock, fed stub messages directly:

- Pose composition from known transforms, including yaw wrap; null until
  both have arrived; `age_s` from the injected clock.
- Map encoding: decode the PNG with Pillow and assert the palette colour at
  known cells (unknown, free, occupied, the 50 threshold), that image row 0 is
  the grid's highest-y row, that the version increments per grid, and that a
  grid with the wrong data length or over the side cap is rejected with the
  previous map kept and a log line.
- Scan: at most 360 points, non-finite and out-of-range returns dropped,
  Cartesian in `base_link`, centimetre rounding.
- Trajectory: extension appends from the matched last point; a head-trimmed
  path still appends; a path without the last point (or an empty one) starts
  a new epoch; the 20000 cap starts a new epoch; `epoch 0, count 0` before
  any message.
- State table: one test per row, including `waiting_for_map` refinement,
  unknown passthrough, and each `reason` string; transition logging once per
  change.

**Python, `tests/python/test_server_api.py`** additions against the real
`ThreadingHTTPServer` the suite starts, substituting `server.slam_bridge` with
a scripted fake: `/api/slam` shape and null cases; `map.png` `ETag` and
`X-Map-*` headers, 304 on matching `If-None-Match`, 404 before the first map;
`trajectory` epoch and `from` semantics including clamping and malformed
query; `Content-Length` on every response; 404 on POST to all three.

**JavaScript, `tests/web/slam.test.mjs`**. The pure layer as unit tests:
reducer transitions for every event type, the failure counter and its reset,
world-to-canvas round trips, zoom about the cursor keeps the world point under
it, follow centring, `slamPlan` decisions, `slamMerge` for same and different
epochs, `slamOverlay` per state. The DOM layer through the vm harness with a
fake `fetch` (returning scripted snapshots, PNG bodies and trajectory replies,
recording every URL) and a recording 2D context (extend the harness's fake
canvas as needed): one poll issues the snapshot, then the PNG only on a
version change, then the trajectory only on a count or epoch change; never two
requests in flight; polling stops when hidden and resumes on
`visibilitychange`; three failures render the unreachable text; a failing SLAM
poll leaves the camera tiles unsuspended and the control breaker untouched.

**Integration** is the live validation below.

## Live validation (to be done on the car, then dated here)

1. Deploy the web service; open the remote; the Map panel shows `mapping`
   with map, pose, scan and trajectory; pan, wheel zoom, Follow and Reset
   behave as specified; reloading the page repopulates within one poll from
   the latched map and trajectory.
2. Drive two minutes with turns: the trajectory grows, the map updates within
   about a second of the keeper's saves, the pose moves smoothly; camera
   tiles stay live throughout and the browser's socket count to the car
   stays at or below six (lsof, as in the 2026-08 investigation).
3. From the host shell, kill `async_slam_toolbox_node`: "slam_toolbox
   restarting" appears, then a new epoch with the trajectory restarting and
   the map continuing; the server log shows the two state transitions and
   the epoch change.
4. Stop the `rosmaster-a1_slam` container: "SLAM service not running" within
   3 s; start it again: the panel recovers without a page reload.
5. Restart the `base` container: the keeper's odometry-reset watchdog opens a
   new session; the panel shows the new epoch and the new map.
6. Record `/api/slam` response size and the web container's CPU before and
   during, for the budget line above.

## Follow-ups and the conversion path (not in this spec)

- **Standalone Vite/React Three Fiber viewer** (if WDY-1638's wording
  stands): a `frontend/` built in a `node:22-slim` stage and copied into the
  web image (the `python/web-app` sample has the pattern), served at
  `/slam/` by the same server, consuming the three routes unchanged; the pure
  layer of `slam.js` ports as ES modules; the scene replaces `drawSlam`.
  Roughly one to two days.
- **Separate Wendy app** reusing the same data: a service whose container
  runs `slam_bridge.py` on its own node behind a ~100-line HTTP shell, with
  the same three routes. Another day for Dockerfile, manifest and deploy
  script. It would not need the driving code unless the viewer must also
  drive.
- **Websocket push** for a standalone viewer, if a 4 Hz pose ever proves
  visibly coarse; the bridge's snapshot method is already the payload.
- **Map deltas**: if grids grow past a few thousand cells a side, send only
  the changed rows or a tiled PNG; not needed for an office.
- Quietening the request log for the 4 Hz poll if it drowns the service log
  in the field.
