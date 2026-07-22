# Swift drone sim in the 🕹 Sim tab — design

**Date:** 2026-07-22
**Author:** Joannis Orlandos (with Claude)
**Status:** Approved design, pre-implementation

## Goal

Run the MuJoCo drone-race sample (`samples/swift/drone/starters/drone-slalom/mujoco_drone_race.py`)
in **Swift instead of Python**, rendering live in the Wendy Sandbox **🕹 Sim tab** exactly
as the Python version does. Do this by building a reusable Swift analog of `wendymujoco.py`,
not a one-off — future sims should be authorable in Swift too.

## Background: why this is feasible

The Sim tab is **already language-agnostic**. The sim process communicates with the
renderer purely through files and a socket under `/tmp/wendy-worldsim` (overridable via
`WENDY_WORLDSIM_DIR`):

- `scene.json` — one-time scene manifest (geometry + mesh buffers), written at startup.
- `state.json` — per-frame poses, written atomically every frame.
- `control.json` — pause/step/reset/poke/ctrl, written by the Sim tab & `wendy-sim-cmd`, polled each frame.
- `ctl.sock` — AF_UNIX stream socket the Wendy AI uses to introspect/observe/drive the sim.

The browser (`wendy-sandbox/image/shell/sim.html`, Three.js) and the native app
(`wendy-sandbox/desktop-native/`, whose `SimProtocol.swift`/`SimModel.swift` already model
this protocol in Swift as a *consumer*) only ever read that JSON. Nothing in the renderer,
the file protocol, `control.json`, or `ctl.sock` is Python-specific.

What *is* Python-specific today:

- `wendy-sandbox/image/sim/wendymujoco.py` — the authoring library (MuJoCo → JSON glue).
- `wendy-sandbox/image/ai/wendy-simrun` — launches `python3 $f` with mtime live-reload.
- The `build-a-sim` skill, `Catalog.swift`, and all `sim-templates/*.py` assume Python.
- MuJoCo reaches Swift only through a **C binding we will add** (the Python `mujoco` wheel
  bundles `libmujoco.so`; nothing binds it for Swift yet).

So "Swift into the Sim tab" = a Swift MuJoCo C binding + a Swift `wendymujoco` analog +
teaching `wendy-simrun` to run a compiled Swift program + making the sandbox image able to
build/run Swift.

## Scope

**In scope**

- `CMuJoCo` — SwiftPM `systemLibrary` binding over the full `mujoco.h` (full module map,
  approved approach).
- `WendyMuJoCo` — reusable Swift library matching `wendymujoco.py`'s surface:
  `load()`/Menagerie resolve+fetch, `Handle`/`launchPassive` with `sync()`/`hud()`/`isRunning()`,
  `buildScene`→`scene.json`, `buildState`→`state.json`, `control.json` polling
  (pause/step/reset/poke/ctrl), the `ctl.sock` endpoint (act/observe/describe/get_state/set_state/reset),
  and `Scene` composition via MjSpec.
- `DroneRace` — the Swift port of the drone sample, built on `WendyMuJoCo`.
- `wendy-simrun` — auto-detect launcher: Swift source / SwiftPM dir → build-then-run with
  rebuild-on-save; prebuilt binary → run + watch its mtime; `.py` unchanged.
- Sandbox image (`wendy-sandbox/image/Dockerfile`) — add a Linux Swift toolchain and the
  MuJoCo C SDK (headers + link path).
- Catalog + `build-a-sim` skill — make a Swift drone template discoverable.

**Non-goals (v1)**

- The Rerun/📊 Viz mirror (`to_rerun`) — no official Rerun Swift SDK. Explicitly dropped in
  v1 (`rerun: false` behavior is the default); revisit later via a bridge.
- Porting the other `sim-templates/*.py` to Swift. The drone is the reference; the library
  makes the rest possible as follow-ups.
- Changing the renderer (`sim.html`, `desktop-native`) — the Swift sim must be
  byte-compatible with the existing protocol, so no renderer change is needed.

## Architecture

### Components

1. **`CMuJoCo` (systemLibrary target)** — `module.modulemap` with a `shim.h` that
   `#include <mujoco/mujoco.h>`, `link "mujoco"`. Swift calls the C API directly and reads
   `mjModel`/`mjData` fields as imported C structs (`model.pointee.ngeom`,
   `data.pointee.geom_xpos`, etc.). All of MjSpec / `mj_getState` / contacts / sensors are
   available with zero hand-written C.

2. **`WendyMuJoCo` (library target)** — the reusable authoring library. Sub-areas:
   - **Model loading**: `load(name)` and Menagerie name-map + `_resolve_model_path` +
     sparse-clone fetch (`git clone --filter=blob:none --sparse`), mirroring `wendymujoco.py`.
     Vendored dir `/opt/sandbox/mujoco-menagerie`.
   - **Scene manifest**: `buildScene(model)` → geoms (type/size/rgba, visibility filter:
     `geom_group < 3` and `rgba[3] != 0`) + deduplicated mesh vertex/face buffers.
   - **Per-frame state**: `buildState(model, data, frame)` → pose array indexed by **full
     geom index** (`geom_xpos` + `mju_mat2Quat(geom_xmat)` → `[x,y,z,qw,qx,qy,qz]`), bounded
     contacts (≤64, `mj_contactForce` magnitude), `hud`, optional `level`.
   - **`Handle` / `launchPassive`**: `sync()` (write state, poll control, service the
     endpoint, honor pause/step), `hud(...)`, `isRunning()`, `setLevel(...)`, `close()`.
   - **Control-file protocol**: read `control.json` each `sync()`; apply reset (counter),
     persistent `ctrl` setpoints (re-applied every frame), one-shot `poke` (qpos/qvel,
     counter-gated, then `mj_forward`).
   - **`ctl.sock` endpoint**: AF_UNIX stream socket; one newline-terminated JSON request →
     one JSON response. Ops: `act` (ctrl/qpos/qvel/force + optional `steps` deferred
     capture), `observe`, `describe`, `get_state`, `set_state`, `reset`. Requests are parked
     from a listener onto a thread-safe queue and executed on the sim's main loop inside
     `sync()`, so every model/data access is atomic between physics steps.
   - **`Scene`**: compose N Menagerie models + props via MjSpec attach; `build()` →
     `(model, data)` posed at home keyframes.
   - **Atomic writes**: write `path.tmp` then rename — the renderer never sees a torn file.

3. **`DroneRace` (executable target)** — the Swift port of `mujoco_drone_race.py`. Builds
   `course.xml` (string interpolation of gate boxes + `<include file="x2.xml"/>`), copies the
   resolved `skydio_x2` model dir + assets to a writable work dir, `mj_loadXML`, then runs the
   geometric controller loop (PD position → desired tilt → reduced-attitude torque → yaw hold →
   rotor mix → `mj_step` → `sync()` → HUD).
   **Linear algebra: no dependency** — the controller uses MuJoCo's own `mju_*` helpers
   (`mju_cross`, `mju_normalize3`, `mju_mulMatTVec`, `mju_mat2Quat`, `mju_quat2Mat`, dot),
   which we already bind. A tiny `Vec3`/`Mat3` value-type wrapper may sit on top for
   readability but pulls in no package.

4. **`wendy-simrun` (auto-detect)** — extend the launcher:
   - `*.py` → unchanged (`python3 $f`, mtime live-reload).
   - `*.swift` single file or a SwiftPM package dir → `swift build -c release` then run the
     product; on source change, rebuild-then-relaunch (reload is seconds, not sub-second —
     accepted). Watch the relevant source tree's mtime.
   - An executable (ELF/mach-o) → run directly, watch the binary's mtime (rebuilt out of band).
   - Preserve single-instance-per-file semantics (pidfile keyed by absolute path) and the
     `/tmp/wendy-worldsim/current` pointer used by the `build-a-sim` skill.

5. **Sandbox image** (`wendy-sandbox/image/Dockerfile`):
   - Add a Linux Swift toolchain (arm64/amd64, matching `dpkg --print-architecture`). Accepts
     a significant image-size increase — noted as a tradeoff.
   - Provide the MuJoCo C SDK (headers + shared lib) at a known prefix (e.g. `/opt/mujoco`)
     from the official DeepMind release tarball for the arch, and expose it to the module map
     / linker (`-I`, `-L`, `-rpath`). The Python `mujoco` wheel stays for the Python path.

6. **Catalog + skill** — add a Swift drone entry to `Catalog.swift` and mention the Swift
   authoring path in the `build-a-sim` skill, so it's discoverable in Library → 🕹 Sims.

### Protocol contract (must match byte-for-byte)

- **`scene.json`**: `{title, up:"z", engine:"mujoco", geoms:[{i, type, size:[…], rgba:[4],
  mesh?:name}], meshes:{name:{vert:[…], face:[…]}}}`. Geom `type` ∈ plane/sphere/capsule/
  ellipsoid/cylinder/box/mesh. Only visible geoms included; `i` is the true MuJoCo geom index.
- **`state.json`**: `{t: epoch-seconds double, frame:int, engine:"mujoco",
  pose:[[x,y,z,qw,qx,qy,qz] × ngeom], contacts:[[x,y,z,forceMag] ≤64], hud:{}, level?}`.
  `pose` is indexed by full geom index (renderer looks up `pose[scene.i]`); quaternion is
  MuJoCo `wxyz` order (renderer handles conversion).
- **`control.json`** (read): `{paused, step, reset, poke, ctrl:{name|idx: val},
  qpos:{name|idx: val}, qvel:{name|idx: val}}`. Missing file → empty.
- **`ctl.sock`**: `/tmp/wendy-worldsim/ctl.sock`, AF_UNIX SOCK_STREAM, one JSON line in / one
  JSON line out per connection; ops as above; clamps to model limits and reports
  clamped/unknown ids.

### Data flow

```
DroneRace (Swift) ──uses──> WendyMuJoCo ──C calls──> CMuJoCo ──> libmujoco
        │                        │
        │  mj_step loop           │  buildScene()  → scene.json   (once)
        │  controller math        │  buildState()  → state.json   (per frame, atomic)
        │                         │  read control.json            (per sync)
        │                         │  ctl.sock listener ⇄ main-loop queue
        ▼                         ▼
                    /tmp/wendy-worldsim/{scene,state,control}.json + ctl.sock
                                   │
                       Caddy /simslot/<slot>/… + ctl-server
                                   │
                    sim.html (Three.js)  &  desktop-native (Swift renderer)
```

### Repo layout

Work spans two repos.

- **`wendy/samples`** — `samples/swift/drone/`: the SwiftPM package (`CMuJoCo`,
  `WendyMuJoCo`, `DroneRace`), plus the existing `starters/drone-slalom/`. This is the
  developer-facing sample.
- **`wendy/wendy-sandbox`** — where the sim actually runs: the same `WendyMuJoCo` library
  (or a vendored/packaged copy), `wendy-simrun` changes, `Dockerfile` toolchain + MuJoCo SDK,
  `Catalog.swift`, and a `sim-templates/` Swift drone entry.
  *Open decision (see below): whether `WendyMuJoCo` lives once and is shared, or is authored
  in the sandbox and mirrored to samples.*

## Error handling

- Follow `wendymujoco.py`'s "best-effort, never break the sim" posture: a failed control
  read, a bad `ctl` key, a socket error, or a mesh-extraction failure is logged and skipped,
  never fatal.
- `mj_loadXML` failure surfaces the MuJoCo error buffer and exits with a clear message.
- Missing `libmujoco`/headers is a build/link error caught by CI on the image, not at runtime.
- Atomic file writes prevent torn reads regardless of renderer timing.

## Testing strategy

- **Protocol golden tests**: run the Swift drone and the Python drone against the same model;
  assert `scene.json` geom/mesh structure and `state.json` shape match (field names, pose
  length = ngeom, quaternion order). Reuse `desktop-native`'s `SimProtocolTests`/`SimModelTests`
  as the schema oracle.
- **Unit tests** (`WendyMuJoCo`): visibility filter, mesh dedup, mat→quat, control parsing,
  `ctl.sock` op round-trips (describe/observe/act/get_state/set_state) against a tiny MJCF.
- **Controller parity**: with a fixed seed/keyframe and disabled control input, compare
  Swift vs Python gate-times / trajectory within tolerance over N steps.
- **simrun**: unit the auto-detect branch selection (`.py` / `.swift` / SwiftPM dir /
  executable) and single-instance replacement.
- **Image smoke**: build the image, `wendy-simrun` the Swift drone headless, assert
  `scene.json` + advancing `state.json` appear.

## Risks / open questions

1. **Swift toolchain image size** — adding the Linux toolchain is a large layer. Acceptable
   per the user; consider a multi-stage build or a separate "sim-dev" image variant if size
   becomes a problem.
2. **MuJoCo C struct import fidelity** — `mjModel`/`mjData` are large structs; Swift's Clang
   importer should map fixed-size array fields fine, but a few (e.g. anonymous unions,
   `mjtNum` typedef) need verification early. First implementation task is a spike:
   `mj_loadXML` → `mj_step` → read `geom_xpos` from Swift.
3. **`WendyMuJoCo` single-source vs mirror** — decide where the canonical library lives to
   avoid drift between `samples` and `wendy-sandbox`.
4. **Rebuild-on-save latency** — a few seconds per reload for Swift; ensure `wendy-simrun`
   shows a clear "building…" state so the Sim tab doesn't look hung.
5. **libmujoco version/ABI** — the C SDK version in the image must match what the module map
   compiles against; pin it.

## Rough phasing (detailed plan follows in writing-plans)

1. Spike: `CMuJoCo` module map + Swift reads `geom_xpos` after `mj_step` (de-risks #2).
2. `WendyMuJoCo` core: load/Menagerie, `buildScene`, `buildState`, `Handle.sync()`, atomic writes.
3. `DroneRace`: course XML, controller via `mju_*`, HUD, gate progression.
4. `control.json` + `ctl.sock` endpoint.
5. `Scene` composition (MjSpec).
6. `wendy-simrun` auto-detect + `Dockerfile` toolchain + MuJoCo SDK.
7. Catalog/skill + tests + image smoke.
