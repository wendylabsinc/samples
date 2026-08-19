# "Open Local Sim…" (native macOS) — design

**Date:** 2026-07-23
**Status:** Approved design, pre-plan
**Repo:** `wendy-sandbox` (native app `desktop-native/` + `ctl-server.py`); design doc kept with the session's others in `wendy/samples/docs/superpowers/`.

## Goal

Add a first-class, one-gesture **"Open Local Sim…"** action to the native macOS WendySandbox app: pick a local Swift MuJoCo sim package on the Mac and run it live in the 🕹 Sim tab. This closes the gap where the app can only run baked-in catalog templates, not a package the user is iterating on locally (e.g. `~/git/wendy/samples/swift/drone-slalom`).

## Constraint that shapes everything

The native app drives a **sandbox container**; a host path is only reachable inside it via the **local-workspace bind-mount**, which the control plane force-disables outside local mode (`config.localWorkspace = LOCAL_WORKSPACE===1 && DRIVER!=='gce'`; `control-plane/src/config.ts:73`, refused at `index.ts:308`). Therefore **"Open Local Sim…" is inherently local-mode only** and is **hidden unless the app is in local-workspace mode** (approved). This is a security boundary, not a limitation to work around: a hosted session must never mount a host path.

## User flow (one gesture)

1. User invokes **"Open Local Sim…"** (command palette + menu; shown only in local-workspace mode).
2. `NSOpenPanel` (directories only) → user picks a folder. The app validates it contains `Package.swift` locally; if not, an alert ("Not a Swift package — pick a folder with a Package.swift").
3. The app bind-mounts the folder as the session workspace via the **existing** path (`WorkspaceClient` → `POST /admin/api/local-workspace` → recycle), reusing `AppShell.applyWorkspace`.
4. After the session recycles and reconnects, the app calls a **new** `ctl-server` action `POST /ctl/sim-open-workspace`, which launches the workspace-root package and returns its slot.
5. The app focuses the 🕹 Sim tab on that slot. `wendy-simrun` auto-detects the `Package.swift`, builds it (~9 s first time), and streams `scene.json`/`state.json` — the sim renders live.

## Architecture / components

**Native (`desktop-native/Sources/WendySandbox/`)**
- **Local-mode signal.** The app must know whether it's in local-workspace mode to show the action. Preferred: the control plane exposes a `localWorkspace: Bool` in a config/status response the app already fetches (add the field if absent); fallback: treat a loopback `controlPlaneOrigin` (`localhost`/`127.0.0.1`) as local. The plan pins the exact source.
- **Command + menu item** "Open Local Sim…" gated on that signal (mirrors where "Choose Folder…" lives, `Command.swift`).
- **Picker + validation:** `NSOpenPanel` (`canChooseDirectories = true`, files off); reject folders without `Package.swift` (local `FileManager` check — no round-trip).
- **Sequencing:** reuse `AppShell.applyWorkspace(path)` to mount + recycle + reconnect, then call `LibraryClient.openWorkspaceSim()`; focus the Sim tab on the returned slot. Handle the recycle→reconnect delay (await reconnect before the run call).
- **`LibraryClient.openWorkspaceSim()`** → `POST /ctl/sim-open-workspace` → `{slot}`.

**Backend (`image/ctl-server.py`)**
- New action **`sim-open-workspace`** (no client-supplied path — traversal-safe by construction): if `WORKDIR/Package.swift` OR `WORKDIR/main.py` exists → `_launch_sim(WORKDIR)` (which sets `WENDY_WORLDSIM_DIR` and execs `wendy-simrun WORKDIR`; `wendy-simrun` already auto-detects a Swift package dir). Else → `{"error": "no sim in workspace"}`. `_launch_sim` is unchanged.

**Reused as-is**
- Local-workspace mount (`control-plane driver.ts`, `/admin/api/local-workspace`).
- `wendy-simrun` Swift auto-detect (built + smoke-tested).

## Error handling

- Not local mode → the action isn't shown (hidden). (No dead-end clicks.)
- Picked folder lacks `Package.swift` → local alert, no session change.
- Workspace has no sim (`sim-open-workspace` guard fails) → the app surfaces `{"error": "no sim in workspace"}`.
- Build failure inside the sandbox → `wendy-simrun` logs it; the Sim tab shows no frames. (Out of scope: streaming build logs to the app — a follow-up.)

## Testing

- **`ctl-server` unit** (extends the existing test approach): `sim-open-workspace` with a `WORKDIR` containing `Package.swift` → calls `_launch_sim(WORKDIR)`; empty `WORKDIR` → error. Path-safety: it never reads a client path.
- **Container integration** (optional, achievable here): `docker run` the image with a Swift package bind-mounted at `/home/dev/workspace`, hit `sim-open-workspace` (or invoke the launch directly), assert `scene.json`/advancing `state.json` — the same harness as the existing smoke test.
- **Native:** parse-check the edited Swift; ⚠️ full build/run is in the user's **Xcode** (this environment has Command Line Tools only — no `xcodebuild`). Stated explicitly.

## Non-goals / follow-ups

- Streaming `wendy-simrun` build logs into the app.
- Multiple concurrent local sims / sub-directory sims within one workspace (the mount is the whole workspace root; one package at a time).
- Making it work on hosted (impossible by the security boundary).

## Validation limitation (call it out)

The native half compiles/ships only through Xcode on the user's machine; this environment can validate the `ctl-server` action (unit + container) and parse-check the Swift, but cannot build or run the macOS app.
