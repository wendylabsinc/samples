# "Open Local Sim…" (native macOS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-gesture "Open Local Sim…" action to the native macOS WendySandbox app that picks a local Swift sim package, mounts it as the session workspace, and runs it live in the 🕹 Sim tab.

**Architecture:** A new `ctl-server` action `sim-open-workspace` launches the workspace-root package (`wendy-simrun /home/dev/workspace`) with no client-supplied path. The native app adds an "Open Local Sim…" command (shown only in local-workspace mode) that reuses the existing folder bind-mount (`AppShell.applyWorkspace`) then calls the new action via `LibraryClient.openWorkspaceSim()` and focuses the Sim tab.

**Tech Stack:** Python 3 (`image/ctl-server.py`), Swift/SwiftUI + AppKit (`desktop-native/`), Docker (container validation), the `sandbox-session` image (Swift 6.3.2 + `wendy-simrun` Swift auto-detect, already built + smoke-tested).

## Global Constraints

- The `sim-open-workspace` action takes **no client-supplied path** — it launches the fixed `WORKDIR` (`/home/dev/workspace`) only (traversal-safe by construction).
- Reuse existing infra: `_launch_sim(dest)` (unchanged), `AppShell.applyWorkspace(path)` (mount + recycle + reconnect), `wendy-simrun`'s Swift-package auto-detect.
- The feature is **local-workspace-mode only** — the native command is **hidden** unless the app is in local mode (the native app runs its own control plane with `LOCAL_WORKSPACE=1`; a loopback `controlPlaneOrigin` host is the signal).
- `WORKDIR = "/home/dev/workspace"` (`ctl-server.py:98`).
- Native commits must NOT stage `desktop-native/Vendor/wendy-companion-ios` (pre-existing dirty submodule pointer); use explicit `git add` paths.
- **Validation ceiling:** this environment has Command Line Tools only (no `xcodebuild`). The `ctl-server` action is validated here (ast.parse + a container check of the launch it performs); the native Swift is parse-checked here and **built/run in the user's Xcode**.

## File Structure

```
wendy-sandbox/
  image/ctl-server.py                                   # + sim-open-workspace action (near sim-open, ~line 1127)
  desktop-native/Sources/WendySandbox/
    LibraryClient.swift                                 # + openWorkspaceSim() on LibraryAPI + LiveLibraryClient
    Command.swift                                       # + CommandContext.{openLocalSim, isLocalWorkspace}; + "Open Local Sim…" in CoreCommands (gated)
    AppShell.swift                                      # + openLocalSim(); wire paletteContext fields
```

---

## Task 1: `ctl-server` `sim-open-workspace` action

**Files:**
- Modify: `image/ctl-server.py` (add a new `if action == "sim-open-workspace":` block immediately after the `sim-open` handler, which ends ~line 1145)

**Interfaces:**
- Consumes: `WORKDIR` (`ctl-server.py:98`), `_launch_sim(dest, cwd=None)` (`ctl-server.py:322`).
- Produces: `POST /ctl/sim-open-workspace` → `{"ok": true, "slot": "<slot>"}` when `WORKDIR` holds a runnable sim (`Package.swift` or `main.py`), else `{"error": "no sim in workspace"}`. No request body is read.

- [ ] **Step 1: Add the handler**

In `image/ctl-server.py`, immediately after the `sim-open` handler's closing `return` (the block that ends with `self._send({"ok": True, "file": fn, "slot": slot})` / `return`, ~line 1145), insert:
```python
        if action == "sim-open-workspace":
            # Run the workspace ROOT as a sim (Swift package or main.py). Used by the native
            # app's "Open Local Sim…" after it bind-mounts a host folder as the workspace.
            # No client path: launches the fixed WORKDIR only (traversal-safe by construction).
            import os
            has_pkg = os.path.isfile(os.path.join(WORKDIR, "Package.swift"))
            has_py = os.path.isfile(os.path.join(WORKDIR, "main.py"))
            if not (has_pkg or has_py):
                self._send({"error": "no sim in workspace"})
                return
            try:
                slot = _launch_sim(WORKDIR)
            except Exception as e:
                self._send({"error": str(e)})
                return
            self._send({"ok": True, "slot": slot})
            return
```

- [ ] **Step 2: Syntax-check**

Run:
```bash
cd wendy-sandbox
python3 -c "import ast; ast.parse(open('image/ctl-server.py').read()); print('parse ok')"
```
Expected: `parse ok`.

- [ ] **Step 3: Container check — the launch the action performs actually streams**

`sim-open-workspace` calls `_launch_sim(WORKDIR)`, which execs `wendy-simrun "$WORKDIR"`. Prove that path end-to-end by bind-mounting a Swift package at `WORKDIR` in the built image and running exactly that:
```bash
cd wendy-sandbox
docker run --rm \
  -v "$HOME/git/wendy/samples/swift/drone-slalom:/home/dev/workspace:ro" \
  sandbox-session bash -c '
    set -e
    export WENDY_WORLDSIM_DIR=/tmp/slot; mkdir -p "$WENDY_WORLDSIM_DIR"
    # workspace is read-only mounted; copy to a writable dir mirroring what a real mount gives
    cp -r /home/dev/workspace /tmp/ws && cd /tmp/ws
    [ -f Package.swift ] && echo "has Package.swift"
    wendy-simrun /tmp/ws >/tmp/log 2>&1 &
    for i in $(seq 1 240); do [ -f "$WENDY_WORLDSIM_DIR/state.json" ] && break; sleep 1; done
    tail -3 /tmp/log
    [ -f "$WENDY_WORLDSIM_DIR/state.json" ] || { echo FAIL; exit 1; }
    f1=$(python3 -c "import json;print(json.load(open(\"$WENDY_WORLDSIM_DIR/state.json\"))[\"frame\"])"); sleep 3
    f2=$(python3 -c "import json;print(json.load(open(\"$WENDY_WORLDSIM_DIR/state.json\"))[\"frame\"])")
    pkill -f wendy-simrun || true
    [ "$f2" -gt "$f1" ] && echo "OK: workspace-root package streams ($f1 -> $f2)" || { echo FAIL; exit 1; }
  '
```
Expected: `has Package.swift`, then `OK: workspace-root package streams (…)`. (A read-only mount is copied to `/tmp/ws` because `swift build` writes `.build`; a real `local-workspace` mount is read-write, so production runs in place — this check validates the `wendy-simrun <dir>` launch the action performs.)

- [ ] **Step 4: Commit** (stage only `ctl-server.py`; never the submodule)
```bash
git add image/ctl-server.py
git commit -m "sandbox: ctl-server sim-open-workspace runs the workspace-root package"
```

---

## Task 2: Native "Open Local Sim…" command + flow

**Files:**
- Modify: `desktop-native/Sources/WendySandbox/LibraryClient.swift` (add `openWorkspaceSim()`)
- Modify: `desktop-native/Sources/WendySandbox/Command.swift` (add `CommandContext.openLocalSim` + `.isLocalWorkspace`; add the gated command)
- Modify: `desktop-native/Sources/WendySandbox/AppShell.swift` (add `openLocalSim()`; wire `paletteContext`)

**Interfaces:**
- Consumes: `POST /ctl/sim-open-workspace` → `{ok, slot}` (Task 1); `SimRunResult` schema (`LibraryClient.swift`, has `ok/file?/slot?/error?`); `AppShell.applyWorkspace(_:)`; `LiveLibraryClient(context:transport:)`; `AppNavigator.go(to: .sim)`; `context.controlPlaneOrigin`.
- Produces: `LibraryAPI.openWorkspaceSim() async throws -> SimRunResult`; `CommandContext.openLocalSim: () -> Void` and `.isLocalWorkspace: Bool`; a `CoreCommands` command `action.openLocalSim` (shown only when `isLocalWorkspace`).

- [ ] **Step 1: Add `openWorkspaceSim()` to the client**

In `LibraryClient.swift`, add to the `LibraryAPI` protocol (after `simRun`):
```swift
    /// POST /ctl/sim-open-workspace {} → runs the workspace-root package/main.py as a sim.
    @discardableResult func openWorkspaceSim() async throws -> SimRunResult
```
And to `LiveLibraryClient` (after the `simRun` method):
```swift
    @discardableResult
    func openWorkspaceSim() async throws -> SimRunResult {
        try SimRunResult.schema.decode(await post("/ctl/sim-open-workspace", body: [:]))
    }
```

- [ ] **Step 2: Extend `CommandContext` and add the gated command**

In `Command.swift`, add two fields to `CommandContext` (after `chooseFolder`):
```swift
    /// Opens the "Open Local Sim…" flow (AppShell.openLocalSim). Only meaningful in local mode.
    let openLocalSim: () -> Void
    /// True when the app runs its own local control plane (host folders can be mounted).
    let isLocalWorkspace: Bool
```
In `CoreCommands.commands(_:)`, right after the existing `action.chooseFolder` command append, add:
```swift
        // Local-only: pick a Swift sim package on disk and run it in the Sim tab.
        if ctx.isLocalWorkspace {
            out.append(Command(
                id: "action.openLocalSim",
                title: "Open Local Sim…",
                subtitle: "Run a local Swift sim package in the Sim tab",
                systemImage: "cube.transparent",
                section: .actions,
                run: { ctx.navigator.closePalette(); ctx.openLocalSim() }
            ))
        }
```

- [ ] **Step 3: Implement `openLocalSim()` in AppShell + wire the context**

In `AppShell.swift`, add the method (next to `chooseFolder`):
```swift
    private func openLocalSim() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Open Sim"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let path = url.path
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("Package.swift").path) else {
            workspaceError = "Not a Swift package — pick a folder containing a Package.swift."
            return
        }
        Task {
            await applyWorkspace(path)                 // mount + recycle + reconnect (reuses existing flow)
            guard let ctx = sessionContext else { return }
            do {
                _ = try await LiveLibraryClient(context: ctx, transport: context.transport).openWorkspaceSim()
                navigator.go(to: .sim)
            } catch {
                workspaceError = "Couldn't run the local sim: \(error)"
            }
        }
    }
```
Then extend `paletteContext` (the `CommandContext(...)` initializer) with the two new fields:
```swift
            openLocalSim: { openLocalSim() },
            isLocalWorkspace: context.controlPlaneOrigin.isLoopbackHost,
```
And add the loopback helper at file scope in `AppShell.swift` (or a small extension file):
```swift
private extension URL {
    /// The native app's local control plane binds PUBLIC_HOST=localhost, so a loopback
    /// origin means local-workspace mode (host folders can be mounted).
    var isLoopbackHost: Bool {
        switch host { case "localhost", "127.0.0.1", "::1": return true; default: return false }
    }
}
```

- [ ] **Step 4: Parse-check the edited Swift**

Run:
```bash
cd wendy-sandbox/desktop-native
swiftc -parse Sources/WendySandbox/LibraryClient.swift Sources/WendySandbox/Command.swift Sources/WendySandbox/AppShell.swift 2>&1 | grep -v "warning:" | head
```
Expected: no syntax errors. (Full type-check + build happens in Xcode — this environment has no `xcodebuild`.)

- [ ] **Step 5: Commit** (stage only the three files; never the submodule)
```bash
git add desktop-native/Sources/WendySandbox/LibraryClient.swift \
        desktop-native/Sources/WendySandbox/Command.swift \
        desktop-native/Sources/WendySandbox/AppShell.swift
git commit -m "sandbox(macOS): Open Local Sim… — pick a local Swift package and run it in the Sim tab"
```

- [ ] **Step 6: Note the Xcode-build handoff**

In the task report, state explicitly that the native change is parse-checked only here; the user must build/run `WendySandbox.xcodeproj` in Xcode to verify the command appears (local mode) and the flow works end-to-end.

---

## Self-Review

**Spec coverage:**
- One-gesture "Open Local Sim…" (picker → validate Package.swift → mount → run → focus Sim) → Task 2 ✓
- Hidden unless local mode (loopback-origin signal) → Task 2 (`isLocalWorkspace` gate) ✓
- `sim-open-workspace` action, no client path, guarded on Package.swift/main.py → Task 1 ✓
- Reuse local-workspace mount + `wendy-simrun` auto-detect → Task 2 (applyWorkspace) + Task 1 (`_launch_sim`) ✓
- Error handling: not-local (hidden), no Package.swift (alert via `workspaceError`), no workspace sim (`{"error"}`), run failure (`workspaceError`) ✓
- Validation ceiling stated (Task 1 container-tested here; Task 2 parse-check + Xcode) ✓

**Placeholder scan:** none — full code in every step. The `workspaceError` property is the existing AppShell error state used by `applyWorkspace` (`AppShell.swift:213`); reused, not invented.

**Type consistency:** `openWorkspaceSim() -> SimRunResult` (Task 1 return `{ok,slot}` decodes into `SimRunResult`'s `ok/slot?`); `CommandContext.openLocalSim`/`isLocalWorkspace` defined in Task 2 Step 2 and set in Task 2 Step 3; `applyWorkspace`/`sessionContext`/`context.transport`/`navigator.go(to:.sim)` all confirmed present in AppShell. `isLoopbackHost` defined in Task 2 Step 3.

**Risk:** the native flow's mount→reconnect→run sequencing depends on `applyWorkspace` having re-published `sessionContext` before the `openWorkspaceSim` call; `applyWorkspace` awaits `resolveSession()` which sets `sessionContext`, so the `guard let ctx = sessionContext` runs after reconnect. If `resolveSession` leaves it nil (boot still loading), the guard bails silently — acceptable (user retries); a follow-up could surface a "session still booting" message.
