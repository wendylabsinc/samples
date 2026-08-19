# wendy-sandbox: run Swift sims in the 🕹 Sim tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the Wendy Sandbox run a **Swift** MuJoCo sim (a SwiftPM package built on `WendyMuJoCo`) live in the 🕹 Sim tab, alongside the existing Python sims — by adding a Swift toolchain + MuJoCo C SDK + vendored `swift-mujoco` to the image, teaching `wendy-simrun` and `ctl-server.py` to build/launch a Swift package, shipping a Swift drone-slalom template, and listing it in the catalog.

**Architecture:** The image vendors `swift-mujoco` at `/opt/sandbox/swift-mujoco` and installs a Swift toolchain + MuJoCo C SDK (pkg-config at `/usr/local`). A Swift sim is a **SwiftPM package directory** (path-dependent on the vendored `swift-mujoco`). `wendy-simrun` auto-detects the artifact kind (`.py` → `python3`; a dir with `Package.swift` → `swift build -c release` then run the product, rebuilding on `Sources/` change; a prebuilt executable → run directly). `ctl-server.py`'s three launch handlers accept a Swift package template; the catalog (native `Catalog.swift` + web `SIM_TEMPLATES`) gains a Swift drone entry tagged with an artifact kind.

**Tech Stack:** Docker (Ubuntu 24.04, root, arch via `dpkg --print-architecture`), swiftly/Swift 6.1 toolchain for Linux, MuJoCo 3.10 C SDK + pkg-config, bash (`wendy-simrun`), Python 3 (`ctl-server.py`), Swift (`desktop-native`), JS (`shell/index.html`), the `swift-mujoco` package (`WendyMuJoCo`).

## Prerequisite (BLOCKING — resolve before Task 3)

The Dockerfile must fetch `swift-mujoco` at build time. It is not yet published. Choose ONE and wire it in Task 3:
- **Publish** `swift-mujoco` to a git URL (e.g. `https://github.com/wendylabsinc/swift-mujoco`) and `git clone` it (mirrors the Menagerie vendoring). *Preferred.*
- **Vendor** a copy into the image build context (`wendy-sandbox/image/swift-mujoco/`) and `COPY` it in. Self-contained but duplicates source.

The plan is written for the git-clone path with an `ARG SWIFT_MUJOCO_REPO`/`ARG SWIFT_MUJOCO_REF`; the COPY variant is noted inline in Task 3.

## Global Constraints

- Ubuntu 24.04 base; every `RUN` is root (no `USER` directive) — write `/usr/local` and `/opt/sandbox` directly, no sudo.
- Arch handling uses `dpkg --print-architecture` (→ `arm64`/`amd64`), matching the existing Caddy/Wokwi/Wendy-CLI download blocks.
- MuJoCo C SDK + pkg-config installed at `/usr/local` (headers `/usr/local/include/mujoco/`, `mujoco.pc` at `/usr/local/lib/pkgconfig`); the vendored `swift-mujoco` and any Swift sim build with `PKG_CONFIG_PATH=/usr/local/lib/pkgconfig` (export it image-wide via `ENV`).
- A Swift sim is a SwiftPM **package directory** whose `Package.swift` has `.package(path: "/opt/sandbox/swift-mujoco")` and one executable product.
- `wendy-simrun`'s language-agnostic pidfile/cleanup machinery (current lines 11-40) stays unchanged; only the `[ -f "$f" ]` guard (line 13) and the launch/reload loop (lines 44-54) get new branches. `.py` behavior is byte-for-byte unchanged.
- `ctl-server.py`'s `_launch_sim` (lines 322-341) is unchanged (it already sets `WENDY_WORLDSIM_DIR` and execs `wendy-simrun <dest>`); only the three `.endswith(".py")` gates (lines 1063, 1087, 1104) are relaxed to also accept a Swift package.
- The native `Catalog.swift` and the web `shell/index.html` `SIM_TEMPLATES` list must stay in sync (both get the same new entry).

## Validation reality (read before executing)

Full end-to-end validation needs a Docker image build + a running Sim tab, which is heavy and likely belongs on CI/host, not this environment. Each task below states its **local** validation (what can be checked without the image) and its **deferred** validation (what needs the image/Sim tab). Do not mark a task "done" on deferred validation alone — record clearly what was and wasn't verified.

## File Structure (touched)

```
wendy-sandbox/
  image/
    Dockerfile                         # + Swift toolchain, MuJoCo C SDK, swift-mujoco vendor+prebuild, ENV PKG_CONFIG_PATH
    ai/wendy-simrun                    # + auto-detect Swift package / binary branch
    ctl-server.py                      # relax 3 .py-only gates to accept a Swift package
    sim-templates/
      drone_slalom_swift/              # NEW Swift sim template package
        Package.swift                  # path dep on /opt/sandbox/swift-mujoco
        Sources/{SlalomCore,DroneRace}/…
  desktop-native/Sources/WendySandbox/Catalog.swift   # + kind discriminator + Swift drone entry
  image/shell/index.html               # + matching SIM_TEMPLATES entry
```

---

## Task 1: Dockerfile — Swift toolchain

**Files:** Modify `image/Dockerfile` (insert a RUN block between the Wokwi block, line 72, and `# Session user + assets`, line 74).

**Local validation:** none without a build. **Deferred:** the layer builds and `swift --version` succeeds.

- [ ] **Step 1: Add the toolchain install**

Insert after line 72:
```dockerfile
# Swift toolchain (Linux) for building Swift-based sims in the Sim tab. Installed via
# swiftly, which resolves the correct arch/build. Pinned for reproducibility.
ENV SWIFTLY_HOME_DIR=/usr/local/swiftly SWIFTLY_BIN_DIR=/usr/local/bin
RUN SWIFT_ARCH=$(dpkg --print-architecture) \
    && case "$SWIFT_ARCH" in arm64) S=aarch64 ;; amd64) S=x86_64 ;; *) echo "unsupported arch: $SWIFT_ARCH" >&2; exit 1 ;; esac \
    && apt-get update && apt-get install -y --no-install-recommends \
         binutils gnupg2 libc6-dev libcurl4-openssl-dev libedit2 libgcc-13-dev \
         libncurses-dev libpython3-dev libstdc++-13-dev libxml2-dev libz3-dev \
         pkg-config tzdata unzip zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://download.swift.org/swiftly/linux/swiftly-${S}.tar.gz" -o /tmp/swiftly.tar.gz \
    && tar -xzf /tmp/swiftly.tar.gz -C /tmp && rm /tmp/swiftly.tar.gz \
    && /tmp/swiftly init --assume-yes --skip-install --no-modify-profile \
    && . "${SWIFTLY_HOME_DIR}/env.sh" \
    && swiftly install --use 6.1.2 \
    && swift --version
```

**Note (must verify at build time):** swiftly's exact init flags (`--skip-install`, `--no-modify-profile`, `--assume-yes`) and the apt dependency list vary by swiftly/Swift version and Ubuntu. The acceptance criterion is that the RUN layer ends with a working `swift --version`. If swiftly's flags differ in the pinned version, adjust them; if a dependency is missing, `swift --version`/a test build will name it — add it. An alternative if swiftly proves unreliable in-build: install the official `swift.org` Ubuntu 24.04 tarball for `$S` directly to `/usr/local/swift` and add its `usr/bin` to `PATH`.

- [ ] **Step 2: Ensure `swift` is on PATH for all users/sessions**

After the install, confirm `/usr/local/bin/swift` (swiftly shim) resolves, and add to the image `PATH` if not already: append `ENV PATH="/usr/local/bin:${PATH}"` (only if needed — `SWIFTLY_BIN_DIR=/usr/local/bin` should already be on PATH).

- [ ] **Step 3: Commit**
```bash
cd wendy-sandbox && git add image/Dockerfile
git commit -m "sandbox: install Swift toolchain (swiftly) in the session image"
```
(Deferred: verify via `docker build` on CI/host; record that local validation was not possible.)

---

## Task 2: Dockerfile — MuJoCo C SDK + pkg-config

**Files:** Modify `image/Dockerfile` (add a RUN block after Task 1's toolchain block); the `mujoco` pip wheel is already installed at line 27, so the C SDK is sourced from that wheel.

**Local validation:** none without a build. **Deferred:** `pkg-config --cflags --libs mujoco` prints `-I/usr/local/include -L/usr/local/lib -lmujoco` in the built image.

- [ ] **Step 1: Add the MuJoCo C SDK install (from the already-installed pip wheel)**

```dockerfile
# MuJoCo C SDK (headers + shared lib + pkg-config) sourced from the installed `mujoco`
# wheel, so Swift links the exact MuJoCo the Python path uses. Mirrors swift-mujoco's
# Scripts/install-mujoco.sh.
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
RUN PKGDIR="$(python3 -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))')" \
    && VER="$(python3 -c 'import mujoco; print(mujoco.__version__)')" \
    && mkdir -p /usr/local/include/mujoco /usr/local/lib/pkgconfig \
    && cp -R "$PKGDIR/include/mujoco/." /usr/local/include/mujoco/ \
    && LIB="$(find "$PKGDIR" -maxdepth 1 -name 'libmujoco*.so*' | head -n1)" \
    && [ -n "$LIB" ] && cp "$LIB" /usr/local/lib/ \
    && ln -sf "$(basename "$LIB")" /usr/local/lib/libmujoco.so \
    && ldconfig \
    && printf 'prefix=/usr/local\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\nName: mujoco\nDescription: MuJoCo\nVersion: %s\nLibs: -L${libdir} -lmujoco\nCflags: -I${includedir}\n' "$VER" > /usr/local/lib/pkgconfig/mujoco.pc \
    && pkg-config --cflags --libs mujoco
```

**Note:** the Linux wheel's headers live at `<pkg>/include/mujoco/`; confirm at build time (the macOS layout was identical). `ldconfig` picks up `/usr/local/lib`; if the loader still can't find it at sim-run time, the vendored-build prebuild in Task 3 will fail loudly and an rpath/`LD_LIBRARY_PATH=/usr/local/lib` can be added to `wendy-simrun`'s Swift branch.

- [ ] **Step 2: Commit**
```bash
git add image/Dockerfile
git commit -m "sandbox: install MuJoCo C SDK + pkg-config from the mujoco wheel"
```

---

## Task 3: Dockerfile — vendor swift-mujoco + prebuild it

**Files:** Modify `image/Dockerfile` (add a RUN block alongside the Menagerie vendoring, near lines 166-176).
**Prerequisite:** resolve the swift-mujoco source (git URL or COPY) per the Prerequisite section.

**Local validation:** none without a build. **Deferred (HIGH VALUE):** if this layer builds, the Swift↔MuJoCo binding + `WendyMuJoCo` compile *in the sandbox* — the strongest single validation of the whole integration.

- [ ] **Step 1: Vendor + prebuild**

```dockerfile
# Vendor swift-mujoco (generic MuJoCo binding + WendyMuJoCo Sim-tab glue) and prebuild it,
# so Swift sims import it via a path dependency and the first sim build is warm. A clean
# build here also proves the toolchain + MuJoCo C SDK are wired correctly.
ARG SWIFT_MUJOCO_REPO=https://github.com/wendylabsinc/swift-mujoco
ARG SWIFT_MUJOCO_REF=main
RUN git clone --depth 1 --branch "$SWIFT_MUJOCO_REF" "$SWIFT_MUJOCO_REPO" /opt/sandbox/swift-mujoco \
    && . "${SWIFTLY_HOME_DIR}/env.sh" \
    && (cd /opt/sandbox/swift-mujoco && swift build -c release --product WendyMuJoCo) \
    && chmod -R a+rX /opt/sandbox/swift-mujoco
```
(COPY variant: replace the `git clone` with `COPY swift-mujoco/ /opt/sandbox/swift-mujoco/`, and add `swift-mujoco/` to the image build context.)

- [ ] **Step 2: Commit**
```bash
git add image/Dockerfile
git commit -m "sandbox: vendor + prebuild swift-mujoco for Swift sims"
```

---

## Task 4: Swift drone-slalom template package

**Files:** Create `image/sim-templates/drone_slalom_swift/` — a SwiftPM package mirroring `samples/swift/drone-slalom` but with the path dependency pointing at the vendored `/opt/sandbox/swift-mujoco`.

**Local validation (POSSIBLE):** on the dev Mac, `swift build` the template against the local `swift-mujoco` (temporarily pointing the path dep at `../../../../swift-mujoco` or an absolute path) proves the sources compile against `WendyMuJoCo`. **Deferred:** building/running inside the image at `/opt/sandbox/...`.

- [ ] **Step 1: Copy the sample sources into the template**

Reuse the reviewed sources from `samples/swift/drone-slalom` (Task set of the DroneRace plan): `Sources/SlalomCore/{DroneController.swift,Course.swift}` and `Sources/DroneRace/main.swift` verbatim.

- [ ] **Step 2: Write the template `Package.swift`** (absolute path dep on the vendored location)
```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "DroneRace",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "/opt/sandbox/swift-mujoco"),
    ],
    targets: [
        .target(name: "SlalomCore", dependencies: [.product(name: "MuJoCo", package: "swift-mujoco")]),
        .executableTarget(name: "DroneRace", dependencies: [
            "SlalomCore",
            .product(name: "WendyMuJoCo", package: "swift-mujoco"),
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
    ]
)
```

- [ ] **Step 3: Local compile check** (dev Mac, temporary path)

Run, temporarily pointing the dep at the local checkout:
```bash
cd wendy-sandbox/image/sim-templates/drone_slalom_swift
# temporarily: .package(path: "/Users/joannisorlandos/git/wendy/swift-mujoco")
PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift build
# then restore the /opt/sandbox/swift-mujoco path before committing
```
Expected: builds `DroneRace` against `WendyMuJoCo`. Restore the `/opt/sandbox/swift-mujoco` path afterward (that's the path the image uses).

- [ ] **Step 4: Copy the template into the image at build time**

Add to the Dockerfile near the existing `COPY sim-templates/ /opt/sandbox/sim-templates/` (line 164) — no change needed if that COPY already recurses (it does: `COPY sim-templates/`), so the new `drone_slalom_swift/` dir ships automatically. Verify the existing COPY is recursive; if the build should also warm the template, optionally add a prebuild RUN.

- [ ] **Step 5: Commit**
```bash
git add image/sim-templates/drone_slalom_swift
git commit -m "sandbox: add Swift drone-slalom sim template (WendyMuJoCo)"
```

---

## Task 5: wendy-simrun — auto-detect Swift package / binary

**Files:** Modify `image/ai/wendy-simrun`.

**Local validation (POSSIBLE):** a shell test of branch selection with fake inputs (a dir with a `Package.swift`, a `.py` file, an executable) — no MuJoCo needed. **Deferred:** an actual Swift build+run+reload inside the image.

**Interfaces / behavior:**
- Keep lines 11-40 (arg capture, pidfile, cleanup) — but relax the line-13 guard to accept a directory OR a file.
- Determine `kind`: directory containing `Package.swift` → `swift`; a regular file that is executable and not `*.py` → `binary`; a `*.py` file → `python` (unchanged).
- `swift`: `(cd "$f" && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig . "${SWIFTLY_HOME_DIR}/env.sh"; swift build -c release)`; resolve the product binary under `$f/.build/release/`; run it (cwd `$f`). Reload-watch the `Sources/` tree (rebuild+relaunch when any source is newer than the last build). Show a "building…" log line.
- `binary`: run `"$f"` directly; watch its own mtime.
- `python`: unchanged (`python3 "$f"`, mtime watch).

- [ ] **Step 1: Write a branch-selection shell test**

Create `image/ai/tests/test-simrun-detect.sh` (a focused unit of the detection logic — extract detection into a function `sim_kind()` in `wendy-simrun` so it's testable):
```bash
#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
KIND_ONLY=1 . "$here/wendy-simrun"   # sourcing with KIND_ONLY defines sim_kind and returns

tmp="$(mktemp -d)"
mkdir -p "$tmp/pkg/Sources"; : > "$tmp/pkg/Package.swift"
: > "$tmp/sim.py"
printf '#!/bin/sh\n' > "$tmp/bin"; chmod +x "$tmp/bin"

[ "$(sim_kind "$tmp/pkg")" = swift ] || { echo "FAIL: dir+Package.swift should be swift"; exit 1; }
[ "$(sim_kind "$tmp/sim.py")" = python ] || { echo "FAIL: .py should be python"; exit 1; }
[ "$(sim_kind "$tmp/bin")" = binary ] || { echo "FAIL: exec should be binary"; exit 1; }
echo "OK: sim_kind detection"
```

- [ ] **Step 2: Refactor detection into `sim_kind()` + early-return when sourced for tests**

At the top of `wendy-simrun` (after `set -u`), add:
```bash
sim_kind() {
  local f="$1"
  if [ -d "$f" ] && [ -f "$f/Package.swift" ]; then echo swift; return; fi
  if [ -f "$f" ] && [ -x "$f" ] && [ "${f%.py}" = "$f" ]; then echo binary; return; fi
  if [ -f "$f" ]; then echo python; return; fi
  echo unknown
}
# Testing hook: `KIND_ONLY=1 . wendy-simrun` defines sim_kind then returns without running.
[ -n "${KIND_ONLY:-}" ] && return 0 2>/dev/null || true
```

- [ ] **Step 3: Run the test to verify it fails, then implement, then passes**

Run: `bash image/ai/tests/test-simrun-detect.sh`
- Before the launch-loop branches exist but after Step 2: the detection test should already PASS (it only needs `sim_kind`). First run it to confirm detection; then implement the launch branches (Step 4) and confirm the script still sources cleanly.

- [ ] **Step 4: Replace the guard (line 13) and the launch loop (lines 44-54) with kind-aware logic**
```bash
# (replaces line 13's `[ -f "$f" ]` guard)
f_abs="$(readlink -f "$f" 2>/dev/null || echo "$f")"
kind="$(sim_kind "$f")"
[ "$kind" = unknown ] && { echo "usage: wendy-simrun <file.py | swiftpm-dir | executable>"; exit 1; }

# ... pidfile/cleanup unchanged (keyed on f_abs) ...

# reload marker for tree-watching
newest() {
  case "$kind" in
    swift) find "$f/Sources" "$f/Package.swift" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 ;;
    *)     stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null ;;
  esac
}
run_once() {
  case "$kind" in
    swift)
      echo "[wendy-simrun] building $f …"
      ( cd "$f" && . "${SWIFTLY_HOME_DIR}/env.sh" 2>/dev/null; PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift build -c release ) || { echo "[wendy-simrun] build failed"; return 1; }
      local bin; bin="$(find "$f/.build/release" -maxdepth 1 -type f -perm -u+x ! -name '*.o' 2>/dev/null | head -1)"
      ( cd "$f" && exec "$bin" ) & child=$! ;;
    binary) "$f" & child=$! ;;
    python) python3 "$f" & child=$! ;;
  esac
}
while true; do
  run_once || { sleep 1; continue; }
  last="$(newest)"
  while kill -0 "$child" 2>/dev/null; do
    if [ "$(newest)" != "$last" ]; then kill "$child" 2>/dev/null; sleep 0.3; break; fi
    sleep 0.3
  done
  wait "$child" 2>/dev/null || true
  sleep 0.3
done
```
(Keep the exact pidfile/cleanup block from the current script between the guard and this loop.)

- [ ] **Step 5: Re-run the detection test; commit**
```bash
bash image/ai/tests/test-simrun-detect.sh   # OK
git add image/ai/wendy-simrun image/ai/tests/test-simrun-detect.sh
git commit -m "sandbox: wendy-simrun auto-detects Swift package / binary / python"
```
(Deferred: real Swift build+run+reload inside the image.)

---

## Task 6: ctl-server.py — accept a Swift package template

**Files:** Modify `image/ctl-server.py` (the three `.endswith(".py")` gates at lines 1063, 1087, 1104; and the template-resolution around 1085-1092).

**Local validation (POSSIBLE):** `python3 -c "import ast; ast.parse(open('image/ctl-server.py').read())"` (syntax) + reasoning; a small unit test of a helper if extracted. **Deferred:** running the server + launching from the Library.

**Behavior:** a template id may be a `.py` file (unchanged) OR a directory name under `/opt/sandbox/sim-templates/` containing `Package.swift`. For a Swift template, `sim-run` copies the whole package dir into `$WORKDIR/<name>/` (not a single file) and calls `_launch_sim(dest_dir)`.

- [ ] **Step 1: Add a resolver helper** (near `_launch_sim`)
```python
def _resolve_template(fn):
    """Return (src_path, is_dir) for a template id: a .py file or a Swift package dir
    under /opt/sandbox/sim-templates. Rejects anything else (path-traversal safe)."""
    import os
    fn = os.path.basename(str(fn or ""))
    base = "/opt/sandbox/sim-templates"
    py = os.path.join(base, fn)
    if fn.endswith(".py") and os.path.isfile(py):
        return py, False
    pkg = os.path.join(base, fn)
    if os.path.isdir(pkg) and os.path.isfile(os.path.join(pkg, "Package.swift")):
        return pkg, True
    return None, False
```

- [ ] **Step 2: Rewrite the `sim-run` handler** (lines 1081-1098) to use the resolver + copy dir-or-file
```python
        if action == "sim-run":
            import os, shutil
            src, is_dir = _resolve_template(body.get("file"))
            if src is None:
                self._send({"error": "unknown sim"}); return
            name = os.path.basename(src)
            dest = os.path.join(WORKDIR, name)
            try:
                if is_dir:
                    if os.path.isdir(dest): shutil.rmtree(dest)
                    shutil.copytree(src, dest)
                else:
                    shutil.copyfile(src, dest)
                slot = _launch_sim(dest)
```
(Apply the analogous relaxation to `sim-open` at 1099-1113 — accept a workspace dir with `Package.swift` — and to `starter-scaffold` at 1059-1067 if a starter's `sim` may be a Swift package.)

- [ ] **Step 3: Syntax check + commit**
```bash
python3 -c "import ast,sys; ast.parse(open('image/ctl-server.py').read()); print('ok')"
git add image/ctl-server.py
git commit -m "sandbox: ctl-server accepts Swift package sim templates"
```

---

## Task 7: Catalog entry (native + web, in sync)

**Files:** Modify `desktop-native/Sources/WendySandbox/Catalog.swift` and `image/shell/index.html`.

**Local validation (POSSIBLE):** `swift build` the `desktop-native` package (if it builds standalone) or at least compile-check `Catalog.swift`; JS is eyeballed / the shell served. **Deferred:** the entry appearing + launching in the running Library.

**Behavior:** add a `kind` discriminator so a sim template can be `.python` (default) or `.swiftPackage`, and add the Swift drone entry pointing at the `drone_slalom_swift` package dir.

- [ ] **Step 1: Extend `SimTemplate`** (`Catalog.swift:33-40`)
```swift
struct SimTemplate: Identifiable, Hashable {
    enum Kind: String, Hashable { case python, swiftPackage }
    let name: String
    let file: String       // sim-run id (a .py filename, or a package dir name)
    let category: String
    let blurb: String
    var kind: Kind = .python
    var id: String { file }
}
```

- [ ] **Step 2: Add the Swift drone entry** (in the "Aerial & Physics" group, after the existing `drone-race`)
```swift
        .init(name: "drone-race-swift", file: "drone_slalom_swift", category: "Aerial & Physics",
              blurb: "Skydio X2 gate slalom in Swift (WendyMuJoCo) — same flight as drone-race, built with the Swift toolchain.",
              kind: .swiftPackage),
```

- [ ] **Step 3: Mirror in the web `SIM_TEMPLATES`** (`shell/index.html:1507-1526`)
```js
{ name: 'drone-race-swift', file: 'drone_slalom_swift', category: 'Aerial & Physics',
  blurb: 'Skydio X2 gate slalom in Swift (WendyMuJoCo).' },
```
(The `ctl('sim-run', { file: t.file })` call at line 2252 already sends `file`; the server-side resolver from Task 6 handles the dir.)

- [ ] **Step 4: Compile-check what's local + commit**
```bash
# if desktop-native builds standalone on this host:
cd wendy-sandbox/desktop-native && swift build 2>&1 | tail -5 || echo "(native build deferred to host)"
git add desktop-native/Sources/WendySandbox/Catalog.swift image/shell/index.html
git commit -m "sandbox: list the Swift drone-race sim in the catalog (native + web)"
```

---

## Self-Review

**Spec coverage** (design components 4 "wendy-simrun auto-detect", 5 "image toolchain + MuJoCo SDK", "catalog/skill"):
- Swift toolchain in image → Task 1 ✓
- MuJoCo C SDK + pkg-config → Task 2 ✓
- swift-mujoco vendored + prebuilt → Task 3 ✓
- Swift sim template → Task 4 ✓
- wendy-simrun auto-detect (swift dir / binary / py) → Task 5 ✓
- launch path accepts Swift (ctl-server gates) → Task 6 ✓ (a real integration point the initial design under-specified)
- catalog discoverability (native + web) → Task 7 ✓
- build-a-sim SKILL update → NOT included; add as a small follow-up (documentation only) once the mechanism works.

**Placeholder scan:** Tasks 1-3 carry explicit "verify at build time" notes for the toolchain/SDK install specifics — these are genuine environment-resolution steps (the swiftly flags / apt deps / wheel layout can't be pinned without a build), not vague TODOs; each has a concrete acceptance check (`swift --version`, `pkg-config …`, the prebuild layer). All code/edits are complete.

**Consistency:** the artifact-kind concept is threaded consistently — `sim_kind()` (bash, Task 5), `_resolve_template` (python, Task 6), `SimTemplate.Kind` (swift, Task 7); the template dir name `drone_slalom_swift` is identical across Task 4 (creation), Task 6 (resolution), and Task 7 (catalog `file`).

**Risks / open items:**
- **BLOCKING prerequisite:** swift-mujoco must be published (git URL) or vendored into the build context (Task 3).
- Image size grows substantially (Swift toolchain ~1GB+); acceptable per the design, but note it.
- swiftly in-build reliability (Task 1) is the least-certain step; the official-tarball fallback is documented.
- Full e2e (image build + Sim-tab launch + live-reload) is deferred to CI/host; local validation covers detection logic, the template compile, and Python/Swift syntax only.
- `build-a-sim` SKILL still says "Python only" — a doc follow-up.
