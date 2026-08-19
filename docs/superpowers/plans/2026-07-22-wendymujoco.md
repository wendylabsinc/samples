# WendyMuJoCo (Sim-tab streaming core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `WendyMuJoCo` library target to the existing `swift-mujoco` package that streams a MuJoCo sim into the Wendy Sandbox 🕹 Sim tab — writing `scene.json` once and `state.json` per frame, reading `control.json` (pause/step/reset/poke/ctrl), and loading MuJoCo Menagerie models — byte-compatibly with the Python `wendymujoco.py` so the existing renderer needs no change.

**Architecture:** `WendyMuJoCo` is a third library target in the `swift-mujoco` package, depending on the generic `MuJoCo` target (for model/data/pose/state access) and on `CMuJoCo` (for raw `ptr` writes to `qpos`/`qvel` and `mj_contactForce`, which `MuJoCo` doesn't wrap). The generic `MuJoCo` module stays free of Wendy specifics. A `Handle` (drop-in shaped like `wendymujoco.launch_passive`) writes the JSON slot files under `/tmp/wendy-worldsim` and services `control.json` each `sync()`.

**Tech Stack:** Swift 6.1, Swift Testing, Foundation (`Data.write(_:options:.atomic)`, `JSONEncoder`/`JSONDecoder`, `Process` for the Menagerie git fetch), the `MuJoCo`/`CMuJoCo` targets from Tasks 1–9 of the swift-mujoco plan.

## Global Constraints

- Swift tools version `6.1`; Swift Testing (`import Testing`), NOT XCTest.
- No `.unsafeFlags` in `Package.swift`.
- All swift commands run with `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig` exported. MuJoCo 3.10.0 at `$HOME/.local`.
- Pristine build — zero compiler warnings.
- The `MuJoCo` module stays generic: put NOTHING Wendy-specific (JSON, sockets, Menagerie, `/tmp/wendy-worldsim`) in it. All of that lives in `WendyMuJoCo`.
- **JSON must be schema-compatible with `wendymujoco.py`** — the reference renderer (`wendy-sandbox/image/shell/sim.html` and `wendy-sandbox/desktop-native`) consumes these files unchanged. Exact key names and array shapes below are binding; JSON object key *order* is irrelevant (objects are unordered).
- The slot directory is `WENDY_WORLDSIM_DIR` if set, else `/tmp/wendy-worldsim`. File names: `scene.json`, `state.json`, `control.json`.
- Quaternions are MuJoCo order `wxyz` in `pose`.
- Numeric rounding mirrors the Python for file size (pos 5, quat 6, size 5, rgba 4, mesh vert 4, contact pos 4 / force 2, hud 2) — replicate it, but tests assert values within tolerance, not exact decimal text.

## Reference: the exact JSON contract (from `wendy-sandbox/image/sim/wendymujoco.py`)

**`scene.json`** (written once at Handle init):
```json
{ "title": "drone race", "up": "z", "engine": "mujoco",
  "geoms": [ {"i": 0, "type": "plane", "size": [40,40,0.1], "rgba": [0.2,0.23,0.28,1]},
             {"i": 3, "type": "mesh",  "size": [1,1,1], "rgba": [...], "mesh": "x2"} ],
  "meshes": { "x2": {"vert": [x,y,z, ...], "face": [i,j,k, ...]} } }
```
- Only **visible** geoms (`group < 3` AND rgba alpha ≠ 0) are listed; `i` is the true MuJoCo geom index (the list is filtered, so indices are sparse).
- `type` ∈ plane/sphere/capsule/ellipsoid/cylinder/box/mesh (from `MjModel.GeomType.rawValue`).
- mesh geoms carry `"mesh": <name>`; each distinct mesh's buffers appear once in `meshes` (deduplicated by name).

**`state.json`** (written every `sync()`):
```json
{ "t": 1721635200.12, "frame": 42, "engine": "mujoco",
  "pose": [ [x,y,z, qw,qx,qy,qz], ... ],
  "contacts": [ [x,y,z, forceMag], ... ],
  "hud": {"gate": "2/5", "t": 1.4, "speed": 3.2},
  "level": 1 }
```
- `t` is **wall-clock epoch seconds** (`Date().timeIntervalSince1970`), NOT sim time.
- `pose` has one entry per geom for **every** geom index `0..<ngeom` (NOT filtered — the renderer looks up `pose[scene_geom.i]`). Each entry is world position (xyz) + world-orientation quaternion (wxyz).
- `contacts`: up to 64 entries, each `[posX, posY, posZ, forceMagnitude]` where forceMagnitude = ‖linear force‖ = norm of the first 3 components of `mj_contactForce`'s 6-vector.
- `hud`: arbitrary string→(number|string) map.
- `level`: optional int; omit the key when nil.

**`control.json`** (read every `sync()`; may be absent → all defaults):
```json
{ "paused": false, "step": 0, "reset": 0, "poke": 0,
  "ctrl": {"mot0": 1.2}, "qpos": {"hinge": 0.3}, "qvel": {"hinge": 0.0} }
```
- `paused` bool; `step`/`reset`/`poke` are monotonically-increasing counters (an action fires when its counter advances past what the Handle last saw).
- `ctrl`/`qpos`/`qvel` map an actuator/joint **name-or-index** to a value.

## File Structure

```
swift-mujoco/
  Package.swift                          # + WendyMuJoCo library product & target
  Sources/WendyMuJoCo/
    WorldSim.swift        # slot dir resolution + atomic write
    Rounding.swift        # round(_:_:) helper used by encoders
    SceneManifest.swift   # SceneManifest/Geom/MeshBuf Codable + buildScene(_:title:)
    StateFrame.swift      # StateFrame Codable + HUDValue + buildState(...)
    Control.swift         # Control Decodable + readControl(in:)
    Handle.swift          # Handle (init/sync/hud/setLevel/isRunning/close) + launchPassive(...)
  Tests/WendyMuJoCoTests/
    WorldSimTests.swift
    SceneManifestTests.swift
    StateFrameTests.swift
    ControlTests.swift
    HandleTests.swift
    MenagerieTests.swift
  Sources/WendyMuJoCo/Menagerie.swift    # load(_:)/name map/resolve/fetch (Task 6)
```

## Out of scope for THIS plan (explicit follow-ups)

- **`ctl.sock` live-control endpoint** (the Unix-socket act/observe/describe/get_state/set_state/reset the Wendy AI uses). Large and separable; the drone renders and is controllable via `control.json` without it. Follow-up plan: `WendyMuJoCo ctl.sock endpoint`.
- **`Scene` multi-robot composition** (`wendymujoco.Scene`) — needs an `MjSpec.attach(fromFile:prefix:frame:)` added to the generic `MuJoCo` target first. The drone doesn't need it (it composes its course via MJCF `<include>` + gate-box XML). Follow-up plan: `MuJoCo.MjSpec attach + WendyMuJoCo.Scene`.
- **Rerun/📊 Viz mirror** — no Swift SDK; dropped per the design.

---

## Task 1: WendyMuJoCo target + WorldSim (paths + atomic write)

**Files:**
- Modify: `Package.swift`
- Create: `Sources/WendyMuJoCo/WorldSim.swift`
- Create: `Sources/WendyMuJoCo/Rounding.swift`
- Test: `Tests/WendyMuJoCoTests/WorldSimTests.swift`

**Interfaces:**
- Produces:
  - `public enum WorldSim` with `static func directory() -> URL` (env `WENDY_WORLDSIM_DIR` else `/tmp/wendy-worldsim`), `static func writeAtomic(_ data: Data, to fileName: String, in dir: URL)`, and `static func path(_ fileName: String, in dir: URL) -> URL`.
  - `func mjRound(_ x: Double, _ places: Int) -> Double` (module-internal).

- [ ] **Step 1: Add the target to `Package.swift`**

Add a product and target (keep everything else):
```swift
        .library(name: "WendyMuJoCo", targets: ["WendyMuJoCo"]),
```
```swift
        .target(name: "WendyMuJoCo", dependencies: ["MuJoCo", "CMuJoCo"]),
        .testTarget(name: "WendyMuJoCoTests", dependencies: ["WendyMuJoCo", "MuJoCo"]),
```

- [ ] **Step 2: Write the failing test**

`Tests/WendyMuJoCoTests/WorldSimTests.swift`:
```swift
import Testing
import Foundation
@testable import WendyMuJoCo

@Test func atomicWriteRoundTrips() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ws-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let payload = Data(#"{"hello":1}"#.utf8)
    WorldSim.writeAtomic(payload, to: "scene.json", in: dir)
    let read = try Data(contentsOf: WorldSim.path("scene.json", in: dir))
    #expect(read == payload)
}

@Test func directoryHonorsEnvOverride() {
    // Default path when unset is the shared slot dir.
    #expect(WorldSim.directory().path.hasSuffix("wendy-worldsim")
            || WorldSim.directory().path == ProcessInfo.processInfo.environment["WENDY_WORLDSIM_DIR"])
}

@Test func roundingMatchesPlaces() {
    #expect(mjRound(1.234567, 2) == 1.23)
    #expect(mjRound(-0.0000004, 5) == 0.0)
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter WorldSimTests`
Expected: FAIL — `WorldSim`/`mjRound` not defined (or target missing).

- [ ] **Step 4: Implement `Rounding.swift`**

```swift
import Foundation

/// Round to `places` decimals, matching Python's round() closely enough for the
/// Sim-tab JSON (used only to shrink files; the renderer parses full precision too).
func mjRound(_ x: Double, _ places: Int) -> Double {
    let p = pow(10.0, Double(places))
    return (x * p).rounded() / p
}
```

- [ ] **Step 5: Implement `WorldSim.swift`**

```swift
import Foundation

public enum WorldSim {
    /// The Sim-tab slot directory: $WENDY_WORLDSIM_DIR or /tmp/wendy-worldsim.
    public static func directory() -> URL {
        if let env = ProcessInfo.processInfo.environment["WENDY_WORLDSIM_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env, isDirectory: true)
        }
        return URL(fileURLWithPath: "/tmp/wendy-worldsim", isDirectory: true)
    }

    public static func path(_ fileName: String, in dir: URL) -> URL {
        dir.appendingPathComponent(fileName)
    }

    /// Write atomically (temp file + rename) so the renderer never reads a torn file.
    /// Foundation's `.atomic` performs exactly the temp-write-then-rename Python's
    /// `os.replace` does. Best-effort: creates the dir; failures are swallowed so a
    /// transient FS hiccup never crashes the sim loop.
    public static func writeAtomic(_ data: Data, to fileName: String, in dir: URL) {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? data.write(to: path(fileName, in: dir), options: .atomic)
    }
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter WorldSimTests`
Expected: PASS (3 tests). Also run `swift build` and confirm zero warnings.

- [ ] **Step 7: Commit**

```bash
git add Package.swift Sources/WendyMuJoCo/WorldSim.swift Sources/WendyMuJoCo/Rounding.swift Tests/WendyMuJoCoTests/WorldSimTests.swift
git commit -m "feat(wendymujoco): target skeleton + WorldSim atomic write"
```

---

## Task 2: SceneManifest (buildScene → scene.json)

**Files:**
- Create: `Sources/WendyMuJoCo/SceneManifest.swift`
- Test: `Tests/WendyMuJoCoTests/SceneManifestTests.swift`

**Interfaces:**
- Consumes: `MjModel` (`ngeom`, `geomIsVisible`, `geomType`→`GeomType`, `geomSize`, `geomRgba`, `geomDataid`, `meshName`, `meshVertices`, `meshFaces`).
- Produces:
  - `struct Geom: Encodable { let i: Int; let type: String; let size: [Double]; let rgba: [Double]; let mesh: String? }`
  - `struct MeshBuf: Encodable { let vert: [Double]; let face: [Int] }`
  - `struct SceneManifest: Encodable { let title: String; let up: String; let engine: String; let geoms: [Geom]; let meshes: [String: MeshBuf] }`
  - `func buildScene(_ model: MjModel, title: String) -> SceneManifest`

- [ ] **Step 1: Write the failing test**

`Tests/WendyMuJoCoTests/SceneManifestTests.swift`:
```swift
import Testing
import Foundation
import MuJoCo
@testable import WendyMuJoCo

private let boxScene = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/>
    <body name="cube" pos="0 0 1">
      <freejoint/>
      <geom name="box" type="box" size="0.1 0.1 0.1" rgba="0.1 0.5 0.9 1" group="0"/>
      <geom name="hidden" type="box" size="0.1 0.1 0.1" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""

@Test func sceneListsOnlyVisibleGeomsWithTrueIndices() throws {
    let m = try MjModel.load(xml: boxScene)
    let scene = buildScene(m, title: "t")
    #expect(scene.up == "z")
    #expect(scene.engine == "mujoco")
    // floor(0) + box(1) visible; hidden(2) excluded
    #expect(scene.geoms.map(\.i) == [0, 1])
    #expect(scene.geoms[0].type == "plane")
    #expect(scene.geoms[1].type == "box")
    #expect(scene.geoms[1].rgba.count == 4)
    #expect(scene.geoms.allSatisfy { $0.mesh == nil })   // no mesh geoms here
    #expect(scene.meshes.isEmpty)
}

@Test func sceneEncodesToExpectedJSONKeys() throws {
    let m = try MjModel.load(xml: boxScene)
    let data = try JSONEncoder().encode(buildScene(m, title: "t"))
    let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(Set(obj.keys) == ["title", "up", "engine", "geoms", "meshes"])
    let g0 = (obj["geoms"] as! [[String: Any]])[0]
    #expect(Set(g0.keys) == ["i", "type", "size", "rgba"])   // no "mesh" key when nil
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter SceneManifestTests`
Expected: FAIL — `buildScene`/types not defined.

- [ ] **Step 3: Implement `SceneManifest.swift`**

```swift
import MuJoCo

public struct Geom: Encodable {
    public let i: Int
    public let type: String
    public let size: [Double]
    public let rgba: [Double]
    public let mesh: String?   // synthesized Encodable omits this key when nil
}

public struct MeshBuf: Encodable {
    public let vert: [Double]
    public let face: [Int]
}

public struct SceneManifest: Encodable {
    public let title: String
    public let up: String
    public let engine: String
    public let geoms: [Geom]
    public let meshes: [String: MeshBuf]
}

/// mjModel -> one-time scene manifest (visible geoms + deduplicated mesh buffers).
public func buildScene(_ model: MjModel, title: String) -> SceneManifest {
    var geoms: [Geom] = []
    var meshes: [String: MeshBuf] = [:]
    for i in 0..<model.ngeom where model.geomIsVisible(i) {
        let kind = model.geomType(i).rawValue
        var meshName: String? = nil
        if model.geomType(i) == .mesh {
            let mid = model.geomDataid(i)
            let name = model.meshName(mid)
            meshName = name
            if meshes[name] == nil {
                meshes[name] = MeshBuf(
                    vert: model.meshVertices(mid).map { mjRound(Double($0), 4) },
                    face: model.meshFaces(mid))
            }
        }
        geoms.append(Geom(
            i: i,
            type: kind,
            size: model.geomSize(i).map { mjRound($0, 5) },
            rgba: model.geomRgba(i).map { mjRound($0, 4) },
            mesh: meshName))
    }
    return SceneManifest(title: title, up: "z", engine: "mujoco", geoms: geoms, meshes: meshes)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter SceneManifestTests`
Expected: PASS (both). (Note: `GeomType.rawValue` for `.other` is `"other"`, which no real renderer geom uses — acceptable, matches "unknown" geoms being skipped in practice since exotic types are rare and still visible-filtered.)

- [ ] **Step 5: Commit**

```bash
git add Sources/WendyMuJoCo/SceneManifest.swift Tests/WendyMuJoCoTests/SceneManifestTests.swift
git commit -m "feat(wendymujoco): buildScene -> scene.json manifest"
```

---

## Task 3: StateFrame (buildState → state.json)

**Files:**
- Create: `Sources/WendyMuJoCo/StateFrame.swift`
- Test: `Tests/WendyMuJoCoTests/StateFrameTests.swift`

**Interfaces:**
- Consumes: `MjModel` (`ngeom`), `MjData` (`geomXpos`, `geomQuat`, `ptr` for contacts), `CMuJoCo` (`mj_contactForce`), `Vec3`, `Quat`.
- Produces:
  - `enum HUDValue: Encodable { case number(Double); case text(String) }` (encodes as a bare JSON number/string via a single-value container).
  - `struct StateFrame: Encodable { let t: Double; let frame: Int; let engine: String; let pose: [[Double]]; let contacts: [[Double]]; let hud: [String: HUDValue]; let level: Int? }`
  - `func buildState(_ model: MjModel, _ data: MjData, frame: Int, hud: [String: HUDValue], level: Int?, now: Double) -> StateFrame` (`now` is injected so tests are deterministic; the Handle passes `Date().timeIntervalSince1970`).

- [ ] **Step 1: Write the failing test**

`Tests/WendyMuJoCoTests/StateFrameTests.swift`:
```swift
import Testing
import Foundation
import MuJoCo
@testable import WendyMuJoCo

private let boxScene = """
<mujoco><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/>
  <body name="cube" pos="0 0 1"><freejoint/>
    <geom name="box" type="box" size="0.1 0.1 0.1"/>
  </body>
</worldbody></mujoco>
"""

@Test func statePoseCoversEveryGeom() throws {
    let m = try MjModel.load(xml: boxScene)
    let d = MjData(m)
    mjForward(m, d)
    let s = buildState(m, d, frame: 7, hud: ["gate": .text("1/5"), "alt": .number(1.0)],
                       level: 2, now: 1721635200.5)
    #expect(s.engine == "mujoco")
    #expect(s.frame == 7)
    #expect(s.t == 1721635200.5)
    #expect(s.pose.count == m.ngeom)          // EVERY geom, not just visible
    #expect(s.pose[1].count == 7)             // x,y,z,qw,qx,qy,qz
    #expect(abs(s.pose[1][2] - 1.0) < 1e-4)   // cube geom at z=1
    #expect(s.level == 2)
}

@Test func stateEncodesHudMixedAndOmitsNilLevel() throws {
    let m = try MjModel.load(xml: boxScene)
    let d = MjData(m); mjForward(m, d)
    let s = buildState(m, d, frame: 0, hud: ["gate": .text("2/5"), "spd": .number(3.25)],
                       level: nil, now: 1.0)
    let obj = try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(s)) as! [String: Any]
    #expect(obj["level"] == nil)                                  // omitted when nil
    let hud = obj["hud"] as! [String: Any]
    #expect(hud["gate"] as? String == "2/5")
    #expect((hud["spd"] as? Double).map { abs($0 - 3.25) < 1e-9 } == true)
    #expect(Set(obj.keys) == ["t", "frame", "engine", "pose", "contacts", "hud"])
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter StateFrameTests`
Expected: FAIL — `buildState`/`HUDValue` not defined.

- [ ] **Step 3: Implement `StateFrame.swift`**

```swift
import MuJoCo
import CMuJoCo

public enum HUDValue: Encodable {
    case number(Double)
    case text(String)
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .number(let n): try c.encode(mjRound(n, 2))
        case .text(let s): try c.encode(s)
        }
    }
}

public struct StateFrame: Encodable {
    public let t: Double
    public let frame: Int
    public let engine: String
    public let pose: [[Double]]
    public let contacts: [[Double]]
    public let hud: [String: HUDValue]
    public let level: Int?   // omitted from JSON when nil (synthesized encodeIfPresent)
}

/// mjData -> per-frame poses (every geom) + bounded contacts.
public func buildState(_ model: MjModel, _ data: MjData, frame: Int,
                       hud: [String: HUDValue], level: Int?, now: Double) -> StateFrame {
    var pose: [[Double]] = []
    pose.reserveCapacity(model.ngeom)
    for i in 0..<model.ngeom {
        let p = data.geomXpos(i)
        let q = data.geomQuat(i)   // wxyz
        pose.append([mjRound(p.x, 5), mjRound(p.y, 5), mjRound(p.z, 5),
                     mjRound(q.w, 6), mjRound(q.x, 6), mjRound(q.y, 6), mjRound(q.z, 6)])
    }
    // Contact points (world) + linear force magnitude. Raw C: MuJoCo's wrapper exposes
    // only the normal component, but the Sim tab shows ‖linear force‖ (= norm of the
    // first 3 of mj_contactForce's 6-vector), so compute it here.
    var contacts: [[Double]] = []
    let n = Swift.min(Int(data.ptr.pointee.ncon), 64)
    if n > 0 {
        var f6 = [Double](repeating: 0, count: 6)
        for i in 0..<n {
            mj_contactForce(model.ptr, data.ptr, Int32(i), &f6)
            let mag = (f6[0]*f6[0] + f6[1]*f6[1] + f6[2]*f6[2]).squareRoot()
            let cp = data.ptr.pointee.contact[i].pos   // (Double,Double,Double)
            contacts.append([mjRound(cp.0, 4), mjRound(cp.1, 4), mjRound(cp.2, 4), mjRound(mag, 2)])
        }
    }
    return StateFrame(t: now, frame: frame, engine: "mujoco",
                      pose: pose, contacts: contacts, hud: hud, level: level)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter StateFrameTests`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add Sources/WendyMuJoCo/StateFrame.swift Tests/WendyMuJoCoTests/StateFrameTests.swift
git commit -m "feat(wendymujoco): buildState -> state.json per-frame poses + contacts"
```

---

## Task 4: Control (control.json decode + read)

**Files:**
- Create: `Sources/WendyMuJoCo/Control.swift`
- Test: `Tests/WendyMuJoCoTests/ControlTests.swift`

**Interfaces:**
- Produces:
  - `struct Control: Decodable { var paused: Bool; var step: Int; var reset: Int; var poke: Int; var ctrl: [String: Double]; var qpos: [String: Double]; var qvel: [String: Double] }` with a custom `init(from:)` that defaults every field when its key is absent, and a memberwise default `init()`.
  - `func readControl(in dir: URL) -> Control` — returns `Control()` (all defaults) if the file is missing or unparseable.

- [ ] **Step 1: Write the failing test**

`Tests/WendyMuJoCoTests/ControlTests.swift`:
```swift
import Testing
import Foundation
@testable import WendyMuJoCo

@Test func controlDefaultsWhenFileMissing() {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ctl-\(UUID().uuidString)")
    let c = readControl(in: dir)   // dir doesn't exist
    #expect(c.paused == false)
    #expect(c.step == 0 && c.reset == 0 && c.poke == 0)
    #expect(c.ctrl.isEmpty && c.qpos.isEmpty && c.qvel.isEmpty)
}

@Test func controlParsesPartialJSON() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ctl-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let json = #"{"paused": true, "reset": 3, "ctrl": {"mot": 1.5}}"#
    try Data(json.utf8).write(to: dir.appendingPathComponent("control.json"))
    let c = readControl(in: dir)
    #expect(c.paused == true)
    #expect(c.reset == 3)
    #expect(c.step == 0)                // absent → default
    #expect(c.ctrl["mot"] == 1.5)
    #expect(c.qpos.isEmpty)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter ControlTests`
Expected: FAIL — `Control`/`readControl` not defined.

- [ ] **Step 3: Implement `Control.swift`**

```swift
import Foundation

public struct Control: Decodable {
    public var paused = false
    public var step = 0
    public var reset = 0
    public var poke = 0
    public var ctrl: [String: Double] = [:]
    public var qpos: [String: Double] = [:]
    public var qvel: [String: Double] = [:]

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case paused, step, reset, poke, ctrl, qpos, qvel
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        paused = (try? c.decodeIfPresent(Bool.self, forKey: .paused)) ?? nil ?? false
        step = (try? c.decodeIfPresent(Int.self, forKey: .step)) ?? nil ?? 0
        reset = (try? c.decodeIfPresent(Int.self, forKey: .reset)) ?? nil ?? 0
        poke = (try? c.decodeIfPresent(Int.self, forKey: .poke)) ?? nil ?? 0
        ctrl = (try? c.decodeIfPresent([String: Double].self, forKey: .ctrl)) ?? nil ?? [:]
        qpos = (try? c.decodeIfPresent([String: Double].self, forKey: .qpos)) ?? nil ?? [:]
        qvel = (try? c.decodeIfPresent([String: Double].self, forKey: .qvel)) ?? nil ?? [:]
    }
}

/// Read control.json from `dir`; any missing file / parse error yields all-defaults.
public func readControl(in dir: URL) -> Control {
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("control.json")),
          let c = try? JSONDecoder().decode(Control.self, from: data)
    else { return Control() }
    return c
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter ControlTests`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add Sources/WendyMuJoCo/Control.swift Tests/WendyMuJoCoTests/ControlTests.swift
git commit -m "feat(wendymujoco): control.json decode with defaults"
```

---

## Task 5: Handle (launchPassive, sync, pause/step/reset/poke/ctrl)

**Files:**
- Create: `Sources/WendyMuJoCo/Handle.swift`
- Test: `Tests/WendyMuJoCoTests/HandleTests.swift`

**Interfaces:**
- Consumes: everything above; `MjModel`/`MjData` (incl. `.ptr`, `setCtrl`, `id(of:name:)`, `joints`, `nkey`), `mjResetData`, `mjResetDataKeyframe`, `mjForward`.
- Produces:
  - `public final class Handle` with:
    - `public init(model: MjModel, data: MjData, title: String = "mujoco sim", hud: [String: HUDValue] = [:], dir: URL = WorldSim.directory())` — writes `scene.json` immediately and baselines the control counters.
    - `public func sync()` — applies reset/ctrl/poke from `control.json`, writes `state.json`, then blocks while paused (honoring reset/poke/single-step).
    - `public func hud(_ fields: [String: HUDValue])`, `public func setLevel(_ level: Int?)`, `public func isRunning() -> Bool`, `public func close()`.
  - `public func launchPassive(_ model: MjModel, _ data: MjData, title: String = "mujoco sim", hud: [String: HUDValue] = [:]) -> Handle`.
- Behavior notes: `reset` resets data (+ keyframe 0 when `nkey > 0`); `ctrl` setpoints reapply every frame; `poke` writes `qpos`/`qvel` by joint name-or-index at that joint's `qposadr`/`dofadr` then `mjForward`; counters fire only when they advance past the last-seen value. `now` for `state.json.t` is `Date().timeIntervalSince1970`.

- [ ] **Step 1: Write the failing test**

`Tests/WendyMuJoCoTests/HandleTests.swift`:
```swift
import Testing
import Foundation
import MuJoCo
@testable import WendyMuJoCo

private let pendulum = """
<mujoco><worldbody>
  <body name="pole" pos="0 0 1">
    <joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"/>
  </body>
</worldbody>
<actuator><motor name="mot" joint="hinge"/></actuator>
</mujoco>
"""

private func tempDir() -> URL {
    let d = FileManager.default.temporaryDirectory.appendingPathComponent("h-\(UUID().uuidString)")
    try! FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    return d
}
private func writeControl(_ json: String, _ dir: URL) {
    try! Data(json.utf8).write(to: dir.appendingPathComponent("control.json"))
}

@Test func initWritesSceneAndSyncWritesState() throws {
    let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
    let m = try MjModel.load(xml: pendulum); let d = MjData(m)
    let h = Handle(model: m, data: d, title: "pend", dir: dir)
    #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("scene.json").path))
    mjStep(m, d); h.sync()
    let st = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appendingPathComponent("state.json"))) as! [String: Any]
    #expect(st["frame"] as? Int == 1)
    #expect((st["pose"] as! [[Double]]).count == m.ngeom)
}

@Test func resetCounterResetsData() throws {
    let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
    let m = try MjModel.load(xml: pendulum); let d = MjData(m)
    let h = Handle(model: m, data: d, dir: dir)
    for _ in 0..<20 { mjStep(m, d) }
    #expect(d.time > 0)
    writeControl(#"{"reset": 1}"#, dir)
    h.sync()                       // sees reset counter advance 0 -> 1
    #expect(d.time == 0)
}

@Test func ctrlSetpointApplied() throws {
    let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
    let m = try MjModel.load(xml: pendulum); let d = MjData(m)
    let h = Handle(model: m, data: d, dir: dir)
    writeControl(#"{"ctrl": {"mot": 0.7}}"#, dir)
    h.sync()
    #expect(d.ctrl[0] == 0.7)      // resolved actuator "mot" -> index 0
}

@Test func pokeSetsJointPosition() throws {
    let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
    let m = try MjModel.load(xml: pendulum); let d = MjData(m)
    let h = Handle(model: m, data: d, dir: dir)
    writeControl(#"{"poke": 1, "qpos": {"hinge": 0.5}}"#, dir)
    h.sync()
    #expect(abs(d.qpos[0] - 0.5) < 1e-9)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter HandleTests`
Expected: FAIL — `Handle` not defined.

- [ ] **Step 3: Implement `Handle.swift`**

```swift
import Foundation
import MuJoCo
import CMuJoCo

public final class Handle {
    private let model: MjModel
    private let data: MjData
    private let dir: URL
    private var hudFields: [String: HUDValue]
    private var level: Int?
    private var frame = 0
    private var running = true

    private var resetSeen = 0
    private var stepSeen = 0
    private var pokeSeen = 0

    // Cache joint name/index -> (qposadr, dofadr) for pokes.
    private let jointQposAdr: [Int]   // by joint id
    private let jointDofAdr: [Int]

    public init(model: MjModel, data: MjData, title: String = "mujoco sim",
                hud: [String: HUDValue] = [:], dir: URL = WorldSim.directory()) {
        self.model = model
        self.data = data
        self.dir = dir
        self.hudFields = hud
        self.jointQposAdr = model.joints.map { $0.qposadr }
        self.jointDofAdr = model.joints.map { $0.dofadr }
        // Baseline counters so a stale flag from a previous sim doesn't fire on this one.
        let c = readControl(in: dir)
        self.resetSeen = c.reset
        self.stepSeen = c.step
        self.pokeSeen = c.poke
        writeScene(title: title)
    }

    public func isRunning() -> Bool { running }
    public func hud(_ fields: [String: HUDValue]) { hudFields = fields }
    public func setLevel(_ level: Int?) { self.level = level }
    public func close() { running = false }

    private func writeScene(title: String) {
        if let d = try? JSONEncoder().encode(buildScene(model, title: title)) {
            WorldSim.writeAtomic(d, to: "scene.json", in: dir)
        }
    }

    private func writeState() {
        frame += 1
        let s = buildState(model, data, frame: frame, hud: hudFields, level: level,
                           now: Date().timeIntervalSince1970)
        if let d = try? JSONEncoder().encode(s) {
            WorldSim.writeAtomic(d, to: "state.json", in: dir)
        }
    }

    private func resolveActuator(_ key: String) -> Int? {
        if let i = Int(key) { return (0..<model.nu).contains(i) ? i : nil }
        return model.id(of: objActuator, name: key)
    }
    private func resolveJoint(_ key: String) -> Int? {
        if let i = Int(key) { return (0..<model.njnt).contains(i) ? i : nil }
        return model.id(of: objJoint, name: key)
    }

    /// Reset when the counter advances. Returns true if it fired.
    private func applyReset(_ c: Control) -> Bool {
        guard c.reset != resetSeen else { return false }
        resetSeen = c.reset
        mjResetData(model, data)
        if model.nkey > 0 { mjResetDataKeyframe(model, data, 0) }
        return true
    }

    /// Persistent actuator setpoints, reapplied every frame so they hold.
    private func applyCtrl(_ c: Control) {
        for (k, v) in c.ctrl {
            if let aid = resolveActuator(k) { data.setCtrl(aid, v) }
        }
    }

    /// One-shot qpos/qvel poke when the counter advances. Returns true if it fired.
    private func applyPoke(_ c: Control) -> Bool {
        guard c.poke != pokeSeen else { return false }
        pokeSeen = c.poke
        for (k, v) in c.qpos {
            if let jid = resolveJoint(k) { data.ptr.pointee.qpos[jointQposAdr[jid]] = v }
        }
        for (k, v) in c.qvel {
            if let jid = resolveJoint(k) { data.ptr.pointee.qvel[jointDofAdr[jid]] = v }
        }
        mjForward(model, data)
        return true
    }

    public func sync() {
        var c = readControl(in: dir)
        _ = applyReset(c)
        applyCtrl(c)
        _ = applyPoke(c)
        writeState()
        // Pause: block after showing the frame until resumed or single-stepped.
        // Reset and pokes still take effect while paused.
        while c.paused && c.step == stepSeen && running {
            Thread.sleep(forTimeInterval: 0.04)
            c = readControl(in: dir)
            let fired = applyReset(c)
            let poked = applyPoke(c)
            if fired || poked { writeState() }
        }
        stepSeen = c.step
    }
}

/// Drop-in shaped like mujoco.viewer.launch_passive that renders in the Sim tab.
public func launchPassive(_ model: MjModel, _ data: MjData, title: String = "mujoco sim",
                          hud: [String: HUDValue] = [:]) -> Handle {
    Handle(model: model, data: data, title: title, hud: hud)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter HandleTests`
Expected: PASS (4 tests). None of these set `paused`, so the pause loop is not exercised destructively.

- [ ] **Step 5: Run the whole suite; commit**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test` — all WendyMuJoCo + MuJoCo tests pass, warning-free.
```bash
git add Sources/WendyMuJoCo/Handle.swift Tests/WendyMuJoCoTests/HandleTests.swift
git commit -m "feat(wendymujoco): Handle stream + control.json reset/ctrl/poke/pause"
```

---

## Task 6: Menagerie model loader

**Files:**
- Create: `Sources/WendyMuJoCo/Menagerie.swift`
- Test: `Tests/WendyMuJoCoTests/MenagerieTests.swift`

**Interfaces:**
- Produces:
  - `enum Menagerie` with:
    - `static let nameMap: [String: String]` (friendly → Menagerie dir), `static let vendorDirs: [String]` (`["/opt/sandbox/mujoco-menagerie"]`).
    - `static func resolveModelPath(_ name: String, searchDirs: [String], robot: Bool = false) -> String?` — returns `scene.xml` by default (nice standalone world), or the shortest non-scene `*.xml` when `robot == true`.
    - `static func fetch(_ name: String, cacheDir: URL) throws -> URL` — sparse-clone one model dir; returns the repo root.
    - `static func load(_ name: String, searchDirs: [String]? = nil, fetch: Bool = true) throws -> MjModel`.

- [ ] **Step 1: Write the failing test** (resolution only — no network)

`Tests/WendyMuJoCoTests/MenagerieTests.swift`:
```swift
import Testing
import Foundation
import MuJoCo
@testable import WendyMuJoCo

@Test func nameMapResolvesFriendlyNames() {
    #expect(Menagerie.nameMap["go2"] == "unitree_go2")
    #expect(Menagerie.nameMap["panda"] == "franka_emika_panda")
}

@Test func resolvePrefersSceneThenRobotXML() throws {
    // Build a fake vendored dir: <root>/toy/{scene.xml, toy.xml}
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("men-\(UUID().uuidString)")
    let toy = root.appendingPathComponent("toy")
    try FileManager.default.createDirectory(at: toy, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data("<mujoco/>".utf8).write(to: toy.appendingPathComponent("scene.xml"))
    try Data("<mujoco/>".utf8).write(to: toy.appendingPathComponent("toy.xml"))

    let scene = Menagerie.resolveModelPath("toy", searchDirs: [root.path], robot: false)
    #expect(scene?.hasSuffix("toy/scene.xml") == true)
    let robot = Menagerie.resolveModelPath("toy", searchDirs: [root.path], robot: true)
    #expect(robot?.hasSuffix("toy/toy.xml") == true)
    #expect(Menagerie.resolveModelPath("missing", searchDirs: [root.path]) == nil)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter MenagerieTests`
Expected: FAIL — `Menagerie` not defined.

- [ ] **Step 3: Implement `Menagerie.swift`**

```swift
import Foundation
import MuJoCo

public enum Menagerie {
    public static let repoURL = "https://github.com/google-deepmind/mujoco_menagerie"
    public static let vendorDirs = ["/opt/sandbox/mujoco-menagerie"]

    public static let nameMap: [String: String] = [
        "franka_panda": "franka_emika_panda", "panda": "franka_emika_panda",
        "franka_emika_panda": "franka_emika_panda",
        "fr3": "franka_fr3", "franka_fr3": "franka_fr3",
        "go2": "unitree_go2", "unitree_go2": "unitree_go2",
        "so101": "robotstudio_so101", "so_arm101": "robotstudio_so101",
        "robotstudio_so101": "robotstudio_so101",
        "so100": "trs_so_arm100", "so_arm100": "trs_so_arm100",
        "trs_so_arm100": "trs_so_arm100",
        "spot": "boston_dynamics_spot", "boston_dynamics_spot": "boston_dynamics_spot",
    ]

    static func dirName(_ name: String) -> String { nameMap[name] ?? name }

    /// scene.xml by default (floor+lights); the shortest non-scene *.xml when robot==true.
    public static func resolveModelPath(_ name: String, searchDirs: [String],
                                        robot: Bool = false) -> String? {
        let fm = FileManager.default
        let d = dirName(name)
        for root in searchDirs {
            let base = (root as NSString).appendingPathComponent(d)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: base, isDirectory: &isDir), isDir.boolValue else { continue }
            let xmls = ((try? fm.contentsOfDirectory(atPath: base)) ?? [])
                .filter { $0.hasSuffix(".xml") }.sorted()
            let nonScene = xmls.filter { $0 != "scene.xml" }
            if robot, let r = nonScene.min(by: { $0.count < $1.count }) {
                return (base as NSString).appendingPathComponent(r)
            }
            if xmls.contains("scene.xml") {
                return (base as NSString).appendingPathComponent("scene.xml")
            }
            if let first = nonScene.first {
                return (base as NSString).appendingPathComponent(first)
            }
        }
        return nil
    }

    /// Sparse-clone one model dir into `cacheDir`; returns the repo root path.
    @discardableResult
    public static func fetch(_ name: String, cacheDir: URL) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let repo = cacheDir.appendingPathComponent("mujoco_menagerie")
        if !fm.fileExists(atPath: repo.appendingPathComponent(".git").path) {
            try run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                     repoURL, repo.path])
        }
        try run(["git", "-C", repo.path, "sparse-checkout", "add", dirName(name)])
        return repo
    }

    public static func load(_ name: String, searchDirs: [String]? = nil,
                            fetch shouldFetch: Bool = true) throws -> MjModel {
        let dirs = searchDirs ?? vendorDirs
        if let p = resolveModelPath(name, searchDirs: dirs) {
            return try MjModel.load(xmlPath: p)
        }
        if shouldFetch {
            let cache = WorldSim.directory().appendingPathComponent("menagerie-cache")
            let repo = try fetch(name, cacheDir: cache)
            if let p = resolveModelPath(name, searchDirs: [repo.path]) {
                return try MjModel.load(xmlPath: p)
            }
        }
        throw MjError("MuJoCo model '\(name)' not found. Vendored: \(Set(nameMap.values).sorted()). "
                      + "Pass a raw Menagerie dir name or load via MjModel.load(xmlPath:).")
    }

    private static func run(_ args: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = args
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw MjError("command failed (\(p.terminationStatus)): \(args.joined(separator: " "))")
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter MenagerieTests`
Expected: PASS (both). The network fetch path is not exercised by unit tests (no network dependency in CI); it is covered later by the drone sample end-to-end.

- [ ] **Step 5: Run whole suite; commit**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test` — full suite passes, warning-free.
```bash
git add Sources/WendyMuJoCo/Menagerie.swift Tests/WendyMuJoCoTests/MenagerieTests.swift
git commit -m "feat(wendymujoco): Menagerie model resolve + sparse-clone fetch"
```

---

## Self-Review

**Spec coverage** (WendyMuJoCo scope minus the explicitly-deferred `ctl.sock` endpoint and `Scene`):
- Model loading / Menagerie resolve+fetch → Task 6 ✓
- scene.json (`buildScene`, visibility filter, mesh dedup) → Task 2 ✓
- state.json (`buildState`, per-geom pose, wxyz quat, bounded contacts, hud, level) → Task 3 ✓
- control.json polling (paused/step/reset/poke/ctrl) → Tasks 4 + 5 ✓
- Handle/launchPassive (sync, hud, isRunning, setLevel, close), atomic writes → Tasks 1 + 5 ✓
- `MuJoCo` module stays Wendy-free (all new code is in the `WendyMuJoCo` target) ✓
- Deferred, stated: `ctl.sock` endpoint, `Scene` composition (needs `MjSpec.attach`), Rerun mirror ✓

**Placeholder scan:** none — every step has full code and exact commands.

**Type consistency across tasks:** `WorldSim.writeAtomic/path/directory` (T1) used by T5; `mjRound` (T1) used by T2/T3; `SceneManifest`/`buildScene` (T2) used by T5.writeScene; `StateFrame`/`HUDValue`/`buildState(now:)` (T3) used by T5.writeState; `Control`/`readControl(in:)` (T4) used by T5.sync; `Menagerie.load` (T6) is independent. `HUDValue` is the same type threaded through `Handle.hud(_:)`. `Handle` uses `MjModel.joints[].qposadr/dofadr`, `id(of:name:)`, `setCtrl`, `.ptr`, `mjResetData`/`mjResetDataKeyframe`/`mjForward` — all real swift-mujoco APIs from the completed Tasks 1–9.

**Known accepted risks:**
- `data.ptr.pointee.qpos[adr] = v` in poke uses the public `ptr` escape hatch by design (the chosen approach over adding setters). Indices come from `MjModel.joints[].qposadr/dofadr`, which are valid for the model that produced them.
- `contacts` force magnitude is recomputed via raw `mj_contactForce` (not `MjData.contacts().forceNormal`, which is only the normal component) to match the Python's ‖linear force‖ exactly.
- `Handle` is not `Sendable` (wraps MuJoCo mutable state); it must be driven from one thread, like the Python. The `ctl.sock` follow-up will add the cross-thread request queue.
