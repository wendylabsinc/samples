# DroneRace Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `samples/swift/drone/starters/drone-slalom/mujoco_drone_race.py` to Swift — a Skydio X2 quadrotor flying a 5-gate slalom under a geometric controller, streaming live to the 🕹 Sim tab via `WendyMuJoCo` — as a buildable, runnable sample in `samples/swift/drone/`.

**Architecture:** A SwiftPM package in `samples/swift/drone/` with a pure, unit-tested `DroneCore` library (controller math, rotor mixer, course-XML builder, gate progression) and a thin `DroneRace` executable that resolves/loads the Skydio X2, builds the gate course, and runs the control loop against `WendyMuJoCo.launchPassive`. It depends on the sibling `swift-mujoco` package (products `WendyMuJoCo` + `MuJoCo`) via a local path dependency.

**Tech Stack:** Swift 6.1, Swift Testing, Foundation; the `swift-mujoco` package (`MuJoCo`: `Vec3`/`Mat3`/`Quat`/`quat2Mat`/`MjModel`/`MjData`/`mjStep`/`mjResetDataKeyframe`; `WendyMuJoCo`: `launchPassive`/`Handle`/`HUDValue`/`Menagerie`). MuJoCo 3.10.0 at `$HOME/.local`.

## Global Constraints

- Swift tools `6.1`; Swift Testing (`import Testing`), NOT XCTest.
- No `.unsafeFlags` in `Package.swift`.
- Every `swift build`/`swift test`/`swift run` runs with `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig` exported (transitively needed by `CMuJoCo`).
- Pristine build — zero warnings.
- Pure logic (controller, rotor mix, course XML, gate progression) lives in `DroneCore` and is unit-tested; the executable `DroneRace` is the thin I/O/loop wiring.
- Controller behavior must match the Python reference (`mujoco_drone_race.py`): same gains, rotor-mixer geometry (`AX=0.14, AY=0.18, CZ=0.0201`), thrust clip `[0, 13]`, gate list, `GATE_OPENING=1.6`, `REACH=1.1`, `G=9.81`, yaw-hold.
- The Skydio X2 free-joint layout: `qpos[0..2]` = world position, `qpos[3..6]` = orientation quaternion (wxyz); `qvel[3..5]` = body-frame angular velocity. World linear velocity is finite-differenced from position (matching the Python), not read from `qvel`.
- The X2 has 4 thrust actuators (`nu == 4`); `MjData.setCtrl([Double])` requires exactly `nu` values.

## Dependency wiring

`samples/swift/drone` is inside the `wendy/samples` git repo; `swift-mujoco` is a sibling repo at `wendy/swift-mujoco`. From `samples/swift/drone`, that is `../../../swift-mujoco`. Use a local path dependency:
```swift
dependencies: [ .package(path: "../../../swift-mujoco") ],
```
(Follow-up, out of scope: once `swift-mujoco` is published, switch to a versioned git URL so external users can build the sample without the sibling checkout. Note this in the sample README.)

## File Structure

```
samples/swift/drone/
  Package.swift
  README.md
  Sources/
    DroneCore/
      DroneController.swift   # gains, rotorMix, control(...)
      Course.swift            # gateFrame, buildCourseXML, advanceGate
    DroneRace/
      main.swift              # resolve X2, build course, load, control loop, launchPassive
  Tests/
    DroneCoreTests/
      ControllerTests.swift
      CourseTests.swift
  starters/drone-slalom/mujoco_drone_race.py   # existing Python reference (unchanged)
```

---

## Task 1: Package + skeleton (builds against swift-mujoco)

**Files:**
- Create: `samples/swift/drone/Package.swift`
- Create: `samples/swift/drone/Sources/DroneCore/DroneController.swift` (stub type so the target compiles)
- Create: `samples/swift/drone/Sources/DroneRace/main.swift` (stub)
- Create: `samples/swift/drone/.gitignore`
- Test: `samples/swift/drone/Tests/DroneCoreTests/ControllerTests.swift` (trivial build-proof test)

**Interfaces:**
- Produces: a `DroneCore` library target, a `DroneRace` executable target (deps `DroneCore` + `WendyMuJoCo` + `MuJoCo`), a `DroneCoreTests` test target; all building against the `swift-mujoco` path dependency.

- [ ] **Step 1: Create `.gitignore`**
```
.build/
.swiftpm/
```

- [ ] **Step 2: Create `Package.swift`**
```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "DroneRace",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../../swift-mujoco"),
    ],
    targets: [
        .target(name: "DroneCore", dependencies: [
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
        .executableTarget(name: "DroneRace", dependencies: [
            "DroneCore",
            .product(name: "WendyMuJoCo", package: "swift-mujoco"),
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
        .testTarget(name: "DroneCoreTests", dependencies: [
            "DroneCore",
            .product(name: "MuJoCo", package: "swift-mujoco"),
        ]),
    ]
)
```

- [ ] **Step 3: Create stub sources**

`Sources/DroneCore/DroneController.swift`:
```swift
import MuJoCo

/// Geometric quadrotor controller (position -> desired tilt -> body torques -> rotor mix).
public struct DroneController {
    public init() {}
}
```

`Sources/DroneRace/main.swift`:
```swift
import DroneCore
import MuJoCo
import WendyMuJoCo

print("DroneRace: MuJoCo \(mujocoVersion())")
```

- [ ] **Step 4: Write a trivial build-proof test**

`Tests/DroneCoreTests/ControllerTests.swift`:
```swift
import Testing
@testable import DroneCore

@Test func controllerConstructs() {
    _ = DroneController()
    #expect(Bool(true))
}
```

- [ ] **Step 5: Build & test**

Run:
```bash
cd samples/swift/drone
PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test
```
Expected: resolves the `swift-mujoco` path dependency, builds `DroneCore`/`DroneRace`/tests, 1 test passes, zero warnings. (If dependency resolution fails, confirm the relative path `../../../swift-mujoco` reaches `/Users/joannisorlandos/git/wendy/swift-mujoco`.)

- [ ] **Step 6: Commit**
```bash
git add samples/swift/drone/Package.swift samples/swift/drone/.gitignore samples/swift/drone/Sources samples/swift/drone/Tests
git commit -m "feat(drone): SwiftPM package skeleton building against swift-mujoco"
```

---

## Task 2: DroneController (rotor mix + geometric control)

**Files:**
- Modify: `Sources/DroneCore/DroneController.swift`
- Test: `Tests/DroneCoreTests/ControllerTests.swift`

**Interfaces:**
- Consumes: `Vec3`, `Mat3` (from `MuJoCo`).
- Produces on `DroneController`:
  - `func rotorMix(_ T: Double, _ tx: Double, _ ty: Double, _ tz: Double) -> [Double]` (4 thrusts, clipped `[0,13]`).
  - `func control(position: Vec3, rotation: Mat3, velocity: Vec3, angularVelocity: Vec3, target: Vec3, mass: Double) -> [Double]` (4 rotor thrusts).
  - public gain/geometry constants matching the Python.

- [ ] **Step 1: Write the failing tests**

`Tests/DroneCoreTests/ControllerTests.swift` (replace the stub test):
```swift
import Testing
import MuJoCo
@testable import DroneCore

private let I = Mat3([1,0,0, 0,1,0, 0,0,1])   // level attitude

@Test func hoverAtTargetGivesEqualThrustsSummingToWeight() {
    let c = DroneController()
    let mass = 2.0
    let p = Vec3(0, 0, 1.5)
    let thrusts = c.control(position: p, rotation: I, velocity: Vec3(0,0,0),
                            angularVelocity: Vec3(0,0,0), target: p, mass: mass)
    #expect(thrusts.count == 4)
    let hover = mass * 9.81 / 4
    for t in thrusts { #expect(abs(t - hover) < 1e-6) }          // all four equal
    #expect(abs(thrusts.reduce(0,+) - mass * 9.81) < 1e-6)        // sum == weight
}

@Test func targetAboveIncreasesTotalThrust() {
    let c = DroneController()
    let mass = 2.0
    let p = Vec3(0, 0, 1.0)
    let hoverSum = mass * 9.81
    let up = c.control(position: p, rotation: I, velocity: Vec3(0,0,0),
                       angularVelocity: Vec3(0,0,0), target: Vec3(0, 0, 2.0), mass: mass)
    #expect(up.reduce(0,+) > hoverSum)   // climbs -> more total thrust than hover
}

@Test func rotorMixClipsToRange() {
    let c = DroneController()
    let t = c.rotorMix(1000, 0, 0, 0)    // huge thrust
    #expect(t.allSatisfy { $0 <= 13.0 && $0 >= 0.0 })
    let z = c.rotorMix(-1000, 0, 0, 0)   // negative -> clipped to 0
    #expect(z.allSatisfy { $0 == 0.0 })
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter ControllerTests`
Expected: FAIL — `control`/`rotorMix` not defined.

- [ ] **Step 3: Implement the controller**

`Sources/DroneCore/DroneController.swift`:
```swift
import Foundation
import MuJoCo

/// Geometric quadrotor controller: position error -> desired thrust axis -> reduced-attitude
/// torque -> rotor thrust mix. Ports mujoco_drone_race.py (Skydio X2).
public struct DroneController {
    // Position PD gains (per axis).
    public let kpPos = Vec3(1.1, 1.1, 10.0)
    public let kdPos = Vec3(2.2, 2.2, 6.0)
    // Attitude / yaw gains.
    public let kpAtt = 9.0, kdAtt = 1.2
    public let kpYaw = 2.0, kdYaw = 0.4
    public let g = 9.81
    // Rotor mixer geometry (from x2.xml): arm half-spans and yaw-reaction coefficient.
    public let ax = 0.14, ay = 0.18, cz = 0.0201
    public let thrustMax = 13.0

    public init() {}

    /// Mix a collective thrust T and body torques (tx,ty,tz) into 4 clipped rotor thrusts.
    public func rotorMix(_ T: Double, _ tx: Double, _ ty: Double, _ tz: Double) -> [Double] {
        let X = tx / (4 * ay), Y = ty / (4 * ax), Z = tz / (4 * cz)
        let t = [T/4 - X + Y - Z, T/4 + X + Y + Z, T/4 + X - Y - Z, T/4 - X - Y + Z]
        return t.map { Swift.min(Swift.max($0, 0.0), thrustMax) }
    }

    /// 4 rotor thrusts for the current state. `rotation` is world<-body; `angularVelocity`
    /// is body-frame; `velocity` is world-frame.
    public func control(position p: Vec3, rotation R: Mat3, velocity v: Vec3,
                        angularVelocity omega: Vec3, target tgt: Vec3, mass: Double) -> [Double] {
        let b3 = R.column(2)                       // body z-axis (thrust dir) in world
        let aDes = Vec3(kpPos.x * (tgt.x - p.x) + kdPos.x * (-v.x),
                        kpPos.y * (tgt.y - p.y) + kdPos.y * (-v.y),
                        kpPos.z * (tgt.z - p.z) + kdPos.z * (-v.z) + g)
        let T = mass * aDes.dot(b3)
        let b3des = aDes.normalized
        // Reduced-attitude control: rotate the current thrust axis toward b3des.
        let eWorld = b3.cross(b3des)
        let eBody = R.transposeTimes(eWorld)
        var tau = Vec3(kpAtt * eBody.x - kdAtt * omega.x,
                       kpAtt * eBody.y - kdAtt * omega.y,
                       kpAtt * eBody.z - kdAtt * omega.z)
        // Hold yaw ~0 (nose down +x).
        let b1 = R.column(0)
        let yaw = atan2(b1.y, b1.x)
        tau.z += -kpYaw * yaw - kdYaw * omega.z
        return rotorMix(T, tau.x, tau.y, tau.z)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter ControllerTests`
Expected: PASS (3 tests). Hover: with level attitude, zero velocity, target==position, `aDes=(0,0,g)`, `b3=(0,0,1)`, `T=mass·g`, all torques zero → four equal `mass·g/4` thrusts.

- [ ] **Step 5: Commit**
```bash
git add samples/swift/drone/Sources/DroneCore/DroneController.swift samples/swift/drone/Tests/DroneCoreTests/ControllerTests.swift
git commit -m "feat(drone): geometric controller + rotor mixer (ported from Python)"
```

---

## Task 3: Course XML + gate progression

**Files:**
- Create: `Sources/DroneCore/Course.swift`
- Test: `Tests/DroneCoreTests/CourseTests.swift`

**Interfaces:**
- Consumes: `Vec3`.
- Produces (free functions in `DroneCore`):
  - `func gateFrame(index: Int, x: Double, y: Double, z: Double, opening: Double) -> String` (4 box `<geom>`s).
  - `func buildCourseXML(gates: [(Double, Double, Double)], opening: Double) -> String` (includes `x2.xml`, a floor, and all gate frames).
  - `func advanceGate(position: Vec3, gates: [Vec3], current: Int, reach: Double) -> Int`.
  - `let defaultGates: [(Double, Double, Double)]` and `let gateOpening`, `let reach` constants matching the Python.

- [ ] **Step 1: Write the failing tests**

`Tests/DroneCoreTests/CourseTests.swift`:
```swift
import Testing
import MuJoCo
@testable import DroneCore

@Test func courseHasFloorIncludeAndFourBoxesPerGate() {
    let gates = [(4.0, 0.0, 1.5), (8.0, 1.0, 1.6)]
    let xml = buildCourseXML(gates: gates, opening: 1.6)
    #expect(xml.contains("<include file=\"x2.xml\"/>"))
    #expect(xml.contains("type=\"plane\""))                       // floor
    // 4 box geoms per gate
    let boxes = xml.components(separatedBy: "type=\"box\"").count - 1
    #expect(boxes == gates.count * 4)
}

@Test func defaultCourseHasFiveGates() {
    #expect(defaultGates.count == 5)
    #expect(gateOpening == 1.6)
    #expect(reach == 1.1)
}

@Test func gateAdvancesWithinReachAndClampsAtEnd() {
    let gates = [Vec3(4,0,1.5), Vec3(8,1,1.6)]
    // far from gate 0 -> stays
    #expect(advanceGate(position: Vec3(0,0,1), gates: gates, current: 0, reach: 1.1) == 0)
    // within reach of gate 0 -> advances to 1
    #expect(advanceGate(position: Vec3(4,0,1.5), gates: gates, current: 0, reach: 1.1) == 1)
    // within reach of last gate -> clamps at last index
    #expect(advanceGate(position: Vec3(8,1,1.6), gates: gates, current: 1, reach: 1.1) == 1)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter CourseTests`
Expected: FAIL — course functions not defined.

- [ ] **Step 3: Implement `Course.swift`**

```swift
import Foundation
import MuJoCo

public let defaultGates: [(Double, Double, Double)] =
    [(4, 0.0, 1.5), (8, 1.0, 1.6), (12, 0.0, 1.5), (16, -1.0, 1.6), (20, 0.0, 1.5)]
public let gateOpening = 1.6
public let reach = 1.1

/// A square gate frame from 4 thin boxes (welded to the world), colour-coded by index.
public func gateFrame(index i: Int, x gx: Double, y gy: Double, z gz: Double,
                      opening w: Double) -> String {
    let h = w / 2 + 0.08
    let col = String(format: "%.2f 0.8 %.2f 1", 0.15 + 0.15 * Double(i), 0.9 - 0.12 * Double(i))
    // top, bottom, left, right bars (x is the depth axis the drone flies along)
    let posts: [(Double, Double, Double, Double, Double, Double)] = [
        (gx, gy, gz + h, 0.06, w/2 + 0.12, 0.06),   // top bar (spans y)
        (gx, gy, gz - h, 0.06, w/2 + 0.12, 0.06),   // bottom bar
        (gx, gy + w/2 + 0.06, gz, 0.06, 0.06, h),   // left post (spans z)
        (gx, gy - w/2 - 0.06, gz, 0.06, 0.06, h),   // right post
    ]
    return posts.map { (px, py, pz, sx, sy, sz) in
        "<geom type=\"box\" pos=\"\(px) \(py) \(pz)\" size=\"\(sx) \(sy) \(sz)\" "
        + "rgba=\"\(col)\" contype=\"1\" conaffinity=\"1\"/>"
    }.joined()
}

/// Wrap the vendored Skydio X2 (included by relative name) in a gate-slalom world.
public func buildCourseXML(gates: [(Double, Double, Double)], opening: Double) -> String {
    let gatesXML = gates.enumerated()
        .map { (i, g) in gateFrame(index: i, x: g.0, y: g.1, z: g.2, opening: opening) }
        .joined()
    return """
    <mujoco>
      <include file="x2.xml"/>
      <worldbody>
        <geom name="floor" type="plane" size="40 40 0.1" rgba="0.2 0.23 0.28 1"/>
        \(gatesXML)
      </worldbody>
    </mujoco>
    """
}

/// Advance to the next gate once within `reach` of the current one; clamp at the last gate.
public func advanceGate(position p: Vec3, gates: [Vec3], current: Int, reach: Double) -> Int {
    guard current < gates.count else { return current }
    if (p - gates[current]).norm < reach {
        return Swift.min(current + 1, gates.count - 1)
    }
    return current
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift test --filter CourseTests`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add samples/swift/drone/Sources/DroneCore/Course.swift samples/swift/drone/Tests/DroneCoreTests/CourseTests.swift
git commit -m "feat(drone): gate-course XML builder + gate progression"
```

---

## Task 4: DroneRace executable (resolve X2, load course, fly, stream)

**Files:**
- Modify: `Sources/DroneRace/main.swift`
- Create: `samples/swift/drone/README.md`

**Interfaces:**
- Consumes: `DroneCore` (`DroneController`, `buildCourseXML`, `advanceGate`, `defaultGates`, `gateOpening`, `reach`), `MuJoCo` (`MjModel`, `MjData`, `Vec3`, `Quat`, `quat2Mat`, `mjStep`, `mjResetDataKeyframe`, `mjForward`), `WendyMuJoCo` (`launchPassive`, `HUDValue`, `Menagerie`, `WorldSim`).
- Produces: a `DroneRace` executable that resolves/fetches the Skydio X2, composes the course, loads it, and runs the control loop streaming to the Sim tab. Honors `DRONE_MAX_STEPS` (bounded headless run for demos/CI) — unset means run until the process is stopped.

- [ ] **Step 1: Implement `main.swift`**

```swift
import Foundation
import DroneCore
import MuJoCo
import WendyMuJoCo

// --- Resolve the Skydio X2 model dir (vendored or fetched), copy it to a writable work
// dir, and drop a course.xml beside it so <include file="x2.xml"/> + its assets/ resolve. ---
func prepareCourse() throws -> String {
    // The bare robot XML (x2.xml), not scene.xml, so we control the world.
    var x2Path = Menagerie.resolveModelPath("skydio_x2", searchDirs: Menagerie.vendorDirs, robot: true)
    if x2Path == nil {
        let cache = WorldSim.directory().appendingPathComponent("menagerie-cache")
        let repo = try Menagerie.fetch("skydio_x2", cacheDir: cache)
        x2Path = Menagerie.resolveModelPath("skydio_x2", searchDirs: [repo.path], robot: true)
    }
    guard let x2 = x2Path else {
        throw MjError("could not resolve or fetch the skydio_x2 model")
    }
    let modelDir = URL(fileURLWithPath: x2).deletingLastPathComponent()
    let work = WorldSim.directory().appendingPathComponent("drone_work")
    let fm = FileManager.default
    try? fm.removeItem(at: work)
    try fm.createDirectory(at: work.deletingLastPathComponent(), withIntermediateDirectories: true)
    try fm.copyItem(at: modelDir, to: work)
    let course = work.appendingPathComponent("course.xml")
    try Data(buildCourseXML(gates: defaultGates, opening: gateOpening).utf8).write(to: course)
    return course.path
}

let coursePath = try prepareCourse()
let model = try MjModel.load(xmlPath: coursePath)
let data = MjData(model)
mjResetDataKeyframe(model, data, 0)   // hover keyframe from x2.xml
mjForward(model, data)

// Total mass via the raw handle (MuJoCo doesn't wrap body_mass).
var mass = 0.0
for i in 0..<model.nbody { mass += model.ptr.pointee.body_mass[i] }

let controller = DroneController()
let dt = model.timestep
let gateVecs = defaultGates.map { Vec3($0.0, $0.1, $0.2) }
var prevP = Vec3(data.qpos[0], data.qpos[1], data.qpos[2])
var targetI = 0
let t0 = data.time
var gateTimes: [Double] = []

let maxSteps = ProcessInfo.processInfo.environment["DRONE_MAX_STEPS"].flatMap { Int($0) }
let handle = launchPassive(model, data, title: "drone race")

var step = 0
while handle.isRunning() {
    let q = data.qpos
    let p = Vec3(q[0], q[1], q[2])
    let R = quat2Mat(Quat(w: q[3], x: q[4], y: q[5], z: q[6]))
    let v = (p - prevP) * (1.0 / dt)
    prevP = p
    let qv = data.qvel
    let omega = Vec3(qv[3], qv[4], qv[5])

    let thrusts = controller.control(position: p, rotation: R, velocity: v,
                                     angularVelocity: omega, target: gateVecs[targetI], mass: mass)
    data.setCtrl(thrusts)
    mjStep(model, data)
    step += 1

    // Gate progression.
    let newTarget = advanceGate(position: p, gates: gateVecs, current: targetI, reach: reach)
    if newTarget != targetI {
        gateTimes.append((data.time - t0 * 1).rounded() )   // placeholder; replaced below
    }
    if (p - gateVecs[targetI]).norm < reach && targetI < gateVecs.count {
        gateTimes.append(((data.time - t0) * 100).rounded() / 100)
    }
    targetI = newTarget

    if step % 5 == 0 {
        let speed = v.norm
        handle.hud([
            "gate": .text("\(Swift.min(targetI + 1, gateVecs.count))/\(gateVecs.count)"),
            "t": .number(((data.time - t0) * 100).rounded() / 100),
            "speed": .number((speed * 100).rounded() / 100),
            "x": .number((p.x * 10).rounded() / 10),
            "alt": .number((p.z * 100).rounded() / 100),
        ])
    }
    handle.sync()

    if let maxSteps, step >= maxSteps {
        print("DroneRace: ran \(step) steps; gates reached: \(Swift.min(targetI + 1, gateVecs.count))/\(gateVecs.count); "
              + "final alt=\(((p.z)*100).rounded()/100)m, x=\((p.x*10).rounded()/10)m")
        break
    }
    if maxSteps == nil { Thread.sleep(forTimeInterval: dt) }   // real-time when streaming live
}
```

Note: simplify the gate-time bookkeeping while implementing — the intent is only to log split times when the drone reaches a gate; keep exactly one append per gate-reach and drop the placeholder line. The controller/course/progression logic is what's tested; this loop is the wiring.

- [ ] **Step 2: Build**

Run:
```bash
cd samples/swift/drone
PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift build
```
Expected: builds `DroneRace`, zero warnings. Fix any compile error against the real `MuJoCo`/`WendyMuJoCo` API (e.g. confirm `data.qpos`/`data.qvel` return `[Double]`, `model.ptr.pointee.body_mass` is accessible, `setCtrl([Double])` takes exactly `nu==4` values).

- [ ] **Step 3: Bounded smoke run**

Run (fetches the X2 on first run — needs network; writes to a temp slot dir so it doesn't clobber a live sim):
```bash
cd samples/swift/drone
WENDY_WORLDSIM_DIR=$(mktemp -d) DRONE_MAX_STEPS=1500 \
  PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig swift run DroneRace
```
Expected: prints a summary line after 1500 steps (≈3s of sim at dt≈0.002) showing gates reached and a plausible altitude (~1.5m, near gate height) — proving the drone loads, is controlled, and climbs/flies rather than falling to the floor. If it immediately drops to alt≈0, the controller wiring (quat order, mass, ctrl mapping) is wrong — debug before declaring done.

- [ ] **Step 4: Write `README.md`**
```markdown
# Drone slalom (Swift)

A Skydio X2 quadrotor flying a 5-gate slalom under a geometric controller, streamed live to
the Wendy Sandbox 🕹 Sim tab via `WendyMuJoCo`. Swift port of `starters/drone-slalom/mujoco_drone_race.py`.

## Build & run
Requires the sibling `swift-mujoco` package and MuJoCo installed (see that repo's README):

    export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig
    swift run DroneRace                       # streams to the Sim tab until stopped
    DRONE_MAX_STEPS=1500 swift run DroneRace  # bounded headless run (prints a summary)

Edit `defaultGates` / controller gains in `Sources/DroneCore/` and rebuild.

> Dependency note: this sample uses a local path dependency on `../../../swift-mujoco`.
> Once `swift-mujoco` is published, switch to a versioned git-URL dependency.
```

- [ ] **Step 5: Commit**
```bash
git add samples/swift/drone/Sources/DroneRace/main.swift samples/swift/drone/README.md
git commit -m "feat(drone): DroneRace executable — X2 slalom streamed to the Sim tab"
```

---

## Self-Review

**Spec coverage** (vs the design's DroneRace component: "Swift port of mujoco_drone_race.py; controller reuses MuJoCo math; course via MJCF include + gate boxes; streams via WendyMuJoCo"):
- Controller (PD position, reduced-attitude, yaw hold, rotor mix) → Task 2 ✓
- Course (gate frames + include x2.xml + floor) + gate progression → Task 3 ✓
- X2 resolve/fetch + copy + course compose + control loop + HUD + launchPassive → Task 4 ✓
- Depends on WendyMuJoCo/MuJoCo via path dependency → Task 1 ✓
- No linear-algebra dependency (uses MuJoCo's `Vec3`/`Mat3`/`quat2Mat`) ✓

**Placeholder scan:** Task 4's loop has a deliberately-flagged simplification note for gate-time logging (the tested logic is in DroneCore); every other step has complete code. The gate-time bookkeeping is cosmetic (HUD/print only) and explicitly called out to tidy during implementation — not a hidden TODO in the load-bearing logic.

**Type consistency:** `DroneController.control(position:rotation:velocity:angularVelocity:target:mass:)` and `rotorMix` (Task 2) are called with exactly those labels in Task 4. `buildCourseXML(gates:opening:)`, `advanceGate(position:gates:current:reach:)`, `defaultGates`, `gateOpening`, `reach` (Task 3) are used verbatim in Task 4. `Vec3`/`Mat3`/`Quat`/`quat2Mat` are the real `MuJoCo` types. `launchPassive(_:_:title:)`, `Handle.hud([String:HUDValue])`, `Handle.sync()`, `Handle.isRunning()`, `Menagerie.resolveModelPath`/`.fetch`/`.vendorDirs`, `WorldSim.directory()` are the real `WendyMuJoCo` API.

**Known risks:**
- Task 4's smoke run needs network (first-run X2 fetch) and is not a `swift test` unit test — the unit-tested surface is DroneCore; the executable is validated by the bounded run. Stated in the task.
- `setCtrl([Double])` requires exactly `nu` values; the X2 has 4 actuators and `rotorMix` returns 4. If a future model differs, the precondition traps — acceptable for this drone-specific sample.
- The controller finite-differences world velocity from position (matching the Python), so the first frame's `v` is ~0 (prevP initialized to the start position) — correct.
