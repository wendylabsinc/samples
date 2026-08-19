# Drone Sim-to-Real: Shared Flight Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the target-agnostic flight core — value types, the `FlightIO` seam, the `SafetyKernel`, the geometric attitude+thrust controller, and the `FlightLoop` driver — verified entirely against a `FakeFlightIO`, so the same code is provably ready to run behind either a sim or a real adapter.

**Architecture:** One SwiftPM package with two libraries. `DroneCore` holds pure value types, the `FlightIO` protocol, and the `Envelope` clamp. `DroneApp` holds the `SafetyKernel` state machine, the `GeometricController`, and the `FlightLoop` that wires `readState → safety check → control → clamp → send`. Nothing here imports MuJoCo or MAVLink; adapters land in later plans and plug into `FlightIO`.

**Tech Stack:** Swift 6.2 (tools-version 6.2), Swift Testing (`import Testing`, `@Test`), no external dependencies. Builds on macOS 14+ and Linux.

## Global Constraints

- Swift tools-version **6.2**; language mode Swift 6 (strict concurrency). Copied from `samples/.swift-version` (6.2.3) and sibling packages.
- Platform floor **`.macOS(.v14)`** (matches sibling swift samples); must also compile on Linux — **no Foundation-only APIs where a Swift-stdlib equivalent exists**, no platform `#if` needed in this plan's code.
- **Quaternion convention is `wxyz`** (scalar-first), matching the drone-sim spec's protocol contract. Every quaternion in every task uses this order.
- **Coordinate frame is z-up** (world up = `(0, 0, 1)`), matching the drone-sim spec (`up:"z"`).
- All shared types are **`Sendable`**. `FlightIO` is an `async`, `Sendable` protocol.
- Tests use **Swift Testing** (`@Test`, `#expect`, `#require`) — never XCTest.
- Package name **`drone`**, located at `samples/swift/drone/`.
- Commit after every task with a Conventional Commits message.

---

## File Structure

```
samples/swift/drone/
├── Package.swift                              # 2 libs + 2 test targets
├── Sources/
│   ├── DroneCore/
│   │   ├── Math.swift                         # Vec3, Quat (wxyz) + operations
│   │   ├── DroneState.swift                   # DroneState, LinkHealth, FlightMode
│   │   ├── AttitudeThrust.swift               # AttitudeThrust setpoint
│   │   ├── FlightIO.swift                     # FlightIO protocol
│   │   └── Envelope.swift                     # Envelope + clamp/breach
│   └── DroneApp/
│       ├── SafetyKernel.swift                 # pure state machine
│       ├── GeometricController.swift          # position → attitude+thrust
│       └── FlightLoop.swift                   # one control iteration
└── Tests/
    ├── DroneCoreTests/
    │   ├── MathTests.swift
    │   └── EnvelopeTests.swift
    └── DroneAppTests/
        ├── FakeFlightIO.swift                 # test double for FlightIO
        ├── SafetyKernelTests.swift
        ├── GeometricControllerTests.swift
        └── FlightLoopTests.swift
```

**Responsibilities:** `Math` = linear algebra only. `DroneState`/`AttitudeThrust` = the seam's data. `FlightIO` = the seam's behavior. `Envelope` = limit enforcement. `SafetyKernel` = *when* commanding is allowed. `GeometricController` = *what* to command. `FlightLoop` = glue that guarantees the loop invariant. `FakeFlightIO` = deterministic, programmable adapter for tests (reused by later plans).

---

### Task 1: Package scaffold + math primitives

**Files:**
- Create: `samples/swift/drone/Package.swift`
- Create: `samples/swift/drone/Sources/DroneCore/Math.swift`
- Test: `samples/swift/drone/Tests/DroneCoreTests/MathTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `struct Vec3: Sendable, Equatable { var x, y, z: Double }` with `+`, `-`, `*` (scalar, both orders), `dot(_:)`, `cross(_:)`, `length: Double`, `normalized(): Vec3`, and `static let zero`, `static let up` (= `Vec3(0,0,1)`).
  - `struct Quat: Sendable, Equatable { var w, x, y, z: Double }` with `static let identity`, `length: Double`, `normalized(): Quat`, `init(desiredZ: Vec3, yaw: Double)` (builds orientation whose body-z aligns with `desiredZ`, at world yaw `yaw`), and `func angle(to axis: Vec3) -> Double` returning the angle (radians) between the quaternion's body-z axis and `axis`.

- [ ] **Step 1: Write the failing test**

`samples/swift/drone/Tests/DroneCoreTests/MathTests.swift`:

```swift
import Testing
@testable import DroneCore

@Test func vec3CrossAndDot() {
    let x = Vec3(x: 1, y: 0, z: 0)
    let y = Vec3(x: 0, y: 1, z: 0)
    #expect(x.cross(y) == Vec3(x: 0, y: 0, z: 1))
    #expect(x.dot(y) == 0)
    #expect(x.dot(x) == 1)
}

@Test func vec3Normalize() {
    let v = Vec3(x: 0, y: 0, z: 5).normalized()
    #expect(abs(v.length - 1) < 1e-12)
    #expect(abs(v.z - 1) < 1e-12)
}

@Test func quatFromDesiredZLevelIsIdentityUp() {
    // Desired body-z pointing straight up, yaw 0 → level attitude.
    let q = Quat(desiredZ: .up, yaw: 0).normalized()
    #expect(abs(q.length - 1) < 1e-9)
    #expect(q.angle(to: .up) < 1e-6)   // body-z coincides with world up
}

@Test func quatFromDesiredZTilted() {
    // Desired body-z tilted toward +x → body-z axis has positive x.
    let desired = Vec3(x: 0.3, y: 0, z: 1).normalized()
    let q = Quat(desiredZ: desired, yaw: 0)
    #expect(q.angle(to: .up) > 0.1)    // tilted away from vertical
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/swift/drone && swift test --filter DroneCoreTests.MathTests`
Expected: FAIL — no such module `DroneCore` / `Package.swift` missing.

- [ ] **Step 3: Write minimal implementation**

`samples/swift/drone/Package.swift`:

```swift
// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "drone",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .target(name: "DroneCore"),
        .target(name: "DroneApp", dependencies: ["DroneCore"]),
        .testTarget(name: "DroneCoreTests", dependencies: ["DroneCore"]),
        .testTarget(name: "DroneAppTests", dependencies: ["DroneApp"]),
    ]
)
```

`samples/swift/drone/Sources/DroneCore/Math.swift`:

```swift
public struct Vec3: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x; self.y = y; self.z = z
    }

    public static let zero = Vec3(x: 0, y: 0, z: 0)
    public static let up = Vec3(x: 0, y: 0, z: 1)

    public static func + (a: Vec3, b: Vec3) -> Vec3 {
        Vec3(x: a.x + b.x, y: a.y + b.y, z: a.z + b.z)
    }
    public static func - (a: Vec3, b: Vec3) -> Vec3 {
        Vec3(x: a.x - b.x, y: a.y - b.y, z: a.z - b.z)
    }
    public static func * (a: Vec3, s: Double) -> Vec3 {
        Vec3(x: a.x * s, y: a.y * s, z: a.z * s)
    }
    public static func * (s: Double, a: Vec3) -> Vec3 { a * s }

    public func dot(_ o: Vec3) -> Double { x * o.x + y * o.y + z * o.z }

    public func cross(_ o: Vec3) -> Vec3 {
        Vec3(x: y * o.z - z * o.y,
             y: z * o.x - x * o.z,
             z: x * o.y - y * o.x)
    }

    public var length: Double { dot(self).squareRoot() }

    public func normalized() -> Vec3 {
        let l = length
        return l > 0 ? self * (1 / l) : self
    }
}

public struct Quat: Sendable, Equatable {
    public var w: Double
    public var x: Double
    public var y: Double
    public var z: Double

    public init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w; self.x = x; self.y = y; self.z = z
    }

    public static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    public var length: Double {
        (w * w + x * x + y * y + z * z).squareRoot()
    }

    public func normalized() -> Quat {
        let l = length
        guard l > 0 else { return .identity }
        return Quat(w: w / l, x: x / l, y: y / l, z: z / l)
    }

    /// Orientation whose body-z axis aligns with `desiredZ`, at world yaw `yaw`.
    /// Columns b1,b2,b3 form the rotation matrix; converted to a wxyz quaternion.
    public init(desiredZ: Vec3, yaw: Double) {
        let b3 = desiredZ.normalized()
        // Desired heading direction in the world x-y plane.
        let b1c = Vec3(x: Foundation_cos(yaw), y: Foundation_sin(yaw), z: 0)
        var b2 = b3.cross(b1c)
        if b2.length < 1e-9 {
            // b3 parallel to b1c (pointing along heading) — pick any orthogonal.
            b2 = b3.cross(Vec3(x: 1, y: 0, z: 0))
            if b2.length < 1e-9 { b2 = b3.cross(Vec3(x: 0, y: 1, z: 0)) }
        }
        b2 = b2.normalized()
        let b1 = b2.cross(b3)
        self = Quat.fromColumns(b1: b1, b2: b2, b3: b3).normalized()
    }

    /// Build a wxyz quaternion from rotation-matrix columns (Shepperd's method).
    static func fromColumns(b1: Vec3, b2: Vec3, b3: Vec3) -> Quat {
        // Matrix m[row][col], columns are b1,b2,b3.
        let m00 = b1.x, m10 = b1.y, m20 = b1.z
        let m01 = b2.x, m11 = b2.y, m21 = b2.z
        let m02 = b3.x, m12 = b3.y, m22 = b3.z
        let trace = m00 + m11 + m22
        if trace > 0 {
            let s = (trace + 1).squareRoot() * 2  // s = 4*w
            return Quat(w: 0.25 * s,
                        x: (m21 - m12) / s,
                        y: (m02 - m20) / s,
                        z: (m10 - m01) / s)
        } else if m00 > m11 && m00 > m22 {
            let s = (1 + m00 - m11 - m22).squareRoot() * 2  // s = 4*x
            return Quat(w: (m21 - m12) / s,
                        x: 0.25 * s,
                        y: (m01 + m10) / s,
                        z: (m02 + m20) / s)
        } else if m11 > m22 {
            let s = (1 + m11 - m00 - m22).squareRoot() * 2  // s = 4*y
            return Quat(w: (m02 - m20) / s,
                        x: (m01 + m10) / s,
                        y: 0.25 * s,
                        z: (m12 + m21) / s)
        } else {
            let s = (1 + m22 - m00 - m11).squareRoot() * 2  // s = 4*z
            return Quat(w: (m10 - m01) / s,
                        x: (m02 + m20) / s,
                        y: (m12 + m21) / s,
                        z: 0.25 * s)
        }
    }

    /// Body-z axis expressed in the world frame (third column of the rotation matrix).
    public var bodyZ: Vec3 {
        Vec3(x: 2 * (x * z + w * y),
             y: 2 * (y * z - w * x),
             z: 1 - 2 * (x * x + y * y))
    }

    /// Angle in radians between this orientation's body-z axis and `axis`.
    public func angle(to axis: Vec3) -> Double {
        let d = bodyZ.normalized().dot(axis.normalized())
        return Foundation_acos(min(1, max(-1, d)))
    }
}
```

Add trig shims at the bottom of `Math.swift` so the file stays Foundation-free at the call sites (Glibc/Darwin provide these):

```swift
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@inline(__always) func Foundation_cos(_ v: Double) -> Double { cos(v) }
@inline(__always) func Foundation_sin(_ v: Double) -> Double { sin(v) }
@inline(__always) func Foundation_acos(_ v: Double) -> Double { acos(v) }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/swift/drone && swift test --filter DroneCoreTests.MathTests`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd samples/swift/drone
git add Package.swift Sources/DroneCore/Math.swift Tests/DroneCoreTests/MathTests.swift
git commit -m "feat(drone): add package scaffold and Vec3/Quat math primitives"
```

---

### Task 2: Seam data types, `FlightIO` protocol, and `Envelope`

**Files:**
- Create: `samples/swift/drone/Sources/DroneCore/DroneState.swift`
- Create: `samples/swift/drone/Sources/DroneCore/AttitudeThrust.swift`
- Create: `samples/swift/drone/Sources/DroneCore/FlightIO.swift`
- Create: `samples/swift/drone/Sources/DroneCore/Envelope.swift`
- Test: `samples/swift/drone/Tests/DroneCoreTests/EnvelopeTests.swift`

**Interfaces:**
- Consumes: `Vec3`, `Quat` (Task 1).
- Produces:
  - `enum FlightMode: Sendable { case unknown, manual, offboard }`
  - `struct LinkHealth: Sendable, Equatable { var armed: Bool; var mode: FlightMode; var lastUpdateAge: Double; var batteryVolts: Double? }`
  - `struct DroneState: Sendable, Equatable { var t: Double; var position: Vec3; var velocity: Vec3; var attitude: Quat; var bodyRates: Vec3; var health: LinkHealth }`
  - `struct AttitudeThrust: Sendable, Equatable { var attitude: Quat; var thrust: Double }`
  - `protocol FlightIO: Sendable` with `func connect() async throws`, `func arm() async throws`, `func engageOffboard() async throws`, `func readState() async throws -> DroneState`, `func send(_ sp: AttitudeThrust) async throws`, `func heartbeat() async throws`, `func handback() async`, `func kill() async`.
  - `struct Envelope: Sendable { var posMin, posMax: Vec3; var maxTiltRadians: Double; var thrustMin, thrustMax: Double }` with `func clamp(_ sp: AttitudeThrust) -> AttitudeThrust` (clamps thrust to `[thrustMin, thrustMax]` and limits tilt of the setpoint's body-z away from `Vec3.up` to `maxTiltRadians`) and `func breaches(position p: Vec3) -> Bool` (true iff `p` is outside the `[posMin, posMax]` box).

- [ ] **Step 1: Write the failing test**

`samples/swift/drone/Tests/DroneCoreTests/EnvelopeTests.swift`:

```swift
import Testing
@testable import DroneCore

private let env = Envelope(
    posMin: Vec3(x: -10, y: -10, z: 0),
    posMax: Vec3(x: 10, y: 10, z: 20),
    maxTiltRadians: 0.5,          // ~28.6°
    thrustMin: 0.05,
    thrustMax: 0.9
)

@Test func clampLimitsThrust() {
    let hot = AttitudeThrust(attitude: .identity, thrust: 2.0)
    #expect(env.clamp(hot).thrust == 0.9)
    let cold = AttitudeThrust(attitude: .identity, thrust: -1.0)
    #expect(env.clamp(cold).thrust == 0.05)
}

@Test func clampLimitsTilt() {
    // Desired body-z tilted 60° from vertical — must be clamped to ~28.6°.
    let steep = Quat(desiredZ: Vec3(x: 1, y: 0, z: 0.577), yaw: 0) // ~60°
    let sp = AttitudeThrust(attitude: steep, thrust: 0.5)
    let clamped = env.clamp(sp)
    #expect(clamped.attitude.angle(to: .up) <= 0.5 + 1e-6)
    #expect(clamped.attitude.angle(to: .up) > 0.4)  // clamped to the limit, not zeroed
}

@Test func clampLeavesGentleTiltUntouched() {
    let gentle = Quat(desiredZ: Vec3(x: 0.1, y: 0, z: 1), yaw: 0) // ~5.7°
    let sp = AttitudeThrust(attitude: gentle, thrust: 0.5)
    let clamped = env.clamp(sp)
    #expect(abs(clamped.attitude.angle(to: .up) - gentle.angle(to: .up)) < 1e-6)
}

@Test func breachesDetectsGeofence() {
    #expect(env.breaches(position: Vec3(x: 0, y: 0, z: 5)) == false)
    #expect(env.breaches(position: Vec3(x: 11, y: 0, z: 5)) == true)   // x past max
    #expect(env.breaches(position: Vec3(x: 0, y: 0, z: -1)) == true)   // below floor
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/swift/drone && swift test --filter DroneCoreTests.EnvelopeTests`
Expected: FAIL — `Envelope` / `AttitudeThrust` undefined.

- [ ] **Step 3: Write minimal implementation**

`samples/swift/drone/Sources/DroneCore/DroneState.swift`:

```swift
public enum FlightMode: Sendable, Equatable {
    case unknown, manual, offboard
}

public struct LinkHealth: Sendable, Equatable {
    public var armed: Bool
    public var mode: FlightMode
    public var lastUpdateAge: Double   // seconds since the estimate was produced
    public var batteryVolts: Double?

    public init(armed: Bool, mode: FlightMode, lastUpdateAge: Double, batteryVolts: Double? = nil) {
        self.armed = armed; self.mode = mode
        self.lastUpdateAge = lastUpdateAge; self.batteryVolts = batteryVolts
    }
}

public struct DroneState: Sendable, Equatable {
    public var t: Double
    public var position: Vec3
    public var velocity: Vec3
    public var attitude: Quat        // wxyz
    public var bodyRates: Vec3       // p, q, r
    public var health: LinkHealth

    public init(t: Double, position: Vec3, velocity: Vec3,
                attitude: Quat, bodyRates: Vec3, health: LinkHealth) {
        self.t = t; self.position = position; self.velocity = velocity
        self.attitude = attitude; self.bodyRates = bodyRates; self.health = health
    }
}
```

`samples/swift/drone/Sources/DroneCore/AttitudeThrust.swift`:

```swift
public struct AttitudeThrust: Sendable, Equatable {
    public var attitude: Quat     // desired orientation, wxyz
    public var thrust: Double     // collective, normalized 0…1

    public init(attitude: Quat, thrust: Double) {
        self.attitude = attitude; self.thrust = thrust
    }
}
```

`samples/swift/drone/Sources/DroneCore/FlightIO.swift`:

```swift
/// The single seam between the shared flight app and the world.
/// Sim (`SimIO`) and real (`MavlinkIO`) adapters conform in later plans.
public protocol FlightIO: Sendable {
    func connect() async throws
    func arm() async throws
    func engageOffboard() async throws
    func readState() async throws -> DroneState
    func send(_ sp: AttitudeThrust) async throws
    func heartbeat() async throws
    func handback() async
    func kill() async
}
```

`samples/swift/drone/Sources/DroneCore/Envelope.swift`:

```swift
public struct Envelope: Sendable {
    public var posMin: Vec3
    public var posMax: Vec3
    public var maxTiltRadians: Double
    public var thrustMin: Double
    public var thrustMax: Double

    public init(posMin: Vec3, posMax: Vec3, maxTiltRadians: Double,
                thrustMin: Double, thrustMax: Double) {
        self.posMin = posMin; self.posMax = posMax
        self.maxTiltRadians = maxTiltRadians
        self.thrustMin = thrustMin; self.thrustMax = thrustMax
    }

    /// Soft clamp: bound thrust and limit setpoint tilt. Keeps flying.
    public func clamp(_ sp: AttitudeThrust) -> AttitudeThrust {
        let thrust = min(thrustMax, max(thrustMin, sp.thrust))
        let tilt = sp.attitude.angle(to: .up)
        guard tilt > maxTiltRadians else {
            return AttitudeThrust(attitude: sp.attitude, thrust: thrust)
        }
        // Re-derive a setpoint at the tilt limit, preserving tilt direction & yaw.
        let z = sp.attitude.bodyZ.normalized()
        // Direction of tilt in the horizontal plane.
        var horiz = Vec3(x: z.x, y: z.y, z: 0)
        if horiz.length < 1e-9 {
            return AttitudeThrust(attitude: sp.attitude, thrust: thrust)
        }
        horiz = horiz.normalized()
        let limited = (horiz * Foundation_sin(maxTiltRadians))
            + (Vec3.up * Foundation_cos(maxTiltRadians))
        let yaw = Foundation_atan2(z.y, z.x)  // keep heading of tilt as yaw proxy
        let q = Quat(desiredZ: limited, yaw: yaw)
        return AttitudeThrust(attitude: q, thrust: thrust)
    }

    /// Hard geofence: is the position outside the allowed box?
    public func breaches(position p: Vec3) -> Bool {
        p.x < posMin.x || p.x > posMax.x
            || p.y < posMin.y || p.y > posMax.y
            || p.z < posMin.z || p.z > posMax.z
    }
}
```

Add the `atan2` shim to the trig block at the bottom of `Math.swift` (Task 1):

```swift
@inline(__always) func Foundation_atan2(_ y: Double, _ x: Double) -> Double { atan2(y, x) }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/swift/drone && swift test --filter DroneCoreTests.EnvelopeTests`
Expected: PASS (4 tests). Also run `swift test --filter DroneCoreTests` — all Task 1 + Task 2 tests pass.

- [ ] **Step 5: Commit**

```bash
cd samples/swift/drone
git add Sources/DroneCore Tests/DroneCoreTests/EnvelopeTests.swift
git commit -m "feat(drone): add seam data types, FlightIO protocol, and Envelope"
```

---

### Task 3: `SafetyKernel` state machine + `FakeFlightIO`

**Files:**
- Create: `samples/swift/drone/Sources/DroneApp/SafetyKernel.swift`
- Create: `samples/swift/drone/Tests/DroneAppTests/FakeFlightIO.swift`
- Test: `samples/swift/drone/Tests/DroneAppTests/SafetyKernelTests.swift`

**Interfaces:**
- Consumes: `Vec3`, `Envelope`, `DroneState`, `AttitudeThrust`, `FlightIO`, `LinkHealth`, `FlightMode` (Tasks 1–2).
- Produces:
  - `enum FlightState: Sendable, Equatable { case disconnected, connected, armed, offboardActive, handback, killed }`
  - `enum GuardDecision: Sendable, Equatable { case command, handback, kill, reject }`
  - `struct SafetyKernel: Sendable` with `private(set) var state: FlightState` (starts `.disconnected`), `let envelope: Envelope`, `let heartbeatDeadline: Double`, and mutating methods:
    - `mutating func didConnect()` (`.disconnected` → `.connected`)
    - `mutating func didArm()` (only `.connected` → `.armed`; otherwise no-op)
    - `mutating func didEngageOffboard()` (only `.armed` → `.offboardActive`; otherwise no-op)
    - `mutating func requestKill()` (any state → `.killed`)
    - `mutating func check(position: Vec3, lastUpdateAge: Double, sendAge: Double) -> GuardDecision`
  - `FakeFlightIO`: `final class FakeFlightIO: FlightIO, @unchecked Sendable` — test double (see Step 3).

- [ ] **Step 1: Write the failing test**

`samples/swift/drone/Tests/DroneAppTests/SafetyKernelTests.swift`:

```swift
import Testing
import DroneCore
@testable import DroneApp

private func makeKernel() -> SafetyKernel {
    let env = Envelope(posMin: Vec3(x: -10, y: -10, z: 0),
                       posMax: Vec3(x: 10, y: 10, z: 20),
                       maxTiltRadians: 0.6, thrustMin: 0.05, thrustMax: 0.9)
    return SafetyKernel(envelope: env, heartbeatDeadline: 0.1)
}

private let inside = Vec3(x: 0, y: 0, z: 5)

@Test func rejectsCommandBeforeOffboard() {
    var k = makeKernel()
    #expect(k.check(position: inside, lastUpdateAge: 0, sendAge: 0) == .reject)
    k.didConnect()
    #expect(k.check(position: inside, lastUpdateAge: 0, sendAge: 0) == .reject)
    k.didArm()
    #expect(k.check(position: inside, lastUpdateAge: 0, sendAge: 0) == .reject)
}

@Test func armGateRequiresConnectFirst() {
    var k = makeKernel()
    k.didArm()                       // ignored — not connected
    #expect(k.state == .disconnected)
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    #expect(k.state == .offboardActive)
}

@Test func commandsWhenHealthyAndOffboard() {
    var k = makeKernel()
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    #expect(k.check(position: inside, lastUpdateAge: 0.01, sendAge: 0.01) == .command)
}

@Test func handbackOnStaleEstimate() {
    var k = makeKernel()
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    let d = k.check(position: inside, lastUpdateAge: 0.5, sendAge: 0.01) // > 0.1 deadline
    #expect(d == .handback)
    #expect(k.state == .handback)
}

@Test func handbackOnStaleSend() {
    var k = makeKernel()
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    #expect(k.check(position: inside, lastUpdateAge: 0.01, sendAge: 0.5) == .handback)
}

@Test func handbackOnGeofenceBreach() {
    var k = makeKernel()
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    let d = k.check(position: Vec3(x: 99, y: 0, z: 5), lastUpdateAge: 0, sendAge: 0)
    #expect(d == .handback)
    #expect(k.state == .handback)
}

@Test func killIsTerminal() {
    var k = makeKernel()
    k.didConnect(); k.didArm(); k.didEngageOffboard()
    k.requestKill()
    #expect(k.state == .killed)
    #expect(k.check(position: inside, lastUpdateAge: 0, sendAge: 0) == .kill)
    k.didConnect(); k.didArm(); k.didEngageOffboard()   // all ignored
    #expect(k.state == .killed)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.SafetyKernelTests`
Expected: FAIL — `SafetyKernel` undefined.

- [ ] **Step 3: Write minimal implementation**

`samples/swift/drone/Sources/DroneApp/SafetyKernel.swift`:

```swift
import DroneCore

public enum FlightState: Sendable, Equatable {
    case disconnected, connected, armed, offboardActive, handback, killed
}

public enum GuardDecision: Sendable, Equatable {
    case command   // safe to send a (clamped) setpoint
    case handback  // hand control to FC/pilot
    case kill      // stop commanding entirely
    case reject    // not yet allowed to command; do nothing
}

/// Pure decision-making state machine. Never performs I/O — the FlightLoop
/// realizes each decision through the adapter. Total: every state × input
/// has a defined outcome.
public struct SafetyKernel: Sendable {
    public private(set) var state: FlightState = .disconnected
    public let envelope: Envelope
    public let heartbeatDeadline: Double

    public init(envelope: Envelope, heartbeatDeadline: Double) {
        self.envelope = envelope
        self.heartbeatDeadline = heartbeatDeadline
    }

    public mutating func didConnect() {
        if state == .disconnected { state = .connected }
    }

    public mutating func didArm() {
        if state == .connected { state = .armed }
    }

    public mutating func didEngageOffboard() {
        if state == .armed { state = .offboardActive }
    }

    public mutating func requestKill() {
        state = .killed
    }

    public mutating func check(position: Vec3,
                               lastUpdateAge: Double,
                               sendAge: Double) -> GuardDecision {
        if state == .killed { return .kill }
        if state == .handback { return .handback }
        guard state == .offboardActive else { return .reject }

        if lastUpdateAge > heartbeatDeadline || sendAge > heartbeatDeadline {
            state = .handback
            return .handback
        }
        if envelope.breaches(position: position) {
            state = .handback
            return .handback
        }
        return .command
    }
}
```

`samples/swift/drone/Tests/DroneAppTests/FakeFlightIO.swift`:

```swift
import DroneCore

/// Deterministic, programmable FlightIO for tests. Records calls and returns
/// a caller-supplied DroneState. Test-only; single-threaded use, hence
/// @unchecked Sendable.
final class FakeFlightIO: FlightIO, @unchecked Sendable {
    var nextState: DroneState
    var connectCalls = 0
    var armCalls = 0
    var offboardCalls = 0
    var sent: [AttitudeThrust] = []
    var heartbeats = 0
    var handbackCalled = false
    var killCalled = false

    init(nextState: DroneState) { self.nextState = nextState }

    func connect() async throws { connectCalls += 1 }
    func arm() async throws { armCalls += 1 }
    func engageOffboard() async throws { offboardCalls += 1 }
    func readState() async throws -> DroneState { nextState }
    func send(_ sp: AttitudeThrust) async throws { sent.append(sp) }
    func heartbeat() async throws { heartbeats += 1 }
    func handback() async { handbackCalled = true }
    func kill() async { killCalled = true }
}

extension DroneState {
    /// Convenience builder for tests.
    static func at(position: Vec3, lastUpdateAge: Double) -> DroneState {
        DroneState(
            t: 0, position: position, velocity: .zero,
            attitude: .identity, bodyRates: .zero,
            health: LinkHealth(armed: true, mode: .offboard,
                               lastUpdateAge: lastUpdateAge)
        )
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.SafetyKernelTests`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
cd samples/swift/drone
git add Sources/DroneApp/SafetyKernel.swift Tests/DroneAppTests/FakeFlightIO.swift Tests/DroneAppTests/SafetyKernelTests.swift
git commit -m "feat(drone): add SafetyKernel state machine and FakeFlightIO test double"
```

---

### Task 4: `GeometricController` (position → attitude+thrust)

**Files:**
- Create: `samples/swift/drone/Sources/DroneApp/GeometricController.swift`
- Test: `samples/swift/drone/Tests/DroneAppTests/GeometricControllerTests.swift`

**Interfaces:**
- Consumes: `Vec3`, `Quat`, `DroneState`, `AttitudeThrust` (Tasks 1–2).
- Produces:
  - `struct ControllerGains: Sendable { var kp, kd: Double }`
  - `struct GeometricController: Sendable` with `init(mass: Double, gravity: Double, maxThrustForce: Double, gains: ControllerGains)` and `func compute(state: DroneState, positionTarget: Vec3, velocityTarget: Vec3 = .zero, yawTarget: Double = 0) -> AttitudeThrust`. Output body-z aligns with the desired acceleration (incl. gravity compensation); thrust is `‖m·a_des‖ / maxThrustForce`, unclamped (the `Envelope` clamps downstream).

- [ ] **Step 1: Write the failing test**

`samples/swift/drone/Tests/DroneAppTests/GeometricControllerTests.swift`:

```swift
import Testing
import DroneCore
@testable import DroneApp

private func makeController() -> GeometricController {
    // Hover thrust fraction = m*g / maxThrustForce = 1*9.81 / 19.62 = 0.5
    GeometricController(mass: 1.0, gravity: 9.81, maxThrustForce: 19.62,
                        gains: ControllerGains(kp: 2.0, kd: 1.5))
}

private func hoverState(at p: Vec3) -> DroneState {
    DroneState(t: 0, position: p, velocity: .zero, attitude: .identity,
               bodyRates: .zero,
               health: LinkHealth(armed: true, mode: .offboard, lastUpdateAge: 0))
}

@Test func hoverAtTargetIsLevelHalfThrust() {
    let c = makeController()
    let s = hoverState(at: Vec3(x: 0, y: 0, z: 5))
    let sp = c.compute(state: s, positionTarget: Vec3(x: 0, y: 0, z: 5))
    #expect(sp.attitude.angle(to: .up) < 1e-6)          // level
    #expect(abs(sp.thrust - 0.5) < 1e-3)                // hover fraction
}

@Test func lateralTargetTiltsTowardIt() {
    let c = makeController()
    let s = hoverState(at: Vec3(x: 0, y: 0, z: 5))
    // Target 2m in +x → must accelerate +x → body-z tilts so its x-component > 0.
    let sp = c.compute(state: s, positionTarget: Vec3(x: 2, y: 0, z: 5))
    #expect(sp.attitude.bodyZ.x > 0.05)
    #expect(sp.attitude.angle(to: .up) > 0.05)          // actually tilted
}

@Test func climbTargetIncreasesThrust() {
    let c = makeController()
    let s = hoverState(at: Vec3(x: 0, y: 0, z: 5))
    let sp = c.compute(state: s, positionTarget: Vec3(x: 0, y: 0, z: 8)) // climb
    #expect(sp.thrust > 0.5)                            // more than hover
    #expect(sp.attitude.angle(to: .up) < 1e-6)          // straight up, still level
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.GeometricControllerTests`
Expected: FAIL — `GeometricController` undefined.

- [ ] **Step 3: Write minimal implementation**

`samples/swift/drone/Sources/DroneApp/GeometricController.swift`:

```swift
import DroneCore

public struct ControllerGains: Sendable {
    public var kp: Double
    public var kd: Double
    public init(kp: Double, kd: Double) { self.kp = kp; self.kd = kd }
}

/// Geometric position controller. Produces an attitude+thrust setpoint —
/// the seam level. The FC (real) or SimIO's mixer model (sim) runs the
/// inner rate/rotor loop below this.
public struct GeometricController: Sendable {
    public let mass: Double
    public let gravity: Double
    public let maxThrustForce: Double
    public let gains: ControllerGains

    public init(mass: Double, gravity: Double, maxThrustForce: Double,
                gains: ControllerGains) {
        self.mass = mass; self.gravity = gravity
        self.maxThrustForce = maxThrustForce; self.gains = gains
    }

    public func compute(state: DroneState,
                        positionTarget: Vec3,
                        velocityTarget: Vec3 = .zero,
                        yawTarget: Double = 0) -> AttitudeThrust {
        // Desired acceleration: PD on position/velocity + gravity compensation.
        let ePos = positionTarget - state.position
        let eVel = velocityTarget - state.velocity
        let aDes = (ePos * gains.kp) + (eVel * gains.kd) + (Vec3.up * gravity)

        // Desired body-z aligns with the total thrust direction.
        let desiredZ = aDes.normalized()
        let attitude = Quat(desiredZ: desiredZ, yaw: yawTarget)

        // Collective thrust magnitude as a fraction of max available force.
        let force = mass * aDes.length
        let thrust = force / maxThrustForce
        return AttitudeThrust(attitude: attitude, thrust: thrust)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.GeometricControllerTests`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd samples/swift/drone
git add Sources/DroneApp/GeometricController.swift Tests/DroneAppTests/GeometricControllerTests.swift
git commit -m "feat(drone): add geometric attitude+thrust position controller"
```

---

### Task 5: `FlightLoop` — one control iteration with the loop invariant

**Files:**
- Create: `samples/swift/drone/Sources/DroneApp/FlightLoop.swift`
- Test: `samples/swift/drone/Tests/DroneAppTests/FlightLoopTests.swift`

**Interfaces:**
- Consumes: `FlightIO`, `DroneState`, `AttitudeThrust`, `SafetyKernel`, `GuardDecision`, `GeometricController`, `Envelope`, `Vec3` (Tasks 1–4).
- Produces:
  - `struct FlightLoop<IO: FlightIO>: Sendable` with `init(io: IO, kernel: SafetyKernel, controller: GeometricController, positionTarget: Vec3)` and:
    - `mutating func bringUp() async throws` — `connect` → `didConnect` → `arm` → `didArm` → `engageOffboard` → `didEngageOffboard`, driving both the adapter and the kernel.
    - `mutating func tick(now: Double) async throws` — one iteration: `readState`, `kernel.check(position:lastUpdateAge:sendAge:)`, then act on the decision. On `.command`: compute → `envelope.clamp` → `send` + `heartbeat`, and record the send time. On `.handback`: `io.handback()`. On `.kill`: `io.kill()`. On `.reject`: nothing. Guarantees the invariant: each tick either sends one clamped setpoint or hands back/kills — never a stale or unclamped setpoint.
    - `private(set) var lastSendTime: Double?`

- [ ] **Step 1: Write the failing test**

`samples/swift/drone/Tests/DroneAppTests/FlightLoopTests.swift`:

```swift
import Testing
import DroneCore
@testable import DroneApp

private func makeParts(statePos: Vec3, lastUpdateAge: Double)
    -> (FakeFlightIO, SafetyKernel, GeometricController) {
    let io = FakeFlightIO(nextState: .at(position: statePos, lastUpdateAge: lastUpdateAge))
    let env = Envelope(posMin: Vec3(x: -10, y: -10, z: 0),
                       posMax: Vec3(x: 10, y: 10, z: 20),
                       maxTiltRadians: 0.3, thrustMin: 0.05, thrustMax: 0.9)
    let kernel = SafetyKernel(envelope: env, heartbeatDeadline: 0.1)
    let controller = GeometricController(mass: 1.0, gravity: 9.81, maxThrustForce: 19.62,
                                         gains: ControllerGains(kp: 2.0, kd: 1.5))
    return (io, kernel, controller)
}

@Test func bringUpDrivesAdapterAndKernel() async throws {
    let (io, kernel, controller) = makeParts(statePos: Vec3(x: 0, y: 0, z: 5), lastUpdateAge: 0)
    var loop = FlightLoop(io: io, kernel: kernel, controller: controller,
                          positionTarget: Vec3(x: 0, y: 0, z: 5))
    try await loop.bringUp()
    #expect(io.connectCalls == 1)
    #expect(io.armCalls == 1)
    #expect(io.offboardCalls == 1)
}

@Test func healthyTickSendsClampedSetpoint() async throws {
    // Big lateral target would demand >0.3 rad tilt; envelope must clamp it.
    let (io, kernel, controller) = makeParts(statePos: Vec3(x: 0, y: 0, z: 5), lastUpdateAge: 0.01)
    var loop = FlightLoop(io: io, kernel: kernel, controller: controller,
                          positionTarget: Vec3(x: 50, y: 0, z: 5))
    try await loop.bringUp()
    try await loop.tick(now: 1.0)
    #expect(io.sent.count == 1)
    #expect(io.heartbeats == 1)
    #expect(io.handbackCalled == false)
    // Invariant: what was sent is within the tilt envelope.
    #expect(io.sent[0].attitude.angle(to: .up) <= 0.3 + 1e-6)
    #expect(io.sent[0].thrust <= 0.9 && io.sent[0].thrust >= 0.05)
}

@Test func staleEstimateHandsBackInsteadOfSending() async throws {
    let (io, kernel, controller) = makeParts(statePos: Vec3(x: 0, y: 0, z: 5), lastUpdateAge: 0.5)
    var loop = FlightLoop(io: io, kernel: kernel, controller: controller,
                          positionTarget: Vec3(x: 0, y: 0, z: 5))
    try await loop.bringUp()
    try await loop.tick(now: 1.0)
    #expect(io.sent.isEmpty)                // never a stale setpoint
    #expect(io.handbackCalled == true)
}

@Test func rejectBeforeBringUpSendsNothing() async throws {
    let (io, kernel, controller) = makeParts(statePos: Vec3(x: 0, y: 0, z: 5), lastUpdateAge: 0.01)
    var loop = FlightLoop(io: io, kernel: kernel, controller: controller,
                          positionTarget: Vec3(x: 0, y: 0, z: 5))
    try await loop.tick(now: 1.0)           // no bringUp() → kernel rejects
    #expect(io.sent.isEmpty)
    #expect(io.handbackCalled == false)
    #expect(io.killCalled == false)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.FlightLoopTests`
Expected: FAIL — `FlightLoop` undefined.

- [ ] **Step 3: Write minimal implementation**

`samples/swift/drone/Sources/DroneApp/FlightLoop.swift`:

```swift
import DroneCore

/// Drives one control iteration through the seam, enforcing the loop
/// invariant: each tick either sends exactly one clamped setpoint, or hands
/// back / kills — never nothing-with-motors-live, never a stale/unclamped one.
public struct FlightLoop<IO: FlightIO>: Sendable {
    public let io: IO
    public var kernel: SafetyKernel
    public let controller: GeometricController
    public let positionTarget: Vec3
    public private(set) var lastSendTime: Double?

    public init(io: IO, kernel: SafetyKernel, controller: GeometricController,
                positionTarget: Vec3) {
        self.io = io; self.kernel = kernel
        self.controller = controller; self.positionTarget = positionTarget
    }

    /// Connect, arm, and engage offboard — advancing the adapter and the
    /// kernel in lockstep so the arm/mode gate is honored.
    public mutating func bringUp() async throws {
        try await io.connect();        kernel.didConnect()
        try await io.arm();            kernel.didArm()
        try await io.engageOffboard(); kernel.didEngageOffboard()
    }

    public mutating func tick(now: Double) async throws {
        let state = try await io.readState()
        let sendAge = lastSendTime.map { now - $0 } ?? 0
        let decision = kernel.check(position: state.position,
                                    lastUpdateAge: state.health.lastUpdateAge,
                                    sendAge: sendAge)
        switch decision {
        case .command:
            let raw = controller.compute(state: state, positionTarget: positionTarget)
            let clamped = kernel.envelope.clamp(raw)
            try await io.send(clamped)
            try await io.heartbeat()
            lastSendTime = now
        case .handback:
            await io.handback()
        case .kill:
            await io.kill()
        case .reject:
            break
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/swift/drone && swift test --filter DroneAppTests.FlightLoopTests`
Expected: PASS (4 tests). Then run the whole suite: `swift test` — all DroneCore + DroneApp tests pass.

- [ ] **Step 5: Commit**

```bash
cd samples/swift/drone
git add Sources/DroneApp/FlightLoop.swift Tests/DroneAppTests/FlightLoopTests.swift
git commit -m "feat(drone): add FlightLoop driver enforcing the send-or-handback invariant"
```

---

## Self-Review

**1. Spec coverage (this plan's slice):**
- `FlightIO` seam + value types → Tasks 1–2. ✓
- `SafetyKernel` with arm/mode gate, heartbeat→handback, geofence→handback, kill-terminal → Task 3. ✓
- Envelope soft-clamp (tilt + thrust) → Task 2; applied in loop → Task 5. ✓
- Geometric controller emitting attitude+thrust (seam level, inner loop left to adapters) → Task 4. ✓
- Loop invariant (valid clamped setpoint or handback, never stale) → Task 5. ✓
- `FakeFlightIO` for the spec's kernel/loop unit tests → Task 3. ✓
- *Deferred by design (other plans):* `SimIO` + corruption/randomization (Plan 2), `MavlinkIO` + NIO codec + SITL + deploy (Plan 3), `ReplayIO` + cross-end parity (Plan 4). The `main.swift` + `--io` launch flag lands with the first real adapter (Plan 2), since it needs an adapter to be meaningful.

**2. Placeholder scan:** No TBD/TODO; every code and test step carries complete code. ✓

**3. Type consistency:** `FlightIO` method names (`connect`/`arm`/`engageOffboard`/`readState`/`send`/`heartbeat`/`handback`/`kill`) identical across `FlightIO.swift`, `FakeFlightIO`, and `FlightLoop`. `SafetyKernel.check(position:lastUpdateAge:sendAge:)` signature matches its call in `FlightLoop`. `Envelope.clamp`/`breaches`, `Quat.bodyZ`/`angle(to:)`, `Vec3.up`/`.zero` used consistently. `AttitudeThrust`/`DroneState`/`LinkHealth` field names stable across tasks. ✓

**4. Frame/convention:** z-up and wxyz honored everywhere; hover-thrust arithmetic in Task 4 tests (`m*g/maxThrustForce = 0.5`) is self-consistent with the controller code. ✓
