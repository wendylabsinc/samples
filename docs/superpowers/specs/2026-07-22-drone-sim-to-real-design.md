# Drone sim-to-real — one app, both ends — design

**Date:** 2026-07-22
**Author:** Joannis Orlandos (with Claude)
**Status:** Approved design, pre-implementation

## Goal

Fly the **same Swift flight app** in the Wendy 🕹 Sim tab (MuJoCo) and on a real drone,
with only a hardware-abstraction layer (HAL) swapped underneath. The whole app —
mission logic, geometric controller, and safety state machine — is compiled once; a single
narrow protocol seam (`FlightIO`) chooses at launch whether it flies MuJoCo or a real
PX4/ArduPilot flight controller over MAVLink.

This builds directly on the approved
[Swift drone sim design](2026-07-22-swift-drone-sim-design.md): that spec puts a Swift
MuJoCo drone in the Sim tab; this spec refactors it so the identical binary also flies a
real aircraft.

## Decisions locked in brainstorming

- **Shared boundary:** the *whole flight app* runs on both ends; only the HAL is per-target.
- **Output seam:** the app emits **attitude + collective thrust**. The FC (real) or a
  rotor-mix model (sim) owns the inner rate/motor loop below the seam.
- **Real I/O:** WendyOS companion computer ⇄ **MAVLink** ⇄ PX4/ArduPilot FC.
- **Sim input fidelity:** **domain randomization** in v1 — the sim state is corrupted
  (noise/latency/rate) and physics/sensing parameters are randomized per episode.
- **Safety (all four in scope):** offboard heartbeat + failsafe, arming + mode gate,
  geofence + envelope limits, software kill switch.
- **Structure:** Approach A — one binary, one `FlightIO` protocol, runtime-selected HAL.
  A `ReplayIO` adapter (Approach C) is designed-for but deferred to a testing follow-up.

## Architecture

### The seam

Everything above `FlightIO` is shared, target-agnostic Swift compiled once. Everything below
is a per-target adapter. MuJoCo `wxyz` quaternion convention is used throughout, matching the
drone-sim spec's protocol contract.

```
protocol FlightIO: Sendable {
  func connect() async throws
  func arm() async throws                 // returns when FC confirms armed
  func engageOffboard() async throws      // returns when mode confirmed
  func readState() async throws -> DroneState
  func send(_ sp: AttitudeThrust) async throws
  func heartbeat() async throws           // liveness tick
  func handback() async                   // hand control to FC/pilot (RTL/land/hold)
  func kill() async                       // immediate stop-commanding; terminal
}

struct DroneState {
  var t: Double                 // seconds, monotonic
  var position: Vec3            // world / local-NED frame
  var velocity: Vec3
  var attitude: Quat            // wxyz
  var bodyRates: Vec3           // p, q, r
  var health: LinkHealth        // armed, mode, lastUpdateAge, batteryV?
}

struct AttitudeThrust {
  var attitude: Quat            // desired orientation (wxyz)
  var thrust: Double            // collective, normalized 0…1
}
```

`async` because MAVLink I/O is inherently async; `SimIO` conforms trivially.

### Layering

Extends — does not rework — the drone-sim spec's `DroneRace → WendyMuJoCo → swift-mujoco`
stack. The controller that currently calls `WendyMuJoCo` directly moves *above* the seam;
the rotor mixer moves *below* it into `SimIO`.

```
DroneApp (shared) ──> FlightIO ──┬─ SimIO   ──> WendyMuJoCo ──> swift-mujoco ──> libmujoco
  · mission / loop               └─ MavlinkIO ──> MAVLink ──> PX4/ArduPilot FC
  · geometric controller (attitude+thrust)
  · SafetyKernel
```

### Control loop

`DroneApp` runs one fixed-rate loop:

```
readState → SafetyKernel.check → controller → SafetyKernel.clamp → send + heartbeat
```

Backend selected at launch: `--io sim` (default in the Sim tab) or
`--io mavlink --endpoint <serial|udp>` (on the WendyOS device). `swift-mujoco` and
`WendyMuJoCo` are unchanged from the drone-sim spec; `SimIO` is a new consumer of
`WendyMuJoCo`.

## Components

### 1. `DroneApp` (shared, compiled once)

Mission/loop driver, the geometric controller (PD position → desired tilt → reduced-attitude
→ **attitude+thrust setpoint**, stopping at the seam), and the `SafetyKernel`. Depends only on
`FlightIO` and the shared value types — never on MuJoCo or MAVLink directly.

### 2. `SafetyKernel` (shared, pure state machine)

Sits between controller and `FlightIO`. Decides; never does I/O itself (the adapter realizes
consequences). Pure and **total** — every (state × event) has a defined transition, so
"unexpected" is a modeled event that trips handback.

**States:** `Disconnected → Connected → Armed → Offboard(Active) → {Handback, Killed}`.
`send()` is permitted only in `Offboard(Active)`.

Mechanisms:

1. **Arming + mode gate.** No commanding until `arm()` and `engageOffboard()` both confirm via
   `readState().health`. PX4 requires setpoints streaming *before* offboard engages — the loop
   pre-streams a hold setpoint during the transition (identical code both ends). `MavlinkIO`
   maps to real arm/mode + ACK; `SimIO` stubs as always-ready.
2. **Offboard heartbeat + failsafe.** Loop must `send()`/`heartbeat()` at ≥ configured rate
   (e.g. 50 Hz). Kernel tracks its send cadence and `lastUpdateAge`; either stalling past a
   deadline → `Handback`. `MavlinkIO.handback()` triggers FC failsafe (RTL/land); PX4's own
   offboard-loss failsafe is the backstop if the process dies. `SimIO.handback()` holds/pauses.
   Wired first (non-optional).
3. **Geofence + envelope limits.** Before every `send()`, clamp against a shared `Envelope`
   (position box, max horizontal/vertical velocity, max tilt, thrust `[min,max]`). Two tiers:
   soft clamp (limit setpoint, keep flying) and hard breach (outside geofence → `Handback`).
   The `Envelope` is config and a domain-randomization knob.
4. **Software kill switch.** `kill()` reachable from the ground station (a MAVLink message
   `MavlinkIO` listens for) and from the sim's `control.json`/`ctl.sock` channel (`SimIO`).
   Stops sending setpoints and relinquishes; real → FC failsafe catches it, sim → reset.
   Terminal (requires fresh `connect()`), unlike `Handback` which can re-engage.

### 3. `SimIO` (sim adapter — wraps `WendyMuJoCo.Handle`)

Makes MuJoCo look like a MAVLink FC to the shared app.

- **FC inner-loop emulation:** `send(AttitudeThrust)` runs the reduced-attitude → body-torque →
  rotor-mix math (the loop the drone-sim spec's `DroneRace` currently contains) to produce the
  X2's four rotor `ctrl` values, then advances physics via `sync()`/`mj_step`. The geometric
  controller moved up into `DroneApp`; the mixer lives here — mirroring the real FC.
- **State corruption pipeline** turns ground truth into a MAVLink-like estimate:
  `groundTruth → +bias → +Gaussian noise → rate-limit (telemetry Hz) → delay (ring buffer) → DroneState`.
- **Domain randomization** via a per-episode seeded `Randomizer`, applied at reset; seam and
  app unchanged:

  | Group | Randomized each run |
  |---|---|
  | Dynamics | mass, inertia, thrust coefficient, drag, motor time-constant |
  | Sensing | noise σ, bias, telemetry rate (e.g. 30–100 Hz), latency (e.g. 10–80 ms) |
  | Environment | wind/disturbance impulses, ground effect |
  | Safety envelope | geofence size, tilt/velocity limits |

  Dynamics randomization uses `swift-mujoco` model-parameter access (mass/inertia/actuator gain)
  at reset; sensing/latency randomization lives in the corruption pipeline. Seed + sampled
  parameters are logged to the HUD and a sidecar file for reproducibility.

### 4. `MavlinkIO` (real adapter — the only per-target flight code)

| `FlightIO` call | MAVLink |
|---|---|
| `connect()` | open link (serial `/dev/ttyACM*` or UDP), wait for `HEARTBEAT` |
| `arm()` | `MAV_CMD_COMPONENT_ARM_DISARM`, await `COMMAND_ACK` |
| `engageOffboard()` | pre-stream setpoints, then `SET_MODE`/OFFBOARD, await confirm |
| `readState()` | fuse `ATTITUDE_QUATERNION` + `LOCAL_POSITION_NED` + rates → `DroneState` |
| `send(AttitudeThrust)` | `SET_ATTITUDE_TARGET` (attitude + thrust, body-rate field ignored) |
| `heartbeat()` | keep the offboard setpoint stream alive |
| `handback()` / `kill()` | `SET_MODE` RTL/HOLD / stop stream, let FC failsafe catch it |

None of the sim corruption/randomization compiles into this adapter — it reports what the FC
says.

### 5. `ReplayIO` (follow-up, designed-for now)

Replays recorded sim/MAVLink traces deterministically for regression tests and offline
debugging of real flights. `FlightIO` is designed so it slots in without change. Not v1 scope.

## Open decisions (resolve during implementation)

1. **MAVLink transport library.** (a) **MAVSDK-Swift** — official, high-level Offboard/Telemetry
   plugins, fastest to flight, but pulls in a `mavsdk_server` sidecar + gRPC (heavy ARM64 dep);
   (b) **thin pure-Swift MAVLink codec over SwiftNIO** (serial/UDP) — no sidecar, fits the edge
   device and the repo's NIO idioms, but we implement the message subset ourselves.
   **Recommendation: (b)**, scoped to the handful of messages in the `MavlinkIO` table.
2. **`wendy.json` entitlements for the real path.** Serial-device access to the FC, or host
   networking if the FC link is UDP. Confirm exact entitlement names against the `wendy` skill.

## Deploy workflow

The `DroneApp` executable is one artifact. On the device it launches with `--io mavlink`.
Deployment follows the standard Wendy flow (`wendy run` → build ARM64 → deploy to WendyOS
companion computer → containerd). The real `wendy.json` adds the entitlement from open
decision #2, which the Sim-tab build does not need.

Developer loop:

```
1. Author/tune controller + envelope in DroneApp.
2. wendy-simrun --io sim → fly in 🕹 Sim tab, domain-randomized, until robust.
3. Same binary, --io mavlink, against PX4 SITL over UDP (no MuJoCo, no hardware).
4. wendy run → WendyOS companion computer → real drone; arm gate + geofence + kill armed.
5. Any anomaly → replay trace offline (ReplayIO, follow-up).
```

Step 3 is the safety-critical middle rung: it exercises `MavlinkIO` and the `SafetyKernel`
with zero MuJoCo and zero hardware before a motor spins.

## Error handling

- **Sim (`SimIO`)** inherits the drone-sim spec's "best-effort, never break the sim": bad
  control read, corruption-pipeline hiccup, or randomization edge case is logged and skipped.
- **Real (`MavlinkIO`)** is fail-safe, not best-effort: any unhandled error, dropped link, or
  timeout resolves to a `SafetyKernel` transition (`Handback`/`Killed`), never a silent
  continue.
- **Loop invariant:** each iteration either commands a valid, clamped setpoint or hands back —
  never nothing, never a stale setpoint.
- The `SafetyKernel` being pure and total means "unexpected" is a modeled event that trips
  handback rather than undefined behavior.

## Testing strategy

- **`SafetyKernel` unit tests** (highest value, no MuJoCo/MAVLink): every transition against a
  fake `FlightIO` — arm-before-mode rejected, heartbeat-stall → handback, geofence breach →
  handback, kill terminal, soft-clamp keeps flying.
- **Envelope/clamp unit tests:** tilt/velocity/thrust/geofence clamps; soft vs. hard boundary.
- **`SimIO` tests:** FC-inner-loop mixer parity vs. the drone-sim spec's rotor mix; corruption
  pipeline deterministic under fixed seed; randomizer reproducibility from seed.
- **`MavlinkIO` vs PX4 SITL** (CI, no hardware): connect → arm → offboard → short setpoint
  sequence → geofence breach triggers RTL.
- **Cross-end parity:** same `DroneApp` + same seed drives `SimIO` and a SITL-backed `MavlinkIO`
  path; assert controller output sequences match within tolerance, proving the seam is truly
  side-agnostic.
- **Protocol regression (`ReplayIO`, follow-up):** recorded traces replay deterministically.

## Risks / open questions

1. **MAVLink library choice** (open decision #1) — affects image size and edge footprint;
   pure-Swift/NIO recommended but requires implementing a message subset correctly.
2. **PX4 SITL in CI** — needs the SITL binary available to CI; containerize it for the
   `MavlinkIO` integration test.
3. **Attitude-setpoint semantics across firmwares** — PX4 vs ArduPilot differ in
   `SET_ATTITUDE_TARGET` thrust normalization and frame; pin and test against the target
   firmware. v1 targets PX4.
4. **Sim/real inner-loop mismatch** — `SimIO`'s rotor-mix model is an approximation of the real
   FC's rate loop; domain randomization of motor time-constant/thrust coefficient is the
   mitigation, but expect first-flight gain retuning.
5. **Real-flight authorization** — first hardware flight needs a safe test area, a pilot on a
   physical kill/mode switch, and the software kill validated in SITL first.

## Phasing (detailed plan follows in writing-plans)

1. Extract `FlightIO`, value types, and move the geometric controller above the seam;
   refactor the drone-sim `DroneRace` to route I/O through `FlightIO`.
2. `SafetyKernel` + `Envelope` with full unit tests against a fake `FlightIO`.
3. `SimIO`: FC-inner-loop mixer, corruption pipeline, `Randomizer` — fly in the Sim tab.
4. `MavlinkIO` (transport per open decision #1) against PX4 SITL; arm/offboard/failsafe.
5. Cross-end parity tests; deploy via `wendy run` to a WendyOS companion computer.
6. `ReplayIO` + protocol regression (follow-up).
```