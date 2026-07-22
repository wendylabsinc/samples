# swift-mujoco Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `swift-mujoco`, a standalone Swift package that binds the MuJoCo C library and exposes a thin, ergonomic Swift API (model load, physics stepping, geom/mesh/pose accessors, math, name↔id introspection, full-physics state save/restore, and MjSpec scene composition) — the generic foundation the Wendy Sim-tab layer depends on.

**Architecture:** A SwiftPM `systemLibrary` target (`CMuJoCo`) wraps the full `mujoco.h` via a module map, resolved through a generated `mujoco.pc` so the package stays free of unsafe build flags. A pure-Swift `MuJoCo` target sits on top, wrapping the raw C pointers (`mjModel*`/`mjData*`) in reference types with value-typed accessors. Nothing in this repo is Wendy-specific.

**Tech Stack:** Swift 6.1, Swift Testing (`import Testing`), MuJoCo (C library, sourced from the `mujoco` pip wheel for cross-platform version consistency), `pkg-config`.

## Global Constraints

- Swift tools version: `6.1` (verbatim in `Package.swift`).
- Test framework: Swift Testing (`import Testing`, `@Test`, `#expect`) — NOT XCTest.
- The package MUST NOT use `.unsafeFlags` in `Package.swift` (that would forbid downstream versioned dependency use). MuJoCo is located via `pkgConfig: "mujoco"`.
- MuJoCo is located at `$MUJOCO_PREFIX` (default `/usr/local`); builds/tests run with `PKG_CONFIG_PATH=$MUJOCO_PREFIX/lib/pkgconfig` exported.
- MuJoCo version is whatever the installed `mujoco` pip wheel provides — never hard-code a version number; derive it.
- Nothing Wendy-specific: no Sim-tab protocol, no JSON, no sockets, no Menagerie fetching in this repo.
- MuJoCo uses `mjtNum` = `double`. All numeric accessors return Swift `Double`.
- Quaternions are MuJoCo order `wxyz`.
- Every wrapper type that owns a C pointer frees it in `deinit`.

---

## File Structure

```
swift-mujoco/
  Package.swift
  README.md
  .gitignore
  Scripts/install-mujoco.sh        # copy headers+lib from the pip wheel to $MUJOCO_PREFIX; write mujoco.pc
  Sources/
    CMuJoCo/
      module.modulemap             # module CMuJoCo { header "shim.h"; export * }
      shim.h                       # #include <mujoco/mujoco.h>
    MuJoCo/
      MjError.swift                # MjError: Error
      MjModel.swift                # MjModel (load, scalars, geom/mesh accessors, introspection, names)
      MjData.swift                 # MjData (make, time/qpos/qvel/ctrl/sensordata, geom poses, contacts)
      MjPhysics.swift              # step/forward/resetData/resetDataKeyframe free functions
      MjMath.swift                 # Vec3, Mat3, Quat; mat2Quat/quat2Mat via mju_*
      MjState.swift                # fullPhysicsState get/set
      MjSpec.swift                 # MjSpec composition + compile
  Tests/
    MuJoCoTests/
      Fixtures.swift               # inline MJCF strings + a temp-file helper
      BindingTests.swift           # Task 1
      ModelTests.swift             # Task 2
      DataTests.swift              # Task 3
      GeomTests.swift              # Task 4
      MeshTests.swift              # Task 5
      MathTests.swift              # Task 6
      NamesTests.swift             # Task 7
      StateTests.swift             # Task 8
      SpecTests.swift              # Task 9
```

---

## Task 1: Package skeleton, MuJoCo install, and the binding spike

De-risks the whole effort (spec risk #2): prove Swift can call into `libmujoco`.

**Files:**
- Create: `Package.swift`
- Create: `Sources/CMuJoCo/module.modulemap`
- Create: `Sources/CMuJoCo/shim.h`
- Create: `Scripts/install-mujoco.sh`
- Create: `Sources/MuJoCo/MjError.swift`
- Create: `.gitignore`
- Test: `Tests/MuJoCoTests/BindingTests.swift`

**Interfaces:**
- Produces: module `MuJoCo` (re-exporting nothing); `struct MjError: Error { public let message: String }`; and a free function `public func mujocoVersion() -> String`.

- [ ] **Step 1: Create `.gitignore`**

```
.build/
.swiftpm/
*.xcodeproj
```

- [ ] **Step 2: Write the MuJoCo install script**

Create `Scripts/install-mujoco.sh`:

```bash
#!/usr/bin/env bash
# Install MuJoCo headers + shared library from the installed `mujoco` pip wheel
# into $MUJOCO_PREFIX (default /usr/local) and write a pkg-config file.
# The wheel ships identical headers + a versioned shared lib on macOS and Linux,
# so dev and CI/image link the exact same MuJoCo the Python path uses.
set -euo pipefail
PREFIX="${MUJOCO_PREFIX:-/usr/local}"

PKGDIR="$(python3 -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))')"
VER="$(python3 -c 'import mujoco; print(mujoco.__version__)')"
echo "mujoco wheel: $PKGDIR (version $VER) -> $PREFIX"

mkdir -p "$PREFIX/include/mujoco" "$PREFIX/lib/pkgconfig"
cp -R "$PKGDIR/include/mujoco/." "$PREFIX/include/mujoco/"

# Locate the shared library inside the wheel (libmujoco.<ver>.dylib | libmujoco.so.<ver>)
LIB="$(find "$PKGDIR" -maxdepth 1 \( -name 'libmujoco*.dylib' -o -name 'libmujoco*.so*' \) | head -n1)"
[ -n "$LIB" ] || { echo "no libmujoco found in $PKGDIR" >&2; exit 1; }
cp "$LIB" "$PREFIX/lib/"
BASE="$(basename "$LIB")"

case "$(uname -s)" in
  Darwin)
    ln -sf "$BASE" "$PREFIX/lib/libmujoco.dylib"
    install_name_tool -id "$PREFIX/lib/$BASE" "$PREFIX/lib/$BASE" || true
    ;;
  Linux)
    ln -sf "$BASE" "$PREFIX/lib/libmujoco.so"
    ldconfig "$PREFIX/lib" 2>/dev/null || true
    ;;
esac

cat > "$PREFIX/lib/pkgconfig/mujoco.pc" <<EOF
prefix=$PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: mujoco
Description: MuJoCo physics engine
Version: $VER
Libs: -L\${libdir} -lmujoco
Cflags: -I\${includedir}
EOF
echo "wrote $PREFIX/lib/pkgconfig/mujoco.pc"
```

- [ ] **Step 3: Run the install script**

Run:
```bash
python3 -m pip install --quiet mujoco   # if not already installed
chmod +x Scripts/install-mujoco.sh
sudo env MUJOCO_PREFIX=/usr/local python3 -c "import mujoco" >/dev/null   # sanity: importable
sudo MUJOCO_PREFIX=/usr/local ./Scripts/install-mujoco.sh
```
Expected: prints the wheel path/version and "wrote /usr/local/lib/pkgconfig/mujoco.pc". Verify:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --cflags --libs mujoco
```
Expected: `-I/usr/local/include -L/usr/local/lib -lmujoco`.

- [ ] **Step 4: Create the C binding files**

`Sources/CMuJoCo/shim.h`:
```c
#include <mujoco/mujoco.h>
```

`Sources/CMuJoCo/module.modulemap`:
```
module CMuJoCo {
    header "shim.h"
    export *
}
```

- [ ] **Step 5: Create `Package.swift`**

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "swift-mujoco",
    products: [
        .library(name: "MuJoCo", targets: ["MuJoCo"]),
    ],
    targets: [
        .systemLibrary(name: "CMuJoCo", path: "Sources/CMuJoCo", pkgConfig: "mujoco"),
        .target(name: "MuJoCo", dependencies: ["CMuJoCo"]),
        .testTarget(name: "MuJoCoTests", dependencies: ["MuJoCo"]),
    ]
)
```

- [ ] **Step 6: Create `MjError.swift` and the version function**

`Sources/MuJoCo/MjError.swift`:
```swift
import CMuJoCo

public struct MjError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

/// The MuJoCo library version string (proves the C library links and calls).
public func mujocoVersion() -> String {
    String(cString: mj_versionString())
}
```

- [ ] **Step 7: Write the failing test**

`Tests/MuJoCoTests/BindingTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func versionStringIsNonEmpty() {
    let v = mujocoVersion()
    #expect(!v.isEmpty)
}
```

- [ ] **Step 8: Run the test to verify it fails (before build wiring proven)**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter versionStringIsNonEmpty
```
Expected on first run: this is the spike — either it PASSES immediately (binding works) or FAILS with a *build/link* error (e.g. "module 'CMuJoCo' not found" or "library not found for -lmujoco"). If it fails to build, fix the install/pkg-config path until it builds; that resolution IS the spike's deliverable.

- [ ] **Step 9: Make it pass and confirm**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter versionStringIsNonEmpty
```
Expected: PASS. (On macOS, if the test binary can't find the dylib at runtime, confirm the `install_name_tool -id` step ran; on Linux confirm `/usr/local/lib` is on the loader path or set `LD_LIBRARY_PATH=/usr/local/lib`.)

- [ ] **Step 10: Add a README build note and commit**

Create `README.md`:
```markdown
# swift-mujoco

Swift bindings for the MuJoCo physics engine.

## Prerequisites
Install MuJoCo (headers + shared lib) from the pip wheel:

    pip install mujoco
    sudo MUJOCO_PREFIX=/usr/local ./Scripts/install-mujoco.sh

## Build & test
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
    swift build
    swift test
```

Commit:
```bash
git init -q 2>/dev/null || true
git add -A
git commit -m "feat: package skeleton + MuJoCo binding spike (version string)"
```

---

## Task 2: MjModel — load and scalar accessors

**Files:**
- Create: `Sources/MuJoCo/MjModel.swift`
- Create: `Tests/MuJoCoTests/Fixtures.swift`
- Test: `Tests/MuJoCoTests/ModelTests.swift`

**Interfaces:**
- Consumes: `MjError`.
- Produces:
  - `public final class MjModel` wrapping `UnsafeMutablePointer<mjModel>` at `public let ptr`.
  - `public static func MjModel.load(xmlPath: String) throws -> MjModel`
  - `public static func MjModel.load(xml: String) throws -> MjModel` (writes to a temp file, loads, deletes)
  - Scalars: `var ngeom, nq, nv, nu, nbody, njnt, nsensor, nmesh, nkey: Int`
  - `var timestep: Double`, `var gravity: (Double, Double, Double)`

- [ ] **Step 1: Create the fixtures file**

`Tests/MuJoCoTests/Fixtures.swift`:
```swift
import Foundation

enum Fixtures {
    /// A single hinge pole with one actuator — for joints/actuators/data tests.
    static let pendulum = """
    <mujoco>
      <worldbody>
        <body name="pole" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0" range="-3.14 3.14" limited="true"/>
          <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0.8 0.2 0.2 1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="mot" joint="hinge" ctrlrange="-1 1" ctrllimited="true"/>
      </actuator>
    </mujoco>
    """

    /// Floor + a free-floating cube with one visible and one hidden (group 3) geom.
    static let boxScene = """
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

    /// A mesh from inline vertices (MuJoCo builds the convex hull faces).
    static let meshScene = """
    <mujoco>
      <asset>
        <mesh name="tri" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="tri" rgba="1 1 1 1"/>
      </worldbody>
    </mujoco>
    """
}
```

- [ ] **Step 2: Write the failing test**

`Tests/MuJoCoTests/ModelTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func loadsBoxSceneScalars() throws {
    let m = try MjModel.load(xml: Fixtures.boxScene)
    #expect(m.ngeom == 3)          // floor + box + hidden
    #expect(m.nbody == 2)          // world + cube
    #expect(m.timestep > 0)
    #expect(m.gravity.2 < 0)       // default gravity is (0,0,-9.81)
}

@Test func loadInvalidXMLThrows() {
    #expect(throws: MjError.self) {
        _ = try MjModel.load(xml: "<mujoco><worldbody><geom type=\"nonsense\"/></worldbody></mujoco>")
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter ModelTests
```
Expected: FAIL — `MjModel` is not defined.

- [ ] **Step 4: Implement `MjModel`**

`Sources/MuJoCo/MjModel.swift`:
```swift
import CMuJoCo
import Foundation

public final class MjModel {
    public let ptr: UnsafeMutablePointer<mjModel>

    init(owning ptr: UnsafeMutablePointer<mjModel>) { self.ptr = ptr }
    deinit { mj_deleteModel(ptr) }

    public static func load(xmlPath: String) throws -> MjModel {
        var err = [CChar](repeating: 0, count: 1000)
        let m = mj_loadXML(xmlPath, nil, &err, Int32(err.count))
        guard let m else { throw MjError(String(cString: err)) }
        return MjModel(owning: m)
    }

    public static func load(xml: String) throws -> MjModel {
        let dir = FileManager.default.temporaryDirectory
        let file = dir.appendingPathComponent("mj-\(UUID().uuidString).xml")
        try xml.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        return try load(xmlPath: file.path)
    }

    public var ngeom: Int { Int(ptr.pointee.ngeom) }
    public var nq: Int { Int(ptr.pointee.nq) }
    public var nv: Int { Int(ptr.pointee.nv) }
    public var nu: Int { Int(ptr.pointee.nu) }
    public var nbody: Int { Int(ptr.pointee.nbody) }
    public var njnt: Int { Int(ptr.pointee.njnt) }
    public var nsensor: Int { Int(ptr.pointee.nsensor) }
    public var nmesh: Int { Int(ptr.pointee.nmesh) }
    public var nkey: Int { Int(ptr.pointee.nkey) }

    public var timestep: Double { ptr.pointee.opt.timestep }
    public var gravity: (Double, Double, Double) {
        let g = ptr.pointee.opt.gravity
        return (g.0, g.1, g.2)   // mjtNum[3] imports as a Swift tuple
    }
}
```

Note: fixed-size C arrays like `mjtNum gravity[3]` import into Swift as homogeneous tuples (`(Double, Double, Double)`). If the imported member name differs (e.g. `opt` is `mjOption`), confirm with `grep -n "gravity" /usr/local/include/mujoco/mjmodel.h`.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter ModelTests
```
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add Sources/MuJoCo/MjModel.swift Tests/MuJoCoTests/Fixtures.swift Tests/MuJoCoTests/ModelTests.swift
git commit -m "feat: MjModel load + scalar accessors"
```

---

## Task 3: MjData — stepping and state buffers

**Files:**
- Create: `Sources/MuJoCo/MjData.swift`
- Create: `Sources/MuJoCo/MjPhysics.swift`
- Test: `Tests/MuJoCoTests/DataTests.swift`

**Interfaces:**
- Consumes: `MjModel`.
- Produces:
  - `public final class MjData` wrapping `UnsafeMutablePointer<mjData>` at `public let ptr`; `public init(_ model: MjModel)`.
  - `var time: Double { get }`
  - Buffer views (length from the model): `var qpos: [Double]`, `var qvel: [Double]`, `var ctrl: [Double]` (get), plus `func setCtrl(_ index: Int, _ value: Double)` and `func setCtrl(_ values: [Double])`.
  - Free functions in `MjPhysics.swift`: `public func mjStep(_ m: MjModel, _ d: MjData)`, `mjForward`, `mjResetData(_:_:)`, `mjResetDataKeyframe(_:_:_ key: Int)`.

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/DataTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func steppingAdvancesTime() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)
    let d = MjData(m)
    #expect(d.time == 0)
    for _ in 0..<10 { mjStep(m, d) }
    #expect(d.time > 0)
    #expect(abs(d.time - 10 * m.timestep) < 1e-9)
}

@Test func ctrlRoundTrips() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)   // 1 actuator
    let d = MjData(m)
    #expect(d.ctrl.count == m.nu)
    d.setCtrl(0, 0.5)
    #expect(d.ctrl[0] == 0.5)
}

@Test func resetClearsTime() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)
    let d = MjData(m)
    for _ in 0..<5 { mjStep(m, d) }
    mjResetData(m, d)
    #expect(d.time == 0)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter DataTests
```
Expected: FAIL — `MjData`/`mjStep` not defined.

- [ ] **Step 3: Implement `MjData`**

`Sources/MuJoCo/MjData.swift`:
```swift
import CMuJoCo

public final class MjData {
    public let ptr: UnsafeMutablePointer<mjData>
    private let model: MjModel   // keep the model alive for this data's lifetime

    public init(_ model: MjModel) {
        self.model = model
        self.ptr = mj_makeData(model.ptr)
    }
    deinit { mj_deleteData(ptr) }

    public var time: Double { ptr.pointee.time }

    private func buffer(_ base: UnsafeMutablePointer<mjtNum>?, _ n: Int) -> [Double] {
        guard let base, n > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: base, count: n))
    }

    public var qpos: [Double] { buffer(ptr.pointee.qpos, model.nq) }
    public var qvel: [Double] { buffer(ptr.pointee.qvel, model.nv) }
    public var ctrl: [Double] { buffer(ptr.pointee.ctrl, model.nu) }

    public func setCtrl(_ index: Int, _ value: Double) {
        precondition(index >= 0 && index < model.nu)
        ptr.pointee.ctrl[index] = value
    }
    public func setCtrl(_ values: [Double]) {
        for i in 0..<min(values.count, model.nu) { ptr.pointee.ctrl[i] = values[i] }
    }
}
```

- [ ] **Step 4: Implement the physics free functions**

`Sources/MuJoCo/MjPhysics.swift`:
```swift
import CMuJoCo

public func mjStep(_ m: MjModel, _ d: MjData) { mj_step(m.ptr, d.ptr) }
public func mjForward(_ m: MjModel, _ d: MjData) { mj_forward(m.ptr, d.ptr) }
public func mjResetData(_ m: MjModel, _ d: MjData) { mj_resetData(m.ptr, d.ptr) }
public func mjResetDataKeyframe(_ m: MjModel, _ d: MjData, _ key: Int) {
    mj_resetDataKeyframe(m.ptr, d.ptr, Int32(key))
}
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter DataTests
```
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add Sources/MuJoCo/MjData.swift Sources/MuJoCo/MjPhysics.swift Tests/MuJoCoTests/DataTests.swift
git commit -m "feat: MjData buffers + physics stepping functions"
```

---

## Task 4: Geom accessors + visibility

**Files:**
- Modify: `Sources/MuJoCo/MjModel.swift` (add geom accessors)
- Test: `Tests/MuJoCoTests/GeomTests.swift`

**Interfaces:**
- Consumes: `MjModel`.
- Produces on `MjModel`:
  - `enum GeomType: String { case plane, sphere, capsule, ellipsoid, cylinder, box, mesh, other }`
  - `func geomType(_ i: Int) -> GeomType`
  - `func geomSize(_ i: Int) -> [Double]` (3 values)
  - `func geomRgba(_ i: Int) -> [Double]` (4 values; resolves material color if `geom_matid >= 0`, else `geom_rgba`)
  - `func geomGroup(_ i: Int) -> Int`
  - `func geomDataid(_ i: Int) -> Int`  (mesh id for mesh geoms, else -1)
  - `func geomIsVisible(_ i: Int) -> Bool`  (group < 3 AND rgba alpha != 0)

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/GeomTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func geomTypesAndVisibility() throws {
    let m = try MjModel.load(xml: Fixtures.boxScene)
    // geom order matches declaration: 0=floor(plane) 1=box 2=hidden
    #expect(m.geomType(0) == .plane)
    #expect(m.geomType(1) == .box)
    #expect(m.geomIsVisible(0) == true)
    #expect(m.geomIsVisible(1) == true)
    #expect(m.geomIsVisible(2) == false)   // group 3 -> hidden
    let rgba = m.geomRgba(1)
    #expect(rgba.count == 4)
    #expect(abs(rgba[2] - 0.9) < 1e-6)     // blue channel from the fixture
    #expect(m.geomSize(1).count == 3)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter GeomTests
```
Expected: FAIL — geom accessors not defined.

- [ ] **Step 3: Implement geom accessors**

Append to `Sources/MuJoCo/MjModel.swift` (inside the class):
```swift
    public enum GeomType: String {
        case plane, sphere, capsule, ellipsoid, cylinder, box, mesh, other
    }

    public func geomType(_ i: Int) -> GeomType {
        switch Int32(ptr.pointee.geom_type[i]) {
        case mjGEOM_PLANE.rawValue: return .plane
        case mjGEOM_SPHERE.rawValue: return .sphere
        case mjGEOM_CAPSULE.rawValue: return .capsule
        case mjGEOM_ELLIPSOID.rawValue: return .ellipsoid
        case mjGEOM_CYLINDER.rawValue: return .cylinder
        case mjGEOM_BOX.rawValue: return .box
        case mjGEOM_MESH.rawValue: return .mesh
        default: return .other
        }
    }

    public func geomSize(_ i: Int) -> [Double] {
        [ptr.pointee.geom_size[i * 3 + 0],
         ptr.pointee.geom_size[i * 3 + 1],
         ptr.pointee.geom_size[i * 3 + 2]]
    }

    public func geomGroup(_ i: Int) -> Int { Int(ptr.pointee.geom_group[i]) }
    public func geomDataid(_ i: Int) -> Int { Int(ptr.pointee.geom_dataid[i]) }

    public func geomRgba(_ i: Int) -> [Double] {
        let matid = Int(ptr.pointee.geom_matid[i])
        let base: UnsafeMutablePointer<Float>
        let off: Int
        if matid >= 0 { base = ptr.pointee.mat_rgba; off = matid * 4 }
        else { base = ptr.pointee.geom_rgba; off = i * 4 }
        return (0..<4).map { Double(base[off + $0]) }
    }

    public func geomIsVisible(_ i: Int) -> Bool {
        geomGroup(i) < 3 && geomRgba(i)[3] != 0.0
    }
```

Note: `mjGEOM_*` and `geom_type` element type are enums/`int`; if a `case` label errors on type, wrap with `Int32(mjGEOM_PLANE.rawValue)` consistently — confirm the imported enum's `rawValue` type with a quick `swift build`. `geom_rgba`/`mat_rgba` are `float*` in `mjModel` (hence `Float`), while `mjtNum` fields are `Double`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter GeomTests
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/MuJoCo/MjModel.swift Tests/MuJoCoTests/GeomTests.swift
git commit -m "feat: geom type/size/rgba/group accessors + visibility rule"
```

---

## Task 5: Mesh vertex/face buffers

**Files:**
- Modify: `Sources/MuJoCo/MjModel.swift`
- Test: `Tests/MuJoCoTests/MeshTests.swift`

**Interfaces:**
- Produces on `MjModel`:
  - `func meshVertices(_ meshId: Int) -> [Float]` (flat xyz triples)
  - `func meshFaces(_ meshId: Int) -> [Int]` (flat triangle indices)
  - `func meshName(_ meshId: Int) -> String` (name or `"mesh<id>"`)

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/MeshTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func meshBuffersExtractable() throws {
    let m = try MjModel.load(xml: Fixtures.meshScene)
    #expect(m.nmesh == 1)
    #expect(m.geomType(0) == .mesh)
    let meshId = m.geomDataid(0)
    #expect(meshId >= 0)
    let verts = m.meshVertices(meshId)
    let faces = m.meshFaces(meshId)
    #expect(verts.count == 4 * 3)      // 4 inline vertices
    #expect(faces.count % 3 == 0)      // triangles
    #expect(faces.count >= 3)          // convex hull built by MuJoCo
    #expect(m.meshName(meshId) == "tri")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter MeshTests
```
Expected: FAIL — mesh accessors not defined.

- [ ] **Step 3: Implement mesh accessors**

Append to `Sources/MuJoCo/MjModel.swift` (inside the class):
```swift
    public func meshVertices(_ meshId: Int) -> [Float] {
        let v0 = Int(ptr.pointee.mesh_vertadr[meshId])
        let vn = Int(ptr.pointee.mesh_vertnum[meshId])
        let base = ptr.pointee.mesh_vert   // float*
        return (0..<(vn * 3)).map { base[(v0 * 3) + $0] }
    }

    public func meshFaces(_ meshId: Int) -> [Int] {
        let f0 = Int(ptr.pointee.mesh_faceadr[meshId])
        let fn = Int(ptr.pointee.mesh_facenum[meshId])
        let base = ptr.pointee.mesh_face   // int*
        return (0..<(fn * 3)).map { Int(base[(f0 * 3) + $0]) }
    }

    public func meshName(_ meshId: Int) -> String {
        name(of: mjOBJ_MESH, id: meshId) ?? "mesh\(meshId)"
    }
```

Note: `name(of:id:)` is added in Task 7. Until then this file references it — implement a temporary local until Task 7, OR reorder so Task 7 precedes if executing strictly in order. To keep Task 5 independently green, add this minimal private helper now and REMOVE it in Task 7 when the public `name(of:id:)` lands:
```swift
    private func _tmpName(_ obj: mjtObj, _ id: Int) -> String? {
        guard let c = mj_id2name(ptr, obj, Int32(id)) else { return nil }
        return String(cString: c)
    }
```
and use `_tmpName(mjOBJ_MESH, meshId)` in `meshName` for now.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter MeshTests
```
Expected: PASS. (`mesh_vert`/`mesh_face` are `float*`/`int*`; `mesh_vertadr` counts vertices, so byte-offset math uses `*3`.)

- [ ] **Step 5: Commit**

```bash
git add Sources/MuJoCo/MjModel.swift Tests/MuJoCoTests/MeshTests.swift
git commit -m "feat: mesh vertex/face buffer extraction"
```

---

## Task 6: Math — Vec3/Mat3/Quat and rotation conversions

**Files:**
- Create: `Sources/MuJoCo/MjMath.swift`
- Modify: `Sources/MuJoCo/MjData.swift` (add geom world-pose accessors)
- Test: `Tests/MuJoCoTests/MathTests.swift`

**Interfaces:**
- Produces:
  - `public struct Vec3 { public var x, y, z: Double; init(_:_:_:) ; init(_ a:[Double]) }` with `+`, `-`, `*` (scalar), `dot`, `cross`, `norm`, `normalized`, `var array: [Double]`.
  - `public struct Mat3 { public let m: [Double] /* 9, row-major */ ; func column(_ i: Int) -> Vec3 ; func transposeTimes(_ v: Vec3) -> Vec3 }`
  - `public struct Quat { public var w, x, y, z: Double }`
  - `public func mat2Quat(_ mat: [Double]) -> Quat`  (via `mju_mat2Quat`)
  - `public func quat2Mat(_ q: Quat) -> Mat3`        (via `mju_quat2Mat`)
- Produces on `MjData`:
  - `func geomXpos(_ i: Int) -> Vec3`
  - `func geomXmat(_ i: Int) -> [Double]` (9, row-major)
  - `func geomQuat(_ i: Int) -> Quat` (convenience: `mat2Quat(geomXmat(i))`)

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/MathTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func vec3Ops() {
    let a = Vec3(1, 0, 0), b = Vec3(0, 1, 0)
    #expect(a.cross(b).array == [0, 0, 1])
    #expect(a.dot(b) == 0)
    #expect(abs(Vec3(3, 4, 0).norm - 5) < 1e-12)
}

@Test func identityMatrixIsIdentityQuat() {
    let q = mat2Quat([1,0,0, 0,1,0, 0,0,1])
    #expect(abs(q.w - 1) < 1e-9)
    #expect(abs(q.x) < 1e-9 && abs(q.y) < 1e-9 && abs(q.z) < 1e-9)
}

@Test func geomPoseReadable() throws {
    let m = try MjModel.load(xml: Fixtures.boxScene)
    let d = MjData(m)
    mjForward(m, d)
    let p = d.geomXpos(1)          // the cube geom, declared at body pos z=1
    #expect(abs(p.z - 1) < 1e-6)
    #expect(d.geomXmat(1).count == 9)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter MathTests
```
Expected: FAIL — `Vec3`/`mat2Quat`/`geomXpos` not defined.

- [ ] **Step 3: Implement `MjMath.swift`**

`Sources/MuJoCo/MjMath.swift`:
```swift
import CMuJoCo

public struct Vec3: Equatable {
    public var x, y, z: Double
    public init(_ x: Double, _ y: Double, _ z: Double) { self.x = x; self.y = y; self.z = z }
    public init(_ a: [Double]) { self.init(a[0], a[1], a[2]) }
    public var array: [Double] { [x, y, z] }
    public var norm: Double { (x*x + y*y + z*z).squareRoot() }
    public var normalized: Vec3 { let n = norm; return n > 0 ? self * (1/n) : self }
    public func dot(_ o: Vec3) -> Double { x*o.x + y*o.y + z*o.z }
    public func cross(_ o: Vec3) -> Vec3 {
        Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x)
    }
    public static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x+b.x, a.y+b.y, a.z+b.z) }
    public static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x-b.x, a.y-b.y, a.z-b.z) }
    public static func * (a: Vec3, s: Double) -> Vec3 { Vec3(a.x*s, a.y*s, a.z*s) }
}

public struct Mat3 {
    public let m: [Double]   // 9, row-major
    public init(_ m: [Double]) { precondition(m.count == 9); self.m = m }
    /// Column i of the rotation matrix (e.g. column 2 = body z-axis in world).
    public func column(_ i: Int) -> Vec3 { Vec3(m[i], m[3+i], m[6+i]) }
    /// Rᵀ · v  (world vector into body frame).
    public func transposeTimes(_ v: Vec3) -> Vec3 {
        Vec3(m[0]*v.x + m[3]*v.y + m[6]*v.z,
             m[1]*v.x + m[4]*v.y + m[7]*v.z,
             m[2]*v.x + m[5]*v.y + m[8]*v.z)
    }
}

public struct Quat: Equatable {
    public var w, x, y, z: Double
    public init(w: Double, x: Double, y: Double, z: Double) { self.w = w; self.x = x; self.y = y; self.z = z }
}

public func mat2Quat(_ mat: [Double]) -> Quat {
    precondition(mat.count == 9)
    var q = [Double](repeating: 0, count: 4)
    mat.withUnsafeBufferPointer { mp in
        mju_mat2Quat(&q, mp.baseAddress)
    }
    return Quat(w: q[0], x: q[1], y: q[2], z: q[3])   // MuJoCo order wxyz
}

public func quat2Mat(_ q: Quat) -> Mat3 {
    var m = [Double](repeating: 0, count: 9)
    var qq = [q.w, q.x, q.y, q.z]
    mju_quat2Mat(&m, &qq)
    return Mat3(m)
}
```

- [ ] **Step 4: Add geom world-pose accessors to `MjData`**

Append to `Sources/MuJoCo/MjData.swift` (inside the class):
```swift
    public func geomXpos(_ i: Int) -> Vec3 {
        let b = ptr.pointee.geom_xpos   // mjtNum*, length ngeom*3
        return Vec3(b[i*3+0], b[i*3+1], b[i*3+2])
    }
    public func geomXmat(_ i: Int) -> [Double] {
        let b = ptr.pointee.geom_xmat   // mjtNum*, length ngeom*9, row-major
        return (0..<9).map { b[i*9 + $0] }
    }
    public func geomQuat(_ i: Int) -> Quat { mat2Quat(geomXmat(i)) }
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter MathTests
```
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add Sources/MuJoCo/MjMath.swift Sources/MuJoCo/MjData.swift Tests/MuJoCoTests/MathTests.swift
git commit -m "feat: Vec3/Mat3/Quat math + geom world-pose accessors"
```

---

## Task 7: Names and model introspection

**Files:**
- Modify: `Sources/MuJoCo/MjModel.swift` (add name lookups + joint/actuator/sensor introspection; remove the `_tmpName` helper from Task 5 and route `meshName` through the new public API)
- Test: `Tests/MuJoCoTests/NamesTests.swift`

**Interfaces:**
- Produces on `MjModel`:
  - `func name(of obj: mjtObj, id: Int) -> String?`
  - `func id(of obj: mjtObj, name: String) -> Int?` (returns nil if `< 0`)
  - `struct JointInfo { let id: Int; let name: String; let type: Int; let limited: Bool; let range: (Double, Double); let qposadr: Int; let dofadr: Int }`
  - `struct ActuatorInfo { let id: Int; let name: String; let ctrlLimited: Bool; let ctrlRange: (Double, Double) }`
  - `struct SensorInfo { let id: Int; let name: String; let type: Int; let dim: Int; let adr: Int }`
  - `var joints: [JointInfo]`, `var actuators: [ActuatorInfo]`, `var sensors: [SensorInfo]`, `var bodyNames: [String]`
- Convenience re-exports so callers need not import `CMuJoCo` for the common object types:
  - `public let objJoint = mjOBJ_JOINT`, `objActuator = mjOBJ_ACTUATOR`, `objBody = mjOBJ_BODY`, `objGeom = mjOBJ_GEOM`, `objSensor = mjOBJ_SENSOR`, `objMesh = mjOBJ_MESH`, `objKey = mjOBJ_KEY` (in `MjModel.swift`, module scope).

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/NamesTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func nameIdRoundTrip() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)
    let jid = m.id(of: objJoint, name: "hinge")
    #expect(jid == 0)
    #expect(m.name(of: objJoint, id: 0) == "hinge")
    #expect(m.id(of: objJoint, name: "does-not-exist") == nil)
}

@Test func introspectsJointsAndActuators() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)
    #expect(m.joints.count == 1)
    let j = m.joints[0]
    #expect(j.name == "hinge")
    #expect(j.limited == true)
    #expect(abs(j.range.1 - 3.14) < 1e-3)

    #expect(m.actuators.count == 1)
    let a = m.actuators[0]
    #expect(a.name == "mot")
    #expect(a.ctrlLimited == true)
    #expect(abs(a.ctrlRange.0 + 1) < 1e-6 && abs(a.ctrlRange.1 - 1) < 1e-6)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter NamesTests
```
Expected: FAIL — name/introspection API not defined.

- [ ] **Step 3: Implement names + introspection**

At module scope in `Sources/MuJoCo/MjModel.swift` (top of file, after `import`):
```swift
public let objJoint = mjOBJ_JOINT
public let objActuator = mjOBJ_ACTUATOR
public let objBody = mjOBJ_BODY
public let objGeom = mjOBJ_GEOM
public let objSensor = mjOBJ_SENSOR
public let objMesh = mjOBJ_MESH
public let objKey = mjOBJ_KEY
```

Inside the class:
```swift
    public func name(of obj: mjtObj, id: Int) -> String? {
        guard let c = mj_id2name(ptr, obj, Int32(id)) else { return nil }
        return String(cString: c)
    }
    public func id(of obj: mjtObj, name: String) -> Int? {
        let i = Int(mj_name2id(ptr, obj, name))
        return i >= 0 ? i : nil
    }

    public struct JointInfo { public let id: Int; public let name: String; public let type: Int; public let limited: Bool; public let range: (Double, Double); public let qposadr: Int; public let dofadr: Int }
    public struct ActuatorInfo { public let id: Int; public let name: String; public let ctrlLimited: Bool; public let ctrlRange: (Double, Double) }
    public struct SensorInfo { public let id: Int; public let name: String; public let type: Int; public let dim: Int; public let adr: Int }

    public var joints: [JointInfo] {
        (0..<njnt).map { j in
            JointInfo(id: j, name: name(of: objJoint, id: j) ?? "",
                      type: Int(ptr.pointee.jnt_type[j]),
                      limited: ptr.pointee.jnt_limited[j] != 0,
                      range: (ptr.pointee.jnt_range[j*2+0], ptr.pointee.jnt_range[j*2+1]),
                      qposadr: Int(ptr.pointee.jnt_qposadr[j]),
                      dofadr: Int(ptr.pointee.jnt_dofadr[j]))
        }
    }
    public var actuators: [ActuatorInfo] {
        (0..<nu).map { a in
            ActuatorInfo(id: a, name: name(of: objActuator, id: a) ?? "",
                         ctrlLimited: ptr.pointee.actuator_ctrllimited[a] != 0,
                         ctrlRange: (ptr.pointee.actuator_ctrlrange[a*2+0], ptr.pointee.actuator_ctrlrange[a*2+1]))
        }
    }
    public var sensors: [SensorInfo] {
        (0..<nsensor).map { s in
            SensorInfo(id: s, name: name(of: objSensor, id: s) ?? "",
                       type: Int(ptr.pointee.sensor_type[s]),
                       dim: Int(ptr.pointee.sensor_dim[s]),
                       adr: Int(ptr.pointee.sensor_adr[s]))
        }
    }
    public var bodyNames: [String] { (0..<nbody).map { name(of: objBody, id: $0) ?? "" } }
```

Then update `meshName` to use the public API and delete the `_tmpName` helper added in Task 5:
```swift
    public func meshName(_ meshId: Int) -> String {
        name(of: objMesh, id: meshId) ?? "mesh\(meshId)"
    }
```

Note: `jnt_limited`/`actuator_ctrllimited` are `mjtByte*` (`UInt8`); compare `!= 0`. Confirm field element types with `grep -n "jnt_limited\|actuator_ctrllimited" /usr/local/include/mujoco/mjmodel.h` if a comparison fails to typecheck.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter NamesTests
```
Expected: PASS. Then run the mesh test again to confirm no regression:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter MeshTests
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/MuJoCo/MjModel.swift Tests/MuJoCoTests/NamesTests.swift
git commit -m "feat: name<->id lookups + joint/actuator/sensor introspection"
```

---

## Task 8: Full-physics state + contacts

**Files:**
- Create: `Sources/MuJoCo/MjState.swift`
- Modify: `Sources/MuJoCo/MjData.swift` (add contacts)
- Test: `Tests/MuJoCoTests/StateTests.swift`

**Interfaces:**
- Produces on `MjData` (via `MjState.swift` extension):
  - `func getFullState() -> [Double]`
  - `func setFullState(_ state: [Double])`
- Produces on `MjData` (contacts):
  - `struct Contact { let geom1: Int; let geom2: Int; let dist: Double; let pos: Vec3; let forceNormal: Double }`
  - `var ncon: Int`
  - `func contacts(max: Int = 64) -> [Contact]`  (uses `mj_contactForce`)

- [ ] **Step 1: Write the failing test**

`Tests/MuJoCoTests/StateTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func fullStateSaveRestore() throws {
    let m = try MjModel.load(xml: Fixtures.pendulum)
    let d = MjData(m)
    d.setCtrl(0, 0.3)
    for _ in 0..<20 { mjStep(m, d) }
    let saved = d.getFullState()
    let qposAt20 = d.qpos
    let timeAt20 = d.time

    for _ in 0..<20 { mjStep(m, d) }
    #expect(d.time > timeAt20)

    d.setFullState(saved)
    #expect(abs(d.time - timeAt20) < 1e-9)
    for i in 0..<m.nq { #expect(abs(d.qpos[i] - qposAt20[i]) < 1e-9) }
}

@Test func contactsAppearWhenBoxLands() throws {
    let m = try MjModel.load(xml: Fixtures.boxScene)   // cube starts at z=1 above floor
    let d = MjData(m)
    var sawContact = false
    for _ in 0..<3000 {
        mjStep(m, d)
        if d.ncon > 0 { sawContact = true; break }
    }
    #expect(sawContact)
    let cs = d.contacts()
    #expect(cs.count == d.ncon || cs.count == 64)
    #expect(cs.first!.forceNormal >= 0)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter StateTests
```
Expected: FAIL — state/contacts API not defined.

- [ ] **Step 3: Implement state save/restore**

`Sources/MuJoCo/MjState.swift`:
```swift
import CMuJoCo

extension MjData {
    private var fullSpec: Int32 { Int32(mjSTATE_FULLPHYSICS.rawValue) }

    public func getFullState() -> [Double] {
        let n = Int(mj_stateSize(model.ptr, UInt32(fullSpec)))
        var arr = [Double](repeating: 0, count: n)
        mj_getState(model.ptr, ptr, &arr, fullSpec)
        return arr
    }
    public func setFullState(_ state: [Double]) {
        var s = state
        mj_setState(model.ptr, ptr, &s, fullSpec)
        mj_forward(model.ptr, ptr)
    }
}
```

Note: `mj_stateSize`/`mj_get/setState` take an unsigned spec in some versions and signed in others. If the compiler rejects `UInt32`/`Int32`, match the header: `grep -n "mj_stateSize\|mj_getState" /usr/local/include/mujoco/mujoco.h`. `model` is the private stored property added in Task 3 — make it accessible to this extension by changing its declaration in `MjData.swift` from `private let model` to `let model` (same file/module, so still internal).

- [ ] **Step 4: Implement contacts**

Append to `Sources/MuJoCo/MjData.swift` (inside the class):
```swift
    public struct Contact {
        public let geom1: Int, geom2: Int
        public let dist: Double
        public let pos: Vec3
        public let forceNormal: Double
    }

    public var ncon: Int { Int(ptr.pointee.ncon) }

    public func contacts(max: Int = 64) -> [Contact] {
        let n = Swift.min(ncon, max)
        guard n > 0 else { return [] }
        var out: [Contact] = []
        out.reserveCapacity(n)
        var f6 = [Double](repeating: 0, count: 6)
        for i in 0..<n {
            let con = ptr.pointee.contact[i]
            mj_contactForce(model.ptr, ptr, Int32(i), &f6)
            let p = con.pos   // mjtNum[3] tuple
            out.append(Contact(geom1: Int(con.geom1), geom2: Int(con.geom2),
                               dist: con.dist, pos: Vec3(p.0, p.1, p.2),
                               forceNormal: f6[0]))
        }
        return out
    }
```

Note: `ptr.pointee.contact` is a `mjContact*`; indexing gives a `mjContact` value whose `pos` is a `(Double,Double,Double)` tuple. Confirm `geom1`/`geom2` exist (older MuJoCo used `geom[2]`); if so, read `con.geom.0`/`con.geom.1`. Check with `grep -n "geom1\|int geom" /usr/local/include/mujoco/mjdata.h`.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter StateTests
```
Expected: PASS (both). If `contactsAppearWhenBoxLands` never sees a contact, increase the step budget; landing from z=1 takes ~0.45 s (~225 steps at dt=0.002).

- [ ] **Step 6: Commit**

```bash
git add Sources/MuJoCo/MjState.swift Sources/MuJoCo/MjData.swift Tests/MuJoCoTests/StateTests.swift
git commit -m "feat: full-physics state save/restore + contact readout"
```

---

## Task 9: MjSpec scene composition

Riskiest binding surface (the `mjs_*` procedural-model API drifts across MuJoCo versions). Confirm exact symbol/field names against the installed headers before implementing.

**Files:**
- Create: `Sources/MuJoCo/MjSpec.swift`
- Test: `Tests/MuJoCoTests/SpecTests.swift`

**Interfaces:**
- Produces:
  - `public final class MjSpec` wrapping `UnsafeMutablePointer<mjSpec>`; `public init(floor: Bool = true, light: Bool = true)`.
  - `func addGeom(type: MjModel.GeomType, size: [Double], pos: [Double], rgba: [Double], toBody body: String? = nil)`
  - `func addBody(name: String, pos: [Double]) -> String` (returns the body name)
  - `func compile() throws -> MjModel`

- [ ] **Step 1: Confirm the MjSpec C API names**

Run:
```bash
grep -nE "mj_makeSpec|mjs_addBody|mjs_addGeom|mjs_findBody|mj_compile|mj_deleteSpec" /usr/local/include/mujoco/*.h
```
Expected: prints the exact declarations (typically in `mujoco/mjspec.h`). Use these signatures verbatim in Step 4; adjust field assignments (`->type`, `->size`, `->pos`, `->rgba`) to the `mjsGeom`/`mjsBody` struct members shown by:
```bash
grep -nA25 "struct mjsGeom_" /usr/local/include/mujoco/mjspec.h
```

- [ ] **Step 2: Write the failing test**

`Tests/MuJoCoTests/SpecTests.swift`:
```swift
import Testing
@testable import MuJoCo

@Test func buildAndCompileScene() throws {
    let spec = MjSpec(floor: true, light: true)   // floor plane geom in worldbody
    let body = spec.addBody(name: "crate", pos: [0, 0, 0.5])
    spec.addGeom(type: .box, size: [0.25, 0.25, 0.25], pos: [0, 0, 0],
                 rgba: [0.4, 0.6, 0.9, 1], toBody: body)
    let model = try spec.compile()
    #expect(model.ngeom == 2)                 // floor + crate box
    #expect(model.id(of: objBody, name: "crate") != nil)
}
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter SpecTests
```
Expected: FAIL — `MjSpec` not defined.

- [ ] **Step 4: Implement `MjSpec.swift`**

`Sources/MuJoCo/MjSpec.swift` (adjust `mjs_*` names/fields to match Step 1's grep output):
```swift
import CMuJoCo

public final class MjSpec {
    public let ptr: UnsafeMutablePointer<mjSpec>

    public init(floor: Bool = true, light: Bool = true) {
        self.ptr = mj_makeSpec()
        let world = mjs_findBody(ptr, "world")
        if light { _ = mjs_addLight(world, nil) }
        if floor {
            let g = mjs_addGeom(world, nil)
            g!.pointee.type = mjGEOM_PLANE
            g!.pointee.size = (12, 12, 0.1)
            g!.pointee.rgba = (0.26, 0.27, 0.32, 1)
        }
    }
    deinit { mj_deleteSpec(ptr) }

    @discardableResult
    public func addBody(name: String, pos: [Double]) -> String {
        let world = mjs_findBody(ptr, "world")
        let b = mjs_addBody(world, nil)
        mjs_setName(b!.pointee.element, name)       // or b!.pointee.name — see Step 1
        b!.pointee.pos = (pos[0], pos[1], pos[2])
        return name
    }

    public func addGeom(type: MjModel.GeomType, size: [Double], pos: [Double],
                        rgba: [Double], toBody body: String? = nil) {
        let parent = mjs_findBody(ptr, body ?? "world")
        let g = mjs_addGeom(parent, nil)
        g!.pointee.type = cGeom(type)
        g!.pointee.size = (size[0], size[1], size[2])
        g!.pointee.pos = (pos[0], pos[1], pos[2])
        g!.pointee.rgba = (Float(rgba[0]), Float(rgba[1]), Float(rgba[2]), Float(rgba[3]))
    }

    public func compile() throws -> MjModel {
        guard let m = mj_compile(ptr, nil) else {
            throw MjError("mj_compile failed: " + String(cString: mjs_getError(ptr)))
        }
        return MjModel(owning: m)
    }

    private func cGeom(_ t: MjModel.GeomType) -> mjtGeom {
        switch t {
        case .plane: return mjGEOM_PLANE
        case .sphere: return mjGEOM_SPHERE
        case .capsule: return mjGEOM_CAPSULE
        case .ellipsoid: return mjGEOM_ELLIPSOID
        case .cylinder: return mjGEOM_CYLINDER
        case .box, .mesh, .other: return mjGEOM_BOX
        }
    }
}
```

Notes: `MjModel(owning:)` is the internal initializer from Task 2 — it's `init(owning:)` with default (internal) access, callable here in-module. `mjsGeom.size`/`pos` are `double[3]`/`rgba` is `float[4]`, importing as tuples. Setting a name may be `mjs_setName(element, cString)` or a direct `->name` string field depending on version (Step 1 shows which). If `mjs_addLight`/`mjs_getError` names differ, use the grep output.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test --filter SpecTests
```
Expected: PASS. If the `mjs_*` API in this MuJoCo version differs substantially, capture the working call sequence — it is the reference for `WendyMuJoCo.Scene` in the next plan.

- [ ] **Step 6: Run the whole suite and commit**

Run:
```bash
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig swift test
```
Expected: ALL tests PASS.

```bash
git add Sources/MuJoCo/MjSpec.swift Tests/MuJoCoTests/SpecTests.swift
git commit -m "feat: MjSpec scene composition + compile"
```

---

## Self-Review

**Spec coverage** (against the design's `swift-mujoco` scope: "CMuJoCo systemLibrary + generic ergonomic MuJoCo wrapper: value/reference wrappers for model & data, loadXML, step/forward, geom accessors, mju_* math helpers, MjSpec, getState/setState, contacts, name↔id lookups; nothing Wendy-specific"):
- CMuJoCo systemLibrary → Task 1 ✓
- model & data wrappers, loadXML → Tasks 2, 3 ✓
- step/forward/reset → Task 3 ✓
- geom accessors (type/size/rgba/group/visibility) → Task 4 ✓
- mesh buffers → Task 5 ✓
- math + geom world poses + mat↔quat → Task 6 ✓
- name↔id + introspection → Task 7 ✓
- getState/setState + contacts → Task 8 ✓
- MjSpec → Task 9 ✓
- "nothing Wendy-specific" → no JSON/socket/Menagerie anywhere ✓

**Placeholder scan:** No "TBD"/"handle appropriately". The version-sensitive C symbol confirmations (Tasks 8, 9) are explicit grep-then-adjust steps with concrete expected code, not deferrals.

**Type consistency:** `MjModel.load` (Tasks 2), `MjData.init(_:)` and `MjData.model` (Task 3, made non-private in Task 8), `GeomType` (Task 4, reused Task 9), `name(of:id:)`/`id(of:name:)` (Task 7, used in Tasks 5 fix + 9 test), `Vec3` (Task 6, used Task 8 `Contact.pos`), `MjModel.init(owning:)` (Task 2, used Task 9) — all consistent. The Task 5 `_tmpName` helper is explicitly introduced then removed in Task 7, with `meshName` re-routed.

**Cross-task ordering caveat:** Task 5's `meshName` temporarily depends on a private helper until Task 7. If executed out of order, implement Task 7's `name(of:id:)` first. Noted in Task 5, Step 3.
