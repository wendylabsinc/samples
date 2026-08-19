import Foundation
import SlalomCore
import MuJoCo
import WendyMuJoCo

// Resolve the Skydio X2 model dir (vendored or fetched), copy it to a writable work dir,
// and drop a course.xml beside it so <include file="x2.xml"/> + its assets/ resolve.
func prepareCourse() throws -> String {
    var x2Path = Menagerie.resolveModelPath("skydio_x2", searchDirs: Menagerie.vendorDirs, robot: true)
    if x2Path == nil {
        let cache = WorldSim.directory().appendingPathComponent("menagerie-cache")
        let repo = try Menagerie.fetch("skydio_x2", cacheDir: cache)
        x2Path = Menagerie.resolveModelPath("skydio_x2", searchDirs: [repo.path], robot: true)
    }
    guard let x2 = x2Path else { throw MjError("could not resolve or fetch the skydio_x2 model") }
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

    targetI = advanceGate(position: p, gates: gateVecs, current: targetI, reach: reach)

    if step % 5 == 0 {
        handle.hud([
            "gate": .text("\(Swift.min(targetI + 1, gateVecs.count))/\(gateVecs.count)"),
            "t": .number((data.time - t0)),
            "speed": .number(v.norm),
            "x": .number(p.x),
            "alt": .number(p.z),
        ])
    }
    handle.sync()

    if let maxSteps, step >= maxSteps {
        let alt = (p.z * 100).rounded() / 100, x = (p.x * 10).rounded() / 10
        print("DroneRace: ran \(step) steps; gate \(Swift.min(targetI + 1, gateVecs.count))/\(gateVecs.count); alt=\(alt)m x=\(x)m")
        break
    }
    if maxSteps == nil { Thread.sleep(forTimeInterval: dt) }   // real-time when streaming live
}
