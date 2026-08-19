import Testing
import MuJoCo
@testable import SlalomCore

@Test func hoverAtTargetGivesEqualThrustsSummingToWeight() {
    let c = DroneController()
    let mass = 2.0
    let p = Vec3(0, 0, 1.5)
    let I = Mat3([1,0,0, 0,1,0, 0,0,1])   // level attitude
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
    let I = Mat3([1,0,0, 0,1,0, 0,0,1])   // level attitude
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
