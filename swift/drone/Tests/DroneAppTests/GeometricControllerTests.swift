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
