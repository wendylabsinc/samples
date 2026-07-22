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
