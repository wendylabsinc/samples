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
