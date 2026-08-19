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
