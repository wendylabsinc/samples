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
