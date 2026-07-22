public struct AttitudeThrust: Sendable, Equatable {
    public var attitude: Quat     // desired orientation, wxyz
    public var thrust: Double     // collective, normalized 0…1

    public init(attitude: Quat, thrust: Double) {
        self.attitude = attitude; self.thrust = thrust
    }
}
