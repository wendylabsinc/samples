public struct Vec3: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x; self.y = y; self.z = z
    }

    public static let zero = Vec3(x: 0, y: 0, z: 0)
    public static let up = Vec3(x: 0, y: 0, z: 1)

    public static func + (a: Vec3, b: Vec3) -> Vec3 {
        Vec3(x: a.x + b.x, y: a.y + b.y, z: a.z + b.z)
    }
    public static func - (a: Vec3, b: Vec3) -> Vec3 {
        Vec3(x: a.x - b.x, y: a.y - b.y, z: a.z - b.z)
    }
    public static func * (a: Vec3, s: Double) -> Vec3 {
        Vec3(x: a.x * s, y: a.y * s, z: a.z * s)
    }
    public static func * (s: Double, a: Vec3) -> Vec3 { a * s }

    public func dot(_ o: Vec3) -> Double { x * o.x + y * o.y + z * o.z }

    public func cross(_ o: Vec3) -> Vec3 {
        Vec3(x: y * o.z - z * o.y,
             y: z * o.x - x * o.z,
             z: x * o.y - y * o.x)
    }

    public var length: Double { dot(self).squareRoot() }

    public func normalized() -> Vec3 {
        let l = length
        return l > 0 ? self * (1 / l) : self
    }
}

public struct Quat: Sendable, Equatable {
    public var w: Double
    public var x: Double
    public var y: Double
    public var z: Double

    public init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w; self.x = x; self.y = y; self.z = z
    }

    public static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    public var length: Double {
        (w * w + x * x + y * y + z * z).squareRoot()
    }

    public func normalized() -> Quat {
        let l = length
        guard l > 0 else { return .identity }
        return Quat(w: w / l, x: x / l, y: y / l, z: z / l)
    }

    /// Orientation whose body-z axis aligns with `desiredZ`, at world yaw `yaw`.
    /// Columns b1,b2,b3 form the rotation matrix; converted to a wxyz quaternion.
    public init(desiredZ: Vec3, yaw: Double) {
        let b3 = desiredZ.normalized()
        // Desired heading direction in the world x-y plane.
        let b1c = Vec3(x: Foundation_cos(yaw), y: Foundation_sin(yaw), z: 0)
        var b2 = b3.cross(b1c)
        if b2.length < 1e-9 {
            // b3 parallel to b1c (pointing along heading) — pick any orthogonal.
            b2 = b3.cross(Vec3(x: 1, y: 0, z: 0))
            if b2.length < 1e-9 { b2 = b3.cross(Vec3(x: 0, y: 1, z: 0)) }
        }
        b2 = b2.normalized()
        let b1 = b2.cross(b3)
        self = Quat.fromColumns(b1: b1, b2: b2, b3: b3).normalized()
    }

    /// Build a wxyz quaternion from rotation-matrix columns (Shepperd's method).
    static func fromColumns(b1: Vec3, b2: Vec3, b3: Vec3) -> Quat {
        // Matrix m[row][col], columns are b1,b2,b3.
        let m00 = b1.x, m10 = b1.y, m20 = b1.z
        let m01 = b2.x, m11 = b2.y, m21 = b2.z
        let m02 = b3.x, m12 = b3.y, m22 = b3.z
        let trace = m00 + m11 + m22
        if trace > 0 {
            let s = (trace + 1).squareRoot() * 2  // s = 4*w
            return Quat(w: 0.25 * s,
                        x: (m21 - m12) / s,
                        y: (m02 - m20) / s,
                        z: (m10 - m01) / s)
        } else if m00 > m11 && m00 > m22 {
            let s = (1 + m00 - m11 - m22).squareRoot() * 2  // s = 4*x
            return Quat(w: (m21 - m12) / s,
                        x: 0.25 * s,
                        y: (m01 + m10) / s,
                        z: (m02 + m20) / s)
        } else if m11 > m22 {
            let s = (1 + m11 - m00 - m22).squareRoot() * 2  // s = 4*y
            return Quat(w: (m02 - m20) / s,
                        x: (m01 + m10) / s,
                        y: 0.25 * s,
                        z: (m12 + m21) / s)
        } else {
            let s = (1 + m22 - m00 - m11).squareRoot() * 2  // s = 4*z
            return Quat(w: (m10 - m01) / s,
                        x: (m02 + m20) / s,
                        y: (m12 + m21) / s,
                        z: 0.25 * s)
        }
    }

    /// Body-z axis expressed in the world frame (third column of the rotation matrix).
    public var bodyZ: Vec3 {
        Vec3(x: 2 * (x * z + w * y),
             y: 2 * (y * z - w * x),
             z: 1 - 2 * (x * x + y * y))
    }

    /// Angle in radians between this orientation's body-z axis and `axis`.
    public func angle(to axis: Vec3) -> Double {
        let d = bodyZ.normalized().dot(axis.normalized())
        return Foundation_acos(min(1, max(-1, d)))
    }
}

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@inline(__always) func Foundation_cos(_ v: Double) -> Double { cos(v) }
@inline(__always) func Foundation_sin(_ v: Double) -> Double { sin(v) }
@inline(__always) func Foundation_acos(_ v: Double) -> Double { acos(v) }
@inline(__always) func Foundation_atan2(_ y: Double, _ x: Double) -> Double { atan2(y, x) }
