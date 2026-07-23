import Foundation
import MuJoCo

public let defaultGates: [(Double, Double, Double)] =
    [(4, 0.0, 1.5), (8, 1.0, 1.6), (12, 0.0, 1.5), (16, -1.0, 1.6), (20, 0.0, 1.5)]
public let gateOpening = 1.6
public let reach = 1.1

/// A square gate frame from 4 thin boxes (welded to the world), colour-coded by index.
public func gateFrame(index i: Int, x gx: Double, y gy: Double, z gz: Double,
                      opening w: Double) -> String {
    let h = w / 2 + 0.08
    let col = String(format: "%.2f 0.8 %.2f 1", 0.15 + 0.15 * Double(i), 0.9 - 0.12 * Double(i))
    let posts: [(Double, Double, Double, Double, Double, Double)] = [
        (gx, gy, gz + h, 0.06, w/2 + 0.12, 0.06),   // top bar (spans y)
        (gx, gy, gz - h, 0.06, w/2 + 0.12, 0.06),   // bottom bar
        (gx, gy + w/2 + 0.06, gz, 0.06, 0.06, h),   // left post (spans z)
        (gx, gy - w/2 - 0.06, gz, 0.06, 0.06, h),   // right post
    ]
    return posts.map { (px, py, pz, sx, sy, sz) in
        "<geom type=\"box\" pos=\"\(px) \(py) \(pz)\" size=\"\(sx) \(sy) \(sz)\" "
        + "rgba=\"\(col)\" contype=\"1\" conaffinity=\"1\"/>"
    }.joined()
}

/// Wrap the vendored Skydio X2 (included by relative name) in a gate-slalom world.
public func buildCourseXML(gates: [(Double, Double, Double)], opening: Double) -> String {
    let gatesXML = gates.enumerated()
        .map { (i, g) in gateFrame(index: i, x: g.0, y: g.1, z: g.2, opening: opening) }
        .joined()
    return """
    <mujoco>
      <include file="x2.xml"/>
      <worldbody>
        <geom name="floor" type="plane" size="40 40 0.1" rgba="0.2 0.23 0.28 1"/>
        \(gatesXML)
      </worldbody>
    </mujoco>
    """
}

/// Advance to the next gate once within `reach` of the current one; clamp at the last gate.
public func advanceGate(position p: Vec3, gates: [Vec3], current: Int, reach: Double) -> Int {
    guard current < gates.count else { return current }
    if (p - gates[current]).norm < reach {
        return Swift.min(current + 1, gates.count - 1)
    }
    return current
}
