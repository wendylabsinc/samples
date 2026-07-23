import Testing
import MuJoCo
@testable import SlalomCore

@Test func courseHasFloorIncludeAndFourBoxesPerGate() {
    let gates = [(4.0, 0.0, 1.5), (8.0, 1.0, 1.6)]
    let xml = buildCourseXML(gates: gates, opening: 1.6)
    #expect(xml.contains("<include file=\"x2.xml\"/>"))
    #expect(xml.contains("type=\"plane\""))                       // floor
    let boxes = xml.components(separatedBy: "type=\"box\"").count - 1
    #expect(boxes == gates.count * 4)                             // 4 box geoms per gate
}

@Test func defaultCourseHasFiveGates() {
    #expect(defaultGates.count == 5)
    #expect(gateOpening == 1.6)
    #expect(reach == 1.1)
}

@Test func gateAdvancesWithinReachAndClampsAtEnd() {
    let gates = [Vec3(4,0,1.5), Vec3(8,1,1.6)]
    #expect(advanceGate(position: Vec3(0,0,1), gates: gates, current: 0, reach: 1.1) == 0)
    #expect(advanceGate(position: Vec3(4,0,1.5), gates: gates, current: 0, reach: 1.1) == 1)
    #expect(advanceGate(position: Vec3(8,1,1.6), gates: gates, current: 1, reach: 1.1) == 1)
}
