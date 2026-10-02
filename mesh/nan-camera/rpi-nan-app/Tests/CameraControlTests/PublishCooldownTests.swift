import XCTest
import CameraWire
@testable import RPiNANCamera

final class PublishCooldownTests: XCTestCase {
    func testInitialCooldownIsInProcessBoundedAndCancelable() throws {
        var clock = 10.0
        var longestPause = 0.0
        try waitForNANPublishRetry(delay: NANPublicationLease.uncertainRetryDelay,
            now: { clock }, pause: { longestPause = max(longestPause, $0); clock += $0 }, stopped: { false })
        XCTAssertEqual(clock, 193, accuracy: 0.001)
        XCTAssertLessThanOrEqual(longestPause, 0.1)
        clock = 0
        XCTAssertThrowsError(try waitForNANPublishRetry(delay: 183,
            now: { clock }, pause: { clock += $0 }, stopped: { clock >= 0.3 }))
        XCTAssertLessThan(clock, 0.5)
    }

    func testUnknownRefreshDoesNotBlockKnownLeaseOrImmediatelyReplay() {
        var lease = NANPublicationLease(initialID: "101", started: 0)
        let token = lease.beginRefresh(now: 120)!
        XCTAssertEqual(lease.completeRefresh(token: token, id: nil, now: 123, outcomeUnknown: true), [])
        XCTAssertTrue(lease.acceptingIDs(now: 130).contains("101"))
        XCTAssertNil(lease.beginRefresh(now: 128))
        XCTAssertNil(lease.beginRefresh(now: 305.99))
        XCTAssertNotNil(lease.beginRefresh(now: 306))
        // Unknown ID is never guessed/canceled; expired original is discarded.
        XCTAssertEqual(lease.stop(now: 307), [])
    }

    func testDefinitiveRefreshFailureRetainsFiveSecondRetryAndStop() {
        var lease = NANPublicationLease(initialID: "101", started: 0)
        let token = lease.beginRefresh(now: 120)!
        _ = lease.completeRefresh(token: token, id: nil, now: 123)
        XCTAssertNil(lease.beginRefresh(now: 127.99))
        XCTAssertNotNil(lease.beginRefresh(now: 128))
        XCTAssertEqual(lease.stop(now: 129), ["101"])
        XCTAssertNil(lease.beginRefresh(now: 500))
    }
    func testInitialUnknownWaitsBeforeNextMutationAndBeforeRestart() throws {
        var calls = 0
        var delays: [Double] = []
        let result = try initialNANPublication(command: {
            calls += 1
            if calls == 1 { throw NANError.ambiguous("lost publish response") }
            XCTAssertEqual(delays, [183, 1])
            return "101"
        }, now: { 10 }, wait: { delays.append($0) })
        XCTAssertEqual(result.0, "101"); XCTAssertEqual(calls, 2)
        calls = 0; delays = []
        XCTAssertThrowsError(try initialNANPublication(command: {
            calls += 1; throw NANError.ambiguous("lost publish response")
        }, wait: { delays.append($0) }))
        XCTAssertEqual(calls, 6)
        XCTAssertEqual(delays.filter { $0 == 183 }.count, 6)
        XCTAssertEqual(delays.last, 183) // no exit/restart before final cooldown
    }

    func testInitialDefinitiveFailureAndCancellation() throws {
        var calls = 0
        var delays: [Double] = []
        _ = try initialNANPublication(command: {
            calls += 1
            if calls == 1 { throw NANError.failed("FAIL-BUSY") }
            return "101"
        }, wait: { delays.append($0) })
        XCTAssertEqual(delays, [1])
        calls = 0
        XCTAssertThrowsError(try initialNANPublication(command: {
            calls += 1; throw NANError.ambiguous("lost response")
        }, wait: { _ in throw NANError.failed("Stop requested") }))
        XCTAssertEqual(calls, 1)
    }

}
