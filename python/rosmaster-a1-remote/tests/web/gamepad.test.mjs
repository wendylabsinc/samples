import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  GAMEPAD_BUTTONS,
  STOP_MAX_ATTEMPTS,
  STOP_RETRY_DELAY_MS,
  applyDeadzone,
  commandReadoutText,
  computeGamepadStep,
  computeDisconnectStep,
  computeMissingPadStep,
  controlModeText,
  cameraFeedState,
  feedReconnectDelayMs,
  nextExpandedFeed,
  nextPeriodicFeedIndex,
  gamepadClamp,
  nextControlState,
  planStatusStopFollowUp,
  planStopAttempt,
  shouldPushAutoParameters,
  stopConfirmedByStatus,
} = require("../../rosmaster-a1-web-remote-wendy/app/static/gamepad.js");

// Helpers ==================================================================

function button(pressed, value) {
  return { pressed: Boolean(pressed), value: value === undefined ? (pressed ? 1 : 0) : value };
}

function makePad({ axes = [0, 0], buttons = {} } = {}) {
  const list = [];
  for (let i = 0; i <= 16; i += 1) {
    list[i] = buttons[i] ? button(buttons[i].pressed, buttons[i].value) : button(false, 0);
  }
  return { id: "test-pad", axes, buttons: list };
}

function noPrev() {
  return { buttons: [] };
}

function uiState(overrides = {}) {
  return {
    gamepadEnabled: true,
    auto: false,
    armed: false,
    manualSpeed: 0.35,
    autoSpeed: 1.0,
    steerScale: 70,
    feedIds: ["hp60c_depth", "hp60c_rgb"],
    expandedFeed: null,
    ...overrides,
  };
}

// press builds the pair of frames a rising edge needs: prev with the button
// released, pad with it pressed.
function press(index, extra = {}) {
  return {
    prev: noPrev(),
    pad: makePad({ buttons: { [index]: { pressed: true }, ...extra } }),
  };
}

// hold builds the frame after a rising edge, with the button still down.
function hold(index) {
  const prev = { buttons: [] };
  prev.buttons[index] = true;
  return { prev, pad: makePad({ buttons: { [index]: { pressed: true } } }) };
}

// applyDeadzone ============================================================

test("applyDeadzone returns 0 below the threshold", () => {
  assert.equal(applyDeadzone(0.05, 0.12), 0);
});

test("applyDeadzone passes rescaled value through above the threshold", () => {
  const result = applyDeadzone(0.5, 0.12);
  assert.ok(Math.abs(result - 0.43181818) < 1e-6);
});

test("applyDeadzone handles negative values symmetrically", () => {
  const result = applyDeadzone(-0.5, 0.12);
  assert.ok(Math.abs(result - -0.43181818) < 1e-6);
});

test("applyDeadzone returns 0 exactly at the threshold boundary", () => {
  assert.equal(Object.is(applyDeadzone(0.12, 0.12), 0) || Object.is(applyDeadzone(0.12, 0.12), -0), true);
  assert.equal(Object.is(applyDeadzone(-0.12, 0.12), 0) || Object.is(applyDeadzone(-0.12, 0.12), -0), true);
});

// GAMEPAD_BUTTONS ==========================================================

test("GAMEPAD_BUTTONS matches the standard Gamepad API layout", () => {
  assert.equal(GAMEPAD_BUTTONS[0], "A");
  assert.equal(GAMEPAD_BUTTONS[1], "B");
  assert.equal(GAMEPAD_BUTTONS[9], "Menu");
  assert.equal(GAMEPAD_BUTTONS.length, 17);
});

// computeGamepadStep: disabled / auto ======================================

test("disabled pad produces no actions and drive is null", () => {
  const pad = makePad({ buttons: { 1: { pressed: true } } });
  const result = computeGamepadStep(pad, noPrev(), uiState({ gamepadEnabled: false }));
  assert.deepEqual(result.actions, []);
  assert.equal(result.drive, null);
});

test("auto mode gates the arm button and returns no drive", () => {
  const pad = makePad({ axes: [0.9, 0, 0.9], buttons: { 0: { pressed: true }, 7: { pressed: true, value: 1 } } });
  const result = computeGamepadStep(pad, noPrev(), uiState({ auto: true }));
  assert.deepEqual(result.actions, []);
  assert.equal(result.drive, null, "auto drives the car; the pad must not overwrite its commands");
});

test("auto mode still returns no drive when the pad is also armed", () => {
  const pad = makePad({ axes: [0.9, 0, 0.9], buttons: { 7: { pressed: true, value: 1 } } });
  const result = computeGamepadStep(pad, noPrev(), uiState({ auto: true, armed: true }));
  assert.equal(result.drive, null);
});

// computeGamepadStep: B / Menu -> hardStop =================================

test("B rising edge produces hardStop", () => {
  const prev = { buttons: [false, false] };
  const pad = makePad({ buttons: { 1: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState());
  assert.deepEqual(result.actions, [{ type: "hardStop" }]);
});

test("holding B does not repeat hardStop", () => {
  const prev = { buttons: [false, true] };
  const pad = makePad({ buttons: { 1: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState());
  assert.deepEqual(result.actions, []);
});

test("SAFETY: B rising edge with the throttle held returns a zeroed drive, not the stale one", () => {
  // The stale drive used to be the dispatcher's problem: the reducer returned
  // the live stick and trigger values on a stop frame and pollGamepad was
  // obliged to throw them away. Deleting that guard left the whole suite
  // green while a held throttle survived a stop, so the decision is the
  // reducer's now and there is no obligation left to forget.
  const prev = { buttons: [false] };
  const pad = makePad({ axes: [0.9, 0, 0.9], buttons: { 1: { pressed: true }, 7: { pressed: true, value: 1 } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: true }));
  assert.deepEqual(result.actions, [{ type: "hardStop" }]);
  assert.deepEqual(result.drive, { left: { x: 0, y: 0 } });
});

test("SAFETY: a stop frame during Auto Nav also returns a zeroed drive rather than nothing", () => {
  const pad = makePad({ axes: [0.9, 0, 0.9], buttons: { 9: { pressed: true } } });
  const result = computeGamepadStep(pad, noPrev(), uiState({ auto: true, armed: true }));
  assert.deepEqual(result.actions, [{ type: "hardStop" }]);
  assert.deepEqual(result.drive, { left: { x: 0, y: 0 } });
});

test("Menu rising edge produces hardStop", () => {
  const prev = { buttons: [] };
  const pad = makePad({ buttons: { 9: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState());
  assert.deepEqual(result.actions, [{ type: "hardStop" }]);
});

test("holding Menu does not repeat hardStop", () => {
  const prev = { buttons: [] };
  prev.buttons[9] = true;
  const pad = makePad({ buttons: { 9: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState());
  assert.deepEqual(result.actions, []);
});

// computeGamepadStep: A -> startManual ======================================

test("A rising edge produces startManual when not armed", () => {
  const prev = { buttons: [] };
  const pad = makePad({ buttons: { 0: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: false }));
  assert.deepEqual(result.actions, [{ type: "startManual" }]);
});

test("A rising edge produces no startManual when already armed", () => {
  const prev = { buttons: [] };
  const pad = makePad({ buttons: { 0: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: true }));
  assert.deepEqual(result.actions, []);
});

test("holding A does not repeat startManual", () => {
  const prev = { buttons: [] };
  prev.buttons[0] = true;
  const pad = makePad({ buttons: { 0: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: false }));
  assert.deepEqual(result.actions, []);
});

// computeGamepadStep: drive, armed ==========================================

test("armed: steering from axes[0] with 0.12 deadzone", () => {
  const prev = { buttons: [] };
  const pad = makePad({ axes: [0.5, 0] });
  const result = computeGamepadStep(pad, prev, uiState({ armed: true }));
  assert.ok(Math.abs(result.drive.left.x - 0.43181818) < 1e-6);
});

test("armed: throttle is RT minus LT with 0.05 deadzone", () => {
  const prev = { buttons: [] };
  const pad = makePad({ axes: [0, 0], buttons: { 6: { pressed: true, value: 0.2 }, 7: { pressed: true, value: 0.9 } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: true }));
  const expectedDrive = applyDeadzone(0.9 - 0.2, 0.05);
  assert.ok(Math.abs(result.drive.left.y - -expectedDrive) < 1e-9);
});

test("armed: small trigger differential within 0.05 deadzone yields zero drive", () => {
  const prev = { buttons: [] };
  const pad = makePad({ axes: [0, 0], buttons: { 6: { pressed: true, value: 0.5 }, 7: { pressed: true, value: 0.52 } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: true }));
  assert.equal(result.drive.left.y, -0);
  assert.equal(Math.abs(result.drive.left.y), 0);
});

test("armed: the right stick is not read at all, because this chassis cannot turn in place", () => {
  // Measured on the car: angular_z at 1.0 for two seconds produced an encoder
  // delta of exactly zero on all four channels, while steering_y at 0.12
  // moved it. The channel is gone rather than zeroed, so nothing downstream
  // can quietly start sending it again.
  const hardOver = computeGamepadStep(makePad({ axes: [0, 0, -1, -1] }), noPrev(), uiState({ armed: true }));
  const centred = computeGamepadStep(makePad({ axes: [0, 0, 0, 0] }), noPrev(), uiState({ armed: true }));
  assert.deepEqual(hardOver.drive, centred.drive);
  assert.deepEqual(Object.keys(hardOver.drive), ["left"]);
});

// computeGamepadStep: Y -> toggleAuto ========================================

test("Y rising edge toggles Auto Nav on", () => {
  const { prev, pad } = press(3);
  const result = computeGamepadStep(pad, prev, uiState({ auto: false }));
  assert.deepEqual(result.actions, [{ type: "toggleAuto", enabled: true }]);
});

test("Y rising edge toggles Auto Nav back off while it is running", () => {
  const { prev, pad } = press(3);
  const result = computeGamepadStep(pad, prev, uiState({ auto: true }));
  assert.deepEqual(result.actions, [{ type: "toggleAuto", enabled: false }]);
});

test("holding Y does not repeat toggleAuto", () => {
  const { prev, pad } = hold(3);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// computeGamepadStep: X -> expandFeed ========================================
//
// The gallery shows every feed at once, so cycling which one is on screen no
// longer means anything. X now cycles which feed is expanded, and the cycle
// includes the gallery itself, so the pad has a way back rather than only a way
// deeper in.

test("X rising edge expands the first feed when the gallery is showing", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({ expandedFeed: null }));
  assert.deepEqual(result.actions, [{ type: "expandFeed", id: "hp60c_depth" }]);
});

test("X moves the expansion on to the next feed", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({ expandedFeed: "hp60c_depth" }));
  assert.deepEqual(result.actions, [{ type: "expandFeed", id: "hp60c_rgb" }]);
});

test("X on the last feed returns to the gallery rather than wrapping straight round", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({ expandedFeed: "hp60c_rgb" }));
  assert.deepEqual(result.actions, [{ type: "expandFeed", id: null }]);
});

test("X expands a feed the car reports as stale, because a tile the operator can see is one they may inspect", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({
    feedIds: ["hp60c_depth", "hp60c_rgb"],
    expandedFeed: "hp60c_depth",
  }));
  assert.deepEqual(result.actions, [{ type: "expandFeed", id: "hp60c_rgb" }]);
});

test("X falls back to the first feed when the expanded one went away", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({
    feedIds: ["hp60c_rgb"],
    expandedFeed: "hp60c_ir",
  }));
  assert.deepEqual(result.actions, [{ type: "expandFeed", id: "hp60c_rgb" }]);
});

test("X produces no expandFeed when the car reports no feeds at all", () => {
  const { prev, pad } = press(2);
  const result = computeGamepadStep(pad, prev, uiState({ feedIds: [], expandedFeed: null }));
  assert.deepEqual(result.actions, []);
});

test("holding X does not repeat expandFeed", () => {
  const { prev, pad } = hold(2);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// The camera feed registry ===================================================

test("nextExpandedFeed walks the feeds and then the gallery", () => {
  const ids = ["a", "b", "c"];
  assert.equal(nextExpandedFeed(null, ids), "a");
  assert.equal(nextExpandedFeed("a", ids), "b");
  assert.equal(nextExpandedFeed("b", ids), "c");
  assert.equal(nextExpandedFeed("c", ids), null);
});

test("nextExpandedFeed has nowhere to go with no feeds", () => {
  assert.equal(nextExpandedFeed(null, []), null);
  assert.equal(nextExpandedFeed("a", null), null);
});

test("cameraFeedState calls a live feed live", () => {
  const view = cameraFeedState({ id: "hp60c_depth", ok: true, age_s: 0.06 });
  assert.equal(view.level, "live");
});

test("cameraFeedState names a stale feed and its age, so a frozen frame is not read as a live one", () => {
  const view = cameraFeedState({ id: "hp60c_rgb", ok: false, age_s: 5.24 });
  assert.equal(view.level, "stale");
  assert.match(view.text, /stale/i);
  assert.match(view.text, /5\.2/, "the operator watches the age climb to tell a hiccup from a dead feed");
});

test("cameraFeedState tells a feed that has never delivered a frame from one that went stale", () => {
  const waiting = cameraFeedState({ id: "hp60c_rgb", ok: false, age_s: null });
  assert.equal(waiting.level, "waiting");
  assert.doesNotMatch(waiting.text, /stale/i);
  assert.notEqual(waiting.text, cameraFeedState({ id: "hp60c_rgb", ok: false, age_s: 5.24 }).text);
});

test("cameraFeedState survives a malformed entry rather than blanking the tile", () => {
  for (const bad of [null, undefined, {}, { ok: "yes" }]) {
    const view = cameraFeedState(bad);
    assert.ok(view.level, "every tile must have a state");
    assert.ok(view.text);
  }
});

// Reconnect pacing ===========================================================
//
// Several MJPEG streams over a laggy link is the pressure that starved the
// server of threads earlier today. Every tile reconnecting on the same 1 s
// boundary is exactly that burst, so the delay carries the tile's own offset.

test("feedReconnectDelayMs staggers one tile behind the next", () => {
  const first = feedReconnectDelayMs(0);
  const second = feedReconnectDelayMs(1);
  const third = feedReconnectDelayMs(2);
  assert.ok(first >= 1000, "the existing one second wait before a retry is kept");
  assert.ok(second > first, "two tiles failing together must not retry together");
  assert.equal(third - second, second - first, "the offsets are evenly spaced");
});

test("feedReconnectDelayMs treats junk as the first slot rather than producing NaN", () => {
  for (const bad of [null, undefined, -3, "x"]) {
    assert.equal(feedReconnectDelayMs(bad), feedReconnectDelayMs(0));
  }
});

test("nextPeriodicFeedIndex refreshes one tile per tick, round robin", () => {
  assert.equal(nextPeriodicFeedIndex(-1, 2), 0);
  assert.equal(nextPeriodicFeedIndex(0, 2), 1);
  assert.equal(nextPeriodicFeedIndex(1, 2), 0);
});

test("nextPeriodicFeedIndex has nothing to refresh with no tiles", () => {
  assert.equal(nextPeriodicFeedIndex(-1, 0), null);
  assert.equal(nextPeriodicFeedIndex(3, null), null);
});

test("nextPeriodicFeedIndex recovers when the tile list shrank under it", () => {
  assert.equal(nextPeriodicFeedIndex(7, 2), 0);
});

// computeGamepadStep: D-pad Up / Down -> manual speed ========================

test("D-pad Up nudges manual speed by plus 0.05", () => {
  const { prev, pad } = press(12);
  const result = computeGamepadStep(pad, prev, uiState({ manualSpeed: 0.35 }));
  assert.deepEqual(result.actions, [{ type: "nudgeManualSpeed", value: 0.4 }]);
});

test("D-pad Down nudges manual speed by minus 0.05", () => {
  const { prev, pad } = press(13);
  const result = computeGamepadStep(pad, prev, uiState({ manualSpeed: 0.35 }));
  assert.deepEqual(result.actions, [{ type: "nudgeManualSpeed", value: 0.3 }]);
});

test("D-pad Up clamps manual speed at the slider maximum of 2", () => {
  const { prev, pad } = press(12);
  const result = computeGamepadStep(pad, prev, uiState({ manualSpeed: 1.98 }));
  assert.deepEqual(result.actions, [{ type: "nudgeManualSpeed", value: 2 }]);
});

test("D-pad Down clamps manual speed at the slider minimum of 0", () => {
  const { prev, pad } = press(13);
  const result = computeGamepadStep(pad, prev, uiState({ manualSpeed: 0.02 }));
  assert.deepEqual(result.actions, [{ type: "nudgeManualSpeed", value: 0 }]);
});

test("holding D-pad Up does not repeat nudgeManualSpeed", () => {
  const { prev, pad } = hold(12);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

test("holding D-pad Down does not repeat nudgeManualSpeed", () => {
  const { prev, pad } = hold(13);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// computeGamepadStep: D-pad Right / Left -> auto speed =======================

test("D-pad Right nudges auto speed by plus 0.05", () => {
  const { prev, pad } = press(15);
  const result = computeGamepadStep(pad, prev, uiState({ autoSpeed: 1.0 }));
  assert.deepEqual(result.actions, [{ type: "nudgeAutoSpeed", value: 1.05 }]);
});

test("D-pad Left nudges auto speed by minus 0.05", () => {
  const { prev, pad } = press(14);
  const result = computeGamepadStep(pad, prev, uiState({ autoSpeed: 1.0 }));
  assert.deepEqual(result.actions, [{ type: "nudgeAutoSpeed", value: 0.95 }]);
});

test("D-pad Right clamps auto speed at the slider maximum of 2", () => {
  const { prev, pad } = press(15);
  const result = computeGamepadStep(pad, prev, uiState({ autoSpeed: 2 }));
  assert.deepEqual(result.actions, [{ type: "nudgeAutoSpeed", value: 2 }]);
});

test("D-pad Left clamps auto speed at the slider minimum of 0", () => {
  const { prev, pad } = press(14);
  const result = computeGamepadStep(pad, prev, uiState({ autoSpeed: 0.01 }));
  assert.deepEqual(result.actions, [{ type: "nudgeAutoSpeed", value: 0 }]);
});

test("holding D-pad Right does not repeat nudgeAutoSpeed", () => {
  const { prev, pad } = hold(15);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

test("holding D-pad Left does not repeat nudgeAutoSpeed", () => {
  const { prev, pad } = hold(14);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// computeGamepadStep: LB / RB -> steering scale ==============================

test("RB nudges steering scale up by 10 percent", () => {
  const { prev, pad } = press(5);
  const result = computeGamepadStep(pad, prev, uiState({ steerScale: 70 }));
  assert.deepEqual(result.actions, [{ type: "nudgeSteerScale", value: 80 }]);
});

test("LB nudges steering scale down by 10 percent", () => {
  const { prev, pad } = press(4);
  const result = computeGamepadStep(pad, prev, uiState({ steerScale: 70 }));
  assert.deepEqual(result.actions, [{ type: "nudgeSteerScale", value: 60 }]);
});

test("RB clamps steering scale at the slider maximum of 100 percent", () => {
  const { prev, pad } = press(5);
  const result = computeGamepadStep(pad, prev, uiState({ steerScale: 95 }));
  assert.deepEqual(result.actions, [{ type: "nudgeSteerScale", value: 100 }]);
});

test("LB clamps steering scale at the slider minimum of 10 percent", () => {
  const { prev, pad } = press(4);
  const result = computeGamepadStep(pad, prev, uiState({ steerScale: 15 }));
  assert.deepEqual(result.actions, [{ type: "nudgeSteerScale", value: 10 }]);
});

test("holding RB does not repeat nudgeSteerScale", () => {
  const { prev, pad } = hold(5);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

test("holding LB does not repeat nudgeSteerScale", () => {
  const { prev, pad } = hold(4);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// computeGamepadStep: View -> reconnectCamera ================================

test("View rising edge asks the camera stream to reconnect", () => {
  const { prev, pad } = press(8);
  const result = computeGamepadStep(pad, prev, uiState());
  assert.deepEqual(result.actions, [{ type: "reconnectCamera" }]);
});

test("holding View does not repeat reconnectCamera", () => {
  const { prev, pad } = hold(8);
  assert.deepEqual(computeGamepadStep(pad, prev, uiState()).actions, []);
});

// computeGamepadStep: unmapped buttons =======================================

test("LS, RS and the Xbox button stay unmapped", () => {
  for (const index of [10, 11, 16]) {
    const { prev, pad } = press(index);
    assert.deepEqual(
      computeGamepadStep(pad, prev, uiState()).actions,
      [],
      `button ${index} must stay unmapped`,
    );
  }
});

// computeGamepadStep: mode and tuning buttons survive auto ===================

test("Y, X, View, the D-pad and the bumpers all stay live during Auto Nav", () => {
  const cases = [
    [3, { type: "toggleAuto", enabled: false }],
    [2, { type: "expandFeed", id: "hp60c_depth" }],
    [8, { type: "reconnectCamera" }],
    [12, { type: "nudgeManualSpeed", value: 0.4 }],
    [13, { type: "nudgeManualSpeed", value: 0.3 }],
    [15, { type: "nudgeAutoSpeed", value: 1.05 }],
    [14, { type: "nudgeAutoSpeed", value: 0.95 }],
    [5, { type: "nudgeSteerScale", value: 80 }],
    [4, { type: "nudgeSteerScale", value: 60 }],
  ];
  for (const [index, expected] of cases) {
    const { prev, pad } = press(index);
    assert.deepEqual(
      computeGamepadStep(pad, prev, uiState({ auto: true })).actions,
      [expected],
      `button ${index} must stay live during auto`,
    );
  }
});

// computeGamepadStep: stop wins ==============================================

test("a hardStop in the same frame suppresses toggleAuto and startManual", () => {
  const pad = makePad({ buttons: { 1: { pressed: true }, 3: { pressed: true }, 0: { pressed: true } } });
  const result = computeGamepadStep(pad, noPrev(), uiState());
  assert.deepEqual(result.actions, [{ type: "hardStop" }]);
});

test("a hardStop in the same frame still allows camera and tuning actions", () => {
  const pad = makePad({ buttons: { 1: { pressed: true }, 8: { pressed: true } } });
  const result = computeGamepadStep(pad, noPrev(), uiState());
  assert.deepEqual(result.actions, [{ type: "hardStop" }, { type: "reconnectCamera" }]);
});

test("a hardStop in the same frame suppresses both auto speed nudges, which would re-POST /api/auto with enabled true", () => {
  for (const index of [15, 14]) {
    const pad = makePad({ buttons: { 1: { pressed: true }, [index]: { pressed: true } } });
    const result = computeGamepadStep(pad, noPrev(), uiState({ auto: true }));
    assert.deepEqual(
      result.actions,
      [{ type: "hardStop" }],
      `button ${index} must not nudge auto speed on a stop frame`,
    );
  }
});

test("a hardStop in the same frame leaves the manual speed nudges alone, they re-POST nothing", () => {
  const pad = makePad({ buttons: { 1: { pressed: true }, 12: { pressed: true } } });
  const result = computeGamepadStep(pad, noPrev(), uiState({ manualSpeed: 0.35 }));
  assert.deepEqual(result.actions, [
    { type: "hardStop" },
    { type: "nudgeManualSpeed", value: 0.4 },
  ]);
});

// computeGamepadStep: nudge arithmetic stays on the slider step ==============

test("repeated manual speed nudges stay on the 0.01 slider step without float drift", () => {
  let speed = 0;
  for (let i = 0; i < 7; i += 1) {
    const { prev, pad } = press(12);
    speed = computeGamepadStep(pad, prev, uiState({ manualSpeed: speed })).actions[0].value;
  }
  assert.equal(speed, 0.35);
});

// computeGamepadStep: drive, not armed =======================================

test("not armed: drive is zeroed even with stick and trigger input", () => {
  const prev = { buttons: [] };
  const pad = makePad({ axes: [0.9, 0.9], buttons: { 7: { pressed: true, value: 1 } } });
  const result = computeGamepadStep(pad, prev, uiState({ armed: false }));
  assert.deepEqual(result.drive, { left: { x: 0, y: 0 } });
});

// computeGamepadStep: nextPadState ===========================================

test("nextPadState reflects the current frame's pressed buttons when enabled", () => {
  const prev = { buttons: [] };
  const pad = makePad({ buttons: { 0: { pressed: true }, 1: { pressed: false } } });
  const result = computeGamepadStep(pad, prev, uiState());
  assert.equal(result.nextPadState.buttons[0], true);
  assert.equal(result.nextPadState.buttons[1], false);
});

test("nextPadState is unchanged from prevPadState when disabled (edge state freezes)", () => {
  const prev = { buttons: [true, false] };
  const pad = makePad({ buttons: { 0: { pressed: true } } });
  const result = computeGamepadStep(pad, prev, uiState({ gamepadEnabled: false }));
  assert.deepEqual(result.nextPadState.buttons, [true, false]);
});

// The page wiring these functions feed used to be covered by tests that read
// index.html as text and matched substrings. Those are gone: the page script
// is tests/web/wiring.test.mjs's subject now, running for real against a fake
// DOM, so the wiring is checked by what the page does rather than by what its
// source happens to contain.

// Safety rules ==============================================================
// The three tests below are the safety contract of this file. Each one guards
// a way the operator could previously lose the ability to stop a moving car.

test("SAFETY: hard stop still works during Auto Nav (B and Menu are never gated by auto)", () => {
  for (const index of [1, 9]) {
    const pad = makePad({ buttons: { [index]: { pressed: true } } });
    const result = computeGamepadStep(pad, noPrev(), uiState({ auto: true }));
    assert.deepEqual(
      result.actions,
      [{ type: "hardStop" }],
      `button ${index} must produce hardStop while auto is engaged`,
    );
  }
});

// nextControlState is the funnel for the on-screen Stop button, B, Menu, a
// controller disconnect and pagehide, so every transition is asserted whole.
// Asserting only .gamepadEnabled here once let hardStop return armed and auto
// both true with the suite still green.
test("SAFETY: nextControlState returns the whole resulting state for every transition", () => {
  const armedLive = { armed: true, auto: false, gamepadEnabled: true };
  const autoLive = { armed: false, auto: true, gamepadEnabled: true };

  // hardStop is the one that matters most: it must leave nothing running.
  assert.deepEqual(nextControlState("hardStop", armedLive), { armed: false, auto: false, gamepadEnabled: true });
  assert.deepEqual(nextControlState("hardStop", autoLive), { armed: false, auto: false, gamepadEnabled: true });
  assert.deepEqual(
    nextControlState("hardStop", { armed: true, auto: true, gamepadEnabled: false }),
    { armed: false, auto: false, gamepadEnabled: false },
  );

  // autoOn clears armed: Auto Nav owns the motors, and a manual arm left
  // standing would hand the operator a live throttle the moment auto ended.
  assert.deepEqual(nextControlState("autoOn", armedLive), { armed: false, auto: true, gamepadEnabled: true });
  assert.deepEqual(
    nextControlState("autoOn", { armed: false, auto: false, gamepadEnabled: false }),
    { armed: false, auto: true, gamepadEnabled: false },
  );

  // autoOff carries armed through, which the arm checkbox handler depends on.
  assert.deepEqual(nextControlState("autoOff", autoLive), { armed: false, auto: false, gamepadEnabled: true });
  assert.deepEqual(
    nextControlState("autoOff", { armed: true, auto: true, gamepadEnabled: true }),
    { armed: true, auto: false, gamepadEnabled: true },
  );

  assert.deepEqual(nextControlState("startManual", autoLive), { armed: true, auto: false, gamepadEnabled: true });

  // An unknown transition changes nothing at all.
  assert.deepEqual(nextControlState("nonsense", autoLive), { armed: false, auto: true, gamepadEnabled: true });
  assert.deepEqual(nextControlState("nonsense", armedLive), { armed: true, auto: false, gamepadEnabled: true });
});

test("SAFETY: no transition can switch the pad off, because the pad is how the car gets stopped", () => {
  for (const transition of ["autoOn", "autoOff", "hardStop", "startManual", "nonsense"]) {
    assert.equal(
      nextControlState(transition, { armed: true, auto: true, gamepadEnabled: true }).gamepadEnabled,
      true,
      `${transition} switched the pad off`,
    );
  }
});

test("SAFETY: a controller disconnect stops the car whenever it is armed or in auto", () => {
  assert.deepEqual(computeDisconnectStep({ armed: true, auto: false }).actions, [{ type: "hardStop" }]);
  assert.deepEqual(computeDisconnectStep({ armed: false, auto: true }).actions, [{ type: "hardStop" }]);
  assert.deepEqual(computeDisconnectStep({ armed: true, auto: true }).actions, [{ type: "hardStop" }]);
  assert.deepEqual(computeDisconnectStep({ armed: false, auto: false }).actions, []);
});

test("SAFETY: a poll frame that finds no pad stops the car and zeroes the drive, without waiting for a disconnect event", () => {
  const zero = { left: { x: 0, y: 0 } };
  for (const mode of [{ armed: true, auto: false }, { armed: false, auto: true }]) {
    const step = computeMissingPadStep({ hadPad: true, gamepadEnabled: true, ...mode });
    assert.deepEqual(step.actions, [{ type: "hardStop" }], `${JSON.stringify(mode)} must stop the car`);
    assert.deepEqual(step.drive, zero, `${JSON.stringify(mode)} must zero the outgoing drive`);
  }
});

test("SAFETY: a poll frame that finds no pad zeroes the drive even when the car is idle, so nothing latches the last command", () => {
  const step = computeMissingPadStep({ hadPad: true, gamepadEnabled: true, armed: false, auto: false });
  assert.deepEqual(step.actions, []);
  assert.deepEqual(step.drive, { left: { x: 0, y: 0 } });
});

test("a missing pad does nothing on a frame that never had one, so the stop does not repeat every frame", () => {
  const step = computeMissingPadStep({ hadPad: false, gamepadEnabled: true, armed: true, auto: true });
  assert.deepEqual(step.actions, []);
  assert.equal(step.drive, null);
});

test("SAFETY: a missing pad stops the car even with the Xbox toggle off", () => {
  // The toggle governs whether the pad may drive the car, not whether its
  // disappearance is a safety event. The two disconnect paths used to disagree
  // about this: the event stopped regardless of the toggle and the poll loop
  // refused to, so whether a sleeping Bluetooth pad aborted an autonomous run
  // came down to whether gamepaddisconnected happened to fire.
  for (const mode of [{ armed: true, auto: false }, { armed: false, auto: true }]) {
    const step = computeMissingPadStep({ hadPad: true, gamepadEnabled: false, ...mode });
    assert.deepEqual(step.actions, [{ type: "hardStop" }], `${JSON.stringify(mode)} must stop the car`);
    assert.equal(
      step.drive,
      null,
      "the drive is still left alone though: with the toggle off the touch UI owns the sticks",
    );
  }
});

test("a missing pad with the toggle off and nothing running does nothing at all", () => {
  const step = computeMissingPadStep({ hadPad: true, gamepadEnabled: false, armed: false, auto: false });
  assert.deepEqual(step.actions, []);
  assert.equal(step.drive, null);
});

// Stop path confirmation =====================================================
// A status field that lies is worse than no status field at all. These pin the
// rule that the page never presents the car as stopped on a stop it has not
// had confirmed.

test("SAFETY: an unconfirmed stop never reads as Disarmed", () => {
  const text = controlModeText({ armed: false, auto: false, stopUnconfirmed: true });
  assert.notEqual(text, "Disarmed");
  assert.match(text, /UNCONFIRMED/);
});

test("SAFETY: a stop still in flight never reads as Disarmed", () => {
  const text = controlModeText({ armed: false, auto: false, stopPending: { auto: true } });
  assert.notEqual(text, "Disarmed");
});

test("controlModeText reports the plain modes when no stop is outstanding", () => {
  assert.equal(controlModeText({ auto: true, armed: false }), "Auto Nav");
  assert.equal(controlModeText({ auto: false, armed: true }), "Manual Armed");
  assert.equal(controlModeText({ auto: false, armed: false }), "Disarmed");
  assert.equal(controlModeText(), "Disarmed");
});

test("controlModeText puts an unconfirmed stop above every other mode", () => {
  assert.match(controlModeText({ auto: true, armed: true, stopUnconfirmed: true }), /UNCONFIRMED/);
});

test("planStopAttempt confirms as soon as a POST resolves", () => {
  assert.deepEqual(planStopAttempt(1, true), { outcome: "confirmed", retry: false, delayMs: 0 });
  assert.deepEqual(planStopAttempt(STOP_MAX_ATTEMPTS, true), { outcome: "confirmed", retry: false, delayMs: 0 });
});

test("planStopAttempt retries a failed stop with a bounded delay", () => {
  const plan = planStopAttempt(1, false);
  assert.equal(plan.outcome, "retry");
  assert.equal(plan.retry, true);
  assert.equal(plan.delayMs, STOP_RETRY_DELAY_MS);
  assert.ok(plan.delayMs > 0 && plan.delayMs < 1000, "the retry must be prompt; the car is moving");
});

test("SAFETY: planStopAttempt gives up as unconfirmed rather than silently succeeding", () => {
  const plan = planStopAttempt(STOP_MAX_ATTEMPTS, false);
  assert.equal(plan.outcome, "unconfirmed");
  assert.equal(plan.retry, false);
});

test("the stop retry budget is bounded and larger than one attempt", () => {
  assert.ok(STOP_MAX_ATTEMPTS > 1, "one attempt is the behavior this finding replaced");
  assert.ok(STOP_MAX_ATTEMPTS <= 5, "the operator must not wait on an unbounded retry loop");
  assert.equal(planStopAttempt(STOP_MAX_ATTEMPTS - 1, false).outcome, "retry");
});

test("SAFETY: a status poll confirms an unconfirmed stop only on the server's own evidence", () => {
  const duringAuto = { auto: true, armed: false };
  assert.equal(stopConfirmedByStatus(duringAuto, { auto: false, armed: false }), true);
  assert.equal(stopConfirmedByStatus(duringAuto, { auto: true, armed: false }), false);
  assert.equal(stopConfirmedByStatus(duringAuto, {}), false, "a status missing the field proves nothing");

  const duringManual = { auto: false, armed: true };
  assert.equal(stopConfirmedByStatus(duringManual, { auto: false, armed: false }), true);
  assert.equal(stopConfirmedByStatus(duringManual, { auto: false, armed: true }), false);
});

test("stopConfirmedByStatus reports nothing to confirm when no stop is outstanding", () => {
  assert.equal(stopConfirmedByStatus(null, { auto: false, armed: false }), false);
});

test("a stop issued while nothing was live still needs the server to report both channels quiet", () => {
  const idle = { auto: false, armed: false };
  assert.equal(stopConfirmedByStatus(idle, { auto: false, armed: false }), true);
  assert.equal(stopConfirmedByStatus(idle, { auto: true, armed: false }), false);
  assert.equal(stopConfirmedByStatus(idle, { auto: false, armed: true }), false);
  assert.equal(stopConfirmedByStatus(idle, {}), false);
});

// planStatusStopFollowUp ======================================================

test("SAFETY: an unconfirmed stop is re-issued until the server's own evidence closes it out", () => {
  const pending = { stopPending: { armed: false, auto: true }, stopUnconfirmed: true };
  assert.deepEqual(
    planStatusStopFollowUp(pending, { auto: true, armed: false }),
    { confirmed: false, reissue: true },
    "an autonomous run has no background sender that would retry the stop for us",
  );
  assert.deepEqual(
    planStatusStopFollowUp(pending, {}),
    { confirmed: false, reissue: true },
    "a status missing the field proves nothing, so the stop goes out again",
  );
  assert.deepEqual(
    planStatusStopFollowUp(pending, { auto: false, armed: false }),
    { confirmed: true, reissue: false },
  );
});

test("planStatusStopFollowUp does nothing when no stop is outstanding", () => {
  assert.deepEqual(
    planStatusStopFollowUp({ stopPending: null }, { auto: true, armed: true }),
    { confirmed: false, reissue: false },
    "a poll must not start sending stops at a car nobody asked to stop",
  );
  assert.deepEqual(planStatusStopFollowUp(), { confirmed: false, reissue: false });
});

test("a manual stop is re-issued on the same evidence rule as an autonomous one", () => {
  const pending = { stopPending: { armed: true, auto: false } };
  assert.deepEqual(planStatusStopFollowUp(pending, { auto: false, armed: true }), { confirmed: false, reissue: true });
  assert.deepEqual(planStatusStopFollowUp(pending, { auto: false, armed: false }), { confirmed: true, reissue: false });
});

// shouldPushAutoParameters ====================================================

test("SAFETY: a speed nudge may not re-command Auto Nav while a stop is outstanding", () => {
  assert.equal(shouldPushAutoParameters({ auto: true, stopPending: { auto: true } }), false);
  assert.equal(shouldPushAutoParameters({ auto: true, stopUnconfirmed: true }), false);
  assert.equal(
    shouldPushAutoParameters({ auto: true, stopPending: { auto: true }, stopUnconfirmed: true }),
    false,
    "clearing a stop warning takes a deliberate act, not a slider nudge",
  );
});

test("a speed nudge does push to a running Auto Nav with nothing outstanding", () => {
  assert.equal(shouldPushAutoParameters({ auto: true, stopPending: null, stopUnconfirmed: false }), true);
});

test("a speed nudge pushes nothing when Auto Nav is not running", () => {
  assert.equal(shouldPushAutoParameters({ auto: false }), false);
  assert.equal(shouldPushAutoParameters(), false);
});

// commandReadoutText ==========================================================

test("SAFETY: the Command readout shows the server's published command during Auto Nav", () => {
  const local = { linear_x: 0, steering_y: 0 };
  const published = { linear_x: 0.42, steering_y: -0.03 };
  assert.equal(
    commandReadoutText(true, local, published),
    "0.42 / -0.03",
    "the local manual command is zero throughout an autonomous run; showing it would be a readout that lies",
  );
});

test("the Command readout shows this page's own command outside Auto Nav", () => {
  assert.equal(commandReadoutText(false, { linear_x: 0.35, steering_y: 0.02 }, { linear_x: 9.99, steering_y: 9.99 }), "0.35 / 0.02");
});

test("the Command readout falls back to the local command when the server has published nothing", () => {
  assert.equal(commandReadoutText(true, { linear_x: 0.1, steering_y: 0 }, null), "0.10 / 0.00");
  assert.equal(commandReadoutText(true, null, null), "0.00 / 0.00");
});

// gamepadClamp ================================================================

test("gamepadClamp is the one clamp both static files share", () => {
  assert.equal(gamepadClamp(5, 0, 1), 1);
  assert.equal(gamepadClamp(-5, 0, 1), 0);
  assert.equal(gamepadClamp(0.5, 0, 1), 0.5);
});
