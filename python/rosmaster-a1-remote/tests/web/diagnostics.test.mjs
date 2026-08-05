// Tests for the pure helpers behind the Controller panel and the drive send
// strategy. Both are decisions, not rendering, so they live in gamepad.js and
// are tested here without a DOM. wiring.test.mjs covers whether app.js
// actually paints and sends what these functions decide.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  DRIVE_HEARTBEAT_MS,
  DRIVE_ZERO_REPEATS,
  GAMEPAD_LOG_LIMIT,
  agoText,
  describeGamepad,
  describeGamepadAction,
  drivePayloadText,
  driveCommandsEqual,
  formatAxisValues,
  formatPressedButtons,
  gamepadContextNotice,
  gamepadLogLine,
  gamepadSelectionNotice,
  isZeroDriveCommand,
  newDriveSendState,
  noteDriveSent,
  planDriveSend,
  pushGamepadLog,
  selectGamepad,
} = require("../../rosmaster-a1-web-remote-wendy/app/static/gamepad.js");

// gamepadContextNotice =======================================================
// The operator cannot open devtools. The panel has to say, in words, which of
// the three failures they are looking at.

test("context: an insecure origin is called out as the reason every pad is hidden", () => {
  const notice = gamepadContextNotice({
    hasGetGamepads: true,
    isSecureContext: false,
    origin: "http://car.local:8091",
  });
  assert.equal(notice.level, "error");
  assert.match(notice.headline, /Insecure context/);
  assert.match(notice.detail, /http:\/\/car\.local:8091/);
  assert.match(notice.detail, /hide/i);
  assert.match(notice.detail, /localhost/);
  assert.match(notice.detail, /HTTPS/);
});

test("context: a secure origin says so and does not cry wolf", () => {
  const notice = gamepadContextNotice({
    hasGetGamepads: true,
    isSecureContext: true,
    origin: "http://localhost:8091",
  });
  assert.equal(notice.level, "ok");
  assert.match(notice.headline, /Secure context/);
  assert.match(notice.detail, /http:\/\/localhost:8091/);
});

test("context: a browser with no getGamepads at all is its own failure, not an insecure one", () => {
  const notice = gamepadContextNotice({
    hasGetGamepads: false,
    isSecureContext: true,
    origin: "http://localhost:8091",
  });
  assert.equal(notice.level, "error");
  assert.match(notice.headline, /getGamepads/);
});

test("context: a missing getGamepads on an insecure origin still names the insecure origin", () => {
  // Chrome does exactly this: on a plain HTTP LAN origin the property is
  // absent. Reporting only "no Gamepad API" would send the operator hunting
  // for a browser bug instead of opening the page over localhost.
  const notice = gamepadContextNotice({
    hasGetGamepads: false,
    isSecureContext: false,
    origin: "http://car.local:8091",
  });
  assert.equal(notice.level, "error");
  assert.match(notice.detail, /localhost/);
});

// selectGamepad ==============================================================

function fakePad(overrides = {}) {
  return {
    index: 0,
    id: "Xbox Wireless Controller (STANDARD GAMEPAD)",
    mapping: "standard",
    connected: true,
    buttons: new Array(17).fill({ pressed: false, value: 0 }),
    axes: [0, 0, 0, 0],
    ...overrides,
  };
}

test("select: an empty pad list selects nothing and reports no candidates", () => {
  const selection = selectGamepad([], null);
  assert.equal(selection.index, null);
  assert.equal(selection.pad, null);
  assert.deepEqual(selection.considered, []);
});

test("select: the browser's null slots are reported as empty slots, not as pads", () => {
  const selection = selectGamepad([null, null, null, null], null);
  assert.equal(selection.index, null);
  assert.equal(selection.considered.length, 0, "empty slots are not pads and must not inflate the pad count");
});

test("select: a single connected pad is selected", () => {
  const selection = selectGamepad([fakePad()], null);
  assert.equal(selection.index, 0);
  assert.equal(selection.considered.length, 1);
  assert.equal(selection.considered[0].accepted, true);
});

test("select: a pad the browser reports as disconnected is rejected with a reason", () => {
  const selection = selectGamepad([fakePad({ connected: false })], null);
  assert.equal(selection.index, null);
  assert.equal(selection.pad, null);
  assert.equal(selection.considered.length, 1, "a rejected pad is still a pad the operator can see");
  assert.equal(selection.considered[0].accepted, false);
  assert.match(selection.considered[0].reason, /connected/i);
});

test("select: connected undefined is treated as connected, because most fakes and some browsers omit it", () => {
  const pad = fakePad();
  delete pad.connected;
  assert.equal(selectGamepad([pad], null).index, 0);
});

test("select: a named controller wins over an unnamed device in another slot", () => {
  const selection = selectGamepad(
    [fakePad({ index: 0, id: "Some HID thing" }), fakePad({ index: 1, id: "Xbox Wireless Controller" })],
    null,
  );
  assert.equal(selection.index, 1);
  const unchosen = selection.considered.find((entry) => entry.index === 0);
  assert.equal(unchosen.accepted, false);
  assert.match(unchosen.reason, /not selected/i);
});

test("select: an unnamed device is still selected when it is the only one", () => {
  // Rejecting it would leave the operator with a pad the page refuses to use
  // and no way to know why. Selecting it and saying so is the honest answer.
  const selection = selectGamepad([fakePad({ id: "Some HID thing" })], null);
  assert.equal(selection.index, 0);
});

test("select: the preferred index sticks even when a better named pad appears later", () => {
  const selection = selectGamepad(
    [fakePad({ index: 0, id: "Some HID thing" }), fakePad({ index: 1, id: "Xbox Wireless Controller" })],
    0,
  );
  assert.equal(selection.index, 0, "the pad the operator is already driving with must not be swapped out mid drive");
});

test("select: a preferred index that has gone away falls back to a live pad", () => {
  const selection = selectGamepad([fakePad({ index: 0 })], 3);
  assert.equal(selection.index, 0);
});

// gamepadSelectionNotice =====================================================
// A pad present but unmatched must not look like no pad at all.

test("notice: no pads at all reads as no pads at all", () => {
  const notice = gamepadSelectionNotice(selectGamepad([], null));
  assert.equal(notice.level, "warn");
  assert.match(notice.text, /No pad/i);
});

test("notice: a pad present but rejected reads differently from no pad, and names the reason", () => {
  const notice = gamepadSelectionNotice(selectGamepad([fakePad({ connected: false })], null));
  assert.equal(notice.level, "error");
  assert.doesNotMatch(notice.text, /^No pad/i);
  assert.match(notice.text, /1 pad/);
  assert.match(notice.text, /connected/i);
});

test("notice: a selected pad names its index and id", () => {
  const notice = gamepadSelectionNotice(selectGamepad([fakePad({ index: 2, id: "Xbox Wireless Controller" })], null));
  assert.equal(notice.level, "ok");
  assert.match(notice.text, /#2/);
  assert.match(notice.text, /Xbox Wireless Controller/);
});

test("notice: a selected pad with a non standard mapping is flagged, because the button map will be wrong", () => {
  const notice = gamepadSelectionNotice(selectGamepad([fakePad({ mapping: "" })], null));
  assert.equal(notice.level, "warn");
  assert.match(notice.text, /mapping/i);
});

// describeGamepad ============================================================

test("describe: one pad line carries index, id, mapping, connected, button and axis counts", () => {
  const line = describeGamepad(fakePad({ index: 1, id: "Xbox Wireless Controller", mapping: "standard" }));
  assert.match(line, /#1/);
  assert.match(line, /Xbox Wireless Controller/);
  assert.match(line, /standard/);
  assert.match(line, /17 buttons/);
  assert.match(line, /4 axes/);
  assert.match(line, /connected/);
});

test("describe: a pad with no mapping string says so rather than printing an empty gap", () => {
  assert.match(describeGamepad(fakePad({ mapping: "" })), /mapping none/);
});

// Axis and button readouts ===================================================

test("axes: every axis is shown with its index, at two decimals", () => {
  assert.equal(formatAxisValues([0, -0.5, 0.25, 1]), "A0 0.00  A1 -0.50  A2 0.25  A3 1.00");
});

test("axes: no axes at all says so instead of rendering an empty string", () => {
  assert.equal(formatAxisValues([]), "none");
});

test("buttons: pressed buttons are listed by name, index and value", () => {
  const buttons = new Array(17).fill(null).map(() => ({ pressed: false, value: 0 }));
  buttons[0] = { pressed: true, value: 1 };
  buttons[7] = { pressed: true, value: 0.42 };
  assert.equal(formatPressedButtons(buttons), "A[0] 1.00  RT[7] 0.42");
});

test("buttons: nothing pressed says none", () => {
  assert.equal(formatPressedButtons(new Array(17).fill({ pressed: false, value: 0 })), "none");
});

// The action log =============================================================

test("log: every dispatched action type gets a human label", () => {
  assert.equal(describeGamepadAction({ type: "hardStop" }), "hard stop");
  assert.equal(describeGamepadAction({ type: "startManual" }), "arm manual");
  assert.equal(describeGamepadAction({ type: "toggleAuto", enabled: true }), "auto nav on");
  assert.equal(describeGamepadAction({ type: "toggleAuto", enabled: false }), "auto nav off");
  assert.equal(describeGamepadAction({ type: "expandFeed", id: "hp60c_rgb" }), "expand hp60c_rgb");
  assert.equal(describeGamepadAction({ type: "expandFeed", id: null }), "camera gallery");
  assert.equal(describeGamepadAction({ type: "reconnectCamera" }), "camera reconnect");
  assert.equal(describeGamepadAction({ type: "nudgeManualSpeed", value: 0.4 }), "manual speed 0.40");
  assert.equal(describeGamepadAction({ type: "nudgeAutoSpeed", value: 1.05 }), "auto speed 1.05");
  assert.equal(describeGamepadAction({ type: "nudgeSteerScale", value: 80 }), "steering 80%");
});

test("log: an unknown action still produces a line rather than swallowing the event", () => {
  assert.equal(describeGamepadAction({ type: "somethingNew" }), "somethingNew");
});

test("log: newest first", () => {
  let log = pushGamepadLog([], { at: 1000, label: "first" });
  log = pushGamepadLog(log, { at: 2000, label: "second" });
  assert.deepEqual(log.map((entry) => entry.label), ["second", "first"]);
});

test("log: the cap holds at GAMEPAD_LOG_LIMIT and drops the oldest", () => {
  assert.equal(GAMEPAD_LOG_LIMIT, 20);
  let log = [];
  for (let i = 0; i < 25; i += 1) log = pushGamepadLog(log, { at: i, label: `entry ${i}` });
  assert.equal(log.length, GAMEPAD_LOG_LIMIT);
  assert.equal(log[0].label, "entry 24");
  assert.equal(log[GAMEPAD_LOG_LIMIT - 1].label, "entry 5");
});

test("log: pushing does not mutate the array it was handed", () => {
  const original = [{ at: 1, label: "kept" }];
  pushGamepadLog(original, { at: 2, label: "new" });
  assert.equal(original.length, 1);
});

test("log: a line carries a wall clock time and the label", () => {
  const line = gamepadLogLine({ at: Date.UTC(2026, 7, 3, 12, 34, 56, 780), label: "hard stop" });
  assert.match(line, /\d{2}:\d{2}:\d{2}\.\d{3}/);
  assert.match(line, /hard stop$/);
});

// The drive payload readout ==================================================

test("payload: the last drive payload is rendered field by field", () => {
  const text = drivePayloadText({ enabled: true, linear_x: 0.28, steering_y: -0.05, seq: 412, age_ms: 3 });
  assert.match(text, /enabled/);
  assert.match(text, /0\.28/);
  assert.match(text, /-0\.05/);
  assert.match(text, /412/);
});

test("payload: nothing sent yet says so", () => {
  assert.match(drivePayloadText(null), /none/i);
});

test("ago: a never seen timestamp reads never, a recent one reads in seconds", () => {
  assert.match(agoText(0, 5000), /never/i);
  assert.match(agoText(4000, 5000), /1\.0 s ago/);
});

// planDriveSend ==============================================================
// The old sender POSTed the same command eight times a second whatever it was.
// On a 300 ms link that is most of the budget spent resending nothing.

function command(overrides = {}) {
  return { enabled: true, linear_x: 0, steering_y: 0, ...overrides };
}

test("send: the first command always goes out", () => {
  const plan = planDriveSend(newDriveSendState(), command({ linear_x: 0.3 }), 1000);
  assert.equal(plan.send, true);
  assert.equal(plan.reason, "changed");
});

test("send: an unchanged nonzero command goes out again only once the heartbeat is due", () => {
  assert.equal(DRIVE_HEARTBEAT_MS, 200);
  const moving = command({ linear_x: 0.3 });
  let state = planDriveSend(newDriveSendState(), moving, 1000).next;

  const early = planDriveSend(state, moving, 1000 + DRIVE_HEARTBEAT_MS - 1);
  assert.equal(early.send, false, "resending inside the heartbeat window is the stutter we are removing");

  const due = planDriveSend(state, moving, 1000 + DRIVE_HEARTBEAT_MS);
  assert.equal(due.send, true);
  assert.equal(due.reason, "heartbeat");
});

test("send: the heartbeat stays comfortably inside the server's 0.5 s command watchdog", () => {
  assert.ok(DRIVE_HEARTBEAT_MS * 2 <= 500, "two heartbeats must fit inside CMD_TIMEOUT_S so one lost POST cannot time the car out");
});

test("send: a change goes out immediately, without waiting for the heartbeat", () => {
  const state = planDriveSend(newDriveSendState(), command({ linear_x: 0.3 }), 1000).next;
  const plan = planDriveSend(state, command({ linear_x: 0.4 }), 1010);
  assert.equal(plan.send, true);
  assert.equal(plan.reason, "changed");
});

test("send: releasing the stick sends the zero at once and then repeats it", () => {
  // A stop must not depend on one message being delivered.
  let state = planDriveSend(newDriveSendState(), command({ linear_x: 0.3 }), 1000).next;

  const released = planDriveSend(state, command({ linear_x: 0 }), 1010);
  assert.equal(released.send, true);
  assert.equal(released.reason, "changed");
  state = released.next;

  const sent = [];
  for (let tick = 1010; tick <= 3000; tick += 50) {
    const plan = planDriveSend(state, command({ linear_x: 0 }), tick);
    if (plan.send) sent.push({ at: tick, reason: plan.reason });
    state = plan.next;
  }
  assert.equal(sent.length, DRIVE_ZERO_REPEATS, "the zero is repeated a bounded number of times, then stops");
  assert.ok(sent.every((entry) => entry.reason === "zero repeat"));
  assert.equal(sent[0].at, 1010 + DRIVE_HEARTBEAT_MS);
});

test("send: an idle zero command that never moved sends nothing at all after the first", () => {
  const zero = command({ enabled: false });
  let state = planDriveSend(newDriveSendState(), zero, 1000).next;
  for (let tick = 1000; tick <= 10000; tick += 50) {
    const plan = planDriveSend(state, zero, tick);
    assert.equal(plan.send, false, `an idle page must not POST at ${tick}; the watchdog already holds the car stopped`);
    state = plan.next;
  }
});

test("send: enabled with all channels zero counts as zero, so a parked armed page stays quiet", () => {
  assert.equal(isZeroDriveCommand({ enabled: true, linear_x: 0, steering_y: 0 }), true);
  assert.equal(isZeroDriveCommand({ enabled: false, linear_x: 0.9, steering_y: 0.9 }), true);
  assert.equal(isZeroDriveCommand({ enabled: true, linear_x: 0.01, steering_y: 0 }), false);
  assert.equal(isZeroDriveCommand({ enabled: true, linear_x: 0, steering_y: -0.01 }), false);
});

test("send: commands are compared on the fields the car acts on, not on identity", () => {
  assert.equal(driveCommandsEqual({ enabled: true, linear_x: 0.3, steering_y: 0 }, { enabled: true, linear_x: 0.3, steering_y: 0 }), true);
  assert.equal(driveCommandsEqual({ enabled: true, linear_x: 0.3, steering_y: 0 }, { enabled: false, linear_x: 0.3, steering_y: 0 }), false);
  assert.equal(driveCommandsEqual(null, { enabled: true, linear_x: 0, steering_y: 0 }), false);
});

test("send: a fresh move after the zero repeats have run out starts the heartbeat again", () => {
  let state = newDriveSendState();
  state = planDriveSend(state, command({ linear_x: 0.3 }), 1000).next;
  state = planDriveSend(state, command({ linear_x: 0 }), 1010).next;
  for (let tick = 1010; tick <= 3000; tick += 50) state = planDriveSend(state, command({ linear_x: 0 }), tick).next;

  const moving = planDriveSend(state, command({ linear_x: 0.5 }), 3050);
  assert.equal(moving.send, true);
  assert.equal(moving.reason, "changed");
  const beat = planDriveSend(moving.next, command({ linear_x: 0.5 }), 3050 + DRIVE_HEARTBEAT_MS);
  assert.equal(beat.send, true);
  assert.equal(beat.reason, "heartbeat");
});

test("send: a command sent outside planDriveSend still resets the heartbeat clock", () => {
  // app.js posts immediately from event handlers such as the stick release.
  // If those sends did not register, the next tick would see a change that is
  // already on the wire and send it a second time.
  const moving = command({ linear_x: 0.3 });
  const state = noteDriveSent(newDriveSendState(), moving, 1000);
  const plan = planDriveSend(state, moving, 1100);
  assert.equal(plan.send, false);
  assert.equal(plan.reason, "recent");
});

test("send: a zero sent outside planDriveSend still schedules its repeats", () => {
  let state = noteDriveSent(newDriveSendState(), command({ linear_x: 0.3 }), 1000);
  state = noteDriveSent(state, command({ linear_x: 0 }), 1010);
  assert.equal(state.zeroRepeatsLeft, DRIVE_ZERO_REPEATS);
});
