// Tests for the page wiring: app.js running for real against a fake DOM, fake
// gamepads and a fake car. gamepad.test.mjs covers the pure decisions; this
// file covers whether the page actually carries them out.
//
// This file replaces a set of tests that read index.html as text and matched
// substrings. Those could not tell the difference between code that runs and
// code that merely appears in the file, which is how a reviewer's mutation run
// broke twelve safety behaviors without turning the suite red. Every test here
// drives a real call and asserts on a real effect: an HTTP request the page
// made, or a value it rendered.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { loadPage, pad } from "./harness.mjs";

const require = createRequire(import.meta.url);
const { computeGamepadStep, STOP_MAX_ATTEMPTS } = require("../../rosmaster-a1-web-remote-wendy/app/static/gamepad.js");

// A page with the first status poll already settled and its calls forgotten,
// so each test's assertions see only what that test caused.
async function freshPage() {
  const page = loadPage();
  await page.settle();
  page.clearCalls();
  return page;
}

function padFrame(page, buttons, axes = [0, 0, 0, 0]) {
  page.setPads([pad({ buttons, axes })]);
  page.run("pollGamepad()");
}

// Part 1: live behavior ======================================================

test("BUG I7: an auto speed nudge during an unconfirmed stop does not re-command Auto Nav", async () => {
  const page = await freshPage();
  page.run(`
    state.gamepadEnabled = true;
    state.auto = true;
    state.stopPending = { armed: false, auto: true };
    state.stopUnconfirmed = true;
  `);

  padFrame(page, { 15: { pressed: true } });
  await page.settle();

  assert.deepEqual(
    page.posts("/api/auto"),
    [],
    "a D-pad nudge must not re-POST /api/auto while a stop is unresolved; the operator thinks they moved a slider",
  );
  assert.equal(page.state.stopUnconfirmed, true, "the nudge must not clear the STOP UNCONFIRMED warning");
  assert.deepEqual(page.state.stopPending, { armed: false, auto: true }, "the nudge must not clear the pending stop");
  assert.match(page.el("modeValue").textContent, /UNCONFIRMED/);
  // The guard is narrow on purpose: the slider still moves, only the car is
  // left alone.
  assert.equal(page.el("autoSpeed").value, "1.05");
});

test("BUG I7: an auto speed nudge with no stop outstanding still pushes the new speed", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.auto = true;");

  padFrame(page, { 15: { pressed: true } });
  await page.settle();

  const posted = page.posts("/api/auto");
  assert.equal(posted.length, 1);
  assert.equal(posted[0].body.enabled, true);
  assert.equal(posted[0].body.speed, 1.05);
});

test("BUG I7: the on-screen auto sliders do not clear an unconfirmed stop either", async () => {
  const page = await freshPage();
  page.run(`
    state.auto = true;
    state.stopPending = { armed: false, auto: true };
    state.stopUnconfirmed = true;
  `);

  page.el("stopRange").value = "50";
  page.fireElement("stopRange", "input", {});
  page.el("autoSpeed").value = "1.5";
  page.fireElement("autoSpeed", "input", {});
  await page.settle();

  assert.deepEqual(page.posts("/api/auto"), []);
  assert.equal(page.state.stopUnconfirmed, true);
});

test("BUG I6: a stop the server has not confirmed goes out again on every status poll", async () => {
  const page = await freshPage();
  page.run(`
    state.stopPending = { armed: false, auto: true };
    state.stopUnconfirmed = true;
  `);
  // The server still reports the autonomous run the stop was issued to end.
  page.fake.status.auto.enabled = true;

  await page.run("refreshStatus()");
  await page.settle();
  assert.equal(page.posts("/api/stop").length, 1, "an outstanding stop must be re-issued, not just polled for");

  await page.run("refreshStatus()");
  await page.settle();
  assert.equal(page.posts("/api/stop").length, 2, "and again on the next poll, for as long as it stays unconfirmed");
  assert.equal(page.state.stopUnconfirmed, true);
});

test("BUG I6: a confirmed stop is not re-issued", async () => {
  const page = await freshPage();
  page.run("state.stopPending = { armed: false, auto: true }; state.stopUnconfirmed = true;");
  page.fake.status.auto.enabled = false;

  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(page.posts("/api/stop"), []);
  assert.equal(page.state.stopPending, null);
});

test("BUG I8: the disconnect event stops an autonomous run even with the Xbox toggle off", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = false; state.auto = true; state.gamepadIndex = 0;");

  page.fireWindow("gamepaddisconnected", { gamepad: { index: 0 } });
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1, "a pad that vanishes mid auto drive stops the car whatever the toggle says");
});

test("BUG I8: the poll loop applies exactly the same rule as the event", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = false; state.auto = true; state.gamepadIndex = 0;");

  page.setPads([]);
  page.run("pollGamepad()");
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1, "the poll path must not need the Xbox toggle to stop the car either");
  assert.deepEqual(
    page.posts("/api/drive"),
    [],
    "with the toggle off the touch UI owns the sticks, so the drive is left alone even though the stop went out",
  );
});

test("BUG I8: with the toggle on, a vanished pad also zeroes the outgoing drive", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.armed = true; state.gamepadIndex = 0; state.left = { x: 0.4, y: -0.8 };");

  page.setPads([]);
  page.run("pollGamepad()");
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1);
  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1, "the last stick values must not keep going out at 120 ms");
  assert.equal(drive[0].body.linear_x, 0);
  assert.equal(drive[0].body.steering_y, 0);
});

test("BUG M6: the Command readout shows what the server is publishing during Auto Nav", async () => {
  const page = await freshPage();
  page.run("state.auto = true;");
  page.fake.status.auto.enabled = true;
  page.fake.status.control.last_published = { linear_x: 0.42, steering_y: -0.03, angular_z: 0.0 };

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(
    page.el("commandValue").textContent,
    "0.42 / -0.03",
    "the local manual command is zero during an autonomous run and must not be painted over the real one",
  );
});

test("BUG M6: outside Auto Nav the Command readout still shows this page's own command", async () => {
  const page = await freshPage();
  page.run("state.armed = true; state.left = { x: 0, y: -1 };");
  page.el("speedInput").value = "0.40";
  page.fake.status.control.last_published = { linear_x: 9.99, steering_y: 9.99, angular_z: 0.0 };

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.el("commandValue").textContent, "0.40 / 0.00");
});

test("WIRING: the dashboard shows a direct controller and its active command source", async () => {
  const page = await freshPage();
  page.fake.status.direct_gamepad = {
    worker_ok: true,
    connected: true,
    compatible: true,
    owned: true,
    stable_id: "usb-xbox-event-joystick",
    name: "Xbox Wireless Controller",
    reason: "direct_gamepad_active",
  };
  page.fake.status.control.active_source = "direct_gamepad";

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.el("directGamepadValue").textContent, "Active Xbox Wireless Controller");
  assert.equal(page.el("sourceValue").textContent, "direct gamepad");
});

test("WIRING: ambiguous direct controllers are shown as fail-closed, not ready", async () => {
  const page = await freshPage();
  page.fake.status.direct_gamepad = {
    worker_ok: true,
    connected: false,
    compatible: true,
    compatible_devices: 2,
    owned: false,
    stable_id: "",
    name: "",
    reason: "multiple_compatible_gamepads",
  };

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.el("directGamepadValue").textContent, "multiple compatible gamepads");
});

// Part 2: the wiring the mutation run walked through =========================

test("WIRING: a hardStop frame never lets the held throttle reach the motors", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.armed = true;");

  // B and RT down on the same frame, sticks hard over.
  padFrame(page, { 1: { pressed: true }, 7: { pressed: true, value: 1 } }, [0.9, 0, 0.9, 0]);
  await page.settle();

  assert.deepEqual(page.state.left, { x: 0, y: 0 }, "the stale pre-stop drive must not survive the stop");
  assert.equal(page.posts("/api/stop").length, 1);
});

test("WIRING: the reducer's drive reaches the car on an ordinary frame", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.armed = true;");
  page.el("speedInput").value = "0.40";

  // Left stick half over, RT held down.
  padFrame(page, { 7: { pressed: true, value: 1 } }, [0.5, 0, 0, 0]);
  await page.run("sendDrive()");
  await page.settle();

  assert.ok(Math.abs(page.state.left.x - 0.43181818) < 1e-6, "the stick must reach the drive state");
  assert.equal(page.state.left.y, -1);
  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1);
  assert.equal(drive[0].body.enabled, true);
  assert.ok(Math.abs(drive[0].body.linear_x - 0.4) < 1e-9, "throttle times the manual speed setting");
  // Stick right must produce negative steering_y. The chassis steers opposite
  // to the stick, so scaledCommand negates; driving with it the wrong way round
  // was reported from the car as "steering is inverted". Pin the sign, not just
  // the magnitude, so a future refactor cannot quietly flip it back.
  assert.ok(drive[0].body.steering_y < 0, "stick right must steer the car right, which is negative steering_y");
  const expectedSteer = -0.43181818 * page.state.limits.maxSteeringY * (Number(page.el("steer").value) / 100);
  assert.ok(Math.abs(drive[0].body.steering_y - expectedSteer) < 1e-6,
    "steering must be scaled by the steering slider");
});

test("WIRING: applyGamepadAction carries out every action the reducer can emit", async () => {
  // Each entry is one action plus the effect on the car or the screen that
  // proves the dispatcher branch really ran, rather than merely existing.
  const dispatched = {
    hardStop: [
      { type: "hardStop" },
      (page) => assert.equal(page.posts("/api/stop").length, 1),
    ],
    startManual: [
      { type: "startManual" },
      (page) => assert.equal(page.posts("/api/start").length, 1),
    ],
    toggleAuto: [
      { type: "toggleAuto", enabled: true },
      (page) => assert.equal(page.posts("/api/auto").length, 1),
    ],
    expandFeed: [
      { type: "expandFeed", id: "hp60c_rgb" },
      (page) => assert.equal(page.state.expandedFeed, "hp60c_rgb"),
    ],
    reconnectCamera: [
      { type: "reconnectCamera" },
      (page) => assert.ok(page.timeouts.length >= 1, "every tile is asked to reopen, on its own offset"),
    ],
    nudgeManualSpeed: [
      { type: "nudgeManualSpeed", value: 0.4 },
      (page) => assert.equal(page.el("speed").value, "0.4"),
    ],
    nudgeAutoSpeed: [
      { type: "nudgeAutoSpeed", value: 1.05 },
      (page) => assert.equal(page.el("autoSpeed").value, "1.05"),
    ],
    nudgeSteerScale: [
      { type: "nudgeSteerScale", value: 80 },
      (page) => assert.equal(page.el("steer").value, "80"),
    ],
  };

  // The vocabulary comes from the reducer itself, so an action type added
  // there without a dispatcher branch fails here instead of silently doing
  // nothing on the car.
  const emitted = new Set();
  const uiState = {
    gamepadEnabled: true, auto: false, armed: false,
    manualSpeed: 0.35, autoSpeed: 1.0, steerScale: 70,
    feedIds: ["hp60c_depth", "hp60c_rgb"], expandedFeed: null,
  };
  for (const index of [1, 0, 3, 2, 8, 12, 15, 5]) {
    const buttons = [];
    for (let i = 0; i <= 16; i += 1) buttons[i] = { pressed: i === index, value: i === index ? 1 : 0 };
    for (const action of computeGamepadStep({ axes: [0, 0], buttons }, { buttons: [] }, uiState).actions) {
      emitted.add(action.type);
    }
  }
  assert.deepEqual([...emitted].sort(), Object.keys(dispatched).sort(), "the reducer's action vocabulary changed");

  for (const type of emitted) {
    const [action, check] = dispatched[type];
    const page = await freshPage();
    page.run("state.gamepadEnabled = true;");
    page.clearCalls();
    page.run(`applyGamepadAction(${JSON.stringify(action)})`);
    await page.settle();
    check(page);
  }
});

test("WIRING: the disconnect event dispatches the stop it decides on", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.armed = true; state.gamepadIndex = 0;");

  page.fireWindow("gamepaddisconnected", { gamepad: { index: 0 } });
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1);
});

test("WIRING: the disconnect event reads the live mode, not a fixed one", async () => {
  const page = await freshPage();
  page.run("state.gamepadIndex = 0;");

  page.fireWindow("gamepaddisconnected", { gamepad: { index: 0 } });
  await page.settle();
  assert.deepEqual(page.posts("/api/stop"), [], "an idle car has nothing to stop");

  page.run("state.auto = true; state.gamepadIndex = 0;");
  page.fireWindow("gamepaddisconnected", { gamepad: { index: 0 } });
  await page.settle();
  assert.equal(page.posts("/api/stop").length, 1, "a car in auto does");
});

test("WIRING: the poll loop only treats a pad as lost once it had one", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.auto = true; state.gamepadIndex = null;");

  page.setPads([]);
  page.run("pollGamepad()");
  page.run("pollGamepad()");
  await page.settle();
  assert.deepEqual(page.posts("/api/stop"), [], "a machine that never had a pad must not stop the car 60 times a second");

  page.setPads([pad({})]);
  page.run("pollGamepad()");
  page.setPads([]);
  page.run("pollGamepad()");
  await page.settle();
  assert.equal(page.posts("/api/stop").length, 1, "losing a pad that was there must stop the car");
});

test("WIRING: a status poll is the automatic way out of an unconfirmed stop", async () => {
  const page = await freshPage();
  page.run(`
    state.stopPending = { armed: false, auto: true };
    state.stopUnconfirmed = true;
    updateReadouts();
  `);
  assert.match(page.el("modeValue").textContent, /UNCONFIRMED/);

  page.fake.status.auto.enabled = false;
  page.fake.status.control.command = { enabled: false };
  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.state.stopPending, null);
  assert.equal(page.state.stopUnconfirmed, false);
  assert.equal(page.el("modeValue").textContent, "Disarmed");
});

test("WIRING: a status poll that still reports the mode running leaves the stop unresolved", async () => {
  const page = await freshPage();
  page.run("state.stopPending = { armed: false, auto: true }; state.stopUnconfirmed = true;");
  page.fake.status.auto.enabled = true;

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.state.stopUnconfirmed, true);
  assert.match(page.el("statusText").textContent, /STOP UNCONFIRMED/);
});

test("WIRING: hardStop records what the stop has to undo, so it reads as Stopping and can be confirmed later", async () => {
  const page = await freshPage();
  page.run("state.auto = true;");

  const inFlight = page.run("hardStop()");
  // hardStop runs synchronously up to its first await, so the page is already
  // rendering the in-flight state here.
  assert.equal(
    page.el("modeValue").textContent,
    "Stopping",
    "a stop in flight must not read as a calm Disarmed",
  );
  assert.deepEqual(page.state.stopPending, { armed: false, auto: true });
  await inFlight;
  await page.settle();
  assert.equal(page.state.stopPending, null, "a POST that resolved confirms the stop");
});

test("WIRING: a stop whose POSTs all fail stays recorded and visibly unresolved", async () => {
  const page = await freshPage();
  page.run("state.auto = true;");
  page.fake.failing.add("/api/stop");

  await page.run("hardStop()");
  await page.settle();

  assert.equal(page.posts("/api/stop").length, STOP_MAX_ATTEMPTS, "the retry budget must actually be spent");
  assert.equal(page.state.stopUnconfirmed, true);
  assert.deepEqual(
    page.state.stopPending,
    { armed: false, auto: true },
    "without the record of what was running, no later status poll could ever confirm this stop",
  );
  assert.match(page.el("modeValue").textContent, /UNCONFIRMED/);
  assert.ok(page.el("modeValue").classList.contains("unresolved"));
});

test("WIRING: no mode change switches the pad off, so the stop button is always there", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; els.gamepad.checked = true;");

  for (const call of ["setAuto(true)", "setAuto(false)", "hardStop()", "startManual()"]) {
    await page.run(call);
    await page.settle();
    assert.equal(page.state.gamepadEnabled, true, `${call} switched the pad off`);
    assert.equal(page.el("gamepad").checked, true, `${call} unticked the Xbox checkbox`);
  }
});

test("WIRING: ticking the Xbox toggle does not disengage Auto Nav", async () => {
  const page = await freshPage();
  page.run("state.auto = true;");

  page.el("gamepad").checked = true;
  page.fireElement("gamepad", "change", {});
  await page.settle();

  assert.equal(page.state.auto, true, "reaching for the B button mid auto drive must not end the run");
  assert.deepEqual(page.posts("/api/auto"), []);
});

test("WIRING: hard stop from the on-screen button and from the pad take the same path", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true; state.armed = true;");

  page.fireElement("stop", "click", {});
  await page.settle();
  assert.equal(page.posts("/api/stop").length, 1);
  assert.equal(page.state.armed, false);
  assert.equal(page.el("arm").checked, false);
});

test("WIRING: the mode readout comes from controlModeText for every state", async () => {
  const page = await freshPage();
  const cases = [
    ["state.armed = false; state.auto = false;", "Disarmed"],
    ["state.armed = true; state.auto = false;", "Manual Armed"],
    ["state.armed = false; state.auto = true;", "Auto Nav"],
    ["state.stopPending = { auto: true };", "Stopping"],
    ["state.stopUnconfirmed = true;", "STOP UNCONFIRMED"],
  ];
  for (const [setup, expected] of cases) {
    page.run(`${setup} updateReadouts();`);
    assert.equal(page.el("modeValue").textContent, expected);
  }
});

// Part 3: the Controller panel ===============================================
// The panel exists because the operator has a working Xbox pad, a page that
// reports nothing, and no devtools console. Every assertion below is a thing
// they would otherwise have to guess at.

test("PANEL: an insecure origin is named on screen as the reason every pad is hidden", async () => {
  const page = loadPage();
  page.setSecureContext(false, "http://car.local:8091");
  await page.settle();

  page.run("renderControllerPanel();");
  const text = page.el("padContext").textContent;
  assert.match(text, /Insecure context/);
  assert.match(text, /car\.local:8091/);
  assert.match(text, /localhost/);
  assert.equal(page.el("padContext").classList.contains("bad"), true);
});

test("PANEL: a browser with no getGamepads at all says so", async () => {
  const page = loadPage();
  page.removeGetGamepads();
  await page.settle();

  page.run("renderControllerPanel();");
  assert.match(page.el("padContext").textContent, /getGamepads/);
});

test("PANEL: a secure origin reads as secure and does not raise a false alarm", async () => {
  const page = await freshPage();
  page.run("renderControllerPanel();");
  assert.match(page.el("padContext").textContent, /Secure context/);
  assert.equal(page.el("padContext").classList.contains("bad"), false);
});

test("PANEL: pad detection populates with the Xbox toggle switched off", async () => {
  // The operator not realising the pad was never detected is a failure in its
  // own right, so the panel cannot wait for the toggle.
  const page = await freshPage();
  // The pad is enabled by default now, so switch it off explicitly: this test
  // is about the panel reporting detection regardless of the toggle, not about
  // whatever the toggle happens to default to.
  page.run("state.gamepadEnabled = false; els.gamepad.checked = false;");
  assert.equal(page.state.gamepadEnabled, false);

  page.setPads([pad({ index: 0, id: "Xbox Wireless Controller (STANDARD GAMEPAD)" })]);
  page.run("pollGamepad()");
  await page.settle();

  assert.match(page.el("padCount").textContent, /1/);
  assert.match(page.el("padList").textContent, /Xbox Wireless Controller/);
  assert.match(page.el("padList").textContent, /17 buttons/);
  assert.match(page.el("padSelection").textContent, /Using pad #0/);
});

test("PANEL: a pad present but rejected looks different from no pad at all", async () => {
  const page = await freshPage();

  page.setPads([]);
  page.run("pollGamepad()");
  const empty = page.el("padSelection").textContent;
  assert.match(empty, /No pads/i);

  page.setPads([pad({ index: 0, connected: false })]);
  page.run("pollGamepad()");
  const rejected = page.el("padSelection").textContent;
  assert.notEqual(rejected, empty);
  assert.match(rejected, /none usable/);
  assert.match(rejected, /connected false/);
  assert.match(page.el("padList").textContent, /disconnected/);
});

test("PANEL: live axes and pressed buttons follow the sticks", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true;");

  padFrame(page, { 0: { pressed: true } }, [0.5, -0.25, 0, 0]);
  await page.settle();

  assert.match(page.el("padAxes").textContent, /A0 0\.50/);
  assert.match(page.el("padAxes").textContent, /A1 -0\.25/);
  assert.match(page.el("padButtons").textContent, /A\[0\]/);
});

test("PANEL: dispatched controller actions land in the rolling log, most recent first", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true;");

  // A rising edge on RB nudges the steering scale, then B hard stops.
  padFrame(page, { 5: { pressed: true } });
  await page.settle();
  padFrame(page, { 1: { pressed: true } });
  await page.settle();

  const lines = page.el("padLog").textContent.split("\n").filter(Boolean);
  assert.match(lines[0], /hard stop/);
  assert.match(lines[1], /steering 80%/);
  assert.match(lines[0], /^\d{2}:\d{2}:\d{2}\.\d{3}/, "every log line carries a timestamp");
});

test("PANEL: the log is capped so a long drive cannot grow it without bound", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true;");
  for (let i = 0; i < 30; i += 1) {
    padFrame(page, { 8: { pressed: true } });
    padFrame(page, {});
  }
  await page.settle();
  assert.equal(page.state.gamepadLog.length, 20);
});

test("PANEL: the last drive payload and the last good response separate a dead network from a dead pad", async () => {
  const page = await freshPage();
  page.run("state.armed = true;");
  page.el("speedInput").value = "0.40";
  page.run("state.left = { x: 0, y: -1 };");

  await page.run("sendDrive()");
  await page.settle();
  page.run("renderControllerPanel();");

  assert.match(page.el("padDrive").textContent, /linear_x 0\.40/);
  assert.match(page.el("padDrive").textContent, /seq \d+/);
  assert.doesNotMatch(page.el("padResponse").textContent, /never/i);

  // With the car unreachable the payload line still shows what we tried to
  // send, and the response line stops advancing.
  page.fake.failing.add("/api/drive");
  const before = page.el("padResponse").textContent;
  page.run("state.left = { x: 0, y: -0.5 };");
  await page.run("sendDrive()");
  await page.settle();
  page.run("renderControllerPanel();");
  assert.match(page.el("padDrive").textContent, /linear_x 0\.20/);
  assert.equal(page.el("padResponse").textContent, before, "a failed POST must not look like a fresh response");
});

// Part 5: the camera gallery =================================================
//
// The tile list is not written down in the page. It comes from the feed
// registry in /api/status, so a feed this hardware does not have produces no
// tile and a feed added to the car later appears with no change here.

function feedsOf(page) {
  return page.tileIds();
}

test("GALLERY: one tile per feed the car reports, in the order it reports them", async () => {
  const page = await freshPage();
  assert.deepEqual(feedsOf(page), ["hp60c_depth", "hp60c_rgb"]);
  assert.equal(page.tile("hp60c_depth").label.textContent, "Depth");
  // One stream at a time: the first feed auto-expands and streams; the other
  // tile exists but rents no socket (see tests/web/feeds.test.mjs for why).
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg/);
  assert.equal(page.tile("hp60c_rgb").img.src, "");
});

test("GALLERY: a feed the car does not report gets no tile at all", async () => {
  const page = loadPage();
  page.fake.status.cameras = [
    { id: "hp60c_depth", label: "Depth", path: "/stream_hp60c_depth.mjpg", ok: true, age_s: 0.06 },
  ];
  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(
    feedsOf(page),
    ["hp60c_depth"],
    "the hardware has no infrared or stereo image, so shipping tiles for them would be two permanently black rectangles",
  );
});

test("GALLERY: a feed added to the registry later appears with no change to the page", async () => {
  const page = await freshPage();
  page.fake.status.cameras = [
    ...page.fake.status.cameras,
    { id: "hp60c_ir", label: "Infrared", path: "/stream_hp60c_ir.mjpg", ok: true, age_s: 0.1 },
  ];

  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(feedsOf(page), ["hp60c_depth", "hp60c_rgb", "hp60c_ir"]);
  assert.equal(page.tile("hp60c_ir").img.src, "", "a new tile appears disconnected; only the expanded feed streams");
});

test("GALLERY: a stale feed is marked on screen rather than left as a silently frozen frame", async () => {
  const page = await freshPage();
  page.fake.status.cameras[1] = { ...page.fake.status.cameras[1], ok: false, age_s: 5.2 };

  await page.run("refreshStatus()");
  await page.settle();

  const tile = page.tile("hp60c_rgb");
  assert.match(tile.state.textContent, /stale/i);
  assert.match(tile.state.textContent, /5\.2/);
  assert.ok(tile.root.classList.contains("stale"), "the tile itself has to look wrong, not just read wrong");
  assert.ok(page.tile("hp60c_depth").root.classList.contains("live"));
});

test("GALLERY: a stale feed keeps its tile, because the RGB stream recovers on its own", async () => {
  const page = await freshPage();
  page.fake.status.cameras[1] = { ...page.fake.status.cameras[1], ok: false, age_s: 5.2 };
  await page.run("refreshStatus()");
  await page.settle();
  assert.deepEqual(feedsOf(page), ["hp60c_depth", "hp60c_rgb"], "stale is not absent");

  page.fake.status.cameras[1] = { ...page.fake.status.cameras[1], ok: true, age_s: 0.08 };
  await page.run("refreshStatus()");
  await page.settle();
  assert.match(page.tile("hp60c_rgb").state.textContent, /live/i);
});

test("GALLERY: a feed that has never produced a frame reads as waiting, not as stale", async () => {
  const page = await freshPage();
  page.fake.status.cameras[1] = { ...page.fake.status.cameras[1], ok: false, age_s: null };

  await page.run("refreshStatus()");
  await page.settle();

  const tile = page.tile("hp60c_rgb");
  assert.doesNotMatch(tile.state.textContent, /stale/i);
  assert.ok(tile.root.classList.contains("waiting"));
});

test("GALLERY: a feed that drops out of the registry loses its tile", async () => {
  const page = await freshPage();
  page.fake.status.cameras = page.fake.status.cameras.slice(0, 1);

  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(feedsOf(page), ["hp60c_depth"]);
  assert.equal(page.el("cameraGallery").children.length, 1);
});

test("GALLERY: an ordinary status poll does not restart the streams", async () => {
  // Every tile is an open MJPEG connection. Re-pointing them once every 750 ms
  // poll would reopen every stream on the car eighty times a minute, which is
  // the traffic that starved the server of request threads.
  const page = await freshPage();
  const before = ["hp60c_depth", "hp60c_rgb"].map((id) => page.tile(id).img.src);

  await page.run("refreshStatus()");
  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(["hp60c_depth", "hp60c_rgb"].map((id) => page.tile(id).img.src), before);
});

test("GALLERY: clicking a tile expands it and clicking again returns to the gallery", async () => {
  const page = await freshPage();

  page.tile("hp60c_rgb").root.dispatch("click", {});
  assert.equal(page.state.expandedFeed, "hp60c_rgb");
  assert.ok(page.el("cameraGallery").classList.contains("expanded"));
  assert.ok(page.tile("hp60c_rgb").root.classList.contains("is-expanded"));
  assert.equal(page.el("galleryBack").classList.contains("hidden"), false, "there must be a visible way back");

  page.tile("hp60c_rgb").root.dispatch("click", {});
  assert.equal(page.state.expandedFeed, null);
  assert.equal(page.el("cameraGallery").classList.contains("expanded"), false);
});

test("GALLERY: the back control returns to the gallery from an expanded feed", async () => {
  const page = await freshPage();
  page.run("setExpandedFeed('hp60c_depth');");

  page.fireElement("galleryBack", "click", {});
  assert.equal(page.state.expandedFeed, null);
  assert.ok(page.el("galleryBack").classList.contains("hidden"));
});

test("GALLERY: expanding does not reopen the stream it expands", async () => {
  const page = await freshPage();
  const before = page.tile("hp60c_depth").img.src;
  page.run("setExpandedFeed('hp60c_depth');");
  assert.equal(page.tile("hp60c_depth").img.src, before, "an expand is a layout change, not a reconnect");
});

test("GALLERY: X cycles which feed is expanded and then back to the gallery", async () => {
  const page = await freshPage();
  page.run("state.gamepadEnabled = true;");

  // The page now opens with the first feed already expanded, so the cycle
  // starts one step in: rgb, then the gallery, then depth again.
  padFrame(page, { 2: { pressed: true } });
  padFrame(page, {});
  assert.equal(page.state.expandedFeed, "hp60c_rgb");

  padFrame(page, { 2: { pressed: true } });
  padFrame(page, {});
  assert.equal(page.state.expandedFeed, null, "the pad needs a way out of an expanded feed, not only a way in");

  padFrame(page, { 2: { pressed: true } });
  await page.settle();
  assert.equal(page.state.expandedFeed, "hp60c_depth");
});

test("GALLERY: an expanded feed that vanishes from the registry falls back to the gallery", async () => {
  const page = await freshPage();
  page.run("setExpandedFeed('hp60c_rgb');");
  page.fake.status.cameras = page.fake.status.cameras.slice(0, 1);

  await page.run("refreshStatus()");
  await page.settle();

  assert.equal(page.state.expandedFeed, null, "an expanded view of a feed that is gone is a blank screen");
  assert.equal(page.el("cameraGallery").classList.contains("expanded"), false);
});

test("GALLERY: two tiles failing together do not retry together", async () => {
  const page = await freshPage();
  page.clearCalls();

  page.tile("hp60c_depth").img.dispatch("error", {});
  page.tile("hp60c_rgb").img.dispatch("error", {});
  await page.settle();

  // The expanded tile's poll loop also arms a long stall bound per frame;
  // only the sub-stall delays are the retries this test reasons about.
  const delays = page.timeouts.filter((ms) => ms < 3000);
  assert.equal(delays.length, 2, "each tile schedules its own retry");
  assert.notEqual(delays[0], delays[1], "several MJPEG reconnects on the same instant is the burst that starved the server");
  assert.ok(Math.min(...delays) >= 1000, "the one second wait before a retry is kept");
});

test("GALLERY: a tile that errors twice before its retry lands only schedules one", async () => {
  const page = await freshPage();
  page.clearCalls();

  page.tile("hp60c_depth").img.dispatch("error", {});
  page.tile("hp60c_depth").img.dispatch("error", {});

  assert.equal(page.timeouts.length, 1, "a stream that errors in a burst must not queue a reconnect per error");
  await page.settle();
});

test("GALLERY: an errored tile really does reopen its stream", async () => {
  const page = await freshPage();
  const before = page.tile("hp60c_depth").img.src;

  page.tile("hp60c_depth").img.dispatch("error", {});
  await page.settle();

  assert.notEqual(page.tile("hp60c_depth").img.src, before);
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg/);
});

test("GALLERY: the periodic refresh touches only the expanded feed's stream", async () => {
  const page = await freshPage();
  const before = ["hp60c_depth", "hp60c_rgb"].map((id) => page.tile(id).img.src);

  page.run("reconnectNextFeed()");
  const afterFirst = ["hp60c_depth", "hp60c_rgb"].map((id) => page.tile(id).img.src);
  assert.notEqual(afterFirst[0], before[0], "the expanded feed's stream is refreshed on its turn");
  assert.equal(afterFirst[1], "", "a disconnected tile stays disconnected");

  page.run("reconnectNextFeed()");
  const afterSecond = ["hp60c_depth", "hp60c_rgb"].map((id) => page.tile(id).img.src);
  assert.equal(afterSecond[0], afterFirst[0], "the non-expanded tile's turn is a no-op, not a reopen");
  assert.equal(afterSecond[1], "");
});

test("GALLERY: the View button reopens the expanded stream and leaves the rest disconnected", async () => {
  const page = await freshPage();
  page.clearCalls();
  const before = page.tile("hp60c_depth").img.src;

  page.run("applyGamepadAction({ type: 'reconnectCamera' })");
  await page.settle();

  // Sub-stall delays only: the reopened poll loop arms its own long stall
  // bound, which is not one of the staggered retries being counted here.
  const delays = page.timeouts.filter((ms) => ms < 3000);
  assert.equal(delays.length, 2, "each tile still schedules its own staggered retry");
  assert.notEqual(delays[0], delays[1]);
  assert.notEqual(page.tile("hp60c_depth").img.src, before, "the expanded stream reopens");
  assert.equal(page.tile("hp60c_rgb").img.src, "", "a tile that is not expanded must not gain a socket from a reconnect");
});

test("GALLERY: a car that reports no cameras renders an empty gallery rather than breaking the page", async () => {
  const page = loadPage();
  page.fake.status.cameras = [];
  await page.run("refreshStatus()");
  await page.settle();

  assert.deepEqual(feedsOf(page), []);
  assert.equal(page.el("cameraGallery").children.length, 0);
  // The rest of the page still renders, which is the point: the cameras are
  // not what the operator drives by.
  assert.equal(page.el("modeValue").textContent, "Disarmed");
});

// Part 4: the command transport ==============================================

test("TRANSPORT: an unchanged still command is not resent", async () => {
  // setInterval(sendDrive, 120) posted eight times a second whatever the
  // command was. The server's watchdog already holds a car stopped when
  // nothing arrives, so a page that is asking for nothing has nothing to say.
  const page = await freshPage();
  page.run("sendDriveIfNeeded();");
  await page.settle();
  page.clearCalls();

  for (let i = 0; i < 50; i += 1) page.run("sendDriveIfNeeded();");
  await page.settle();
  assert.deepEqual(page.posts("/api/drive"), []);
});

test("TRANSPORT: a change goes out at once and an unchanged throttle waits for the heartbeat", async () => {
  const page = await freshPage();
  page.run("state.armed = true; state.left = { x: 0, y: -1 };");
  page.run("sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1);

  // Same command again on the very next tick: nothing new to say.
  page.run("sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1);

  // A new stick position is a change and does not wait.
  page.run("state.left = { x: 0, y: -0.5 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 2);
});

test("TRANSPORT: releasing the stick sends the zero and keeps repeating it", async () => {
  const page = await freshPage();
  page.run("state.armed = true; state.left = { x: 0, y: -1 }; sendDriveIfNeeded();");
  await page.settle();
  page.clearCalls();

  page.run("state.left = { x: 0, y: 0 }; sendDriveIfNeeded();");
  await page.settle();
  const zeros = page.posts("/api/drive");
  assert.equal(zeros.length, 1, "the release must not wait for a heartbeat");
  assert.equal(zeros[0].body.linear_x, 0);
  assert.ok(page.state.driveSend.zeroRepeatsLeft > 0, "a stop must not depend on one message being delivered");
});

test("TRANSPORT: every drive payload carries the fields the car checks for staleness", async () => {
  const page = await freshPage();
  page.run("state.armed = true; state.left = { x: 0, y: -1 };");
  await page.run("sendDrive()");
  page.run("state.left = { x: 0, y: -0.5 };");
  await page.run("sendDrive()");
  await page.settle();

  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 2);
  assert.ok(drive[0].body.client_id, "a client id, so a page reload is not read as a reordering");
  assert.equal(drive[0].body.client_id, drive[1].body.client_id);
  assert.equal(drive[1].body.seq, drive[0].body.seq + 1, "the sequence must advance so the car can drop what arrives late");
  assert.ok(Number.isFinite(drive[0].body.age_ms));
  assert.ok(drive[0].body.age_ms >= 0);
  assert.equal(drive[0].body.angular_z, undefined, "angular_z moves nothing on this chassis and is no longer sent");
});

test("TRANSPORT: a throttle already in flight does not delay the stop that follows it", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/drive");
  page.run("state.armed = true; state.left = { x: 0, y: -1 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1, "the throttle is on the wire and not yet answered");

  page.run("state.left = { x: 0, y: 0 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 2, "the zero must not queue behind it");
  assert.equal(page.posts("/api/drive")[1].body.linear_x, 0);
  page.releaseHeld();
  await page.settle();
});

test("TRANSPORT: a second throttle does queue behind the first, rather than piling up on the link", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/drive");
  page.run("state.armed = true; state.left = { x: 0, y: -1 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1);

  page.run("state.left = { x: 0, y: -0.5 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1, "a request is still outstanding, so this one waits");

  // Once the link answers, the newest value goes out, not the one that waited.
  page.fake.held.delete("/api/drive");
  page.run("state.left = { x: 0, y: -0.25 };");
  page.releaseHeld();
  await page.settle();
  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 2);
  // Derived from the speed control rather than hardcoded, so changing the
  // default manual speed does not make this test lie about queueing.
  const manualSpeed = Number(page.el("speedInput").value);
  assert.ok(Math.abs(drive[1].body.linear_x - 0.25 * manualSpeed) < 1e-9,
    "the queued send must carry the newest stick position, not a stale one");
});

// Part 3: bounded fetches ====================================================
//
// The server wedged on the car once and every fetch on this page waited on it
// forever, because nothing had a timeout. A hang never rejects, so none of the
// existing catch/finally paths ever ran: driveInFlight above stayed stuck true
// and every gamepad command after the one hung POST was silently dropped, the
// status poll stacked a new connection every 750 ms tick with nothing to stop
// it, and the only failure indicator (setConnection(false, ...)) never fired
// because it only runs on a rejection. These tests hold a fetch open the way
// the wedged server did and prove the page recovers instead of bricking.

test("RESILIENCE: a hung drive POST times out, so the guard it left behind does not block the next command", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/drive");
  page.run("state.armed = true; state.left = { x: 0, y: -1 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1, "the throttle is on the wire and not yet answered");

  // Simulate FETCH_TIMEOUT_MS actually elapsing on a server that never answers.
  page.expireFetchTimeouts();
  await page.settle();

  // driveInFlight must be false again: a second non-zero command, which would
  // otherwise queue behind an outstanding request forever, has to go out
  // immediately rather than being silently dropped like every command after
  // the one hung POST in the field.
  page.run("state.left = { x: 0, y: -0.5 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 2,
    "a timed-out drive must self-heal, not drop every command that follows it");

  page.expireFetchTimeouts();
  await page.settle();
});

test("RESILIENCE: a throttle queued behind a drive that then times out still goes out, not lost with it", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/drive");
  page.run("state.armed = true; state.left = { x: 0, y: -1 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1);

  // A second, different throttle arrives while the first is still in flight,
  // so it queues rather than piling a second connection on top of the first.
  page.run("state.left = { x: 0, y: -0.5 }; sendDriveIfNeeded();");
  await page.settle();
  assert.equal(page.posts("/api/drive").length, 1, "still queued, the first request has not answered");

  // The first request times out instead of ever answering. The finally in
  // sendDrive that clears driveQueued and re-fires it must run all the same,
  // the same way it already does when a held request is released normally
  // (see the TRANSPORT test above); a timeout must not be a second way for a
  // queued command to be dropped on the floor.
  page.expireFetchTimeouts();
  await page.settle();

  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 2, "the queued throttle must still go out once the guard clears");
  const manualSpeed = Number(page.el("speedInput").value);
  assert.ok(Math.abs(drive[1].body.linear_x - 0.5 * manualSpeed) < 1e-9,
    "and it must carry the value that was queued, not a stale one");

  page.expireFetchTimeouts();
  await page.settle();
});

test("RESILIENCE: a hung status fetch is not re-issued by the next poll tick", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/status");

  page.run("refreshStatus()");
  await page.settle();
  const gets = () => page.calls.filter((call) => call.path === "/api/status" && call.method === "GET");
  assert.equal(gets().length, 1, "the first poll's fetch is on the wire");

  // The 750 ms setInterval tick calls refreshStatus() again; the in-flight
  // guard must find the previous fetch still outstanding and skip it rather
  // than stacking a second connection on top of it, the way the wedged
  // server's poll did in the field.
  page.run("refreshStatus()");
  await page.settle();
  assert.equal(gets().length, 1, "a tick while the previous poll is still outstanding must not stack a request");

  page.expireFetchTimeouts();
  await page.settle();
});

test("RESILIENCE: a hung /api/gamepad post is not re-issued while one is outstanding", async () => {
  const page = await freshPage();
  page.fake.held.add("/api/gamepad");

  page.run("state.gamepadLastReportAt = -1000;");
  padFrame(page, { 0: { pressed: true } });
  await page.settle();
  assert.equal(page.posts("/api/gamepad").length, 1, "the first report is on the wire");

  // Force past the 100 ms throttle again so it is the in-flight guard under
  // test here, not the throttle, that stops the second post.
  page.run("state.gamepadLastReportAt = -1000;");
  padFrame(page, { 0: { pressed: true } });
  await page.settle();
  assert.equal(page.posts("/api/gamepad").length, 1, "a post already outstanding must not be stacked");

  page.expireFetchTimeouts();
  await page.settle();
});

test("RESILIENCE: a status fetch that times out flips the connection indicator offline, the path a rejection already took", async () => {
  const page = await freshPage();
  assert.equal(page.state.lastStatusOk, true, "the initial poll in freshPage() already succeeded");
  page.fake.held.add("/api/status");

  page.run("refreshStatus()");
  await page.settle();
  assert.equal(page.state.lastStatusOk, true, "still waiting on the hung fetch, nothing has failed yet");

  // A hang never rejects on its own, which is exactly why the operator saw no
  // failure indicator in the field: setConnection(false, ...) only runs from
  // a catch. The timeout is what turns the hang into a rejection.
  page.expireFetchTimeouts();
  await page.settle();

  assert.equal(page.el("statusDot").className, "dot bad");
  assert.equal(page.el("statusText").textContent, "Remote offline");
  assert.equal(page.state.lastStatusOk, false);
});
