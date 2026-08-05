// Tests for the WebHID fallback input source.
//
// The operator's Xbox Series pad is invisible to navigator.getGamepads on their
// machine, measured in three browsers and on a bare page with none of our code
// in it. WebHID is a different permission path to the same hardware, so the
// page can open the device directly. What arrives there is not a Gamepad
// snapshot, it is a raw input report, and decoding it is the one piece of this
// work with no way to check itself: the byte layout is documentation plus
// inference until the operator pushes a stick.
//
// So the decoder is pure and every claim about the layout is a test with the
// bytes written out in it. When the operator finds a byte we placed wrongly,
// the fix is one line here and one line in gamepad.js, and the rest of the
// suite says whether it broke anything.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { hidDevice, loadPage, pad } from "./harness.mjs";

const require = createRequire(import.meta.url);
const {
  HID_SILENCE_MS,
  HID_XBOX_VENDOR_ID,
  XBOX_HID_MIN_LENGTH,
  chooseInputSource,
  computeGamepadStep,
  decodeXboxHidReport,
  describeHidDevice,
  emptyPadSnapshot,
  formatHidReportBytes,
  hidPadSnapshot,
  hidReportBytes,
  hidStatusNotice,
  selectHidDevice,
} = require("../../rosmaster-a1-web-remote-wendy/app/static/gamepad.js");

// Helpers ==================================================================

// A whole Xbox Series input report, report id already stripped the way WebHID
// hands it over in event.data. Everything defaults to the resting position, so
// each test writes down only the field it is about.
function report({
  lx = 32768,
  ly = 32768,
  rx = 32768,
  ry = 32768,
  lt = 0,
  rt = 0,
  hat = 0,
  face = 0,
  system = 0,
  share = 0,
  length = 16,
} = {}) {
  const le = (value) => [value & 0xff, (value >> 8) & 0xff];
  const bytes = [
    ...le(lx), ...le(ly), ...le(rx), ...le(ry),
    ...le(lt), ...le(rt),
    hat,
    face,
    system,
    share,
  ];
  return bytes.slice(0, length);
}

function near(actual, expected, tolerance = 0.002) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function pressedIndexes(buttons) {
  return (buttons || [])
    .map((button, index) => (button && button.pressed ? index : null))
    .filter((index) => index !== null);
}

// The reducer's own uiState, armed and manual, so a decoded report can be shown
// to drive the same values a Gamepad API frame does.
function armedUi() {
  return {
    gamepadEnabled: true,
    auto: false,
    armed: true,
    manualSpeed: 1,
    autoSpeed: 1,
    steerScale: 70,
    feedIds: ["hp60c_rgb"],
    expandedFeed: null,
  };
}

// Part 1: reading the bytes ==================================================

test("HID: the vendor filter is Microsoft, which is what hidutil reports for this pad", () => {
  assert.equal(HID_XBOX_VENDOR_ID, 0x045e);
});

test("HID: hidReportBytes reads a DataView, a typed array and a plain array alike", () => {
  const source = [0x00, 0x80, 0xff];
  const typed = new Uint8Array(source);
  assert.deepEqual(hidReportBytes(typed), source);
  assert.deepEqual(hidReportBytes(new DataView(typed.buffer)), source);
  assert.deepEqual(hidReportBytes(source), source);
  assert.deepEqual(hidReportBytes(null), []);
});

test("HID: a centred pad decodes to zero on every axis and nothing pressed", () => {
  const decoded = decodeXboxHidReport(report());
  assert.equal(decoded.ok, true, decoded.reason);
  assert.deepEqual(decoded.axes, [0, 0, 0, 0]);
  assert.deepEqual(pressedIndexes(decoded.buttons), []);
  assert.equal(decoded.buttons[6].value, 0);
  assert.equal(decoded.buttons[7].value, 0);
});

test("HID: full stick deflection decodes to exactly the ends of the range", () => {
  const left = decodeXboxHidReport(report({ lx: 0, ly: 0 }));
  assert.equal(left.axes[0], -1);
  assert.equal(left.axes[1], -1);
  const right = decodeXboxHidReport(report({ lx: 65535, ly: 65535 }));
  assert.equal(right.axes[0], 1);
  assert.equal(right.axes[1], 1);
  const rightStick = decodeXboxHidReport(report({ rx: 0, ry: 65535 }));
  assert.equal(rightStick.axes[2], -1);
  assert.equal(rightStick.axes[3], 1);
});

test("HID: the sticks are 16 bit little endian, so the low byte alone does not move the axis to an end", () => {
  // 0x00 0x80 is centred and 0x80 0x00 is nearly hard left. Read big endian,
  // both would decode as the same value, which is the mistake this catches.
  const swapped = decodeXboxHidReport(report({ lx: 0x0080 }));
  near(swapped.axes[0], -0.996);
});

test("HID: the triggers are 10 bit, so 1023 is full travel and not 65535", () => {
  const full = decodeXboxHidReport(report({ lt: 1023, rt: 1023 }));
  assert.equal(full.buttons[6].value, 1);
  assert.equal(full.buttons[7].value, 1);
  const half = decodeXboxHidReport(report({ rt: 512 }));
  near(half.buttons[7].value, 0.5);
  assert.equal(half.buttons[6].value, 0);
});

test("HID: a trigger reading past 1023 is clamped rather than allowed past full throttle", () => {
  const decoded = decodeXboxHidReport(report({ rt: 4095 }));
  assert.equal(decoded.buttons[7].value, 1);
});

test("HID: each face button lands on the Gamepad API index the reducer already uses", () => {
  const cases = [
    [0x01, 0, "A"],
    [0x02, 1, "B"],
    [0x08, 2, "X"],
    [0x10, 3, "Y"],
    [0x40, 4, "LB"],
    [0x80, 5, "RB"],
  ];
  for (const [mask, index, name] of cases) {
    const decoded = decodeXboxHidReport(report({ face: mask }));
    assert.deepEqual(pressedIndexes(decoded.buttons), [index], `${name} must decode to button ${index}`);
    assert.equal(decoded.buttons[index].value, 1);
  }
});

test("HID: View, Menu, the stick clicks and the Xbox button land on their standard indexes", () => {
  const cases = [
    [0x04, 8, "View"],
    [0x08, 9, "Menu"],
    [0x10, 16, "Xbox"],
    [0x20, 10, "LS"],
    [0x40, 11, "RS"],
  ];
  for (const [mask, index, name] of cases) {
    const decoded = decodeXboxHidReport(report({ system: mask }));
    assert.deepEqual(pressedIndexes(decoded.buttons), [index], `${name} must decode to button ${index}`);
  }
});

test("HID: two buttons held at once both decode, so the bitfield is read as bits", () => {
  const decoded = decodeXboxHidReport(report({ face: 0x02 | 0x40 }));
  assert.deepEqual(pressedIndexes(decoded.buttons), [1, 4]);
});

test("HID: the hat switch becomes the four D-pad buttons, diagonals included", () => {
  const cases = [
    [0, []],
    [1, [12]],
    [2, [12, 15]],
    [3, [15]],
    [4, [13, 15]],
    [5, [13]],
    [6, [13, 14]],
    [7, [14]],
    [8, [12, 14]],
  ];
  for (const [hat, expected] of cases) {
    const decoded = decodeXboxHidReport(report({ hat }));
    assert.deepEqual(pressedIndexes(decoded.buttons), expected, `hat ${hat}`);
  }
});

test("HID: an out of range hat value presses no direction rather than guessing one", () => {
  const decoded = decodeXboxHidReport(report({ hat: 15 }));
  assert.deepEqual(pressedIndexes(decoded.buttons), []);
});

test("HID: a 15 byte report decodes, because the Share byte is the only thing missing from it", () => {
  const decoded = decodeXboxHidReport(report({ face: 0x01, length: 15 }));
  assert.equal(decoded.ok, true, decoded.reason);
  assert.equal(decoded.length, 15);
  assert.equal(decoded.hasShare, false);
  assert.deepEqual(pressedIndexes(decoded.buttons), [0]);
});

test("HID: the Share button on a 16 byte report decodes to its own index", () => {
  const decoded = decodeXboxHidReport(report({ share: 0x01 }));
  assert.equal(decoded.hasShare, true);
  assert.deepEqual(pressedIndexes(decoded.buttons), [17]);
});

test("HID: a report too short for the layout is refused and says so, rather than decoding garbage", () => {
  const decoded = decodeXboxHidReport([0x01, 0x02, 0x03]);
  assert.equal(decoded.ok, false);
  assert.equal(decoded.length, 3);
  assert.match(decoded.reason, /3 bytes/);
  assert.match(decoded.reason, new RegExp(String(XBOX_HID_MIN_LENGTH)));
  assert.deepEqual(decoded.axes, []);
  assert.deepEqual(decoded.buttons, []);
});

// Part 2: normalising into the shape the reducer already consumes =============

test("HID: hidPadSnapshot yields the plain axes and buttons shape the reducer takes", () => {
  const snapshot = hidPadSnapshot(decodeXboxHidReport(report({ lx: 65535, rt: 1023, face: 0x01 })));
  assert.equal(snapshot.axes[0], 1);
  assert.equal(snapshot.buttons[0].pressed, true);
  assert.equal(snapshot.buttons[7].value, 1);
  // Nothing but axes and buttons: metadata the reducer does not read must not
  // travel with the snapshot and become something a later frame depends on.
  assert.deepEqual(Object.keys(snapshot).sort(), ["axes", "buttons"]);
  for (const button of snapshot.buttons) {
    assert.deepEqual(Object.keys(button).sort(), ["pressed", "value"]);
  }
});

test("HID: a report that could not be decoded normalises to an empty snapshot, not a stale one", () => {
  const snapshot = hidPadSnapshot(decodeXboxHidReport([0x00]));
  assert.deepEqual(snapshot, emptyPadSnapshot());
  assert.deepEqual(snapshot, { axes: [], buttons: [] });
});

test("HID: an empty snapshot through the live reducer is a zeroed command, not a held throttle", () => {
  const step = computeGamepadStep(emptyPadSnapshot(), { buttons: [] }, armedUi());
  // Compared with === rather than deepEqual or strictEqual because the
  // reducer's armed-and-idle branch negates a zero throttle and yields -0,
  // which both of those treat as a different value from 0 and arithmetic does
  // not. The claim under test is that neither component asks for motion.
  assert.ok(step.drive.left.x === 0, `x was ${step.drive.left.x}`);
  assert.ok(step.drive.left.y === 0, `y was ${step.drive.left.y}`);
  assert.deepEqual(step.actions, []);
});

test("HID: a decoded throttle drives the reducer to the same values as the equivalent Gamepad API frame", () => {
  const decoded = hidPadSnapshot(decodeXboxHidReport(report({ lx: 65535, rt: 1023 })));
  const apiFrame = {
    axes: [1, 0, 0, 0],
    buttons: Array.from({ length: 17 }, (unused, index) => ({
      pressed: index === 7,
      value: index === 7 ? 1 : 0,
    })),
  };
  const fromHid = computeGamepadStep(decoded, { buttons: [] }, armedUi());
  const fromApi = computeGamepadStep(apiFrame, { buttons: [] }, armedUi());
  assert.deepEqual(fromHid.drive, fromApi.drive);
  assert.deepEqual(fromHid.drive, { left: { x: 1, y: -1 } });
});

test("HID: a decoded B press drives the reducer to a hard stop, the same as the Gamepad API path", () => {
  const decoded = hidPadSnapshot(decodeXboxHidReport(report({ face: 0x02, rt: 1023 })));
  const step = computeGamepadStep(decoded, { buttons: [] }, armedUi());
  assert.deepEqual(step.actions, [{ type: "hardStop" }]);
  // The throttle was still held when B went down and the stop still zeroes it.
  assert.deepEqual(step.drive, { left: { x: 0, y: 0 } });
});

test("HID: a decoded Menu press stops the car during an autonomous run", () => {
  const decoded = hidPadSnapshot(decodeXboxHidReport(report({ system: 0x08 })));
  const step = computeGamepadStep(decoded, { buttons: [] }, { ...armedUi(), armed: false, auto: true });
  assert.deepEqual(step.actions, [{ type: "hardStop" }]);
  assert.deepEqual(step.drive, { left: { x: 0, y: 0 } });
});

test("HID: a decoded D-pad Up press nudges the manual speed through the existing rule", () => {
  const decoded = hidPadSnapshot(decodeXboxHidReport(report({ hat: 1 })));
  const step = computeGamepadStep(decoded, { buttons: [] }, { ...armedUi(), manualSpeed: 0.35 });
  assert.deepEqual(step.actions, [{ type: "nudgeManualSpeed", value: 0.4 }]);
});

test("HID: a held button through the reducer twice fires once, so the rising edge rule still holds", () => {
  const decoded = hidPadSnapshot(decodeXboxHidReport(report({ face: 0x01 })));
  const ui = { ...armedUi(), armed: false };
  const first = computeGamepadStep(decoded, { buttons: [] }, ui);
  assert.deepEqual(first.actions, [{ type: "startManual" }]);
  const second = computeGamepadStep(decoded, first.nextPadState, ui);
  assert.deepEqual(second.actions, []);
});

// Part 3: which source drives the frame ======================================

test("HID: the Gamepad API keeps priority whenever it has a pad at all", () => {
  const choice = chooseInputSource({ pad: true, hidOpen: true, hidLastReportAt: 1000, now: 1000 });
  assert.equal(choice.source, "gamepad");
  assert.equal(choice.silent, false);
});

test("HID: with no pad and no open device the frame has no source, which is the existing loss path", () => {
  assert.equal(chooseInputSource({ pad: false, hidOpen: false, now: 1000 }).source, "none");
});

test("HID: an open device with a fresh report drives the frame", () => {
  const choice = chooseInputSource({ pad: false, hidOpen: true, hidLastReportAt: 1000, now: 1100 });
  assert.equal(choice.source, "hid");
  assert.equal(choice.silent, false);
});

test("HID: an open device gone quiet longer than the silence window drives a zeroed frame", () => {
  const choice = chooseInputSource({
    pad: false,
    hidOpen: true,
    hidLastReportAt: 1000,
    now: 1000 + HID_SILENCE_MS + 1,
  });
  assert.equal(choice.source, "hid");
  assert.equal(choice.silent, true, "a pad that has stopped reporting must not keep commanding motion");
});

test("HID: an open device that has never reported is silent, not fresh", () => {
  const choice = chooseInputSource({ pad: false, hidOpen: true, hidLastReportAt: 0, now: 5000 });
  assert.equal(choice.source, "hid");
  assert.equal(choice.silent, true);
});

test("HID: the silence window is well clear of the pad's own report cadence", () => {
  // The pad reports on the order of every 8 to 12 ms, so this is tens of
  // reports of silence before the drive is zeroed, and still fast enough that
  // a pad which has genuinely stopped cannot hold a throttle for long.
  assert.ok(HID_SILENCE_MS >= 100 && HID_SILENCE_MS <= 600, `HID_SILENCE_MS is ${HID_SILENCE_MS}`);
});

// Part 4: what the panel says =================================================

test("HID: a browser with no WebHID says so plainly and names Chrome", () => {
  const notice = hidStatusNotice({ hasWebHid: false });
  assert.equal(notice.level, "error");
  assert.match(notice.detail, /Chrome/);
  assert.match(`${notice.headline} ${notice.detail}`, /WebHID/);
});

test("HID: supported but never authorised tells the operator to press the button", () => {
  const notice = hidStatusNotice({ hasWebHid: true, deviceName: "" });
  assert.equal(notice.level, "warn");
  assert.match(notice.detail, /Connect/);
});

test("HID: authorised but not open reads differently from not authorised at all", () => {
  const notice = hidStatusNotice({ hasWebHid: true, deviceName: "Xbox pad", open: false });
  assert.equal(notice.level, "warn");
  assert.match(notice.detail, /Xbox pad/);
  assert.notEqual(notice.headline, hidStatusNotice({ hasWebHid: true, deviceName: "" }).headline);
});

test("HID: authorised and open but reporting nothing is its own state", () => {
  const notice = hidStatusNotice({
    hasWebHid: true,
    deviceName: "Xbox pad",
    open: true,
    lastReportAt: 0,
    now: 9000,
  });
  assert.equal(notice.level, "warn");
  assert.match(`${notice.headline} ${notice.detail}`, /nothing|no input report/i);
});

test("HID: authorised, open and reporting is the only state that reads as good", () => {
  const notice = hidStatusNotice({
    hasWebHid: true,
    deviceName: "Xbox pad",
    open: true,
    lastReportAt: 1000,
    now: 1050,
    reportLength: 16,
  });
  assert.equal(notice.level, "ok");
  assert.match(notice.detail, /16/);
  // Not a claim that WebHID is driving. With a Gamepad API pad listed as well
  // this device is open, reporting and driving nothing, and only the input
  // source line knows that.
  assert.doesNotMatch(notice.headline, /Driving/);
});

test("HID: a report the decoder refused is named in the panel rather than read as working", () => {
  const notice = hidStatusNotice({
    hasWebHid: true,
    deviceName: "Xbox pad",
    open: true,
    lastReportAt: 1000,
    now: 1050,
    reportLength: 8,
    decodeReason: "report is 8 bytes, the Xbox layout needs at least 15",
  });
  assert.equal(notice.level, "error");
  assert.match(notice.detail, /8 bytes/);
});

test("HID: a device that would not open reads as an error and names what may be holding it", () => {
  const notice = hidStatusNotice({
    hasWebHid: true,
    deviceName: "Xbox pad",
    open: false,
    error: "device is already claimed",
  });
  assert.equal(notice.level, "error");
  assert.match(notice.detail, /already claimed/);
  assert.match(notice.detail, /Steam/);
});

test("HID: a failure before anything was authorised is reported rather than swallowed", () => {
  const notice = hidStatusNotice({ hasWebHid: true, deviceName: "", error: "the chooser failed: blocked" });
  assert.equal(notice.level, "warn");
  assert.match(notice.detail, /chooser failed/);
});

test("HID: an open device reporting fine is not marked bad by an error from before it opened", () => {
  // error is what the page last tried and could not do, decodeReason is a report
  // that arrived and could not be read. Only the second one can be true of a
  // device that is open and reporting, so a stale error must not colour it.
  const notice = hidStatusNotice({
    hasWebHid: true,
    deviceName: "Xbox pad",
    open: true,
    lastReportAt: 1000,
    now: 1010,
    reportLength: 16,
    error: "",
  });
  assert.equal(notice.level, "ok");
});

test("HID: the raw bytes render as offset labelled hex, so the operator can name a byte", () => {
  const text = formatHidReportBytes(report({ lx: 65535, face: 0x02 }));
  const lines = text.split("\n");
  assert.equal(lines.length, 2, text);
  assert.match(lines[0], /^00: ff ff 00 80 00 80 00 80$/);
  assert.match(lines[1], /^08: 00 00 00 00 00 02 00 00$/);
});

test("HID: with no report yet the raw box says so rather than showing an empty box", () => {
  assert.match(formatHidReportBytes([]), /no report/i);
});

test("HID: describeHidDevice carries the ids the operator can check against hidutil", () => {
  const line = describeHidDevice({ productName: "Xbox Wireless Controller", vendorId: 0x045e, productId: 0x0b12 });
  assert.match(line, /Xbox Wireless Controller/);
  assert.match(line, /045e/);
  assert.match(line, /0b12/);
});

test("HID: selectHidDevice takes the first Microsoft device already granted and ignores the rest", () => {
  const keyboard = { vendorId: 0x05ac, productName: "Keyboard" };
  const pad = { vendorId: 0x045e, productName: "Xbox Wireless Controller" };
  assert.equal(selectHidDevice([keyboard, pad]), pad);
  assert.equal(selectHidDevice([keyboard]), null);
  assert.equal(selectHidDevice([]), null);
  assert.equal(selectHidDevice(null), null);
});

// Part 5: the page wiring ====================================================
//
// The pure tests above say the decoder is right about the bytes it was given.
// These say the page actually reaches the device, actually runs the decoded
// report through the one reducer, and actually stops the car when the device
// goes away. app.js runs for real here, against a fake DOM and a fake device,
// because a WebHID path that decodes perfectly and is wired to nothing is worth
// nothing on a car that is being driven.

const XBOX_PRODUCT_NAME = "Xbox Wireless Controller";

async function hidPage(options) {
  const page = loadPage(options);
  await page.settle();
  page.clearCalls();
  return page;
}

// The pad reports every few milliseconds, so a report fired and read on the
// same tick is the normal case and needs no clock control.
function hidFrame(page, device, bytes) {
  device.sendReport(bytes);
  page.run("pollGamepad()");
}

test("WIRING: a browser with no navigator.hid disables the connect button and says why", async () => {
  const page = await hidPage({ hasWebHid: false });
  assert.equal(page.el("hidConnect").disabled, true);
  assert.match(page.el("hidStatus").textContent, /Chrome/);
  assert.equal(page.el("hidStatus").classList.contains("bad"), true);
});

test("WIRING: Chrome with nothing authorised enables the button and asks for a click", async () => {
  const page = await hidPage();
  assert.equal(page.el("hidConnect").disabled, false);
  assert.match(page.el("hidStatus").textContent, /No controller authorised/);
  assert.match(page.el("hidStatus").textContent, /Connect controller/);
  assert.equal(page.el("hidStatus").classList.contains("warn"), true);
});

test("WIRING: the connect click asks the chooser for Microsoft devices and nothing else", async () => {
  const page = await hidPage();
  const device = hidDevice();
  page.hid.requested = [device];

  page.fireElement("hidConnect", "click");
  await page.settle();

  assert.equal(page.hid.requestCalls, 1);
  // Read field by field rather than deepEqual: the options object was built
  // inside the page's own realm, where its prototype is a different
  // Object.prototype and deepEqual rejects it on identity alone.
  const filters = page.hid.requestArgs[0].filters;
  assert.equal(filters.length, 1, "one filter, so the chooser offers nothing but this vendor");
  assert.deepEqual(Object.keys(filters[0]), ["vendorId"]);
  assert.equal(filters[0].vendorId, 0x045e);
  assert.equal(device.opened, true, "the page has to open the device it was handed");
  assert.equal(page.state.hid.open, true);
  assert.match(page.el("hidStatus").textContent, /Open but reporting nothing/);
});

test("WIRING: a device authorised in an earlier session reopens on load with no click", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });

  assert.equal(page.hid.requestCalls, 0, "a granted device must not reopen the chooser");
  assert.equal(device.opened, true);
  assert.equal(page.state.hid.open, true);
  assert.match(page.state.hid.deviceName, /Xbox Wireless Controller/);
});

test("WIRING: a granted device from another vendor is left alone", async () => {
  const keyboard = hidDevice({ vendorId: 0x05ac, productName: "Keyboard" });
  const page = await hidPage({ hidDevices: [keyboard] });

  assert.equal(keyboard.opened, false);
  assert.equal(page.state.hid.open, false);
  assert.match(page.el("hidStatus").textContent, /No controller authorised/);
});

test("WIRING: a dismissed chooser leaves the panel saying what it said before", async () => {
  const page = await hidPage();
  page.hid.requested = [];

  page.fireElement("hidConnect", "click");
  await page.settle();

  assert.equal(page.state.hid.open, false);
  assert.match(page.el("hidStatus").textContent, /No controller authorised/);
});

test("WIRING: a device that will not open reports the failure rather than reading as open", async () => {
  const device = hidDevice({ openFails: true });
  const page = await hidPage({ hidDevices: [device] });

  assert.equal(page.state.hid.open, false);
  assert.match(page.el("hidStatus").textContent, /Could not open the controller/);
  assert.match(page.el("hidStatus").textContent, /already claimed/);
  // Something else holding the pad exclusively is the likeliest way this fails
  // on a machine where the pad works everywhere else, so Steam is named.
  assert.match(page.el("hidStatus").textContent, /Steam/);
  assert.equal(page.el("hidStatus").classList.contains("bad"), true);
});

test("WIRING: pressing connect twice does not leave two listeners decoding every report", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.hid.requested = [device];

  page.fireElement("hidConnect", "click");
  await page.settle();

  assert.equal(device.reportListeners, 1);
  assert.equal(page.state.hid.open, true);
});

test("WIRING: a decoded report drives the car through the same reducer and sender", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");

  // Hard right on the left stick, right trigger at full travel.
  hidFrame(page, device, report({ lx: 65535, rt: 1023 }));
  page.run("sendDriveIfNeeded();");
  await page.settle();

  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1, "a decoded report has to reach /api/drive");
  assert.equal(drive[0].body.enabled, true);
  assert.ok(drive[0].body.linear_x > 0, `linear_x was ${drive[0].body.linear_x}`);
  // Steering is negated on the way out, because the chassis steers the opposite
  // way from the stick. Both input paths go through the one scaledCommand, so
  // this is the same sign the touch joystick and the Gamepad API produce.
  assert.ok(drive[0].body.steering_y < 0, `steering_y was ${drive[0].body.steering_y}`);
  assert.equal(page.state.padSourceText, "WebHID");
});

test("WIRING: the panel shows the live axes and buttons from the WebHID path", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });

  hidFrame(page, device, report({ lx: 0, face: 0x01 }));

  assert.match(page.el("padAxes").textContent, /A0 -1\.00/);
  assert.match(page.el("padButtons").textContent, /A\[0\]/);
  assert.match(page.el("hidStatus").textContent, /Open and reporting/);
  assert.equal(page.state.padSourceText, "WebHID", "the input source line is what names the driver");
  assert.equal(page.el("hidStatus").classList.contains("good"), true);
});

test("WIRING: B on the WebHID pad stops the car, the same button as on the Gamepad API path", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");

  hidFrame(page, device, report({ face: 0x02, rt: 1023 }));
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1, "B must reach /api/stop through WebHID");
  assert.equal(page.state.armed, false);
  assert.deepEqual(page.state.left, { x: 0, y: 0 });
});

test("WIRING: Menu on the WebHID pad stops an autonomous run", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.auto = true; state.gamepadEnabled = true;");

  hidFrame(page, device, report({ system: 0x08 }));
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1);
  assert.equal(page.state.auto, false);
});

test("WIRING: the on-screen STOP button still works while WebHID is the source", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true;");
  hidFrame(page, device, report({ rt: 1023 }));
  page.clearCalls();

  page.fireElement("stop", "click");
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1);
  assert.equal(page.state.armed, false);
});

test("WIRING: A on the WebHID pad arms manual control", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = false; state.auto = false; state.gamepadEnabled = true;");

  hidFrame(page, device, report({ face: 0x01 }));
  await page.settle();

  assert.equal(page.posts("/api/start").length, 1);
  assert.equal(page.state.armed, true);
});

test("WIRING: a WebHID device that disconnects stops the car through the existing loss path", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");
  hidFrame(page, device, report({ rt: 1023 }));
  page.clearCalls();

  page.fireHid("disconnect", { device });
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1, "a vanished device must stop the car");
  assert.equal(page.state.armed, false);
  assert.deepEqual(page.state.left, { x: 0, y: 0 });
  assert.equal(page.state.hid.open, false);
  assert.match(page.el("hidStatus").textContent, /Authorised but not open/);
});

test("WIRING: a disconnect for some other device does not stop the car", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true;");
  page.clearCalls();

  page.fireHid("disconnect", { device: hidDevice({ productName: "Some other pad" }) });
  await page.settle();

  assert.deepEqual(page.posts("/api/stop"), []);
  assert.equal(page.state.hid.open, true);
});

test("WIRING: a device that comes back reopens itself without another chooser", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.fireHid("disconnect", { device });
  await page.settle();
  assert.equal(page.state.hid.open, false);

  page.fireHid("connect", { device });
  await page.settle();

  assert.equal(page.state.hid.open, true);
  assert.equal(page.hid.requestCalls, 0, "the grant survives a replug, so no chooser may open");
  // A device object outlives its own disconnect and so does a listener on it, so
  // reopening must not attach a second one and decode every report twice.
  assert.equal(device.reportListeners, 1);
});

test("WIRING: a report after a replug decodes once, not once per reopen", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  for (let round = 0; round < 3; round += 1) {
    page.fireHid("disconnect", { device });
    await page.settle();
    page.fireHid("connect", { device });
    await page.settle();
  }
  page.clearCalls();

  const handled = device.sendReport(report({ face: 0x01 }));
  assert.equal(handled, 1, "three replugs must still leave exactly one report listener");
});

test("WIRING: an open device gone quiet zeroes the drive without disarming the car", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");
  hidFrame(page, device, report({ rt: 1023 }));
  assert.ok(page.state.left.y < 0, "the throttle is live before the pad goes quiet");
  page.clearCalls();

  // Backdate the last report past the silence window, which is what a pad that
  // has stopped reporting looks like from the poll loop.
  page.run(`state.hid.lastReportAt = performance.now() - ${HID_SILENCE_MS + 50};`);
  page.run("pollGamepad(); sendDriveIfNeeded();");
  await page.settle();

  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1);
  assert.equal(drive[0].body.linear_x, 0, "a pad reporting nothing must not hold the throttle open");
  // Silence is weaker evidence than a disconnect, so it does not disarm. The
  // drive stays at zero for as long as the silence lasts, and the disconnect
  // event carries the hard stop when the pad has genuinely gone.
  assert.equal(page.state.armed, true);
  assert.deepEqual(page.posts("/api/stop"), []);
  assert.match(page.state.padSourceText, /drive held at zero/);
  assert.match(page.el("hidStatus").textContent, /Open but reporting nothing/);
});

test("WIRING: the Gamepad API keeps priority when it has a pad, WebHID open or not", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");

  // A live WebHID throttle, and a listed pad with nothing pressed on it. The
  // listed pad wins, so the car does not move.
  device.sendReport(report({ rt: 1023 }));
  page.setPads([pad({ axes: [0, 0, 0, 0] })]);
  page.run("pollGamepad(); sendDriveIfNeeded();");
  await page.settle();

  assert.equal(page.state.padSourceText, "Gamepad API");
  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1);
  assert.equal(drive[0].body.linear_x, 0, "the Gamepad API path must not be overridden by WebHID");
});

test("WIRING: with no pad listed and no device open the existing loss path still runs", async () => {
  const page = await hidPage();
  page.run("state.armed = true; state.gamepadIndex = 0;");
  page.clearCalls();

  page.run("pollGamepad()");
  await page.settle();

  assert.equal(page.posts("/api/stop").length, 1);
  assert.match(page.state.padSourceText, /^none/);
});

test("WIRING: the debug toggle reveals the raw bytes and the offsets in them", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  assert.equal(page.el("hidRawBox").classList.contains("hidden"), true);

  hidFrame(page, device, report({ lx: 65535, face: 0x02 }));
  page.el("hidDebug").checked = true;
  page.fireElement("hidDebug", "change");

  assert.equal(page.el("hidRawBox").classList.contains("hidden"), false);
  assert.match(page.el("hidRaw").textContent, /^00: ff ff 00 80 00 80 00 80$/m);
  assert.match(page.el("hidRaw").textContent, /^08: 00 00 00 00 00 02 00 00$/m);
});

test("WIRING: a report the decoder refuses shows in the panel and commands nothing", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = true;");

  hidFrame(page, device, [0x01, 0x02, 0x03, 0x04]);
  page.run("sendDriveIfNeeded();");
  await page.settle();

  assert.match(page.el("hidStatus").textContent, /not decodable/);
  assert.equal(page.el("hidStatus").classList.contains("bad"), true);
  const drive = page.posts("/api/drive");
  assert.equal(drive.length, 1);
  assert.equal(drive[0].body.linear_x, 0, "an undecodable report must not drive the car");
  // The bytes are on screen without the debug toggle being touched, because a
  // report we cannot read is exactly the one the operator needs to see.
  assert.equal(page.el("hidDebug").checked, false);
  assert.equal(page.el("hidRawBox").classList.contains("hidden"), false);
  assert.match(page.el("hidRaw").textContent, /^00: 01 02 03 04$/m);
});

test("WIRING: the Xbox toggle silences the WebHID path as it does the Gamepad API path", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });
  page.run("state.armed = true; state.gamepadEnabled = false;");

  hidFrame(page, device, report({ rt: 1023 }));
  page.run("sendDriveIfNeeded();");
  await page.settle();

  const drive = page.posts("/api/drive");
  assert.deepEqual(page.state.left, { x: 0, y: 0 });
  if (drive.length) assert.equal(drive[0].body.linear_x, 0);
});

test("WIRING: the WebHID path files its own telemetry with the car, marked as such", async () => {
  const device = hidDevice();
  const page = await hidPage({ hidDevices: [device] });

  // reportGamepad throttles itself to one post per 100 ms and compares against
  // a zero, so on a process only tens of milliseconds old the first frame is
  // throttled away. Backdating the mark is what a page that has been open for a
  // second already looks like.
  page.run("state.gamepadLastReportAt = -1000;");
  hidFrame(page, device, report({ face: 0x01 }));
  await page.settle();

  const posts = page.posts("/api/gamepad");
  assert.equal(posts.length, 1);
  assert.equal(posts[0].body.mapping, "webhid");
  assert.equal(posts[0].body.index, -1);
  assert.match(posts[0].body.id, new RegExp(XBOX_PRODUCT_NAME));
  assert.deepEqual(posts[0].body.pressed.map((entry) => entry.name), ["A"]);
});
