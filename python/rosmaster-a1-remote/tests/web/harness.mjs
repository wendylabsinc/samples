// A minimal browser stand-in for running the real page script.
//
// rosmaster-a1-web-remote-wendy/app/static/app.js used to be an inline script
// in index.html, which put the whole stop path wiring somewhere no unit test
// could reach. The tests that tried to cover it read index.html as text and
// matched substrings, and a reviewer's mutation run showed what that was worth:
// twelve edits that broke real safety behavior, every one of them green.
//
// So the page script is a file now, and this harness runs it for real in a
// node:vm context with a fake DOM, fake gamepads, and a fake server. Both
// static scripts are evaluated in one context exactly as the two script tags in
// index.html do it, so app.js sees gamepad.js's functions the same way it does
// in the browser. Tests then call the page's own functions and assert on what
// it sent and what it rendered.
//
// Node stdlib only, no build step, matching the rest of this project.
import vm from "node:vm";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const STATIC_DIR = new URL("../../rosmaster-a1-web-remote-wendy/app/static/", import.meta.url);

function read(name) {
  return readFileSync(new URL(name, STATIC_DIR), "utf8");
}

// The initial slider and checkbox values come from index.html itself rather
// than from constants duplicated here, so a fake element never claims a
// starting value the shipped page does not have.
function parseInputDefaults() {
  const html = read("index.html");
  const defaults = new Map();
  for (const tag of html.match(/<input\b[^>]*>/g) || []) {
    const id = /\bid="([^"]+)"/.exec(tag);
    if (!id) continue;
    const attr = (name) => {
      const found = new RegExp(`\\b${name}="([^"]*)"`).exec(tag);
      return found ? found[1] : undefined;
    };
    defaults.set(id[1], { value: attr("value"), min: attr("min"), max: attr("max"), type: attr("type") });
  }
  assert.ok(defaults.has("speed"), "index.html must still carry the speed slider the harness seeds from");
  return defaults;
}

const INPUT_DEFAULTS = parseInputDefaults();

function makeCanvasContext() {
  const noop = () => {};
  return {
    fillStyle: "", strokeStyle: "", lineWidth: 0,
    clearRect: noop, fillRect: noop, beginPath: noop, arc: noop, stroke: noop,
    moveTo: noop, lineTo: noop, closePath: noop, fill: noop,
  };
}

function makeElement(id) {
  const defaults = INPUT_DEFAULTS.get(id) || {};
  const listeners = new Map();
  return {
    id,
    value: defaults.value === undefined ? "" : defaults.value,
    min: defaults.min === undefined ? "" : defaults.min,
    max: defaults.max === undefined ? "" : defaults.max,
    type: defaults.type === undefined ? "" : defaults.type,
    checked: false,
    disabled: false,
    textContent: "",
    innerHTML: "",
    src: "",
    className: "",
    width: 640,
    height: 400,
    style: {},
    children: [],
    classList: {
      names: new Set(),
      add(name) { this.names.add(name); },
      remove(name) { this.names.delete(name); },
      contains(name) { return this.names.has(name); },
      toggle(name, on) { if (on) this.names.add(name); else this.names.delete(name); },
    },
    listeners,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    removeEventListener() {},
    dispatch(type, event) {
      for (const handler of listeners.get(type) || []) handler(event);
    },
    appendChild(child) { this.children.push(child); },
    // The gallery creates and destroys tiles as the car's feed registry
    // changes, so the fake container has to be able to lose a child.
    removeChild(child) {
      const at = this.children.indexOf(child);
      if (at >= 0) this.children.splice(at, 1);
      return child;
    },
    querySelector() { return makeElement(`${id}-child`); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    setPointerCapture() {},
    getContext() { return makeCanvasContext(); },
  };
}

function defaultStatus() {
  return {
    ok: true,
    // The feed registry the gallery renders from. Two entries, because that is
    // what this hardware has: the HP60C driver publishes depth and rgb and no
    // infrared or stereo image at all.
    cameras: [
      { id: "hp60c_depth", label: "Depth", path: "/stream_hp60c_depth.mjpg", ok: true, age_s: 0.06 },
      { id: "hp60c_rgb", label: "RGB", path: "/stream_hp60c_rgb.mjpg", ok: true, age_s: 0.09 },
    ],
    control: {
      command: { enabled: false, linear_x: 0, steering_y: 0, angular_z: 0 },
      last_published: { linear_x: 0, steering_y: 0, angular_z: 0 },
      cmd_vel_subscribers: 1,
      limits: { max_linear_x: 0.65, max_steering_y: 0.12, max_angular_z: 1.0, cmd_timeout_s: 0.5 },
    },
    lidar: { ok: true, points: [], sectors: {}, finite_ranges: 0, min_m: null },
    hp60c: { depth: { ok: true, width: 640, height: 480 }, rgb: { ok: true } },
    sensors: {},
    auto: { enabled: false, decision: null, preview: null },
    // depth_source names the camera the planner actually read, which the page
    // repeats back rather than assuming. This car has an HP60C; a RealSense
    // reports the same shape with a different name in it.
    navigation: { ready: true, depth_source: "hp60c", depth_ok: true },
    gamepad: { ok: false },
    commands: { max_age_ms: 400, rejected: { stale: 0, out_of_order: 0, total: 0 } },
  };
}

// A WebHID device stand-in. There is no WebHID in Node, so the page's fallback
// input path is exercised against this: it opens, it takes an inputreport
// listener, and a test fires reports into it by hand.
export function hidDevice({
  vendorId = 0x045e,
  productId = 0x0b12,
  productName = "Xbox Wireless Controller",
  opened = false,
  openFails = false,
} = {}) {
  const listeners = new Map();
  return {
    vendorId,
    productId,
    productName,
    opened,
    openFails,
    opens: 0,
    listeners,
    async open() {
      this.opens += 1;
      if (this.openFails) throw new Error("device is already claimed");
      this.opened = true;
    },
    async close() { this.opened = false; },
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    removeEventListener() {},
    // reportCount is how a test proves a second Connect press did not leave two
    // listeners decoding every report.
    get reportListeners() { return (listeners.get("inputreport") || []).length; },
    sendReport(bytes, reportId = 1) {
      const handlers = listeners.get("inputreport") || [];
      const data = new DataView(new Uint8Array(bytes).buffer);
      for (const handler of handlers) handler({ reportId, data, device: this });
      return handlers.length;
    },
  };
}

// loadPage evaluates gamepad.js and app.js in one fresh context and returns the
// handles a test needs to drive them.
//
// hidDevices seeds what navigator.hid.getDevices() answers on load, which is the
// path a device already authorised in an earlier session comes back through.
// hasWebHid false is the Firefox case: no navigator.hid at all.
export function loadPage({ hidDevices = [], hidRequest = null, hasWebHid = true } = {}) {
  const elements = new Map();
  const calls = [];
  const fake = {
    status: defaultStatus(),
    // Paths listed here reject, which is how the page sees an offline car.
    failing: new Set(),
    // Paths listed here have their response held until page.releaseHeld(),
    // which is how a test sees the page's behavior while a request is still in
    // flight. The real link's drive POSTs have a 1212 ms 90th percentile, so
    // "still in flight" is the normal case, not an edge one.
    held: new Set(),
    responses: new Map(),
  };
  // Each entry tracks one still-open held request so an abort can settle it
  // out of turn: resolve() is what releaseHeld() calls, and settled guards
  // against a request being resolved by one path and then reached by the
  // other (release after an abort already fired, or vice versa).
  const heldResponses = [];

  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, makeElement(id));
      return elements.get(id);
    },
    createElement(tag) { return makeElement(`created-${tag}`); },
    addEventListener() {},
  };

  const windowListeners = new Map();
  // isSecureContext and location.origin are what the Controller panel reads to
  // explain why a working pad is invisible, so the harness has to be able to
  // present both a secure and an insecure page.
  const windowObject = {
    isSecureContext: true,
    location: { origin: "http://localhost:8091" },
    addEventListener(type, handler) {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(handler);
    },
  };

  let pads = [];
  const navigator = { getGamepads: () => pads };

  // The WebHID side of the fake navigator. requestDevice answers whatever the
  // test loaded into hidRequest, which is the chooser's result: an array, empty
  // for a dismissed chooser.
  const hidListeners = new Map();
  const hid = {
    granted: hidDevices.slice(),
    requested: hidRequest,
    requestCalls: 0,
    requestArgs: [],
    async requestDevice(options) {
      hid.requestCalls += 1;
      hid.requestArgs.push(options);
      if (hid.requestThrows) throw new Error("chooser blocked");
      return hid.requested === null ? [] : hid.requested;
    },
    async getDevices() { return hid.granted.slice(); },
    addEventListener(type, handler) {
      if (!hidListeners.has(type)) hidListeners.set(type, []);
      hidListeners.get(type).push(handler);
    },
    removeEventListener() {},
  };
  if (hasWebHid) navigator.hid = hid;

  function fetchStub(path, init) {
    const method = (init && init.method) || "GET";
    let body = null;
    if (init && typeof init.body === "string" && init.body.length) body = JSON.parse(init.body);
    calls.push({ path, method, body });
    if (fake.failing.has(path)) return Promise.reject(new Error(`offline ${path}`));
    const payload = path === "/api/status"
      ? fake.status
      : fake.responses.has(path) ? fake.responses.get(path) : { ok: true };
    const response = { ok: true, status: 200, json: () => Promise.resolve(payload) };
    if (fake.held.has(path)) {
      return new Promise((resolve, reject) => {
        const entry = { settled: false, resolve: () => resolve(response) };
        heldResponses.push(entry);
        // A real fetch given an AbortSignal rejects the moment that signal
        // fires, hung or not. Mirroring that here is what lets a test drive
        // the page's own timeout path (expireFetchTimeouts, below) instead of
        // a fetch that is held staying held forever no matter what app.js does.
        const signal = init && init.signal;
        if (!signal) return;
        signal.addEventListener("abort", () => {
          if (entry.settled) return;
          entry.settled = true;
          const at = heldResponses.indexOf(entry);
          if (at >= 0) heldResponses.splice(at, 1);
          reject(signal.reason || new Error("The operation was aborted."));
        });
      });
    }
    return Promise.resolve(response);
  }

  // Timers are inert so nothing runs behind a test's back. setTimeout still
  // resolves, because hardStop's retry policy sleeps between attempts and a
  // timeout that never fires would hang the suite rather than test it. The
  // requested delay is recorded before the callback runs, which is how a test
  // sees that two tiles asked to reconnect at different moments rather than
  // together.
  //
  // A fetch timeout is the one setTimeout in this file that must NOT resolve
  // that way: FETCH_TIMEOUT_MS (4000 ms) exists precisely so a still-open
  // request stays open for a while, and every other delay in app.js and
  // gamepad.js (the 150 ms stop retry, the ~1000-2200 ms feed reconnect) sits
  // well under it. So any requested delay at or above FETCH_TIMEOUT_THRESHOLD_MS
  // is treated as a fetch timeout: it does not fire on its own, the way a real
  // 4-second timer would not fire during a test that runs in milliseconds.
  // expireFetchTimeouts() below is what a test uses to say the deadline
  // arrived. Everything shorter keeps firing immediately, unchanged, so every
  // existing delay-based test keeps working exactly as it did before.
  const FETCH_TIMEOUT_THRESHOLD_MS = 3000;
  const timeouts = [];
  let nextTimerId = 1;
  const fetchTimeoutTimers = new Map();
  const sandbox = {
    console,
    document,
    window: windowObject,
    navigator,
    fetch: fetchStub,
    performance: { now: () => performance.now() },
    requestAnimationFrame: () => 0,
    setInterval: () => 0,
    clearInterval: () => {},
    // AbortController is a Node/Web global, not a node:vm builtin, so a
    // fresh vm context has no constructor for it unless one is handed in. The
    // host realm's own class is used as-is: nothing here does a cross-realm
    // instanceof check on it, only new AbortController() and .signal/.abort().
    AbortController,
    setTimeout: (fn, ms) => {
      const delay = Number(ms) || 0;
      timeouts.push(delay);
      if (delay >= FETCH_TIMEOUT_THRESHOLD_MS) {
        const id = nextTimerId++;
        fetchTimeoutTimers.set(id, fn);
        return id;
      }
      queueMicrotask(fn);
      return 0;
    },
    clearTimeout: (id) => { fetchTimeoutTimers.delete(id); },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(read("gamepad.js"), context, { filename: "gamepad.js" });
  vm.runInContext(read("app.js"), context, { filename: "app.js" });

  const page = {
    context,
    calls,
    fake,
    // run evaluates an expression against the page's own scope.
    run(expression) { return vm.runInContext(expression, context); },
    el(id) { return document.getElementById(id); },
    // A plain host-realm copy of the page's state. The live object lives in
    // the vm realm, where its prototype is a different Object.prototype and
    // assert.deepEqual would reject it on identity alone.
    get state() { return JSON.parse(vm.runInContext("JSON.stringify(state)", context)); },
    setPads(next) { pads = next; },
    // The three ways a browser can hide a working controller, so a test can
    // present each one to the page.
    setSecureContext(flag, origin) {
      windowObject.isSecureContext = Boolean(flag);
      if (origin) windowObject.location.origin = origin;
    },
    removeGetGamepads() { delete navigator.getGamepads; },
    // The WebHID fallback's handles. hid.requested is the chooser's answer,
    // hid.granted is what an earlier session already authorised.
    hid,
    fireHid(type, event) {
      const handlers = hidListeners.get(type) || [];
      assert.notEqual(handlers.length, 0, `the page registered no navigator.hid ${type} listener`);
      for (const handler of handlers) handler(event);
    },
    hasHidListener(type) { return (hidListeners.get(type) || []).length > 0; },
    releaseHeld() {
      const pending = heldResponses.splice(0, heldResponses.length);
      for (const entry of pending) {
        entry.settled = true;
        entry.resolve();
      }
      return pending.length;
    },
    // expireFetchTimeouts simulates FETCH_TIMEOUT_MS actually elapsing: it
    // fires every fetch-timeout setTimeout callback registered so far (see
    // FETCH_TIMEOUT_THRESHOLD_MS above), which is what turns a still-held
    // fetch into the same AbortController-driven rejection a real hung
    // request would produce after 4 real seconds. Callbacks already cleared
    // by the page's own clearTimeout are gone from the map and do not fire.
    expireFetchTimeouts() {
      const pending = [...fetchTimeoutTimers.values()];
      fetchTimeoutTimers.clear();
      for (const fn of pending) fn();
      return pending.length;
    },
    fireWindow(type, event) {
      const handlers = windowListeners.get(type) || [];
      assert.notEqual(handlers.length, 0, `the page registered no ${type} listener`);
      for (const handler of handlers) handler(event);
    },
    fireElement(id, type, event) {
      const element = document.getElementById(id);
      assert.ok(element.listeners.has(type), `#${id} has no ${type} listener`);
      element.dispatch(type, event);
    },
    posts(path) { return calls.filter((call) => call.path === path && call.method === "POST"); },
    clearCalls() { calls.length = 0; timeouts.length = 0; },
    // Every delay the page asked setTimeout for, in the order it asked.
    get timeouts() { return timeouts.slice(); },
    // The gallery's live tiles, keyed by feed id, as the page holds them.
    tile(id) { return vm.runInContext(`cameraTiles.get(${JSON.stringify(id)})`, context); },
    tileIds() { return JSON.parse(vm.runInContext("JSON.stringify([...cameraTiles.keys()])", context)); },
    // settle drains the microtask queue, including the ones the fake setTimeout
    // schedules, so an awaited page function has finished before an assertion.
    async settle(rounds = 40) {
      for (let i = 0; i < rounds; i += 1) await Promise.resolve();
    },
  };
  return page;
}

// A Gamepad-shaped snapshot, the same shape navigator.getGamepads returns.
export function pad({ index = 0, id = "Xbox Wireless Controller", axes = [0, 0, 0, 0], buttons = {}, connected = true, mapping = "standard" } = {}) {
  const list = [];
  for (let i = 0; i <= 16; i += 1) {
    const spec = buttons[i];
    list[i] = spec
      ? { pressed: Boolean(spec.pressed), value: spec.value === undefined ? (spec.pressed ? 1 : 0) : spec.value }
      : { pressed: false, value: 0 };
  }
  return { index, id, mapping, connected, axes, buttons: list };
}
