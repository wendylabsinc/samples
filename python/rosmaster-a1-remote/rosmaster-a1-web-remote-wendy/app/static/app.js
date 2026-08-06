const state = {
  armed: false,
  auto: false,
  // On by default. Off by default meant a connected controller did nothing
  // until you found an unlabelled checkbox, which reads as a broken feature
  // rather than a disabled one. Nothing moves just because the pad is enabled:
  // driving still needs the car armed with A, and Auto Nav still overrides.
  gamepadEnabled: true,
  gamepadIndex: null,
  gamepadName: "",
  gamepadLastText: "",
  gamepadLastReportAt: 0,
  gamepadPadState: { buttons: [] },
  // The camera feed registry as the car last reported it, and which of its
  // feeds is expanded. Both are filled in by refreshStatus from /api/status
  // and empty until the first poll lands, so the gallery never renders a tile
  // for a camera that may not exist and the controller's expand cycle never
  // guesses at one either.
  cameraFeeds: [],
  feedIds: [],
  expandedFeed: null,
  left: { x: 0, y: 0 },
  // Everything the Controller panel paints. It is kept on state rather than
  // read off the DOM so a test can assert on what the page decided, and it is
  // filled in whether or not the Xbox toggle is on: an operator who has not
  // realised the pad was never detected is one of the failures the panel is
  // for.
  padLines: [],
  padSelectionText: "Waiting for the first poll",
  padSelectionLevel: "warn",
  padAxesText: "none",
  padButtonsText: "none",
  padSourceText: "none",
  // The WebHID fallback's whole state. It is a fallback and not a replacement:
  // nothing here is consulted while navigator.getGamepads lists a pad, and the
  // decoded report is normalised into the same snapshot shape the Gamepad API
  // path produces so both run through one reducer.
  //
  //   supported      navigator.hid exists, which is Chrome only
  //   deviceName     the device the operator authorised, or empty
  //   open           the page has it open and is listening for reports
  //   lastReportAt   performance.now() of the last inputreport
  //   reportBytes    the last report, raw, for the debug readout
  //   error          what the page last tried and could not do, or empty
  //   decodeReason   why the last report could not be decoded, or empty
  //   snapshot       the last decoded report in the reducer's own shape
  //   debug          the raw bytes toggle in the panel
  hid: {
    supported: false,
    deviceName: "",
    open: false,
    lastReportAt: 0,
    reportId: 0,
    reportLength: 0,
    reportBytes: [],
    error: "",
    decodeReason: "",
    snapshot: { axes: [], buttons: [] },
    debug: false,
  },
  gamepadLog: [],
  lastDrivePayload: null,
  lastDriveOkAt: 0,
  driveSeq: 0,
  driveSend: newDriveSendState(),
  // The two candidate sources for the Command readout. lastCommand is what
  // this page last sent for manual driving; lastPublished is what /api/status
  // reports the server last published, which is the only truthful source
  // during an autonomous run. commandReadoutText picks between them.
  lastCommand: { linear_x: 0, steering_y: 0 },
  lastPublished: null,
  // stopPending records what a stop still has to prove it undid, so a
  // later /api/status poll can confirm it. stopUnconfirmed means the
  // retries ran out without an answer and the readout must say so.
  stopPending: null,
  stopUnconfirmed: false,
  limits: { maxLinearX: 0.65, maxSteeringY: 0.12 },
  lastStatusOk: false,
};

const els = {
  arm: document.getElementById("arm"),
  auto: document.getElementById("auto"),
  gamepad: document.getElementById("gamepad"),
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  speed: document.getElementById("speed"),
  speedInput: document.getElementById("speedInput"),
  steer: document.getElementById("steer"),
  autoSpeed: document.getElementById("autoSpeed"),
  autoSpeedInput: document.getElementById("autoSpeedInput"),
  stopRange: document.getElementById("stopRange"),
  avoidRange: document.getElementById("avoidRange"),
  clearRange: document.getElementById("clearRange"),
  speedLabel: document.getElementById("speedLabel"),
  steerLabel: document.getElementById("steerLabel"),
  autoSpeedLabel: document.getElementById("autoSpeedLabel"),
  stopRangeLabel: document.getElementById("stopRangeLabel"),
  avoidRangeLabel: document.getElementById("avoidRangeLabel"),
  clearRangeLabel: document.getElementById("clearRangeLabel"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  cameraGallery: document.getElementById("cameraGallery"),
  galleryStats: document.getElementById("galleryStats"),
  galleryBack: document.getElementById("galleryBack"),
  leftReadout: document.getElementById("leftReadout"),
  padContext: document.getElementById("padContext"),
  padCount: document.getElementById("padCount"),
  padList: document.getElementById("padList"),
  padSelection: document.getElementById("padSelection"),
  padAxes: document.getElementById("padAxes"),
  padButtons: document.getElementById("padButtons"),
  padSource: document.getElementById("padSource"),
  hidStatus: document.getElementById("hidStatus"),
  hidConnect: document.getElementById("hidConnect"),
  hidDebug: document.getElementById("hidDebug"),
  hidRaw: document.getElementById("hidRaw"),
  hidRawBox: document.getElementById("hidRawBox"),
  padLog: document.getElementById("padLog"),
  padDrive: document.getElementById("padDrive"),
  padResponse: document.getElementById("padResponse"),
  modeValue: document.getElementById("modeValue"),
  driverValue: document.getElementById("driverValue"),
  commandValue: document.getElementById("commandValue"),
  watchdogValue: document.getElementById("watchdogValue"),
  autoReadyValue: document.getElementById("autoReadyValue"),
  autoValue: document.getElementById("autoValue"),
  gamepadValue: document.getElementById("gamepadValue"),
  lidarStats: document.getElementById("lidarStats"),
  lidarCanvas: document.getElementById("lidarCanvas"),
  frontValue: document.getElementById("frontValue"),
  leftValue: document.getElementById("leftValue"),
  rightValue: document.getElementById("rightValue"),
  closestValue: document.getElementById("closestValue"),
};

function meters(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} m` : "--";
}

function makeJoystick(root, onChange) {
  const knob = root.querySelector(".knob");
  const stick = { pointerId: null, x: 0, y: 0 };

  function setKnob(x, y) {
    stick.x = x;
    stick.y = y;
    knob.style.left = `${50 + x * 38}%`;
    knob.style.top = `${50 + y * 38}%`;
    onChange({ x, y });
  }

  function updateFromEvent(event) {
    const rect = root.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = rect.width / 2;
    const dx = (event.clientX - cx) / radius;
    const dy = (event.clientY - cy) / radius;
    const distance = Math.hypot(dx, dy);
    const scale = distance > 1 ? 1 / distance : 1;
    setKnob(gamepadClamp(dx * scale, -1, 1), gamepadClamp(dy * scale, -1, 1));
  }

  root.addEventListener("pointerdown", (event) => {
    if (state.auto) return;
    stick.pointerId = event.pointerId;
    root.setPointerCapture(event.pointerId);
    updateFromEvent(event);
  });

  root.addEventListener("pointermove", (event) => {
    if (stick.pointerId !== event.pointerId) return;
    updateFromEvent(event);
  });

  function release(event) {
    if (stick.pointerId !== event.pointerId) return;
    stick.pointerId = null;
    setKnob(0, 0);
  }

  root.addEventListener("pointerup", release);
  root.addEventListener("pointercancel", release);
  root.addEventListener("lostpointercapture", () => {
    stick.pointerId = null;
    setKnob(0, 0);
  });
}

// angular_z is gone. Commanding it for two seconds moved no encoder on any of
// the four channels: this is an Ackermann chassis with no in place turn, and
// the left stick already steers it.
function scaledCommand() {
  const steerScale = Number(els.steer.value) / 100;
  const manualSpeed = finiteNonNegative(els.speedInput.value, Number(els.speed.value) || 0);
  return {
    enabled: state.armed && !state.auto,
    linear_x: -state.left.y * manualSpeed,
    // Negated because the chassis steers the opposite way from the stick:
    // pushing right sent the car left. Both the touch joystick and the pad
    // feed state.left.x, so flipping here fixes both and keeps them agreeing.
    steering_y: -state.left.x * state.limits.maxSteeringY * steerScale,
  };
}

// FETCH_TIMEOUT_MS bounds every fetch this page makes. The server wedged on
// the car once, and nothing here had a timeout: every fetch waited on it
// forever. A hang never rejects, so none of the existing catch/finally paths
// ever ran to notice: driveInFlight below stayed stuck true and silently
// dropped every gamepad command after the one hung POST, the status poll
// stacked a new connection every 750 ms tick, and setConnection(false, ...)
// never fired because it only runs from a rejection. Worse, this hardware
// keeps four MJPEG streams open at all times, which already holds four of
// Chrome's six per-origin sockets; wedged fetches piling up on the rest is
// what made even a page refresh queue forever. 4000 ms sits comfortably above
// the drive POST's 1212 ms 90th-percentile latency (server.py's
// CommandFreshness docstring) and comfortably below where an operator gives
// up and reloads.
const FETCH_TIMEOUT_MS = 4000;

async function postJson(path, payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`${path} ${response.status}`);
    return response.json();
  } finally {
    // Cleared on every outcome, timeout included, so a request that answers
    // just under the wire never leaves a stray abort() scheduled behind it.
    clearTimeout(timer);
  }
}

// The drive sender ==========================================================
//
// The old sender was setInterval(sendDrive, 120): eight POSTs a second whether
// or not anything had changed, on a link that averages 319 ms round trip and
// whose drive POSTs have a 1212 ms 90th percentile. Most of that traffic was
// re-telling the car something it already knew, and the queue it built is what
// the operator felt as stutter and as commands landing after they had moved on.
//
// planDriveSend in gamepad.js holds the policy. Three things drive a send now:
// a change, a 200 ms heartbeat while the command asks for motion, and a
// bounded repeat of a command that has just stopped asking for motion.
//
// Staleness on the wire. The payload carries three extra fields:
//
//   client_id  random per page load, so a reload that restarts the counter is
//              not mistaken for a reordered burst
//   seq        one per command, so the car can drop anything that arrives
//              behind something newer. This is what stops a throttle that was
//              still in flight from overtaking the zero that followed it
//   age_ms     how long ago this page last read its inputs, on
//              performance.now()
//
// Every one of those is either an identifier or a difference between two
// readings of this machine's own monotonic clock. No absolute time crosses the
// wire, so the browser and the car never have to agree on what time it is.
//
// age_ms is deliberately the age of the input sample, not the age of the
// value. A throttle held steady for ten seconds is ten seconds old as a value
// and a few milliseconds old as a sample, and it is the sample that says
// whether the command still reflects the operator's intent. The case this
// catches is the poll loop stopping while the send timer keeps running, a
// backgrounded tab or a stalled main thread, where the page would otherwise go
// on re-sending the last throttle it ever read with nobody at the controls.
const DRIVE_TICK_MS = 50;
const DRIVE_CLIENT_ID = `p${Math.floor(Math.random() * 1e9).toString(36)}`;
let inputSampledAt = 0;
let driveInFlight = false;
let driveQueued = false;

// noteInputSample is called by whatever just read the operator: the poll loop
// on every frame, and each on-screen control as it changes.
function noteInputSample() {
  inputSampledAt = performance.now();
}

// sendDriveIfNeeded is the only caller that may decide not to send. Every
// other path in the page sends unconditionally, because it is reacting to
// something the operator just did.
function sendDriveIfNeeded() {
  if (state.auto) return;
  const command = scaledCommand();
  const now = performance.now();
  const plan = planDriveSend(state.driveSend, command, now);
  if (!plan.send) return;

  // One drive POST at a time. Firing a heartbeat into a request that is still
  // outstanding is how the queue builds in the first place. A command that
  // asks for nothing is exempt: a stop must never wait behind a throttle.
  if (driveInFlight && !isZeroDriveCommand(command)) {
    driveQueued = true;
    return;
  }
  sendDrive(command);
}

async function sendDrive(command) {
  if (state.auto) return;
  const base = command || scaledCommand();
  const now = performance.now();
  state.driveSend = noteDriveSent(state.driveSend, base, now);
  state.driveSeq += 1;
  const payload = {
    ...base,
    client_id: DRIVE_CLIENT_ID,
    seq: state.driveSeq,
    age_ms: Math.max(0, Math.round(now - inputSampledAt)),
  };
  state.lastDrivePayload = payload;
  driveInFlight = true;
  try {
    await postJson("/api/drive", payload);
    state.lastCommand = base;
    state.lastDriveOkAt = Date.now();
    renderCommandReadout();
  } catch {
    setConnection(false, "Command offline");
  } finally {
    driveInFlight = false;
    if (driveQueued) {
      driveQueued = false;
      sendDriveIfNeeded();
    }
  }
}

// applyControlState performs every mode change. It routes armed, auto
// and gamepadEnabled through nextControlState in gamepad.js, which never
// switches the pad off: the operator must not lose the stop button as a
// side effect of changing mode.
//
// Three other places write one of these three fields, and none of them
// is a mode change:
//   refreshStatus mirrors state.auto from /api/status, because auto can
//     be engaged from outside this page by the voice control app
//   the Manual checkbox handler records what the operator just ticked
//   the Xbox checkbox handler records what the operator just ticked
// Nothing else may write them. In particular no mode change may clear
// gamepadEnabled: tests/web/wiring.test.mjs loads this file and runs every
// mode change against the checkbox to prove the pad survives all of them.
function applyControlState(transition) {
  const next = nextControlState(transition, state);
  state.armed = next.armed;
  state.auto = next.auto;
  state.gamepadEnabled = next.gamepadEnabled;
  els.arm.checked = next.armed;
  els.auto.checked = next.auto;
  els.gamepad.checked = next.gamepadEnabled;
  // A deliberate new mode command supersedes an earlier unresolved stop.
  // hardStop reopens its own pending record right after this call.
  state.stopPending = null;
  state.stopUnconfirmed = false;
}

function autoPayload(enabled) {
  return {
    enabled,
    speed: finiteNonNegative(els.autoSpeedInput.value, Number(els.autoSpeed.value) || 0),
    stop_distance: Number(els.stopRange.value) / 100,
    avoid_distance: Number(els.avoidRange.value) / 100,
    clear_distance: Number(els.clearRange.value) / 100,
  };
}

// setAuto is a deliberate mode change: the Auto Nav checkbox, the Y button, or
// the arm checkbox turning auto off. Only a mode change may supersede an
// outstanding stop, which is what applyControlState does on the way through.
async function setAuto(enabled) {
  applyControlState(enabled ? "autoOn" : "autoOff");
  if (enabled) {
    state.left = { x: 0, y: 0 };
  }
  try {
    await postJson("/api/auto", autoPayload(enabled));
  } catch {
    setConnection(false, "Auto offline");
  }
  updateReadouts();
}

// pushAutoParameters re-sends the tuning values to a run that is already going,
// from a slider or a D-pad nudge. It is not a mode change and must never be
// routed through setAuto: shouldPushAutoParameters in gamepad.js says when it
// is allowed, and it is not allowed while a stop is outstanding.
async function pushAutoParameters() {
  if (!shouldPushAutoParameters(state)) return;
  try {
    await postJson("/api/auto", autoPayload(true));
  } catch {
    setConnection(false, "Auto offline");
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// hardStop zeroes the local drive immediately, which is safety positive,
// but it does not get to call the car stopped until something says so.
// The retry and confirmation policy lives in planStopAttempt and
// controlModeText in gamepad.js so it can be tested; this function only
// carries it out. If the retries run out, state.stopPending stays set so
// a later /api/status poll can still confirm the stop, and the readout
// stays visibly unresolved until one of them does.
async function hardStop() {
  const pending = { armed: state.armed, auto: state.auto };
  applyControlState("hardStop");
  state.left = { x: 0, y: 0 };
  state.stopPending = pending;
  state.stopUnconfirmed = false;
  updateReadouts();

  for (let attempt = 1; ; attempt += 1) {
    let ok = true;
    try {
      await postJson("/api/stop", {});
    } catch {
      ok = false;
    }
    const plan = planStopAttempt(attempt, ok);
    if (plan.outcome === "confirmed") {
      state.stopPending = null;
      state.stopUnconfirmed = false;
      break;
    }
    if (plan.outcome === "unconfirmed") {
      state.stopUnconfirmed = true;
      setConnection(false, "STOP UNCONFIRMED, the car may still be moving");
      break;
    }
    setConnection(false, `Stop failed, retry ${attempt + 1} of ${STOP_MAX_ATTEMPTS}`);
    updateReadouts();
    await sleep(plan.delayMs);
  }
  updateReadouts();
}

async function startManual() {
  applyControlState("startManual");
  state.left = { x: 0, y: 0 };
  try {
    await postJson("/api/start", {});
    await sendDrive();
  } catch {
    setConnection(false, "Start failed");
  }
  updateReadouts();
}

function updateReadouts() {
  syncSpeedInputs();
  els.speedLabel.textContent = `${finiteNonNegative(els.speedInput.value, 0).toFixed(2)} m/s`;
  els.steerLabel.textContent = `${els.steer.value}%`;
  els.autoSpeedLabel.textContent = `${finiteNonNegative(els.autoSpeedInput.value, 0).toFixed(2)} m/s`;
  els.stopRangeLabel.textContent = `${(Number(els.stopRange.value) / 100).toFixed(2)} m`;
  els.avoidRangeLabel.textContent = `${(Number(els.avoidRange.value) / 100).toFixed(2)} m`;
  els.clearRangeLabel.textContent = `${(Number(els.clearRange.value) / 100).toFixed(2)} m`;
  els.leftReadout.textContent = `${(-state.left.y).toFixed(2)} / ${state.left.x.toFixed(2)}`;
  // controlModeText in gamepad.js is the only renderer of this readout,
  // so an unresolved stop cannot be painted over as a calm disarmed
  // state by a second code path.
  els.modeValue.textContent = controlModeText(state);
  els.modeValue.classList.toggle("unresolved", Boolean(state.stopUnconfirmed));
  updateGamepadText();
  state.lastCommand = scaledCommand();
  renderCommandReadout();
}

// The Command line has one renderer and commandReadoutText owns the choice of
// source, so the local manual command can no longer be painted over the top of
// what the server says it is publishing during an autonomous run.
function renderCommandReadout() {
  els.commandValue.textContent = commandReadoutText(state.auto, state.lastCommand, state.lastPublished);
}

function finiteNonNegative(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : fallback;
}

function widenRangeToValue(range, value) {
  if (!Number.isFinite(value)) return;
  const currentMax = Number(range.max) || 0;
  if (value > currentMax) range.max = String(Math.ceil(value));
}

function syncSpeedInputs() {
  const manual = finiteNonNegative(els.speedInput.value, Number(els.speed.value) || 0);
  const auto = finiteNonNegative(els.autoSpeedInput.value, Number(els.autoSpeed.value) || 0);
  widenRangeToValue(els.speed, manual);
  widenRangeToValue(els.autoSpeed, auto);
  els.speed.value = String(manual);
  els.autoSpeed.value = String(auto);
  els.speedInput.value = manual.toFixed(2);
  els.autoSpeedInput.value = auto.toFixed(2);
}

function setSpeedFromRange(kind) {
  if (kind === "manual") {
    els.speedInput.value = Number(els.speed.value).toFixed(2);
  } else {
    els.autoSpeedInput.value = Number(els.autoSpeed.value).toFixed(2);
  }
  updateReadouts();
  if (kind === "auto") pushAutoParameters();
}

function setSpeedFromNumber(kind) {
  const input = kind === "manual" ? els.speedInput : els.autoSpeedInput;
  const range = kind === "manual" ? els.speed : els.autoSpeed;
  const value = finiteNonNegative(input.value, Number(range.value) || 0);
  widenRangeToValue(range, value);
  range.value = String(value);
  input.value = value.toFixed(2);
  updateReadouts();
  if (kind === "auto") pushAutoParameters();
}

// setSpeedValue and setSteerScale apply an already-clamped value from the
// gamepad reducer to both the slider and its number field, so the visible
// control and the value the car uses never diverge.
function setSpeedValue(kind, value) {
  const range = kind === "manual" ? els.speed : els.autoSpeed;
  const input = kind === "manual" ? els.speedInput : els.autoSpeedInput;
  range.value = String(value);
  input.value = value.toFixed(2);
  updateReadouts();
  if (kind === "auto") pushAutoParameters();
}

function setSteerScale(value) {
  els.steer.value = String(value);
  updateReadouts();
}

function setConnection(ok, text) {
  state.lastStatusOk = ok;
  els.statusDot.className = ok ? "dot good" : "dot bad";
  els.statusText.textContent = text;
}

function updateGamepadText() {
  if (state.gamepadLastText) {
    els.gamepadValue.textContent = state.gamepadLastText;
  } else if (!state.gamepadEnabled && state.gamepadIndex !== null) {
    els.gamepadValue.textContent = "Seen";
  } else if (!state.gamepadEnabled) {
    els.gamepadValue.textContent = "Off";
  } else if (state.gamepadIndex === null) {
    els.gamepadValue.textContent = "Waiting";
  } else {
    els.gamepadValue.textContent = state.gamepadName ? `On ${state.gamepadIndex}` : "On";
  }
}

// The camera gallery ========================================================
//
// One tile per entry in the feed registry the car reports in /api/status, all
// live at once. Nothing here knows what the feeds are: this hardware publishes
// depth and rgb and no infrared or stereo image at all, and if that changes
// the registry changes with it and the gallery follows with no edit here.
//
// Tiles are kept, not rebuilt. Each one holds an open MJPEG connection, so
// recreating them on the 750 ms status poll would reopen every stream on the
// car eighty times a minute, which is exactly the traffic that starved the
// server of request threads earlier today. A poll updates a tile's text and
// its state class; only an error, the periodic refresh or the View button
// re-points a src.
const cameraTiles = new Map();

// A monotonic cache buster, seeded from the wall clock so a reload cannot
// reuse the previous page's URLs. A counter rather than Date.now() at the call
// site because two reconnects inside the same millisecond would otherwise
// produce the same URL, and the browser may then not reopen the stream at all.
let feedStreamSerial = Date.now();

// Which tile the periodic refresh reaches next.
let periodicFeedIndex = -1;

function feedStreamUrl(feed) {
  feedStreamSerial += 1;
  return `${feed.path}?r=${feedStreamSerial}`;
}

function createCameraTile(feed, index) {
  const root = document.createElement("figure");
  root.classList.add("tile");
  const bar = document.createElement("figcaption");
  bar.classList.add("tile-bar");
  const label = document.createElement("span");
  label.classList.add("tile-label");
  const status = document.createElement("span");
  status.classList.add("tile-state");
  bar.appendChild(label);
  bar.appendChild(status);
  const img = document.createElement("img");
  img.classList.add("tile-image");
  img.alt = `${feed.label || feed.id} camera feed`;
  root.appendChild(bar);
  root.appendChild(img);

  const tile = { root, img, label, state: status, feed, index, reconnectPending: false };
  root.addEventListener("click", () => toggleExpandedFeed(feed.id));
  img.addEventListener("error", () => scheduleFeedReconnect(feed.id));
  img.src = feedStreamUrl(feed);
  return tile;
}

function renderCameraGallery(feeds) {
  const list = Array.isArray(feeds) ? feeds.filter((feed) => feed && feed.id && feed.path) : [];
  const wanted = new Set(list.map((feed) => feed.id));
  for (const [id, tile] of [...cameraTiles]) {
    if (wanted.has(id)) continue;
    els.cameraGallery.removeChild(tile.root);
    cameraTiles.delete(id);
  }
  list.forEach((feed, index) => {
    let tile = cameraTiles.get(feed.id);
    if (!tile) {
      tile = createCameraTile(feed, index);
      cameraTiles.set(feed.id, tile);
      els.cameraGallery.appendChild(tile.root);
    }
    tile.feed = feed;
    tile.index = index;
    tile.label.textContent = feed.label || feed.id;
    // cameraFeedState in gamepad.js is the only place that decides what a feed
    // is doing, so a stale feed cannot be painted as a live one here.
    const view = cameraFeedState(feed);
    tile.state.textContent = view.text;
    for (const level of ["live", "stale", "waiting"]) {
      tile.root.classList.toggle(level, level === view.level);
    }
  });
  // An expanded view of a feed that is no longer reported is a blank screen,
  // so the gallery takes back over.
  if (state.expandedFeed && !wanted.has(state.expandedFeed)) state.expandedFeed = null;
  renderGalleryLayout();
}

function renderGalleryLayout() {
  const expanded = state.expandedFeed;
  els.cameraGallery.classList.toggle("expanded", Boolean(expanded));
  els.galleryBack.classList.toggle("hidden", !expanded);
  for (const [id, tile] of cameraTiles) tile.root.classList.toggle("is-expanded", id === expanded);
  const count = cameraTiles.size;
  const live = state.cameraFeeds.filter((feed) => feed && feed.ok).length;
  els.galleryStats.textContent = count
    ? `${count} feed${count === 1 ? "" : "s"}, ${live} live`
    : "no feeds reported";
}

// setExpandedFeed is the only writer of state.expandedFeed. An id with no tile
// behind it collapses to the gallery rather than showing nothing.
function setExpandedFeed(id) {
  state.expandedFeed = id && cameraTiles.has(id) ? id : null;
  renderGalleryLayout();
}

function toggleExpandedFeed(id) {
  setExpandedFeed(state.expandedFeed === id ? null : id);
}

function reconnectFeed(id) {
  const tile = cameraTiles.get(id);
  if (!tile) return;
  tile.img.src = feedStreamUrl(tile.feed);
}

// scheduleFeedReconnect is the only path from a broken stream back to a live
// one, and it holds the two rules that keep several tiles from becoming a
// burst: one pending retry per tile, and a delay carrying that tile's offset
// so tiles that failed together do not come back on the same instant.
function scheduleFeedReconnect(id) {
  const tile = cameraTiles.get(id);
  if (!tile || tile.reconnectPending) return;
  tile.reconnectPending = true;
  setTimeout(() => {
    const current = cameraTiles.get(id);
    if (!current) return;
    current.reconnectPending = false;
    reconnectFeed(id);
  }, feedReconnectDelayMs(tile.index));
}

function reconnectAllFeeds() {
  for (const id of cameraTiles.keys()) scheduleFeedReconnect(id);
}

// The periodic refresh moves one tile per tick rather than all of them, so the
// interval never produces a burst of its own.
function reconnectNextFeed() {
  const ids = [...cameraTiles.keys()];
  periodicFeedIndex = nextPeriodicFeedIndex(periodicFeedIndex, ids.length);
  if (periodicFeedIndex === null) return;
  reconnectFeed(ids[periodicFeedIndex]);
}

function drawLidar(lidar) {
  const canvas = els.lidarCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#080a09";
  ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height * 0.78;
  const maxRange = 3.0;
  ctx.strokeStyle = "#25312c";
  ctx.lineWidth = 1;
  for (let r = 1; r <= 3; r += 1) {
    ctx.beginPath();
    ctx.arc(cx, cy, (r / maxRange) * height * 0.58, Math.PI, 0);
    ctx.stroke();
  }

  ctx.fillStyle = "#58c897";
  for (const point of lidar.points || []) {
    const rad = (point.a - 90) * Math.PI / 180;
    const range = Math.min(point.r || 0, maxRange);
    const scale = (range / maxRange) * height * 0.58;
    const x = cx + Math.cos(rad) * scale;
    const y = cy + Math.sin(rad) * scale;
    ctx.fillRect(x - 2, y - 2, 4, 4);
  }

  ctx.fillStyle = "#eef2ef";
  ctx.beginPath();
  ctx.moveTo(cx, cy - 14);
  ctx.lineTo(cx - 10, cy + 12);
  ctx.lineTo(cx + 10, cy + 12);
  ctx.closePath();
  ctx.fill();
}

// The Controller panel ======================================================
//
// The operator has a genuine Xbox pad, a page that reports nothing, and no
// devtools console. Everything below exists so the next failure is legible
// from the screen. The wording lives in gamepad.js, tested; this file only
// assigns it to elements.

function controllerEnvironment() {
  return {
    hasGetGamepads: typeof navigator.getGamepads === "function",
    isSecureContext: Boolean(window.isSecureContext),
    origin: (window.location && window.location.origin) || "",
  };
}

function setNoticeLevel(element, level) {
  element.classList.toggle("bad", level === "error");
  element.classList.toggle("warn", level === "warn");
  element.classList.toggle("good", level === "ok");
}

function renderControllerPanel() {
  const notice = gamepadContextNotice(controllerEnvironment());
  els.padContext.textContent = `${notice.headline}. ${notice.detail}`;
  setNoticeLevel(els.padContext, notice.level);

  els.padCount.textContent = String(state.padLines.length);
  els.padList.textContent = state.padLines.length
    ? state.padLines.join("\n")
    : "The browser reports no pads in any slot.";
  els.padSelection.textContent = state.padSelectionText;
  setNoticeLevel(els.padSelection, state.padSelectionLevel);

  renderHidPanel();

  els.padSource.textContent = state.padSourceText;
  els.padAxes.textContent = state.padAxesText;
  els.padButtons.textContent = state.padButtonsText;
  els.padLog.textContent = state.gamepadLog.length
    ? state.gamepadLog.map(gamepadLogLine).join("\n")
    : "No controller actions yet.";
  els.padDrive.textContent = drivePayloadText(state.lastDrivePayload);
  els.padResponse.textContent = agoText(state.lastDriveOkAt, Date.now());
}

// The WebHID fallback =======================================================
//
// navigator.getGamepads() returns nothing on the operator's machine, measured
// in three browsers and on a page with none of our code on it, while hidutil
// describes the pad correctly and Steam reads it. WebHID reaches the same
// hardware down a different permission path, so this is a second way in rather
// than a fix for the first one: the Gamepad API path above is untouched and
// keeps priority whenever it lists a pad.
//
// Nothing here decides anything. decodeXboxHidReport and hidPadSnapshot in
// gamepad.js turn a report into the snapshot shape computeGamepadStep already
// consumes, and the same reducer, the same dispatcher and the same loss path
// carry it from there. There is no second mapping, no second deadzone and no
// second stop rule to keep in step with the first.
let hidDevice = null;
// Which device object the report listener is attached to, rather than whether one
// is attached at all. A device object outlives its own disconnect and so does a
// listener on it, so a pad replugged twice would otherwise end up decoding every
// report two or three times over.
let hidListeningOn = null;

function hidSupported() {
  return Boolean(navigator.hid && typeof navigator.hid.requestDevice === "function");
}

function hidStatusView() {
  return {
    hasWebHid: state.hid.supported,
    deviceName: state.hid.deviceName,
    open: state.hid.open,
    lastReportAt: state.hid.lastReportAt,
    now: performance.now(),
    reportLength: state.hid.reportLength,
    error: state.hid.error,
    decodeReason: state.hid.decodeReason,
  };
}

function renderHidPanel() {
  const notice = hidStatusNotice(hidStatusView());
  els.hidStatus.textContent = `${notice.headline}. ${notice.detail}`;
  setNoticeLevel(els.hidStatus, notice.level);
  // A button that opens no chooser must look unpressable. The operator has
  // already lost hours to a control that appeared to work and did nothing.
  els.hidConnect.disabled = !state.hid.supported;
  // The box opens on the toggle, and on its own for a report the decoder
  // refused: that is precisely the report the operator needs to read, and
  // making them find a checkbox first is friction at the worst moment.
  //
  // The hex is only built while the box is open. This runs on every animation
  // frame with the rest of the panel, and formatting a dump nobody is looking
  // at sixty times a second is work the drive loop can do without.
  const showRaw = state.hid.debug || Boolean(state.hid.decodeReason);
  els.hidRawBox.classList.toggle("hidden", !showRaw);
  if (showRaw) els.hidRaw.textContent = formatHidReportBytes(state.hid.reportBytes);
}

// handleHidInputReport is the only writer of the decoded snapshot. It decodes
// on arrival rather than on the frame that reads it, so the raw bytes on screen
// and the snapshot the reducer sees are always the same report.
function handleHidInputReport(event) {
  const bytes = hidReportBytes(event && event.data);
  const decoded = decodeXboxHidReport(bytes);
  state.hid.reportBytes = bytes;
  state.hid.reportId = Number(event && event.reportId) || 0;
  state.hid.reportLength = decoded.length;
  state.hid.decodeReason = decoded.ok ? "" : decoded.reason;
  // A report that would not decode normalises to an empty snapshot, so a
  // decoder that has lost the thread zeroes the drive rather than leaving the
  // last values it did understand on the wire.
  state.hid.snapshot = hidPadSnapshot(decoded);
  state.hid.lastReportAt = performance.now();
}

// openHidDevice is idempotent, which matters because the operator may press
// Connect controller again on a device the page already holds. Opening twice
// throws, and listening twice would decode every report two times.
async function openHidDevice(device) {
  if (!device) return;
  hidDevice = device;
  state.hid.deviceName = describeHidDevice(device);
  try {
    if (!device.opened) await device.open();
    if (hidListeningOn !== device) {
      device.addEventListener("inputreport", handleHidInputReport);
      hidListeningOn = device;
    }
    state.hid.open = true;
    state.hid.error = "";
  } catch (error) {
    state.hid.open = false;
    state.hid.error = (error && error.message) || String(error);
  }
  renderHidPanel();
}

// connectHid runs from the click and calls requestDevice as its first act.
// Chrome's user gesture requirement is strict: anything awaited before this
// point spends the gesture and the chooser never opens.
async function connectHid() {
  if (!hidSupported()) return;
  try {
    const devices = await navigator.hid.requestDevice({ filters: [{ vendorId: HID_XBOX_VENDOR_ID }] });
    const chosen = selectHidDevice(devices);
    // A dismissed chooser is not a failure and must not overwrite what the page
    // already knows: the panel goes on saying whatever it said before.
    if (!chosen) {
      renderHidPanel();
      return;
    }
    await openHidDevice(chosen);
  } catch (error) {
    state.hid.error = `the chooser failed: ${(error && error.message) || error}`;
    renderHidPanel();
  }
}

// restoreHidDevice is why the operator authorises once rather than on every
// reload: a device already granted comes back from getDevices() with no gesture
// and no chooser.
async function restoreHidDevice() {
  if (!state.hid.supported || typeof navigator.hid.getDevices !== "function") return;
  try {
    const chosen = selectHidDevice(await navigator.hid.getDevices());
    if (chosen) await openHidDevice(chosen);
  } catch (error) {
    state.hid.error = `could not list granted devices: ${(error && error.message) || error}`;
    renderHidPanel();
  }
}

// A WebHID device that vanishes mid drive leaves nobody holding the stop
// button, exactly as a vanished Gamepad API pad does, so it goes through the
// same applyPadLoss path and gets the same hard stop. The pad state is cleared
// with it, so a button held as the pad went away cannot count as a fresh press
// when it comes back.
function handleHidDisconnect(device) {
  if (!hidDevice || (device && device !== hidDevice)) return;
  state.hid.open = false;
  state.hid.lastReportAt = 0;
  state.hid.snapshot = emptyPadSnapshot();
  state.gamepadPadState = { buttons: [] };
  state.padAxesText = "none";
  state.padButtonsText = "none";
  renderHidPanel();
  applyPadLoss(true);
}

// findGamepad now records why it chose what it chose. selectGamepad in
// gamepad.js holds the rules; the only state written here is the sticky index
// and the strings the panel paints.
function findGamepad() {
  const pads = typeof navigator.getGamepads === "function" ? Array.from(navigator.getGamepads()) : [];
  const selection = selectGamepad(pads, state.gamepadIndex);
  state.padLines = pads.filter(Boolean).map(describeGamepad);
  const notice = gamepadSelectionNotice(selection);
  state.padSelectionText = notice.text;
  state.padSelectionLevel = notice.level;
  if (selection.pad) {
    state.gamepadIndex = selection.index;
    state.gamepadName = selection.pad.id || "";
  }
  return selection.pad;
}

function buttonName(index) {
  return GAMEPAD_BUTTONS[index] || `B${index}`;
}

function roundedAxes(pad) {
  return Array.from(pad.axes || []).map((value) => Math.round((Number(value) || 0) * 1000) / 1000);
}

function pressedButtons(pad) {
  return Array.from(pad.buttons || [])
    .map((button, index) => button && button.pressed
      ? { index, name: buttonName(index), value: Math.round((Number(button.value) || 0) * 1000) / 1000 }
      : null)
    .filter(Boolean);
}

// gamepadPostInFlight is the same guard driveInFlight and statusInFlight are:
// reportGamepad already throttles itself to 10/s, but a hung post at that
// rate is a new connection stacked every 100 ms with nothing to stop it,
// exactly like the status poll. The throttle stays; this only skips the POST
// itself when one is still outstanding, so the on-screen readout below still
// updates every frame even while a slow post is in flight.
let gamepadPostInFlight = false;

function reportGamepad(pad) {
  const now = performance.now();
  if (now - state.gamepadLastReportAt < 100) return;
  state.gamepadLastReportAt = now;

  const pressed = pressedButtons(pad);
  const axes = roundedAxes(pad);
  const activeAxes = axes
    .map((value, index) => Math.abs(value) >= 0.12 ? `A${index}:${value.toFixed(2)}` : "")
    .filter(Boolean);

  if (pressed.length) {
    state.gamepadLastText = pressed.map((button) => `${button.name}[${button.index}]`).join(" ");
  } else if (activeAxes.length) {
    state.gamepadLastText = activeAxes.slice(0, 3).join(" ");
  } else {
    state.gamepadLastText = state.gamepadEnabled ? "On" : "Seen";
  }
  updateGamepadText();

  if (gamepadPostInFlight) return;
  gamepadPostInFlight = true;
  postJson("/api/gamepad", {
    enabled: state.gamepadEnabled,
    armed: state.armed,
    auto: state.auto,
    index: pad.index,
    id: pad.id || "",
    mapping: pad.mapping || "",
    buttons: pad.buttons ? pad.buttons.length : 0,
    axes,
    pressed,
  }).catch(() => {}).finally(() => { gamepadPostInFlight = false; });
}

// applyGamepadAction is the whole dispatcher: every decision was already
// made by the pure reducer, so each branch is one call to an existing
// page function. Keep it that way, no logic here.
function applyGamepadAction(action) {
  // Every dispatched action is logged before it is carried out, so an action
  // whose handler throws or hangs still shows on screen as having fired.
  state.gamepadLog = pushGamepadLog(state.gamepadLog, { at: Date.now(), label: describeGamepadAction(action) });
  if (action.type === "hardStop") hardStop();
  else if (action.type === "startManual") startManual();
  else if (action.type === "toggleAuto") setAuto(action.enabled);
  else if (action.type === "expandFeed") setExpandedFeed(action.id);
  else if (action.type === "reconnectCamera") reconnectAllFeeds();
  else if (action.type === "nudgeManualSpeed") setSpeedValue("manual", action.value);
  else if (action.type === "nudgeAutoSpeed") setSpeedValue("auto", action.value);
  else if (action.type === "nudgeSteerScale") setSteerScale(action.value);
}

// applyPadLoss is the one place a vanished pad is acted on, and both observers
// of a vanished pad call it: the poll loop frame that finds none, and the
// gamepaddisconnected event. computeMissingPadStep in gamepad.js holds the
// policy, so the two paths cannot drift into different rules the way they had.
function applyPadLoss(hadPad) {
  const missing = computeMissingPadStep({
    hadPad,
    gamepadEnabled: state.gamepadEnabled,
    armed: state.armed,
    auto: state.auto,
  });
  for (const action of missing.actions) applyGamepadAction(action);
  if (missing.drive) {
    state.left = missing.drive.left;
    updateReadouts();
    sendDrive();
  }
}

function pollGamepad() {
  requestAnimationFrame(pollGamepad);
  // This frame is the input sample every outgoing command is aged against. If
  // this loop stops, age_ms climbs and the car stops trusting what the send
  // timer keeps posting, which is the behavior we want from a page that has
  // stopped reading its controller.
  noteInputSample();
  const pad = findGamepad();
  // chooseInputSource in gamepad.js holds the priority rule, and there is only
  // one: a listed pad always wins. WebHID is consulted only on a frame where the
  // browser lists no pad at all.
  const choice = chooseInputSource({
    pad: Boolean(pad),
    hidOpen: state.hid.open,
    hidLastReportAt: state.hid.lastReportAt,
    now: performance.now(),
  });
  state.padSourceText = inputSourceLabel(choice);

  if (choice.source === "none") {
    // A frame with no source at all is a disconnect observed from the poll loop.
    // gamepaddisconnected is not guaranteed to fire, and returning here
    // without zeroing would leave the sender POSTing the last stick values
    // forever, so the server watchdog never trips.
    const hadPad = state.gamepadIndex !== null;
    state.gamepadIndex = null;
    state.gamepadName = "";
    state.gamepadLastText = "";
    state.gamepadPadState = { buttons: [] };
    state.padAxesText = "none";
    state.padButtonsText = "none";
    updateGamepadText();
    applyPadLoss(hadPad);
    renderControllerPanel();
    return;
  }

  let padSnapshot;
  if (choice.source === "gamepad") {
    reportGamepad(pad);
    padSnapshot = {
      axes: Array.from(pad.axes || []),
      buttons: Array.from(pad.buttons || []).map((button) => ({
        pressed: Boolean(button && button.pressed),
        value: Number(button && button.value) || 0,
      })),
    };
  } else if (choice.silent) {
    // An open device that has gone quiet drives an empty snapshot, which the
    // reducer turns into a zeroed command with no actions. Silence is not the
    // same evidence as a disconnect, so it zeroes the drive without disarming;
    // the disconnect event does the rest.
    padSnapshot = emptyPadSnapshot();
  } else {
    padSnapshot = state.hid.snapshot;
    // The same telemetry POST the Gamepad API path makes, so the car's own
    // record of the controller is filled in whichever way in the pad arrived.
    // index -1 and mapping webhid are how that record says which one it was.
    reportGamepad({
      index: -1,
      id: state.hid.deviceName || "WebHID device",
      mapping: "webhid",
      axes: padSnapshot.axes,
      buttons: padSnapshot.buttons,
    });
  }
  // The live axis and button readouts do not wait for the Xbox toggle. A pad
  // that moves these while the car does nothing tells the operator the pad is
  // fine and the toggle is off, which is a failure they have hit before.
  state.padAxesText = formatAxisValues(padSnapshot.axes);
  state.padButtonsText = formatPressedButtons(padSnapshot.buttons);
  const step = computeGamepadStep(padSnapshot, state.gamepadPadState, {
    gamepadEnabled: state.gamepadEnabled,
    auto: state.auto,
    armed: state.armed,
    manualSpeed: finiteNonNegative(els.speedInput.value, Number(els.speed.value) || 0),
    autoSpeed: finiteNonNegative(els.autoSpeedInput.value, Number(els.autoSpeed.value) || 0),
    steerScale: Number(els.steer.value) || 0,
    feedIds: state.feedIds,
    expandedFeed: state.expandedFeed,
  });
  state.gamepadPadState = step.nextPadState;

  for (const action of step.actions) applyGamepadAction(action);

  // computeGamepadStep already zeroed this drive if a hardStop fired on the
  // same frame, so there is no stale command left for this loop to discard.
  if (step.drive) {
    state.left = step.drive.left;
    updateReadouts();
  }
  renderControllerPanel();
}

// statusInFlight is the status poll's version of driveInFlight above: a status
// fetch that hangs must not get a second one stacked on top of it by the next
// 750 ms tick, the way the wedged server did in the field. Unlike a drive
// command, nothing ever needs to jump this queue, so the guard is a plain
// skip with no exception for it: the interval itself keeps running (below),
// only the fetch inside a tick that finds one already outstanding is skipped.
let statusInFlight = false;

async function refreshStatus() {
  if (statusInFlight) return;
  statusInFlight = true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const status = await fetch("/api/status", { cache: "no-store", signal: controller.signal }).then((r) => r.json());
    const cameras = Array.isArray(status.cameras) ? status.cameras : [];
    const control = status.control || {};
    const limits = control.limits || {};
    const lidar = status.lidar || {};
    const auto = status.auto || {};
    const navigation = status.navigation || {};
    const sectors = lidar.sectors || {};
    state.limits.maxLinearX = limits.max_linear_x || state.limits.maxLinearX;
    state.limits.maxSteeringY = limits.max_steering_y || state.limits.maxSteeringY;
    state.auto = Boolean(auto.enabled);
    els.auto.checked = state.auto;
    // A stop the page could not confirm for itself is closed out here, but
    // only on the server's own evidence that the mode the stop was issued to
    // end is no longer active. Until that evidence arrives the stop goes out
    // again on every poll: an autonomous run has no background sender that
    // would retry it, so a stop whose POST never landed would otherwise leave
    // the car driving itself with nothing but a red banner to show for it.
    const followUp = planStatusStopFollowUp(state, {
      auto: Boolean(auto.enabled),
      armed: Boolean((control.command || {}).enabled),
    });
    if (followUp.confirmed) {
      state.stopPending = null;
      state.stopUnconfirmed = false;
    } else if (followUp.reissue) {
      postJson("/api/stop", {}).catch(() => {});
    }
    if (state.stopUnconfirmed) {
      setConnection(false, "STOP UNCONFIRMED, the car may still be moving");
    } else {
      // Named from the server's own answer about which depth camera is
      // fitted, so this line does not go on claiming an HP60C after the
      // hardware has been swapped for something else.
      const depthOnline = navigation.depth_ok ? `${navigation.depth_source} depth online` : "Sensors pending";
      setConnection(true, lidar.ok ? "LiDAR online" : depthOnline);
    }
    state.cameraFeeds = cameras;
    state.feedIds = cameras.map((feed) => feed && feed.id).filter(Boolean);
    renderCameraGallery(cameras);
    els.autoReadyValue.textContent = navigation.ready ? "Ready" : navigation.reason || "Not ready";
    els.driverValue.textContent = `${control.cmd_vel_subscribers || 0} subs`;
    els.watchdogValue.textContent = `${(limits.cmd_timeout_s || 0.5).toFixed(2)} s`;
    els.lidarStats.textContent = lidar.ok ? `${lidar.finite_ranges || 0} pts` : "waiting";
    els.frontValue.textContent = meters(sectors.front && sectors.front.near_m);
    els.leftValue.textContent = meters(sectors.left && sectors.left.near_m);
    els.rightValue.textContent = meters(sectors.right && sectors.right.near_m);
    els.closestValue.textContent = meters(lidar.min_m);
    const autoRead = state.auto ? auto.decision : auto.preview;
    if (autoRead) {
      const depthText = Number.isFinite(autoRead.depth_obstacle_m) ? ` obstacle ${meters(autoRead.depth_obstacle_m)}` : "";
      const directionText = autoRead.lidar_direction ? ` ${autoRead.lidar_direction}` : "";
      els.autoValue.textContent = `${state.auto ? "" : "preview "}${autoRead.action || "disabled"}${directionText} ${meters(autoRead.front_m)}${depthText}`;
    }
    if (control.last_published) state.lastPublished = control.last_published;
    drawLidar(lidar);
    updateReadouts();
  } catch {
    // A hang never rejects on its own: this catch only runs because the
    // timeout above turned it into one. Without that, a wedged server left
    // this the only failure indicator, and it never fired.
    setConnection(false, "Remote offline");
  } finally {
    clearTimeout(timer);
    statusInFlight = false;
  }
}

makeJoystick(document.getElementById("leftStick"), (value) => {
  noteInputSample();
  state.left = value;
  updateReadouts();
  sendDrive();
});

els.arm.addEventListener("change", () => {
  noteInputSample();
  state.armed = els.arm.checked;
  if (state.armed && state.auto) {
    els.auto.checked = false;
    setAuto(false);
  }
  updateReadouts();
  sendDrive();
});

els.auto.addEventListener("change", () => {
  setAuto(els.auto.checked);
});

// Enabling the pad is orthogonal to mode. It must not disengage Auto Nav:
// an operator who ticks Xbox mid auto drive is reaching for the B button,
// not asking to end the run.
els.gamepad.addEventListener("change", () => {
  noteInputSample();
  state.gamepadEnabled = els.gamepad.checked;
  if (!state.gamepadEnabled) {
    state.left = { x: 0, y: 0 };
    sendDrive();
  }
  updateReadouts();
});

window.addEventListener("gamepadconnected", (event) => {
  state.gamepadIndex = event.gamepad.index;
  state.gamepadName = event.gamepad.id || "";
  state.gamepadLastText = "Seen";
  updateGamepadText();
});

window.addEventListener("gamepaddisconnected", (event) => {
  if (state.gamepadIndex === event.gamepad.index) {
    state.gamepadIndex = null;
    state.gamepadName = "";
    state.gamepadLastText = "";
    state.gamepadPadState = { buttons: [] };
  }
  updateGamepadText();
  // A pad that drops out mid drive leaves nobody holding the stop button. A
  // disconnect event is by definition a pad that was there and is not any
  // more, so hadPad is true, and this runs for any disconnect rather than only
  // the pad we were tracking: stopping a car nobody asked to stop is
  // recoverable, the other way is not.
  applyPadLoss(true);
});

// requestDevice is called from inside this handler, not from anything it
// awaits, because Chrome will not open the chooser without a live user gesture.
els.hidConnect.addEventListener("click", connectHid);

els.hidDebug.addEventListener("change", () => {
  state.hid.debug = els.hidDebug.checked;
  renderHidPanel();
});

// navigator.hid, not the device, is where connect and disconnect are announced,
// and a device object survives its own disconnect, so the events are the only
// notice the page gets that the pad has gone.
if (navigator.hid && typeof navigator.hid.addEventListener === "function") {
  navigator.hid.addEventListener("disconnect", (event) => {
    handleHidDisconnect(event && event.device);
  });
  navigator.hid.addEventListener("connect", (event) => {
    const device = event && event.device;
    // A pad that comes back is reopened without a chooser, because the grant is
    // still there. Only while nothing is open, so a live device is never
    // disturbed by another one being plugged in.
    if (!state.hid.open && selectHidDevice([device])) openHidDevice(device);
  });
}

els.start.addEventListener("click", startManual);
els.stop.addEventListener("click", hardStop);
els.speed.addEventListener("input", () => setSpeedFromRange("manual"));
els.speedInput.addEventListener("change", () => setSpeedFromNumber("manual"));
els.steer.addEventListener("input", updateReadouts);
els.autoSpeed.addEventListener("input", () => setSpeedFromRange("auto"));
els.autoSpeedInput.addEventListener("change", () => setSpeedFromNumber("auto"));
for (const slider of [els.stopRange, els.avoidRange, els.clearRange]) {
  slider.addEventListener("input", () => {
    updateReadouts();
    pushAutoParameters();
  });
}

window.addEventListener("pagehide", hardStop);
els.galleryBack.addEventListener("click", () => setExpandedFeed(null));
// DRIVE_TICK_MS only decides how finely the heartbeat and the zero repeats are
// timed. Every change is already sent by whichever handler caused it, so the
// tick is not the path a stick movement takes to the car.
// Recorded before the first render, so the panel never spends a frame claiming
// WebHID is missing on a browser that has it.
state.hid.supported = hidSupported();
noteInputSample();
setInterval(sendDriveIfNeeded, DRIVE_TICK_MS);
setInterval(refreshStatus, 750);
// One tile per tick, so the periodic refresh of N streams is N ticks apart
// rather than N connections at once.
setInterval(reconnectNextFeed, 15000);
pollGamepad();
updateReadouts();
renderControllerPanel();
renderGalleryLayout();
refreshStatus();
// Last, and deliberately not awaited: a granted device reopens itself on load
// so the operator authorises once, and a browser without WebHID at all simply
// leaves the connect button disabled with the panel saying why.
restoreHidDevice();
