const GAMEPAD_BUTTONS = [
  "A", "B", "X", "Y", "LB", "RB", "LT", "RT",
  "View", "Menu", "LS", "RS", "DUp", "DDown", "DLeft", "DRight", "Xbox",
];

function gamepadClamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function applyDeadzone(value, deadzone = 0.12) {
  if (!Number.isFinite(value) || Math.abs(value) < deadzone) return 0;
  const sign = Math.sign(value);
  return sign * gamepadClamp((Math.abs(value) - deadzone) / (1 - deadzone), 0, 1);
}

function gamepadButtonValue(button) {
  if (!button) return 0;
  return gamepadClamp(Number(button.value) || (button.pressed ? 1 : 0), 0, 1);
}

// The ranges below are what the reducer clamps nudges to, so a D-pad press can
// never push a value outside them and the dispatcher can apply the result
// without rechecking.
//
// The steering range is exactly the #steer slider in index.html. The speed
// range matches the #speed and auto speed sliders as they ship, but not
// always at runtime: widenRangeToValue in index.html raises a slider's max
// when the operator types a larger number into its companion number field,
// and the reducer does not follow it. Starting from a typed 5.00 m/s, a D-pad
// Up press therefore lands on 2.00: a jump down to this ceiling rather than a
// nudge. That is safe, because the result is only ever slower than what the
// operator had, and the displayed and sent values stay in step. It is simply
// not a nudge.
const GAMEPAD_SPEED_STEP = 0.05;
const GAMEPAD_SPEED_MIN = 0;
const GAMEPAD_SPEED_MAX = 2;
const GAMEPAD_STEER_STEP = 10;
const GAMEPAD_STEER_MIN = 10;
const GAMEPAD_STEER_MAX = 100;

// Speeds are carried on a 0.01 slider step, so nudges round to two decimals.
// Without this, 0.35 + 0.05 lands on 0.39999999999999997 and the drift
// compounds over a drive.
function nudgeSpeed(current, delta) {
  const base = Number.isFinite(Number(current)) ? Number(current) : 0;
  const raw = gamepadClamp(base + delta, GAMEPAD_SPEED_MIN, GAMEPAD_SPEED_MAX);
  return Math.round(raw * 100) / 100;
}

function nudgeSteer(current, delta) {
  const base = Number.isFinite(Number(current)) ? Number(current) : GAMEPAD_STEER_MIN;
  return Math.round(gamepadClamp(base + delta, GAMEPAD_STEER_MIN, GAMEPAD_STEER_MAX));
}

// The camera gallery =========================================================
//
// Every feed the car reports is on screen at once, so there is no "current
// source" to cycle any more. What X cycles now is which tile is expanded, and
// the cycle passes back through the gallery: ids in order, then null. A cycle
// that only ever went deeper would leave the pad with no way out of an
// expanded tile.
//
// The candidate ids come from the /api/status feed registry, so a feed the car
// does not have is not in the cycle and a feed added later joins it with no
// change here. Unlike the old source selector this does include feeds the car
// reports as stale: the tile is on screen either way, and looking closely at a
// feed that has stopped is a reasonable thing to want to do.
//
// An expanded feed that has vanished from the registry falls to the first one,
// because indexOf returns -1 and -1 + 1 is 0.
function nextExpandedFeed(current, ids) {
  const feeds = Array.isArray(ids) ? ids.filter(Boolean) : [];
  if (!feeds.length) return null;
  const next = feeds.indexOf(current) + 1;
  return next < feeds.length ? feeds[next] : null;
}

// cameraFeedState turns one registry entry into the state its tile paints.
// Three states, and the difference between them is the whole point of the
// badge: a stale feed leaves its last frame on screen, and a frozen frame with
// no label on it looks exactly like a live one of a car that is not moving.
//
//   live     fresh frames are arriving
//   stale    frames arrived and then stopped, with the age still climbing
//   waiting  no frame has ever arrived on this feed
//
// Absence is not a state here. A feed the car does not report is not in the
// registry, so it has no tile at all.
function cameraFeedState(feed) {
  const view = feed || {};
  // Number(null) is 0, and a null age_s is the server saying no frame has ever
  // arrived. Read through Number alone, a feed that has never produced
  // anything would report as stale by zero seconds, which is the freshest
  // reading there is.
  const raw = view.age_s;
  const age = raw === null || raw === undefined ? NaN : Number(raw);
  if (view.ok === true) return { level: "live", text: "live" };
  if (!Number.isFinite(age)) return { level: "waiting", text: "no frames yet" };
  return { level: "stale", text: `stale ${age.toFixed(1)} s` };
}

// Reconnect pacing.
//
// The page reopens a stream that errors, and reopens each stream periodically.
// With one view that was one reconnect; with a tile per feed it is one per
// tile, and several MJPEG connections opening at the same instant over a laggy
// link is the pressure that starved the server of request threads earlier
// today. Two rules keep the traffic flat:
//
//   an error driven retry waits its tile's own offset past the one second
//   base, so tiles that fail together do not come back together
//
//   the periodic refresh moves one tile per tick rather than all of them, so
//   the interval never produces a burst at all
const FEED_RECONNECT_BASE_MS = 1000;
const FEED_RECONNECT_STAGGER_MS = 400;

function feedReconnectDelayMs(index, baseMs = FEED_RECONNECT_BASE_MS, staggerMs = FEED_RECONNECT_STAGGER_MS) {
  const slot = Number(index);
  const offset = Number.isFinite(slot) && slot > 0 ? Math.floor(slot) : 0;
  const base = Number.isFinite(Number(baseMs)) ? Number(baseMs) : FEED_RECONNECT_BASE_MS;
  return base + offset * staggerMs;
}

function nextPeriodicFeedIndex(previous, count) {
  const total = Math.floor(Number(count) || 0);
  if (total <= 0) return null;
  const prev = Number.isFinite(Number(previous)) ? Math.floor(Number(previous)) : -1;
  const next = prev + 1;
  return next >= 0 && next < total ? next : 0;
}

// computeGamepadStep is a pure reducer: given a plain-object snapshot of the
// live Gamepad, the previous frame's pressed-state, and the relevant slice of
// UI state, it returns what the caller should do this frame. It never touches
// the DOM, fetch, or any global, so it can be unit tested without a browser.
//
// uiState carries:
//   gamepadEnabled  the Xbox toggle
//   auto, armed     current control mode
//   manualSpeed     m/s, the #speed slider value
//   autoSpeed       m/s, the auto speed slider value
//   steerScale      percent, the #steer slider value
//   feedIds         camera feed ids the status poll reports, in gallery order
//   expandedFeed    the feed currently expanded, or null for the gallery
//
// It returns { actions, drive, nextPadState }. The action types are:
//   hardStop         stop and disarm, from B or Menu, never gated by auto
//   startManual      arm manual control, from A
//   toggleAuto       {enabled} turn Auto Nav on or off, from Y
//   expandFeed       {id} expand this feed, or null for the gallery, from X
//   reconnectCamera  force every feed to reopen, from View
//   nudgeManualSpeed {value} clamped manual speed, from D-pad Up/Down
//   nudgeAutoSpeed   {value} clamped auto speed, from D-pad Right/Left
//   nudgeSteerScale  {value} clamped steering percent, from RB/LB
// Each nudge carries the already-clamped absolute `value`, so the dispatcher
// assigns it without re-deriving the range.
function computeGamepadStep(padSnapshot, prevPadState, uiState) {
  const prevButtons = (prevPadState && prevPadState.buttons) || [];

  // Only the pad's own on/off switch silences it entirely. Auto Nav gates the
  // drive axes and the arm button below, never the stop buttons: the operator
  // must always be able to stop a car that is driving itself.
  if (!uiState.gamepadEnabled) {
    return {
      actions: [],
      drive: null,
      nextPadState: { buttons: prevButtons.slice() },
    };
  }

  const buttons = (padSnapshot && padSnapshot.buttons) || [];
  const pressed = buttons.map((button) => Boolean(button && button.pressed));

  // rose is the rising edge test every discrete button goes through, so no
  // action can repeat while its button is held down.
  function rose(index) {
    return Boolean(pressed[index]) && !prevButtons[index];
  }

  const stopPressed = Boolean(pressed[1]) || Boolean(pressed[9]);
  const prevStopPressed = Boolean(prevButtons[1]) || Boolean(prevButtons[9]);
  const stopping = stopPressed && !prevStopPressed;

  const actions = [];
  if (stopping) actions.push({ type: "hardStop" });

  // Mode buttons yield to a stop pressed in the same frame. Engaging auto or
  // arming manual control immediately after a stop request would undo it.
  if (!stopping && rose(3)) actions.push({ type: "toggleAuto", enabled: !uiState.auto });
  if (!stopping && rose(0) && !uiState.auto && !uiState.armed) actions.push({ type: "startManual" });

  if (rose(2)) {
    const feeds = Array.isArray(uiState.feedIds) ? uiState.feedIds.filter(Boolean) : [];
    // With no feeds at all there is nothing to expand and nothing to return
    // to, so X does nothing rather than emitting a no-op the log would show as
    // an action the operator took.
    if (feeds.length) actions.push({ type: "expandFeed", id: nextExpandedFeed(uiState.expandedFeed, feeds) });
  }
  if (rose(8)) actions.push({ type: "reconnectCamera" });

  if (rose(12)) {
    actions.push({ type: "nudgeManualSpeed", value: nudgeSpeed(uiState.manualSpeed, GAMEPAD_SPEED_STEP) });
  }
  if (rose(13)) {
    actions.push({ type: "nudgeManualSpeed", value: nudgeSpeed(uiState.manualSpeed, -GAMEPAD_SPEED_STEP) });
  }
  // The auto speed nudges yield to a stop pressed in the same frame, for the
  // same reason the mode buttons do: on a frame where the operator asked for a
  // stop, nothing else touches the mode the stop is ending. The dispatcher has
  // its own guard for the window afterwards, shouldPushAutoParameters, because
  // the frame gate alone only covers the single frame the button went down.
  // Manual speed re-sends nothing to the car, so it stays live.
  if (!stopping && rose(15)) {
    actions.push({ type: "nudgeAutoSpeed", value: nudgeSpeed(uiState.autoSpeed, GAMEPAD_SPEED_STEP) });
  }
  if (!stopping && rose(14)) {
    actions.push({ type: "nudgeAutoSpeed", value: nudgeSpeed(uiState.autoSpeed, -GAMEPAD_SPEED_STEP) });
  }
  if (rose(5)) {
    actions.push({ type: "nudgeSteerScale", value: nudgeSteer(uiState.steerScale, GAMEPAD_STEER_STEP) });
  }
  if (rose(4)) {
    actions.push({ type: "nudgeSteerScale", value: nudgeSteer(uiState.steerScale, -GAMEPAD_STEER_STEP) });
  }

  let drive;
  if (stopping) {
    // A stop frame ends with a zeroed command, whatever the sticks and
    // triggers were doing when the button went down. The dispatcher used to
    // own this: it computed a hardStopped flag and discarded the live drive
    // itself. Deleting that flag left the suite green while a held throttle
    // survived the stop, so the decision lives here now and the dispatcher
    // only assigns what it is handed.
    drive = { left: { x: 0, y: 0 } };
  } else if (uiState.auto) {
    // Auto Nav owns the motors. Returning null tells the dispatcher to leave
    // the outgoing command alone rather than overwrite it with stick values.
    drive = null;
  } else if (!uiState.armed) {
    drive = { left: { x: 0, y: 0 } };
  } else {
    // The right stick is deliberately unread. Commanding angular_z at 1.0 for
    // two seconds produced an encoder delta of exactly zero on all four
    // channels while steering_y at 0.12 moved the car: this is an Ackermann
    // chassis with no in place turn, and the left stick already steers it. A
    // control that does nothing is worse than no control, because the operator
    // spends the drive believing it might work.
    const axes = (padSnapshot && padSnapshot.axes) || [];
    const steer = applyDeadzone(axes[0] || 0);
    const reverse = gamepadButtonValue(buttons[6]);
    const forward = gamepadButtonValue(buttons[7]);
    const driveValue = applyDeadzone(forward - reverse, 0.05);
    drive = { left: { x: steer, y: -driveValue } };
  }

  return { actions, drive, nextPadState: { buttons: pressed } };
}

// computeDisconnectStep decides what happens when the pad vanishes mid drive.
// A controller that drops out while the car is armed or driving itself leaves
// nobody holding the stop button, so the car stops.
function computeDisconnectStep(uiState) {
  const actions = [];
  if (uiState && (uiState.armed || uiState.auto)) actions.push({ type: "hardStop" });
  return { actions };
}

// computeMissingPadStep is the whole "the pad went away" step, and both places
// that observe a pad going away run it: the gamepaddisconnected listener and
// the poll loop frame that finds no pad. The event is not guaranteed to fire,
// and without the poll path state.left and state.right would keep the last
// stick values while the 120 ms sender POSTs them, so the server's command
// watchdog never trips and the car drives on with nobody holding the stop
// button. Running one function in both places is what stops the two paths from
// drifting into different rules, which is what they had done.
//
// uiState carries hadPad, whether a pad was present before, plus the same
// gamepadEnabled/armed/auto the reducer takes. The two gates do different jobs:
//   hadPad          the stop fires on the transition, not every frame at 60 Hz
//                   on a machine that never had a pad at all
//   gamepadEnabled  gates the zeroed drive only. With the Xbox toggle off the
//                   touch UI owns the sticks, and zeroing them under the
//                   operator's finger would break manual driving. It does not
//                   gate the stop: the toggle says whether the pad may drive
//                   the car, not whether its disappearance is a safety event.
// It returns { actions, drive }, where a non-null drive is the zeroed command
// the caller must apply.
function computeMissingPadStep(uiState) {
  const previous = uiState || {};
  if (!previous.hadPad) return { actions: [], drive: null };
  return {
    actions: computeDisconnectStep(previous).actions,
    drive: previous.gamepadEnabled ? { left: { x: 0, y: 0 } } : null,
  };
}

// The stop path must never present a state it has not confirmed. hardStop
// zeroes the local drive synchronously, which is safety positive, but it also
// clears armed and auto before POST /api/stop has answered. During Auto Nav
// the server's own loop is still driving, so a page that falls back to a calm
// "Disarmed" after a failed POST is telling the operator the car is safe while
// it is moving. A status field that lies is worse than no status field.
const STOP_MAX_ATTEMPTS = 3;
const STOP_RETRY_DELAY_MS = 150;

// planStopAttempt is the retry policy, expressed one attempt at a time so the
// caller holds no rules of its own. attempt is 1 based and ok says whether
// that attempt's POST resolved. The outcomes are:
//   confirmed    the stop landed, the page may show a stopped state
//   retry        send it again after delayMs
//   unconfirmed  attempts exhausted, the page must stay visibly unresolved
function planStopAttempt(attempt, ok) {
  if (ok) return { outcome: "confirmed", retry: false, delayMs: 0 };
  if ((Number(attempt) || 0) >= STOP_MAX_ATTEMPTS) return { outcome: "unconfirmed", retry: false, delayMs: 0 };
  return { outcome: "retry", retry: true, delayMs: STOP_RETRY_DELAY_MS };
}

// stopConfirmedByStatus lets an /api/status poll close out a stop the page
// could not confirm for itself. Only evidence counts: the server has to report
// the mode the stop was issued to end as no longer active. A stop issued
// during Auto Nav is confirmed by auto disengaged; a stop issued from manual
// control is confirmed by the driver command no longer enabled. Anything else
// leaves the page unresolved, which is the honest answer.
function stopConfirmedByStatus(pending, reported) {
  if (!pending) return false;
  const status = reported || {};
  if (pending.auto) return status.auto === false;
  if (pending.armed) return status.armed === false;
  // Nothing was live when the stop went out, so the server reporting both
  // channels quiet is all the confirmation there is to have. This still needs
  // the server to say so: auto can be engaged from outside the page between
  // two polls, and an absent field proves nothing.
  return status.auto === false && status.armed === false;
}

// planStatusStopFollowUp is what an /api/status poll does about a stop that is
// still outstanding, and it is the only automatic way out of the unresolved
// state. Two outcomes, never both:
//   confirmed  the server's own evidence says the mode the stop was issued to
//              end is no longer active, so the page may close the stop out
//   reissue    POST /api/stop again
// Reissuing matters most during Auto Nav. A manual stop self heals, because the
// 120 ms drive sender keeps posting enabled false and the server zeroes the
// command. An autonomous run has no such background path: the server only
// clears auto when an enabled drive command arrives. So a stop whose POST never
// landed used to leave the car driving itself for as long as the page stayed
// open, with only a red banner to show for it. Re-sending a stop the car has
// already carried out costs nothing.
function planStatusStopFollowUp(view, reported) {
  const state = view || {};
  if (!state.stopPending) return { confirmed: false, reissue: false };
  if (stopConfirmedByStatus(state.stopPending, reported)) return { confirmed: true, reissue: false };
  return { confirmed: false, reissue: true };
}

// shouldPushAutoParameters answers one question: may a slider or a D-pad nudge
// re-send the Auto Nav tuning values to the car right now?
//
// It may not while a stop is outstanding. The nudge path used to run through
// the same function as the Auto Nav checkbox, so it re-POSTed /api/auto with
// enabled true and cleared the stop bookkeeping on its way past. During the
// window after a stop the page could not confirm, the status poll has already
// restored state.auto from the server, so a single press of D-pad Right would
// re-command an autonomous run and wipe the red warning, while the operator
// believed they had only nudged a slider. Clearing that warning takes a
// deliberate act: the Y button, the Auto Nav checkbox, or a confirmed stop.
function shouldPushAutoParameters(view) {
  const state = view || {};
  if (!state.auto) return false;
  return !state.stopPending && !state.stopUnconfirmed;
}

// commandReadoutText renders the Drive panel's Command line. During Auto Nav
// the car is driven by the server's own loop, so the only honest source is what
// the server reports it last published; the page's local manual command is zero
// throughout and used to be painted over the top of it, so the readout said
// 0.00 while the car was moving. A readout that lies is worse than no readout.
function commandReadoutText(auto, localCommand, lastPublished) {
  const source = (auto && lastPublished) || localCommand || {};
  const linear = Number(source.linear_x) || 0;
  const steering = Number(source.steering_y) || 0;
  return `${linear.toFixed(2)} / ${steering.toFixed(2)}`;
}

// controlModeText is the only place that turns control state into the Drive
// panel readout, so there is exactly one renderer to keep honest. An
// unconfirmed stop never reads as "Disarmed".
function controlModeText(uiState) {
  const view = uiState || {};
  if (view.stopUnconfirmed) return "STOP UNCONFIRMED";
  if (view.stopPending) return "Stopping";
  if (view.auto) return "Auto Nav";
  if (view.armed) return "Manual Armed";
  return "Disarmed";
}

// nextControlState is the single place that decides armed/auto/gamepadEnabled
// for a mode change. Every branch carries gamepadEnabled through untouched:
// no mode change may switch off the pad, because the pad is how the operator
// stops the car. Turning the pad off is the operator's decision alone.
function nextControlState(transition, prev) {
  const previous = prev || {};
  const gamepadEnabled = Boolean(previous.gamepadEnabled);
  const armed = Boolean(previous.armed);
  switch (transition) {
    case "autoOn":
      return { armed: false, auto: true, gamepadEnabled };
    case "autoOff":
      // autoOff carries armed through on purpose, and the arm checkbox
      // handler in index.html depends on it: ticking Manual while Auto Nav is
      // running sets state.armed and then calls setAuto(false), so the arm it
      // just set has to survive this transition.
      //
      // The precondition is that armed can only be true here because the
      // operator armed manually, since autoOn clears it. Auto engaged from
      // outside the page breaks that precondition: the voice control app
      // posts /api/auto directly, and refreshStatus mirrors state.auto without
      // touching state.armed. An operator who was already armed when that
      // happened gets a live throttle back the moment they press Y, with no
      // explicit arm step. Do not change this branch without changing the arm
      // checkbox handler with it.
      return { armed, auto: false, gamepadEnabled };
    case "hardStop":
      return { armed: false, auto: false, gamepadEnabled };
    case "startManual":
      return { armed: true, auto: false, gamepadEnabled };
    default:
      return { armed, auto: Boolean(previous.auto), gamepadEnabled };
  }
}

// Controller diagnostics ====================================================
//
// The operator drives from a laptop with no devtools console open, and the
// controller has failed for at least two different reasons already: Chrome
// hides every pad outside a secure context, and something else besides. These
// functions turn what the browser reports into lines the Controller panel can
// paint. They are pure, so the panel's wording is testable and app.js only
// assigns strings to elements.

const GAMEPAD_LOG_LIMIT = 20;

// gamepadContextNotice explains the single most likely reason a working
// controller looks dead. On a plain HTTP LAN origin Chrome does not merely
// return an empty pad list, it can omit navigator.getGamepads altogether, so
// the insecure origin has to be named even when the missing property is what
// we detected. Reporting only "no Gamepad API" would send the operator hunting
// for a browser bug.
function gamepadContextNotice(env) {
  const view = env || {};
  const origin = view.origin || "this page";
  const insecure = `${origin} is not a secure context. The browser hides every gamepad from it. `
    + "Open the remote over http://localhost (an SSH or TCP forward to the car works) or over HTTPS.";
  if (!view.hasGetGamepads) {
    return {
      level: "error",
      headline: "navigator.getGamepads is missing",
      detail: view.isSecureContext
        ? `${origin} is a secure context but this browser exposes no Gamepad API, so no controller can ever be seen here.`
        : insecure,
    };
  }
  if (!view.isSecureContext) {
    return { level: "error", headline: "Insecure context", detail: insecure };
  }
  return {
    level: "ok",
    headline: "Secure context",
    detail: `${origin} is a secure context, so gamepads are visible to this page.`,
  };
}

// selectGamepad is the pad choice, lifted out of app.js so the panel can show
// what it rejected and why. It returns the chosen index and one `considered`
// entry per real pad, which is what makes a present but unmatched pad look
// different from no pad at all.
//
// Only two things get a pad rejected outright: an empty slot, which is not a
// pad and is not reported at all, and the browser saying connected is false.
// An unrecognised id is never a rejection, because refusing to drive with the
// operator's only pad and saying nothing is how we got here. `connected` is
// only disqualifying when it is explicitly false: some browsers and every test
// fake omit the field.
function selectGamepad(pads, preferredIndex) {
  const list = Array.isArray(pads) ? pads : [];
  const present = [];
  for (let slot = 0; slot < list.length; slot += 1) {
    const entry = list[slot];
    if (!entry) continue;
    present.push({ pad: entry, index: Number.isFinite(entry.index) ? entry.index : slot });
  }

  const usable = present.filter((item) => item.pad.connected !== false);
  const preferred = usable.find((item) => item.index === preferredIndex);
  const named = usable.find((item) => /xbox|controller|gamepad/i.test(item.pad.id || ""));
  const chosen = preferred || named || usable[0] || null;

  const considered = present.map((item) => {
    if (chosen && item.index === chosen.index) {
      return { index: item.index, id: item.pad.id || "", accepted: true, reason: "selected" };
    }
    if (item.pad.connected === false) {
      return { index: item.index, id: item.pad.id || "", accepted: false, reason: "the browser reports connected false" };
    }
    return {
      index: item.index,
      id: item.pad.id || "",
      accepted: false,
      reason: `not selected, pad #${chosen.index} was preferred`,
    };
  });

  return {
    index: chosen ? chosen.index : null,
    pad: chosen ? chosen.pad : null,
    considered,
  };
}

// gamepadSelectionNotice is the one line that answers "did findGamepad pick a
// pad". No pads, pads that were all rejected, and a working pad have to read
// as three different things.
function gamepadSelectionNotice(selection) {
  const view = selection || { considered: [], index: null, pad: null };
  const considered = view.considered || [];
  if (!considered.length) {
    return { level: "warn", text: "No pads reported. Press a button on the controller: the browser only reveals a pad after it sends input." };
  }
  const count = `${considered.length} pad${considered.length === 1 ? "" : "s"}`;
  if (view.index === null || !view.pad) {
    const reasons = considered.map((entry) => `#${entry.index} ${entry.reason}`).join("; ");
    return { level: "error", text: `${count} reported, none usable: ${reasons}` };
  }
  const mapping = view.pad.mapping || "";
  const base = `Using pad #${view.index} ${view.pad.id || "unnamed"} of ${count} reported`;
  if (mapping !== "standard") {
    return {
      level: "warn",
      text: `${base}. Its mapping is "${mapping || "none"}", not standard, so the button and axis numbers below may not match the Xbox layout.`,
    };
  }
  return { level: "ok", text: `${base}.` };
}

// describeGamepad is one line per pad the browser reports, carrying everything
// needed to tell a real Xbox pad from a keyboard that enumerated itself.
function describeGamepad(pad) {
  const view = pad || {};
  const buttons = (view.buttons && view.buttons.length) || 0;
  const axes = (view.axes && view.axes.length) || 0;
  const connected = view.connected === false ? "disconnected" : "connected";
  return `#${Number.isFinite(view.index) ? view.index : "?"} `
    + `"${view.id || "unnamed"}" `
    + `mapping ${view.mapping || "none"} `
    + `${connected} ${buttons} buttons ${axes} axes`;
}

function formatAxisValues(axes) {
  const list = Array.isArray(axes) ? axes : [];
  if (!list.length) return "none";
  return list.map((value, index) => `A${index} ${(Number(value) || 0).toFixed(2)}`).join("  ");
}

function formatPressedButtons(buttons) {
  const list = Array.isArray(buttons) ? buttons : [];
  const pressed = [];
  for (let index = 0; index < list.length; index += 1) {
    const button = list[index];
    if (!button || !button.pressed) continue;
    const name = GAMEPAD_BUTTONS[index] || `B${index}`;
    pressed.push(`${name}[${index}] ${(Number(button.value) || 0).toFixed(2)}`);
  }
  return pressed.length ? pressed.join("  ") : "none";
}

// describeGamepadAction labels a reducer action for the rolling log. The
// default branch prints the raw type rather than dropping the entry: an action
// the log cannot name is exactly the one worth seeing.
function describeGamepadAction(action) {
  const view = action || {};
  switch (view.type) {
    case "hardStop": return "hard stop";
    case "startManual": return "arm manual";
    case "toggleAuto": return view.enabled ? "auto nav on" : "auto nav off";
    case "expandFeed": return view.id ? `expand ${view.id}` : "camera gallery";
    case "reconnectCamera": return "camera reconnect";
    case "nudgeManualSpeed": return `manual speed ${(Number(view.value) || 0).toFixed(2)}`;
    case "nudgeAutoSpeed": return `auto speed ${(Number(view.value) || 0).toFixed(2)}`;
    case "nudgeSteerScale": return `steering ${Math.round(Number(view.value) || 0)}%`;
    default: return String(view.type || "unknown");
  }
}

// pushGamepadLog returns a new array, newest first, capped. It never mutates
// what it was handed, so the caller cannot end up sharing a growing array with
// a render pass.
function pushGamepadLog(log, entry, limit = GAMEPAD_LOG_LIMIT) {
  const previous = Array.isArray(log) ? log : [];
  return [entry, ...previous].slice(0, limit);
}

function padTwo(value) {
  return String(value).padStart(2, "0");
}

// gamepadLogLine renders one entry. The time is a local wall clock, because
// the operator is correlating it with what they just did with their thumbs,
// not with anything on the car.
function gamepadLogLine(entry) {
  const view = entry || {};
  const when = new Date(Number(view.at) || 0);
  const time = `${padTwo(when.getHours())}:${padTwo(when.getMinutes())}:${padTwo(when.getSeconds())}`
    + `.${String(when.getMilliseconds()).padStart(3, "0")}`;
  return `${time}  ${view.label || ""}`;
}

// drivePayloadText renders the last body actually POSTed to /api/drive, so a
// dead network can be told apart from a dead controller.
function drivePayloadText(payload) {
  if (!payload) return "none sent yet";
  const parts = [
    `enabled ${payload.enabled ? "true" : "false"}`,
    `linear_x ${(Number(payload.linear_x) || 0).toFixed(2)}`,
    `steering_y ${(Number(payload.steering_y) || 0).toFixed(2)}`,
  ];
  if (Number.isFinite(Number(payload.seq))) parts.push(`seq ${Number(payload.seq)}`);
  if (Number.isFinite(Number(payload.age_ms))) parts.push(`age ${Math.round(Number(payload.age_ms))} ms`);
  return parts.join("  ");
}

function agoText(then, now) {
  if (!then) return "never";
  const seconds = Math.max(0, (Number(now) || 0) - Number(then)) / 1000;
  return `${seconds.toFixed(1)} s ago`;
}

// The WebHID fallback input source ==========================================
//
// Why this exists at all. The operator's Xbox Series pad is bound and described
// correctly by hidutil, Steam reads it, and navigator.getGamepads() returns an
// empty list in Firefox, in Chrome, and on a bare secure-context page with none
// of our code on it. Something between macOS and the browsers refuses to hand
// this device over through the Gamepad API, and no amount of work in
// selectGamepad can reach a pad the browser never lists.
//
// WebHID is a different permission path to the same hardware: Chrome will open
// the device directly once the operator has picked it from a chooser. So the
// page gains a second way in, and everything below turns what comes out of it
// into the snapshot computeGamepadStep already consumes. The mapping, the
// deadzone, the stop rules and the arming logic are not duplicated here and
// must not be: this is a decoder and a normaliser, nothing else.
//
// What is verified and what is not. Every function here is tested against
// bytes written out by hand, so the code does what these tests say. Whether the
// tests describe the operator's actual pad is a separate question, and the
// answer is only partly known: the layout below is the publicly documented
// Xbox One S / Series X|S HID input report, and no report from the operator's
// own controller has been read yet. That is why the panel shows the raw bytes
// behind a debug toggle. The offsets in one place, the tests in another, and
// the operator's eyes on the bytes is the fastest route from a wrong guess to a
// right one.
const HID_XBOX_VENDOR_ID = 0x045e;

// The report as WebHID hands it over, which is with the report id already
// stripped off into event.reportId:
//
//   0  1   left stick X, uint16 little endian
//   2  3   left stick Y
//   4  5   right stick X
//   6  7   right stick Y
//   8  9   left trigger, 10 bit, so 0 to 1023
//   10 11  right trigger
//   12     hat switch, 0 centred then 1 to 8 clockwise from north
//   13     A B X Y LB RB bitfield
//   14     View Menu Xbox LS RS bitfield
//   15     Share, on the Series pads only, which is why 15 bytes is enough
const XBOX_HID_AXIS_OFFSETS = [0, 2, 4, 6];
const XBOX_HID_TRIGGER_OFFSETS = [8, 10];
const XBOX_HID_HAT_OFFSET = 12;
const XBOX_HID_SHARE_OFFSET = 15;
const XBOX_HID_MIN_LENGTH = 15;
const XBOX_HID_TRIGGER_MAX = 1023;
// The sticks are unsigned with the rest position in the middle. Dividing by
// 32767 rather than 32768 is what makes a centred stick decode to exactly 0,
// and the clamp is what keeps the one extra count at the low end from reading
// past full deflection.
const XBOX_HID_STICK_CENTRE = 32768;
const XBOX_HID_STICK_SPAN = 32767;
// Cosmetic only. The reducer reads the triggers through their analogue value,
// never through pressed, so this threshold decides what the Buttons down line
// in the panel shows and nothing about how the car is driven.
const XBOX_HID_TRIGGER_PRESS = 0.12;

// The bitfields, as a table rather than a run of if statements, because this is
// the part most likely to be wrong and a table is the part easiest to correct.
// The indexes on the left are the Gamepad API standard mapping the reducer
// already speaks, so nothing downstream has to learn a second numbering.
const XBOX_HID_BUTTON_BITS = [
  { index: 0, offset: 13, mask: 0x01 },
  { index: 1, offset: 13, mask: 0x02 },
  { index: 2, offset: 13, mask: 0x08 },
  { index: 3, offset: 13, mask: 0x10 },
  { index: 4, offset: 13, mask: 0x40 },
  { index: 5, offset: 13, mask: 0x80 },
  { index: 8, offset: 14, mask: 0x04 },
  { index: 9, offset: 14, mask: 0x08 },
  { index: 10, offset: 14, mask: 0x20 },
  { index: 11, offset: 14, mask: 0x40 },
  { index: 16, offset: 14, mask: 0x10 },
];

// The hat switch is one direction or one diagonal, so it becomes up to two of
// the four D-pad buttons. An unlisted value presses nothing: a hat reading we
// do not recognise is not evidence of any direction, and guessing one would put
// a speed nudge on the log the operator never asked for.
const XBOX_HID_HAT_BUTTONS = {
  1: [12],
  2: [12, 15],
  3: [15],
  4: [13, 15],
  5: [13],
  6: [13, 14],
  7: [14],
  8: [12, 14],
};

// The Share button is not in the standard 17 button mapping, so it goes on the
// end where the panel can name it B17 and the reducer ignores it.
const XBOX_HID_SHARE_INDEX = 17;

// hidReportBytes accepts whatever the event handed over. WebHID gives a
// DataView; a test gives an array; a caller that has already copied gives a
// Uint8Array. All three read the same way here so no caller has to convert.
function hidReportBytes(data) {
  if (!data) return [];
  if (typeof data.getUint8 === "function" && Number.isFinite(data.byteLength)) {
    const out = [];
    for (let at = 0; at < data.byteLength; at += 1) out.push(data.getUint8(at));
    return out;
  }
  if (Number.isFinite(data.length)) {
    const out = [];
    for (let at = 0; at < data.length; at += 1) out.push(Number(data[at]) & 0xff);
    return out;
  }
  return [];
}

function hidUint16(bytes, offset) {
  return (Number(bytes[offset]) & 0xff) | ((Number(bytes[offset + 1]) & 0xff) << 8);
}

function hidStickAxis(raw) {
  return gamepadClamp((raw - XBOX_HID_STICK_CENTRE) / XBOX_HID_STICK_SPAN, -1, 1);
}

// decodeXboxHidReport is the whole layout, pure, in one place. It returns the
// buttons already numbered the way the Gamepad API numbers them, plus enough
// metadata for the panel to explain itself.
//
// A report shorter than the layout is refused rather than decoded from whatever
// bytes did arrive. Half a report read as a whole one is not a smaller input,
// it is a wrong one, and a wrong stick value drives the car.
function decodeXboxHidReport(data) {
  const bytes = hidReportBytes(data);
  if (bytes.length < XBOX_HID_MIN_LENGTH) {
    return {
      ok: false,
      length: bytes.length,
      hasShare: false,
      hat: null,
      reason: `report is ${bytes.length} bytes, the Xbox layout needs at least ${XBOX_HID_MIN_LENGTH}`,
      axes: [],
      buttons: [],
    };
  }

  const axes = XBOX_HID_AXIS_OFFSETS.map((offset) => hidStickAxis(hidUint16(bytes, offset)));

  const buttons = [];
  for (let index = 0; index <= 16; index += 1) buttons[index] = { pressed: false, value: 0 };
  for (const bit of XBOX_HID_BUTTON_BITS) {
    const on = Boolean(bytes[bit.offset] & bit.mask);
    buttons[bit.index] = { pressed: on, value: on ? 1 : 0 };
  }
  XBOX_HID_TRIGGER_OFFSETS.forEach((offset, slot) => {
    const value = gamepadClamp(hidUint16(bytes, offset) / XBOX_HID_TRIGGER_MAX, 0, 1);
    buttons[6 + slot] = { pressed: value >= XBOX_HID_TRIGGER_PRESS, value };
  });
  const hat = Number(bytes[XBOX_HID_HAT_OFFSET]) & 0xff;
  for (const index of XBOX_HID_HAT_BUTTONS[hat] || []) buttons[index] = { pressed: true, value: 1 };

  const hasShare = bytes.length > XBOX_HID_SHARE_OFFSET;
  if (hasShare) {
    const on = Boolean(bytes[XBOX_HID_SHARE_OFFSET] & 0x01);
    buttons[XBOX_HID_SHARE_INDEX] = { pressed: on, value: on ? 1 : 0 };
  }

  return { ok: true, length: bytes.length, hasShare, hat, reason: "", axes, buttons };
}

// emptyPadSnapshot is what a frame with no readable input looks like. Run
// through computeGamepadStep it produces a zeroed drive and no actions, which
// is why every "we do not know what the pad is doing" path in the page returns
// this rather than growing a stop rule of its own.
function emptyPadSnapshot() {
  return { axes: [], buttons: [] };
}

// hidPadSnapshot is the normalisation, and it is deliberately lossy: the
// reducer takes axes and buttons and nothing else, so nothing else travels.
// A report that failed to decode becomes an empty snapshot rather than the last
// good one, because a decoder that has lost the thread must not keep the car
// driving on the last values it understood.
function hidPadSnapshot(decoded) {
  const view = decoded || {};
  if (!view.ok) return emptyPadSnapshot();
  return {
    axes: (view.axes || []).map((value) => Number(value) || 0),
    buttons: (view.buttons || []).map((button) => ({
      pressed: Boolean(button && button.pressed),
      value: Number(button && button.value) || 0,
    })),
  };
}

// How long an open device may say nothing before its frame is driven as an
// empty snapshot. The pad reports on the order of every 8 to 12 ms whether or
// not anything moved, so this is tens of missing reports rather than a pause
// between two of them, and it is short enough that a pad which has genuinely
// stopped cannot hold a throttle open for long.
//
// This one number is the part of the WebHID path with the least evidence behind
// it, because it depends on the pad reporting continuously rather than only on
// change. If it reports only on change, a steady throttle would be zeroed after
// this long and the fix is to raise it. The failure is visible and recoverable
// in that direction and is not in the other, which is why it is set this way
// round.
const HID_SILENCE_MS = 300;

// chooseInputSource is the priority rule, and there is only one: the Gamepad
// API path wins whenever it has a pad. It is the path with a drive's worth of
// hours behind it, so WebHID is consulted only when the browser lists no pad at
// all, which is the operator's situation on this machine.
//
// silent is not the same as absent. An open device that has gone quiet still
// drives the frame, with an empty snapshot, so the drive is zeroed without
// disarming: silence is ambiguous evidence and a disconnect is not. The
// disconnect event carries the full loss path, hard stop included.
function chooseInputSource(view) {
  const state = view || {};
  if (state.pad) return { source: "gamepad", silent: false };
  if (!state.hidOpen) return { source: "none", silent: false };
  const last = Number(state.hidLastReportAt) || 0;
  const age = (Number(state.now) || 0) - last;
  const fresh = last > 0 && age >= 0 && age <= HID_SILENCE_MS;
  return { source: "hid", silent: !fresh };
}

// hidStatusNotice is the Controller panel's WebHID line. Each outcome has a
// different next move for the operator, so each one has to read differently:
// use Chrome, press the button, quit whatever is holding the pad, press it
// again, wiggle the pad, send us the bytes, or drive.
//
// The view carries error and decodeReason separately on purpose. An error is
// something the page tried and could not do; a decodeReason is a report that
// arrived and could not be read. They have different fixes and neither may be
// shown as the other.
function hidStatusNotice(view) {
  const state = view || {};
  if (!state.hasWebHid) {
    return {
      level: "error",
      headline: "WebHID is not available here",
      detail: "This browser exposes no navigator.hid. WebHID is Chrome only, so on this browser the "
        + "Gamepad API above is the only way in and the connect button stays disabled.",
    };
  }
  if (!state.deviceName) {
    return {
      level: "warn",
      headline: "No controller authorised",
      detail: state.error
        ? `Nothing is authorised yet and the last attempt failed: ${state.error}`
        : "Press Connect controller and pick the Xbox pad from the chooser. Chrome only opens that "
          + "chooser from a click, and the choice is remembered across reloads.",
    };
  }
  if (!state.open) {
    // An open that failed is the likeliest way this path breaks on a machine
    // where the pad works everywhere else: another process holding the device
    // exclusively, Steam being the obvious candidate on this one. Saying only
    // "not open" would send the operator back to the button that just failed.
    if (state.error) {
      return {
        level: "error",
        headline: "Could not open the controller",
        detail: `${state.deviceName} is authorised and would not open: ${state.error}. Another `
          + "process may be holding it exclusively, Steam included, so close what else is reading "
          + "the pad and press Connect controller again.",
      };
    }
    return {
      level: "warn",
      headline: "Authorised but not open",
      detail: `${state.deviceName} is authorised and the page has not got it open. Press Connect `
        + "controller again, or replug the pad.",
    };
  }
  const last = Number(state.lastReportAt) || 0;
  const age = (Number(state.now) || 0) - last;
  if (!(last > 0 && age >= 0 && age <= HID_SILENCE_MS)) {
    return {
      level: "warn",
      headline: "Open but reporting nothing",
      detail: `${state.deviceName} is open and no input report has arrived in the last `
        + `${HID_SILENCE_MS} ms, so the drive is held at zero. Press a button on the pad.`,
    };
  }
  if (state.decodeReason) {
    return {
      level: "error",
      headline: "Reports arriving but not decodable",
      detail: `${state.deviceName} is reporting and the decoder refused the last one: ${state.decodeReason}. `
        + "The drive is held at zero, and the bytes it refused are on screen below.",
    };
  }
  // Not "driving from WebHID". This function only knows what the device is
  // doing, and with a Gamepad API pad listed as well the device is open,
  // reporting, and driving nothing. The Input source line is the one that
  // answers that, so this points at it rather than guessing.
  return {
    level: "ok",
    headline: "Open and reporting",
    detail: `${state.deviceName} is open and reporting ${Number(state.reportLength) || 0} byte reports. `
      + "It drives the car on any frame where the browser lists no pad of its own, and the input source "
      + "line below says which of the two is driving right now.",
  };
}

// inputSourceLabel names the source that drove the last frame. It exists
// because hidStatusNotice can only say what the WebHID device is doing, not
// whether it is the path in use: with a Gamepad API pad present as well, an
// open and reporting device is not driving anything, and a panel that said
// "Driving from WebHID" there would be wrong about the one thing it is for.
function inputSourceLabel(choice) {
  const view = choice || {};
  if (view.source === "gamepad") return "Gamepad API";
  if (view.source === "hid") {
    return view.silent
      ? `WebHID, nothing reported in the last ${HID_SILENCE_MS} ms, drive held at zero`
      : "WebHID";
  }
  return "none, no pad listed and no WebHID device open";
}

// formatHidReportBytes is the debug readout, and it is the point of the whole
// panel: the byte layout above is documentation rather than measurement, so the
// operator confirming which byte moves when they push a stick is how it becomes
// measurement. Offsets are labelled per row so a byte can be named out loud.
function formatHidReportBytes(data) {
  const bytes = hidReportBytes(data);
  if (!bytes.length) return "no report received yet";
  const hex = bytes.map((byte) => byte.toString(16).padStart(2, "0"));
  const lines = [];
  for (let at = 0; at < hex.length; at += 8) {
    lines.push(`${String(at).padStart(2, "0")}: ${hex.slice(at, at + 8).join(" ")}`);
  }
  return lines.join("\n");
}

// describeHidDevice names the device the way hidutil does, so the operator can
// check the page opened the same thing they were looking at on the command line.
function describeHidDevice(device) {
  const view = device || {};
  const vendor = (Number(view.vendorId) || 0).toString(16).padStart(4, "0");
  const product = (Number(view.productId) || 0).toString(16).padStart(4, "0");
  return `${view.productName || "unnamed HID device"} (${vendor}:${product})`;
}

// selectHidDevice picks from what navigator.hid.getDevices() has already
// granted, so the operator authorises once rather than on every reload. The
// vendor test is the same one the chooser filter uses: a granted device from
// some other vendor is not this pad and must not be opened as though it were.
function selectHidDevice(devices, vendorId = HID_XBOX_VENDOR_ID) {
  const list = Array.isArray(devices) ? devices : [];
  for (const device of list) {
    if (device && Number(device.vendorId) === Number(vendorId)) return device;
  }
  return null;
}

// The drive send strategy ====================================================
//
// The page used to POST the current command every 120 ms whatever it was. On a
// link that averages 319 ms round trip that is most of the budget spent
// resending a value the car already has, and the queue it builds is what makes
// the car stutter and respond late.
//
// The rules now:
//   a change goes out at once
//   an unchanged command that is asking for motion is repeated every
//     DRIVE_HEARTBEAT_MS, comfortably inside the server's CMD_TIMEOUT_S so one
//     lost POST cannot time the car out
//   an unchanged command that is asking for nothing stops entirely, once it
//     has been repeated DRIVE_ZERO_REPEATS times
//
// The zero repeats are the safety half. Stopping the heartbeat on a zero is
// safe in itself, because the server's watchdog holds the car stopped if
// nothing arrives, but a stop must not depend on one message being delivered,
// so the zero goes out again a bounded number of times before the page falls
// silent.
const DRIVE_HEARTBEAT_MS = 200;
const DRIVE_ZERO_REPEATS = 3;

function isZeroDriveCommand(command) {
  const view = command || {};
  if (!view.enabled) return true;
  return (Number(view.linear_x) || 0) === 0 && (Number(view.steering_y) || 0) === 0;
}

function driveCommandsEqual(a, b) {
  if (!a || !b) return false;
  return Boolean(a.enabled) === Boolean(b.enabled)
    && (Number(a.linear_x) || 0) === (Number(b.linear_x) || 0)
    && (Number(a.steering_y) || 0) === (Number(b.steering_y) || 0);
}

function newDriveSendState() {
  return { last: null, lastSentAt: 0, zeroRepeatsLeft: 0 };
}

// planDriveSend is the whole sender policy, one call per candidate moment. The
// caller passes its state back in, so the function holds nothing itself and a
// test can run a whole drive through it without a clock.
// noteDriveSent is "we just put this command on the wire", and every send goes
// through it, including the ones app.js issues directly from an event handler
// rather than from planDriveSend. Without that the heartbeat clock and the
// change detector would drift out of step with what the car actually received,
// and the next tick would resend a command it already has.
function noteDriveSent(state, command, now) {
  const previous = state || newDriveSendState();
  const zero = isZeroDriveCommand(command);
  const wasMoving = Boolean(previous.last) && !isZeroDriveCommand(previous.last);
  const repeat = driveCommandsEqual(previous.last, command);
  return {
    last: {
      enabled: Boolean(command.enabled),
      linear_x: Number(command.linear_x) || 0,
      steering_y: Number(command.steering_y) || 0,
    },
    lastSentAt: now,
    // A command that has just stopped asking for motion owes the car a few
    // more copies, so a stop does not depend on one message being delivered.
    // One that was already still, such as a page parked since it loaded, owes
    // nothing and falls silent.
    zeroRepeatsLeft: !zero
      ? 0
      : wasMoving
        ? DRIVE_ZERO_REPEATS
        : Math.max(0, (previous.zeroRepeatsLeft || 0) - (repeat ? 1 : 0)),
  };
}

function planDriveSend(state, command, now) {
  const previous = state || newDriveSendState();

  if (!driveCommandsEqual(previous.last, command)) {
    return { send: true, reason: "changed", next: noteDriveSent(previous, command, now) };
  }
  if (now - previous.lastSentAt < DRIVE_HEARTBEAT_MS) {
    return { send: false, reason: "recent", next: previous };
  }
  if (!isZeroDriveCommand(command)) {
    return { send: true, reason: "heartbeat", next: noteDriveSent(previous, command, now) };
  }
  if (previous.zeroRepeatsLeft > 0) {
    return { send: true, reason: "zero repeat", next: noteDriveSent(previous, command, now) };
  }
  return { send: false, reason: "idle", next: previous };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DRIVE_HEARTBEAT_MS,
    DRIVE_ZERO_REPEATS,
    GAMEPAD_BUTTONS,
    GAMEPAD_LOG_LIMIT,
    HID_SILENCE_MS,
    HID_XBOX_VENDOR_ID,
    XBOX_HID_MIN_LENGTH,
    agoText,
    cameraFeedState,
    chooseInputSource,
    decodeXboxHidReport,
    describeHidDevice,
    emptyPadSnapshot,
    formatHidReportBytes,
    hidPadSnapshot,
    hidReportBytes,
    hidStatusNotice,
    inputSourceLabel,
    selectHidDevice,
    describeGamepad,
    describeGamepadAction,
    driveCommandsEqual,
    drivePayloadText,
    feedReconnectDelayMs,
    formatAxisValues,
    formatPressedButtons,
    gamepadContextNotice,
    gamepadLogLine,
    gamepadSelectionNotice,
    isZeroDriveCommand,
    newDriveSendState,
    nextExpandedFeed,
    nextPeriodicFeedIndex,
    noteDriveSent,
    planDriveSend,
    pushGamepadLog,
    selectGamepad,
    STOP_MAX_ATTEMPTS,
    STOP_RETRY_DELAY_MS,
    applyDeadzone,
    commandReadoutText,
    computeGamepadStep,
    computeDisconnectStep,
    computeMissingPadStep,
    controlModeText,
    gamepadClamp,
    nextControlState,
    planStatusStopFollowUp,
    planStopAttempt,
    shouldPushAutoParameters,
    stopConfirmedByStatus,
  };
}
