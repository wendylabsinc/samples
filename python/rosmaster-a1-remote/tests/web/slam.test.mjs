// The Map panel: the pure layer of slam.js as plain unit tests (this file's
// first half), and the page wiring through the vm harness (second half,
// appended by the tasks that build the DOM layer).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  SLAM_UNREACHABLE_FAILURES,
  SLAM_MIN_SCALE,
  SLAM_MAX_SCALE,
  newSlamModel,
  slamReduce,
  slamView,
  slamPlan,
  slamMerge,
  slamOverlay,
  slamMapBounds,
  slamFitView,
  slamMapMetaFromHeaders,
  slamStateText,
  slamStatsText,
  slamReadoutText,
} = require("../../rosmaster-a1-web-remote-wendy/app/static/slam.js");

// A mapping snapshot: a 10 x 5 m map whose origin is (-5, -2.5), so the map
// is centred on the world origin, and a robot at (1, 2) facing +y.
export function snapshot(overrides = {}) {
  return {
    ok: true,
    bridge: { state: "mapping", reason: null },
    slam: { state: "mapping", saves: 3 },
    slam_age_s: 0.2,
    pose: { x: 1.0, y: 2.0, yaw: Math.PI / 2, age_s: 0.01 },
    map_odom: { x: 0, y: 0, yaw: 0, age_s: 0.05 },
    map: { version: 4, width: 200, height: 100, resolution: 0.05, origin: { x: -5, y: -2.5, yaw: 0 }, age_s: 0.5 },
    scan: { age_s: 0.1, points: [1, 0, 0, 1] },
    trajectory: { epoch: 1, count: 2 },
    ...overrides,
  };
}

const FIT_SCALE = 640 / 12; // a 10 x 5 m map with a 20 % margin into 640 x 400

function near(actual, expected, eps = 1e-6) {
  assert.ok(Math.abs(actual - expected) < eps, `${actual} is not within ${eps} of ${expected}`);
}

test("SLAM model: a snapshot fills state, pose, scan and what the bridge holds", () => {
  const model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(model.state, "mapping");
  assert.deepEqual(model.pose, { x: 1.0, y: 2.0, yaw: Math.PI / 2, age_s: 0.01 });
  assert.equal(model.mapAvailable.version, 4);
  assert.deepEqual(model.trajectoryAvailable, { epoch: 1, count: 2 });
  assert.deepEqual(model.scan.points, [1, 0, 0, 1]);
  assert.equal(model.failures, 0);
});

test("SLAM model: following keeps the view centred on the pose", () => {
  const model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(model.view.cx, 1.0);
  assert.equal(model.view.cy, 2.0);
});

test("SLAM model: the first snapshot with a map fits the view to it, once", () => {
  let model = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot({ pose: null }) });
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  assert.equal(model.view.fitted, true);
  model = slamReduce(model, { type: "wheel", factor: 2, atX: 320, atY: 200 });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ pose: null }) });
  near(model.view.scale, FIT_SCALE * 2, 1e-6);
});

test("SLAM model: failures count up, three mean unreachable, one completed cycle clears them", () => {
  let model = newSlamModel();
  for (let i = 0; i < SLAM_UNREACHABLE_FAILURES; i += 1) model = slamReduce(model, { type: "failure", error: "offline" });
  assert.deepEqual(slamOverlay(model), { title: "Car not reachable", detail: "offline" });
  assert.equal(slamStateText(model), "car unreachable");
  // A snapshot alone does not clear it (that was the bug: a snapshot can
  // succeed while a later step in the same cycle keeps failing); only
  // cycleDone, dispatched after every step of a cycle has succeeded, does.
  model = slamReduce(model, { type: "snapshot", body: snapshot() });
  assert.equal(model.failures, 3);
  model = slamReduce(model, { type: "cycleDone" });
  assert.equal(model.failures, 0);
  assert.equal(model.lastError, null);
  assert.equal(slamOverlay(model), null);
});

test("SLAM view: canvas and world round trip, north up, east right", () => {
  const model = { ...newSlamModel(), view: { scale: 50, cx: 1, cy: 2, follow: false, fitted: true } };
  const view = slamView(model, 640, 400);
  assert.deepEqual(view.toCanvas(1, 2), [320, 200]);
  assert.deepEqual(view.toCanvas(2, 3), [370, 150]);
  const [wx, wy] = view.toWorld(370, 150);
  near(wx, 2, 1e-9);
  near(wy, 3, 1e-9);
});

test("SLAM view: wheel zoom keeps the world point under the cursor fixed and multiplies the scale", () => {
  let model = { ...newSlamModel(640, 400), view: { scale: 50, cx: 1, cy: 2, follow: false, fitted: true } };
  const before = slamView(model, 640, 400).toWorld(500, 100);
  model = slamReduce(model, { type: "wheel", factor: 1.15, atX: 500, atY: 100 });
  const after = slamView(model, 640, 400).toWorld(500, 100);
  near(before[0], after[0], 1e-9);
  near(before[1], after[1], 1e-9);
  near(model.view.scale, 57.5, 1e-9);
});

test("SLAM view: zoom is clamped to the scale range", () => {
  let model = { ...newSlamModel(), view: { scale: SLAM_MAX_SCALE, cx: 0, cy: 0, follow: false, fitted: true } };
  model = slamReduce(model, { type: "zoom", factor: 2 });
  assert.equal(model.view.scale, SLAM_MAX_SCALE);
  model = slamReduce({ ...model, view: { ...model.view, scale: SLAM_MIN_SCALE } }, { type: "zoom", factor: 0.5 });
  assert.equal(model.view.scale, SLAM_MIN_SCALE);
});

test("SLAM view: dragging pans in canvas pixels and turns follow off; follow on recentres", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  model = slamReduce({ ...model, view: { ...model.view, scale: 100 } }, { type: "drag", dx: 50, dy: -20 });
  assert.equal(model.view.follow, false);
  near(model.view.cx, 0.5, 1e-9);
  near(model.view.cy, 1.8, 1e-9);
  model = slamReduce(model, { type: "follow", on: true });
  assert.equal(model.view.cx, 1.0);
  assert.equal(model.view.cy, 2.0);
});

test("SLAM view: reset fits the map and turns follow off; with no map it frames the pose", () => {
  let model = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot() });
  model = slamReduce(model, { type: "wheel", factor: 3, atX: 10, atY: 10 });
  model = slamReduce(model, { type: "reset" });
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  assert.equal(model.view.follow, false);
  let bare = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot({ map: null }) });
  bare = slamReduce(bare, { type: "reset" });
  assert.equal(bare.view.cx, 1);
  assert.equal(bare.view.cy, 2);
  near(bare.view.scale, 400 / 6);
});

test("SLAM plan: the PNG only when the version moved, the trajectory only when it grew or reset", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.deepEqual(slamPlan(model), { map: true, trajectory: true, trajectoryQuery: { epoch: 1, from: 0 } });
  model = slamReduce(model, { type: "map", meta: { ...snapshot().map }, image: { bitmap: true } });
  model = slamReduce(model, { type: "trajectory", reply: { epoch: 1, from: 0, total: 2, points: [0, 0, 0.1, 0] } });
  assert.deepEqual(slamPlan(model), { map: false, trajectory: false, trajectoryQuery: { epoch: 1, from: 2 } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ trajectory: { epoch: 1, count: 3 } }) });
  assert.deepEqual(slamPlan(model), { map: false, trajectory: true, trajectoryQuery: { epoch: 1, from: 2 } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ map: { ...snapshot().map, version: 5 }, trajectory: { epoch: 2, count: 0 } }) });
  assert.deepEqual(slamPlan(model), { map: true, trajectory: true, trajectoryQuery: { epoch: 2, from: 0 } });
});

test("SLAM plan: a same-epoch count smaller than what is held means a different server, and resyncs from 0", () => {
  // The web service restarted under an open page: the new bridge relatches
  // the keeper's path as epoch 1 again, this time with fewer points than the
  // client already holds from the old epoch 1. `from: haveCount` would ask
  // for points past the end of the relatched path and get nothing back.
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ trajectory: { epoch: 1, count: 4 } }) });
  model = slamReduce(model, { type: "map", meta: { ...snapshot().map }, image: { bitmap: true } });
  model = slamReduce(model, { type: "trajectory", reply: { epoch: 1, from: 0, total: 4, points: [0, 0, 0.1, 0, 0.2, 0, 0.3, 0] } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ trajectory: { epoch: 1, count: 2 } }) });
  assert.deepEqual(slamPlan(model), { map: false, trajectory: true, trajectoryQuery: { epoch: 1, from: 0 } });
});

test("SLAM plan: a null map in the snapshot fetches nothing and keeps the held image", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  model = slamReduce(model, { type: "map", meta: snapshot().map, image: { bitmap: true } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ map: null, bridge: { state: "waiting_for_map", reason: null } }) });
  assert.equal(slamPlan(model).map, false);
  assert.deepEqual(model.mapImage, { bitmap: true });
  assert.deepEqual(slamOverlay(model), { title: "Waiting for first map", detail: "" });
  model = slamReduce(model, { type: "mapMissing" });
  assert.deepEqual(model.mapImage, { bitmap: true });
  assert.equal(slamPlan(model).map, false);
});

test("SLAM merge: same epoch appends from the index, another epoch replaces", () => {
  const have = { epoch: 1, points: [0, 0, 0.1, 0] };
  assert.deepEqual(slamMerge(have, { epoch: 1, from: 2, total: 3, points: [0.2, 0] }), { epoch: 1, points: [0, 0, 0.1, 0, 0.2, 0] });
  assert.deepEqual(slamMerge(have, { epoch: 1, from: 1, total: 2, points: [0.5, 0.5] }), { epoch: 1, points: [0, 0, 0.5, 0.5] });
  assert.deepEqual(slamMerge(have, { epoch: 2, from: 0, total: 1, points: [9, 9] }), { epoch: 2, points: [9, 9] });
});

test("SLAM overlay: one title per state, none while mapping", () => {
  const at = (state, reason = null) => slamOverlay({ ...newSlamModel(), state, reason });
  assert.deepEqual(at("connecting"), { title: "Connecting to the car", detail: "" });
  assert.deepEqual(at("slam_unreachable", "no /slam/status for 4.2 s"), { title: "SLAM service not running", detail: "no /slam/status for 4.2 s" });
  assert.deepEqual(at("slam_down", "x"), { title: "slam_toolbox restarting", detail: "x" });
  assert.deepEqual(at("waiting_for_scan"), { title: "Waiting for LiDAR", detail: "" });
  assert.deepEqual(at("waiting_for_odom_tf"), { title: "Waiting for odometry", detail: "" });
  assert.deepEqual(at("waiting_for_map"), { title: "Waiting for first map", detail: "" });
  assert.deepEqual(at("relocalising", "r"), { title: "relocalising", detail: "r" });
  assert.equal(at("mapping"), null);
});

test("SLAM bounds: the map's corners, with the origin yaw honoured", () => {
  assert.deepEqual(slamMapBounds({ width: 4, height: 2, resolution: 0.5, origin: { x: 1, y: 1, yaw: 0 } }), { minX: 1, maxX: 3, minY: 1, maxY: 2 });
  const turned = slamMapBounds({ width: 4, height: 2, resolution: 0.5, origin: { x: 0, y: 0, yaw: Math.PI / 2 } });
  near(turned.minX, -1, 1e-9);
  near(turned.maxX, 0, 1e-9);
  near(turned.minY, 0, 1e-9);
  near(turned.maxY, 2, 1e-9);
});

test("SLAM fit: margin, centre and clamping", () => {
  assert.deepEqual(slamFitView({ minX: 0, maxX: 10, minY: 0, maxY: 5 }, 600, 300), { scale: 50, cx: 5, cy: 2.5 });
  assert.equal(slamFitView({ minX: 0, maxX: 0.1, minY: 0, maxY: 0.1 }, 600, 300).scale, SLAM_MAX_SCALE);
  assert.equal(slamFitView({ minX: 0, maxX: 1000, minY: 0, maxY: 1000 }, 600, 300).scale, SLAM_MIN_SCALE);
});

test("SLAM headers: placement metadata is read from the PNG response", () => {
  const headers = new Map([
    ["X-Map-Version", "7"], ["X-Map-Width", "40"], ["X-Map-Height", "30"], ["X-Map-Resolution", "0.05"],
    ["X-Map-Origin-X", "-1.25"], ["X-Map-Origin-Y", "0.5"], ["X-Map-Origin-Yaw", "0"],
  ]);
  assert.deepEqual(slamMapMetaFromHeaders((name) => headers.get(name)), {
    version: 7, width: 40, height: 30, resolution: 0.05, origin: { x: -1.25, y: 0.5, yaw: 0 },
  });
});

test("SLAM text: the pill, the stats and the readout", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(slamStateText(model), "mapping");
  assert.equal(slamStatsText(model), "3 saves, 2 poses");
  assert.equal(slamReadoutText(model), "x 1.00 m  y 2.00 m  heading 90°  map 10.0 × 5.0 m");
  model = slamReduce(model, { type: "hidden", hidden: true });
  assert.equal(slamStateText(model), "hidden");
  assert.equal(slamReadoutText(newSlamModel()), "Waiting for the car");
  assert.equal(slamStateText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ bridge: { state: "waiting_for_odom_tf", reason: null } }) })), "waiting for odom tf");
  // slam_unreachable gets the spec's exact pill wording rather than the
  // generic underscore-to-space form every other state gets.
  assert.equal(slamStateText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ bridge: { state: "slam_unreachable", reason: null } }) })), "SLAM service not running");
  assert.equal(slamStatsText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ slam: { saves: 1 }, trajectory: { epoch: 1, count: 1 } }) })), "1 save, 1 pose");
});


// The page wiring ===========================================================
import { loadPage, response } from "./harness.mjs";

function mapResponse(version = 4, body = "png-bytes", yaw = 0) {
  return response({
    status: 200,
    headers: {
      "X-Map-Version": String(version), "X-Map-Width": "200", "X-Map-Height": "100", "X-Map-Resolution": "0.05",
      "X-Map-Origin-X": "-5", "X-Map-Origin-Y": "-2.5", "X-Map-Origin-Yaw": String(yaw), ETag: `"${version}"`,
    },
    blob: body,
  });
}

// A loaded page whose load-time poll has settled against the harness default
// (no map yet), then switched to a mapping car with a map and two trajectory
// poses. Calls are cleared so a test sees only what it triggers.
async function mappingPage(snapshotOverrides = {}) {
  const page = loadPage();
  await page.settle();
  page.fake.slam = snapshot(snapshotOverrides);
  page.fake.responses.set("/api/slam/map.png", mapResponse(4));
  page.fake.responses.set("/api/slam/trajectory", { epoch: 1, from: 0, total: 2, points: [0, 0, 0.1, 0] });
  page.clearCalls();
  return page;
}

test("SLAM wiring: the page polls the snapshot once on load and fetches nothing else while there is no map", async () => {
  const page = loadPage();
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1);
  assert.equal(page.gets("/api/slam/map.png").length, 0);
  assert.equal(page.gets("/api/slam/trajectory").length, 0);
  assert.equal(page.el("slamState").textContent, "waiting for map");
  assert.equal(page.el("slamStats").textContent, "0 saves, 0 poses");
});

test("SLAM wiring: one cycle fetches the snapshot, then the PNG, then the trajectory, in that order", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  assert.deepEqual(page.calls.filter((c) => c.method === "GET").map((c) => c.path), [
    "/api/slam", "/api/slam/map.png", "/api/slam/trajectory?epoch=1&from=0",
  ]);
  const model = page.slam;
  assert.equal(model.map.version, 4);
  assert.deepEqual(model.mapImage, { bitmap: true, blob: "png-bytes" });
  assert.deepEqual(model.trajectory, { epoch: 1, points: [0, 0, 0.1, 0] });
  assert.equal(page.el("slamState").textContent, "mapping");
  assert.equal(page.el("slamStats").textContent, "3 saves, 2 poses");
  assert.equal(page.el("slamReadout").textContent, "x 1.00 m  y 2.00 m  heading 90°  map 10.0 × 5.0 m");
});

test("SLAM wiring: an unchanged map and trajectory cost only the snapshot", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  assert.deepEqual(page.calls.map((c) => c.path), ["/api/slam"]);
});

test("SLAM wiring: a new map version refetches the PNG with If-None-Match, and a 304 keeps the image", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.fake.slam = snapshot({ map: { ...snapshot().map, version: 5 } });
  page.fake.responses.set("/api/slam/map.png", response({ status: 304, headers: { ETag: '"5"' } }));
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  const pngCall = page.calls.find((c) => c.path === "/api/slam/map.png");
  assert.ok(pngCall, "the version moved, so the PNG was asked for");
  assert.equal(pngCall.headers["If-None-Match"], '"4"');
  assert.equal(page.slam.map.version, 4, "a 304 leaves the held image alone");
  assert.equal(page.slam.failures, 0, "a 304 is not a failure");
});

test("SLAM wiring: a PNG 404 is not a failure and keeps whatever image is held", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.fake.slam = snapshot({ map: { ...snapshot().map, version: 6 } });
  page.fake.responses.set("/api/slam/map.png", response({ status: 404 }));
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.slam.failures, 0);
  assert.equal(page.slam.map.version, 4);
  assert.deepEqual(page.slam.mapImage, { bitmap: true, blob: "png-bytes" });
});

test("SLAM wiring: never two requests in flight", async () => {
  const page = await mappingPage();
  page.fake.held.add("/api/slam");
  page.run("refreshSlam()");
  page.run("refreshSlam()");
  page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "the guard swallows overlapping polls");
  assert.equal(page.releaseHeld(), 1);
  await page.settle();
  assert.equal(page.gets("/api/slam/map.png").length, 1, "the held cycle carried on to the PNG");
});

test("SLAM wiring: three failed polls show the unreachable state on the canvas, one success clears it", async () => {
  const page = await mappingPage();
  page.fake.failing.add("/api/slam");
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshSlam()");
    await page.settle();
  }
  assert.equal(page.el("slamState").textContent, "car unreachable");
  const texts = page.canvasFrame("slamCanvas").filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(texts.includes("Car not reachable"), `overlay drawn: ${texts.join(" | ")}`);
  page.fake.failing.delete("/api/slam");
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.el("slamState").textContent, "mapping");
  assert.equal(page.slam.failures, 0);
});

test("SLAM wiring: a snapshot that keeps succeeding while the PNG keeps failing still reaches unreachable", async () => {
  // The bug: failures used to reset on a successful snapshot alone, so a
  // snapshot/PNG-failure/snapshot/PNG-failure cycle oscillated the counter
  // between 0 and 1 and never reached SLAM_UNREACHABLE_FAILURES, leaving the
  // pill on "mapping" while the map silently never arrived.
  const page = await mappingPage();
  page.fake.responses.set("/api/slam/map.png", response({ status: 500 }));
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshSlam()");
    await page.settle();
  }
  assert.equal(page.el("slamState").textContent, "car unreachable");
  const texts = page.canvasFrame("slamCanvas").filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(texts.includes("Car not reachable"), `overlay drawn: ${texts.join(" | ")}`);
  assert.match(page.slam.lastError, /\/api\/slam\/map\.png/);
});

test("SLAM wiring: a timed-out snapshot counts as one failure", async () => {
  const page = await mappingPage();
  page.fake.held.add("/api/slam");
  page.run("refreshSlam()");
  await page.settle();
  assert.ok(page.expireFetchTimeouts() >= 1);
  await page.settle();
  assert.equal(page.slam.failures, 1);
  assert.equal(page.run("slamInFlight"), false, "the guard is released after a failure");
});

test("SLAM wiring: SLAM failures never trip the control breaker or blank the camera tiles", async () => {
  const page = await mappingPage();
  await page.run("refreshStatus()");
  await page.settle();
  assert.deepEqual(page.tileIds(), ["hp60c_depth", "hp60c_rgb"]);
  page.fake.failing.add("/api/slam");
  for (let i = 0; i < 5; i += 1) {
    await page.run("refreshSlam()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, false);
  assert.equal(page.run("controlFailStreak"), 0);
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg/);
});

test("SLAM wiring: polling stops while the panel is hidden or the tab is in the background, and resumes at once", async () => {
  const page = await mappingPage();
  page.fireElement("slamHide", "click", {});
  assert.equal(page.el("slamState").textContent, "hidden");
  assert.ok(page.el("slamBody").classList.contains("hidden"));
  assert.equal(page.el("slamHide").textContent, "Show");
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 0);
  page.fireElement("slamHide", "click", {});
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "unhiding polls immediately");
  assert.equal(page.el("slamHide").textContent, "Hide");
  page.clearCalls();
  page.setDocumentHidden(true);
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 0);
  page.setDocumentHidden(false);
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "coming back to the tab polls immediately");
});

// Drawing and interaction ====================================================

test("SLAM drawing: the map image is placed at its origin, north up, one cell per resolution", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  const translate = frame.find((c) => c.op === "translate");
  const draw = frame.find((c) => c.op === "drawImage");
  assert.ok(translate && draw, "the frame translated to the origin and drew the image");
  // Follow is on, so the view is centred on the pose (1, 2) at the fitted
  // scale. The origin (-5, -2.5) is 6 m west and 4.5 m south of it.
  near(translate.args[0], 320 - 6 * FIT_SCALE);
  near(translate.args[1], 200 + 4.5 * FIT_SCALE);
  const cell = FIT_SCALE * 0.05;
  assert.deepEqual(draw.args[0], { bitmap: true, blob: "png-bytes" });
  near(draw.args[1], 0);
  near(draw.args[2], -100 * cell, 1e-6);
  near(draw.args[3], 200 * cell, 1e-6);
  near(draw.args[4], 100 * cell, 1e-6);
  const rotate = frame.find((c) => c.op === "rotate");
  near(rotate.args[0], 0);
});

test("SLAM drawing: a non-zero origin yaw rotates the map and still translates to the origin", async () => {
  const yaw = Math.PI / 2;
  const page = await mappingPage({ map: { ...snapshot().map, origin: { x: -5, y: -2.5, yaw } } });
  page.fake.responses.set("/api/slam/map.png", mapResponse(4, "png-bytes", yaw));
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  const rotate = frame.find((c) => c.op === "rotate");
  near(rotate.args[0], -yaw, 1e-9);
  // World angles turn counter-clockwise, canvas y points down, so drawSlamMap
  // rotates by -yaw; the origin still has to land at its own canvas point
  // under whatever scale and centre the view actually has, not a hardcoded
  // north-up constant.
  const { scale, cx, cy } = page.slam.view;
  const translate = frame.find((c) => c.op === "translate");
  near(translate.args[0], (-5 - cx) * scale + 320, 1e-9);
  near(translate.args[1], 200 - (-2.5 - cy) * scale, 1e-9);
});

test("SLAM drawing: scan points go through the pose and the robot marker points along its heading", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  // Scan (1, 0) and (0, 1) in base_link with the robot at (1, 2) facing +y
  // land at world (1, 3) and (0, 2): one metre north and one metre west of
  // the centred pose.
  const dots = frame.filter((c) => c.op === "fillRect" && c.args[2] === 2 && c.args[3] === 2).map((c) => [c.args[0] + 1, c.args[1] + 1]);
  assert.equal(dots.length, 2);
  assert.ok(dots.some(([x, y]) => Math.abs(x - 320) < 1e-6 && Math.abs(y - (200 - FIT_SCALE)) < 1e-6), `north dot in ${JSON.stringify(dots)}`);
  assert.ok(dots.some(([x, y]) => Math.abs(x - (320 - FIT_SCALE)) < 1e-6 && Math.abs(y - 200) < 1e-6), `west dot in ${JSON.stringify(dots)}`);
  // moveTo order in a frame: the trajectory's first point, the robot's tip,
  // the scale bar. Heading +y is straight up on the canvas.
  const moves = frame.filter((c) => c.op === "moveTo");
  assert.equal(moves.length, 3);
  const tip = moves[1];
  near(tip.args[0], 320, 1e-6);
  assert.ok(tip.args[1] < 200, "the tip is above the centre");
  const scaleBar = frame.filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(scaleBar.includes("1 m"), "at 53 px/m the bar is one metre");
});

test("SLAM drawing: the trajectory is one polyline through its points", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  const firstMove = frame.findIndex((c) => c.op === "moveTo");
  const line = frame[firstMove + 1];
  assert.equal(line.op, "lineTo");
  // Points (0, 0) then (0.1, 0), seen from the centre (1, 2).
  near(frame[firstMove].args[0], 320 - 1 * FIT_SCALE);
  near(frame[firstMove].args[1], 200 + 2 * FIT_SCALE);
  near(line.args[0], 320 - 0.9 * FIT_SCALE);
});

test("SLAM drawing: a state other than mapping dims the scene and writes the state and reason over it", async () => {
  const page = await mappingPage({ bridge: { state: "slam_down", reason: "map -> odom 4.1 s old" } });
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  assert.ok(frame.some((c) => c.op === "drawImage"), "the last map is still drawn");
  assert.ok(frame.some((c) => c.op === "globalAlpha" && c.args[0] === 0.5), "the scene is dimmed");
  const texts = frame.filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(texts.includes("slam_toolbox restarting"));
  assert.ok(texts.includes("map -> odom 4.1 s old"));
  assert.equal(page.el("slamReason").textContent, "map -> odom 4.1 s old");
  assert.equal(page.el("slamState").textContent, "slam down");
});

test("SLAM drawing: no pose means no scan and no robot, and a wide view uses the five metre bar", async () => {
  const page = await mappingPage({ pose: null });
  await page.run("refreshSlam()");
  await page.settle();
  for (let i = 0; i < 12; i += 1) page.fireElement("slamZoomOut", "click", {});
  const frame = page.canvasFrame("slamCanvas");
  assert.equal(frame.filter((c) => c.op === "fillRect" && c.args[2] === 2).length, 0);
  assert.equal(frame.filter((c) => c.op === "moveTo").length, 2, "trajectory and scale bar only");
  assert.ok(frame.filter((c) => c.op === "fillText").map((c) => c.args[0]).includes("5 m"));
});

test("SLAM interaction: drag pans in canvas pixels and turns Follow off; wheel and buttons zoom; Reset refits; Follow recentres", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.el("slamFollow").checked, true);
  // The fake canvas is 640 x 400 drawn into a 100 x 100 box, so one CSS
  // pixel is 6.4 canvas pixels across and 4 down.
  page.fireElement("slamCanvas", "pointerdown", { clientX: 10, clientY: 10, pointerId: 1 });
  page.fireElement("slamCanvas", "pointermove", { clientX: 20, clientY: 10, pointerId: 1 });
  page.fireElement("slamCanvas", "pointerup", { pointerId: 1 });
  let model = page.slam;
  assert.equal(model.view.follow, false);
  assert.equal(page.el("slamFollow").checked, false);
  near(model.view.cx, 1 - 64 / FIT_SCALE);
  near(model.view.cy, 2);
  page.fireElement("slamCanvas", "pointermove", { clientX: 30, clientY: 10, pointerId: 1 });
  near(page.slam.view.cx, model.view.cx, 1e-9);
  const before = model.view.scale;
  page.fireElement("slamCanvas", "wheel", { clientX: 50, clientY: 50, deltaY: -100, preventDefault() {} });
  near(page.slam.view.scale, before * 1.15);
  page.fireElement("slamZoomOut", "click", {});
  near(page.slam.view.scale, before);
  page.fireElement("slamZoomIn", "click", {});
  near(page.slam.view.scale, before * 1.15);
  page.fireElement("slamReset", "click", {});
  model = page.slam;
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  page.el("slamFollow").checked = true;
  page.fireElement("slamFollow", "change", {});
  assert.equal(page.slam.view.cx, 1);
  assert.equal(page.slam.view.cy, 2);
});
