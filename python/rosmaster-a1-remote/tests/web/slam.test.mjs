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

test("SLAM model: failures count up, three mean unreachable, one snapshot clears them", () => {
  let model = newSlamModel();
  for (let i = 0; i < SLAM_UNREACHABLE_FAILURES; i += 1) model = slamReduce(model, { type: "failure", error: "offline" });
  assert.deepEqual(slamOverlay(model), { title: "Car not reachable", detail: "offline" });
  assert.equal(slamStateText(model), "car unreachable");
  model = slamReduce(model, { type: "snapshot", body: snapshot() });
  assert.equal(model.failures, 0);
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
  assert.equal(slamStatsText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ slam: { saves: 1 }, trajectory: { epoch: 1, count: 1 } }) })), "1 save, 1 pose");
});
