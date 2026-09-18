// The Map panel: the car's SLAM map, pose, LiDAR scan and trajectory on a 2D
// canvas, fed by three polled routes: /api/slam, /api/slam/map.png and
// /api/slam/trajectory. Spec: docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md.
//
// Two layers, like gamepad.js and app.js. Everything above "The DOM layer" is
// pure (no DOM, no fetch, no page globals) and exported for node --test.
// Everything below touches the page and is exercised through the vm harness
// in tests/web/harness.mjs. app.js calls startSlamPanel(els) once at load.

const SLAM_POLL_MS = 250;
const SLAM_FETCH_TIMEOUT_MS = 4000; // the same bound app.js puts on its own fetches
const SLAM_UNREACHABLE_FAILURES = 3;
const SLAM_ZOOM_STEP = 1.15;
const SLAM_MIN_SCALE = 20; // canvas pixels per metre
const SLAM_MAX_SCALE = 400;
const SLAM_DEFAULT_SCALE = 60;
const SLAM_FIT_MARGIN = 1.2;

function slamClamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function newSlamModel(width = 640, height = 400) {
  return {
    state: "connecting",
    reason: null,
    slam: null,
    pose: null,
    mapOdom: null,
    scan: null,
    // The PNG currently decoded: its placement metadata (from the PNG's own
    // response headers) and the image handle.
    map: null,
    mapImage: null,
    // What the last snapshot said the bridge holds. slamPlan compares it
    // with `map` to decide whether the PNG needs fetching.
    mapAvailable: null,
    trajectory: { epoch: 0, points: [] },
    trajectoryAvailable: { epoch: 0, count: 0 },
    view: { scale: SLAM_DEFAULT_SCALE, cx: 0, cy: 0, follow: true, fitted: false },
    canvas: { width, height },
    failures: 0,
    lastError: null,
    hidden: false,
    tabHidden: false,
  };
}

// The world-to-canvas transform. y is up in the world and down on the canvas.
function slamView(model, width, height) {
  const { scale, cx, cy } = model.view;
  return {
    scale,
    toCanvas(wx, wy) {
      return [(wx - cx) * scale + width / 2, height / 2 - (wy - cy) * scale];
    },
    toWorld(px, py) {
      return [cx + (px - width / 2) / scale, cy - (py - height / 2) / scale];
    },
  };
}

function slamMapBounds(meta) {
  const w = meta.width * meta.resolution;
  const h = meta.height * meta.resolution;
  const yaw = meta.origin.yaw || 0;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const corners = [[0, 0], [w, 0], [0, h], [w, h]].map(([u, v]) => [meta.origin.x + c * u - s * v, meta.origin.y + s * u + c * v]);
  const xs = corners.map((p) => p[0]);
  const ys = corners.map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

function slamFitView(bounds, width, height) {
  const w = Math.max(bounds.maxX - bounds.minX, 0.5);
  const h = Math.max(bounds.maxY - bounds.minY, 0.5);
  const scale = slamClamp(Math.min(width / (w * SLAM_FIT_MARGIN), height / (h * SLAM_FIT_MARGIN)), SLAM_MIN_SCALE, SLAM_MAX_SCALE);
  return { scale, cx: (bounds.minX + bounds.maxX) / 2, cy: (bounds.minY + bounds.maxY) / 2 };
}

function fitViewInto(model, meta) {
  Object.assign(model.view, slamFitView(slamMapBounds(meta), model.canvas.width, model.canvas.height));
  model.view.fitted = true;
}

function slamReduce(model, event) {
  const next = { ...model, view: { ...model.view } };
  switch (event.type) {
    case "resize":
      next.canvas = { width: event.width, height: event.height };
      return next;
    case "snapshot": {
      const body = event.body || {};
      const bridge = body.bridge || {};
      next.state = typeof bridge.state === "string" ? bridge.state : "unknown";
      next.reason = bridge.reason || null;
      next.slam = body.slam || null;
      next.pose = body.pose || null;
      next.mapOdom = body.map_odom || null;
      next.scan = body.scan || null;
      next.mapAvailable = body.map || null;
      const trajectory = body.trajectory || {};
      next.trajectoryAvailable = { epoch: Number(trajectory.epoch) || 0, count: Number(trajectory.count) || 0 };
      next.failures = 0;
      next.lastError = null;
      if (!next.view.fitted && next.mapAvailable) fitViewInto(next, next.mapAvailable);
      if (next.view.follow && next.pose) {
        next.view.cx = next.pose.x;
        next.view.cy = next.pose.y;
      }
      return next;
    }
    case "map":
      next.map = event.meta;
      next.mapImage = event.image;
      if (!next.view.fitted) fitViewInto(next, event.meta);
      return next;
    case "mapMissing":
      // The bridge has no grid right now (a 404). Whatever image is held
      // stays on screen under the state overlay; there is nothing to fetch.
      next.mapAvailable = null;
      return next;
    case "trajectory":
      next.trajectory = slamMerge(model.trajectory, event.reply);
      return next;
    case "failure":
      next.failures = model.failures + 1;
      next.lastError = event.error || "request failed";
      return next;
    case "hidden":
      next.hidden = Boolean(event.hidden);
      return next;
    case "tabHidden":
      next.tabHidden = Boolean(event.hidden);
      return next;
    case "drag":
      // Dragging the picture right moves the centre of view left.
      next.view.cx = model.view.cx - event.dx / model.view.scale;
      next.view.cy = model.view.cy + event.dy / model.view.scale;
      next.view.follow = false;
      return next;
    case "wheel": {
      const { width, height } = model.canvas;
      const [wx, wy] = slamView(model, width, height).toWorld(event.atX, event.atY);
      const scale = slamClamp(model.view.scale * event.factor, SLAM_MIN_SCALE, SLAM_MAX_SCALE);
      next.view.scale = scale;
      // Whatever was under the cursor stays under it.
      next.view.cx = wx - (event.atX - width / 2) / scale;
      next.view.cy = wy + (event.atY - height / 2) / scale;
      return next;
    }
    case "zoom":
      next.view.scale = slamClamp(model.view.scale * event.factor, SLAM_MIN_SCALE, SLAM_MAX_SCALE);
      return next;
    case "follow":
      next.view.follow = Boolean(event.on);
      if (next.view.follow && model.pose) {
        next.view.cx = model.pose.x;
        next.view.cy = model.pose.y;
      }
      return next;
    case "reset": {
      const target = model.map || model.mapAvailable;
      if (target) {
        fitViewInto(next, target);
      } else if (model.pose) {
        const { x, y } = model.pose;
        Object.assign(next.view, slamFitView({ minX: x - 2.5, maxX: x + 2.5, minY: y - 2.5, maxY: y + 2.5 }, model.canvas.width, model.canvas.height));
        next.view.fitted = true;
      }
      next.view.follow = false;
      return next;
    }
    default:
      return model;
  }
}

// Which fetches the next cycle needs beyond the snapshot.
function slamPlan(model) {
  const available = model.mapAvailable;
  const map = Boolean(available) && (!model.map || model.map.version !== available.version);
  const have = model.trajectory;
  const want = model.trajectoryAvailable;
  const haveCount = have.points.length / 2;
  const sameEpoch = have.epoch === want.epoch;
  const trajectory = !sameEpoch || want.count !== haveCount;
  return { map, trajectory, trajectoryQuery: { epoch: want.epoch, from: sameEpoch ? haveCount : 0 } };
}

function slamMerge(list, reply) {
  const points = Array.isArray(reply.points) ? reply.points : [];
  if (reply.epoch !== list.epoch) return { epoch: reply.epoch, points: points.slice() };
  const from = Math.max(0, Number(reply.from) || 0);
  return { epoch: list.epoch, points: list.points.slice(0, from * 2).concat(points) };
}

function slamOverlay(model) {
  if (model.failures >= SLAM_UNREACHABLE_FAILURES) return { title: "Car not reachable", detail: model.lastError || "" };
  switch (model.state) {
    case "mapping":
      return null;
    case "connecting":
      return { title: "Connecting to the car", detail: "" };
    case "slam_unreachable":
      return { title: "SLAM service not running", detail: model.reason || "" };
    case "slam_down":
      return { title: "slam_toolbox restarting", detail: model.reason || "" };
    case "waiting_for_scan":
      return { title: "Waiting for LiDAR", detail: "" };
    case "waiting_for_odom_tf":
      return { title: "Waiting for odometry", detail: "" };
    case "waiting_for_map":
      return { title: "Waiting for first map", detail: "" };
    default:
      return { title: String(model.state), detail: model.reason || "" };
  }
}

// Placement metadata from the PNG response's own headers, never from an
// earlier snapshot, so the picture and its metres cannot disagree.
function slamMapMetaFromHeaders(get) {
  const number = (name) => Number(get(name));
  return {
    version: number("X-Map-Version"),
    width: number("X-Map-Width"),
    height: number("X-Map-Height"),
    resolution: number("X-Map-Resolution"),
    origin: { x: number("X-Map-Origin-X"), y: number("X-Map-Origin-Y"), yaw: number("X-Map-Origin-Yaw") },
  };
}

function slamStateText(model) {
  if (model.hidden) return "hidden";
  if (model.failures >= SLAM_UNREACHABLE_FAILURES) return "car unreachable";
  return String(model.state).replaceAll("_", " ");
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function slamStatsText(model) {
  const saves = model.slam && Number.isFinite(model.slam.saves) ? model.slam.saves : 0;
  return `${plural(saves, "save")}, ${plural(model.trajectoryAvailable.count, "pose")}`;
}

function slamReadoutText(model) {
  const parts = [];
  if (model.pose) {
    const heading = Math.round((model.pose.yaw * 180) / Math.PI);
    parts.push(`x ${model.pose.x.toFixed(2)} m`, `y ${model.pose.y.toFixed(2)} m`, `heading ${heading}°`);
  }
  const map = model.map || model.mapAvailable;
  if (map) parts.push(`map ${(map.width * map.resolution).toFixed(1)} × ${(map.height * map.resolution).toFixed(1)} m`);
  if (!parts.length) parts.push("Waiting for the car");
  return parts.join("  ");
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SLAM_POLL_MS,
    SLAM_FETCH_TIMEOUT_MS,
    SLAM_UNREACHABLE_FAILURES,
    SLAM_ZOOM_STEP,
    SLAM_MIN_SCALE,
    SLAM_MAX_SCALE,
    SLAM_DEFAULT_SCALE,
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
  };
}
