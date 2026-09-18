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
    case "cycleDone":
      next.failures = 0;
      next.lastError = null;
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
  // A same-epoch count smaller than what the client already holds cannot be
  // the same server: epoch counters are per process, and within one
  // incarnation the trajectory is append-only. It is a new bridge that
  // relatched the keeper's path as epoch 1 again after a web-service
  // restart, so resync from 0 rather than asking for a `from` the relatched
  // path can never satisfy.
  const from = sameEpoch && want.count >= haveCount ? haveCount : 0;
  return { map, trajectory, trajectoryQuery: { epoch: want.epoch, from } };
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
  if (model.state === "slam_unreachable") return "SLAM service not running";
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

// The DOM layer ==============================================================
//
// One model, replaced wholesale by the reducer; one in-flight guard; one
// render per completed cycle or interaction. Nothing below runs under
// node --test except through the vm harness.

let slamModel = newSlamModel();
let slamInFlight = false;
let slamEls = null;

async function slamFetch(path, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SLAM_FETCH_TIMEOUT_MS);
  try {
    return await fetch(path, { cache: "no-store", signal: controller.signal, headers });
  } finally {
    clearTimeout(timer);
  }
}

async function slamFetchJson(path) {
  const response = await slamFetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}

function slamDecodeImage(blob) {
  if (typeof createImageBitmap === "function") return createImageBitmap(blob);
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("map image failed to decode")); };
    image.src = url;
  });
}

async function slamFetchMap(heldVersion) {
  const headers = heldVersion === null ? undefined : { "If-None-Match": `"${heldVersion}"` };
  const response = await slamFetch("/api/slam/map.png", headers);
  if (response.status === 304 || response.status === 404) return { status: response.status };
  if (!response.ok) throw new Error(`/api/slam/map.png answered ${response.status}`);
  const meta = slamMapMetaFromHeaders((name) => response.headers.get(name));
  const image = await slamDecodeImage(await response.blob());
  return { status: 200, meta, image };
}

// One cycle: the snapshot, then the PNG only if its version moved, then the
// trajectory only if it grew or reset, then one redraw. Sequential behind
// one guard, so the panel never holds more than one browser socket: the
// 2026-08 freeze was the per-origin connection budget, and this panel must
// not spend it.
async function refreshSlam() {
  if (slamInFlight || slamModel.hidden || slamModel.tabHidden) return;
  slamInFlight = true;
  try {
    const body = await slamFetchJson("/api/slam");
    slamModel = slamReduce(slamModel, { type: "snapshot", body });
    const plan = slamPlan(slamModel);
    if (plan.map) {
      const result = await slamFetchMap(slamModel.map ? slamModel.map.version : null);
      if (result.status === 200) {
        // Captured before the reduce, which replaces slamModel.mapImage with
        // the new bitmap; the old one is only reachable through this local
        // once the reduce runs. The reducer stays pure: this is a side
        // effect the reduce itself must not perform.
        const previousImage = slamModel.mapImage;
        slamModel = slamReduce(slamModel, { type: "map", meta: result.meta, image: result.image });
        if (previousImage && typeof previousImage.close === "function") previousImage.close();
      } else if (result.status === 404) {
        slamModel = slamReduce(slamModel, { type: "mapMissing" });
      }
    }
    if (plan.trajectory) {
      const { epoch, from } = plan.trajectoryQuery;
      const reply = await slamFetchJson(`/api/slam/trajectory?epoch=${epoch}&from=${from}`);
      slamModel = slamReduce(slamModel, { type: "trajectory", reply });
    }
    slamModel = slamReduce(slamModel, { type: "cycleDone" });
  } catch (error) {
    // Deliberately not noteControlFailure(): a slow map must never trip the
    // control breaker and blank the camera tiles.
    slamModel = slamReduce(slamModel, { type: "failure", error: String((error && error.message) || error) });
  } finally {
    slamInFlight = false;
  }
  renderSlamPanel();
}

function drawSlamOverlay(ctx, overlay, width, height) {
  ctx.globalAlpha = 1;
  ctx.textAlign = "center";
  ctx.fillStyle = "#eef2ef";
  ctx.font = "800 20px system-ui, sans-serif";
  ctx.fillText(overlay.title, width / 2, height / 2 - 6);
  if (overlay.detail) {
    ctx.fillStyle = "#b7c3bd";
    ctx.font = "14px system-ui, sans-serif";
    ctx.fillText(overlay.detail, width / 2, height / 2 + 18);
  }
}

function drawSlam(ctx, model, width, height) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#080a09";
  ctx.fillRect(0, 0, width, height);
  const view = slamView(model, width, height);
  const overlay = slamOverlay(model);
  // Anything but mapping dims the scene under the state text; the last map
  // stays visible, so a slam restart never blanks the panel.
  ctx.globalAlpha = overlay ? 0.5 : 1;
  if (model.mapImage && model.map) drawSlamMap(ctx, model.map, model.mapImage, view);
  drawSlamTrajectory(ctx, model.trajectory.points, view);
  if (model.pose) {
    if (model.scan) drawSlamScan(ctx, model.scan.points, model.pose, view);
    drawSlamRobot(ctx, model.pose, view);
  }
  ctx.globalAlpha = 1;
  drawSlamScaleBar(ctx, view, width, height);
  if (overlay) drawSlamOverlay(ctx, overlay, width, height);
}

function drawSlamMap(ctx, meta, image, view) {
  const [ox, oy] = view.toCanvas(meta.origin.x, meta.origin.y);
  const cell = view.scale * meta.resolution;
  ctx.save();
  ctx.translate(ox, oy);
  // World angles turn counter-clockwise; canvas y points down, so the same
  // turn is clockwise on screen.
  ctx.rotate(-(meta.origin.yaw || 0));
  ctx.imageSmoothingEnabled = false;
  // The PNG is north-up (row 0 is the grid's highest y), so its top edge
  // sits `height` cells above the origin and its bottom edge on it.
  ctx.drawImage(image, 0, -meta.height * cell, meta.width * cell, meta.height * cell);
  ctx.restore();
}

function drawSlamTrajectory(ctx, points, view) {
  if (points.length < 4) return;
  ctx.strokeStyle = "#f0b429";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < points.length; i += 2) {
    const [px, py] = view.toCanvas(points[i], points[i + 1]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function drawSlamScan(ctx, points, pose, view) {
  const c = Math.cos(pose.yaw);
  const s = Math.sin(pose.yaw);
  ctx.fillStyle = "#58c897";
  for (let i = 0; i < points.length; i += 2) {
    const sx = points[i];
    const sy = points[i + 1];
    const [px, py] = view.toCanvas(pose.x + c * sx - s * sy, pose.y + s * sx + c * sy);
    ctx.fillRect(px - 1, py - 1, 2, 2);
  }
}

function drawSlamRobot(ctx, pose, view) {
  // 0.30 m long by 0.20 m wide in world units, never under 10 px.
  const length = Math.max(0.3 * view.scale, 10);
  const half = length / 3;
  const [cx, cy] = view.toCanvas(pose.x, pose.y);
  const dx = Math.cos(pose.yaw);
  const dy = -Math.sin(pose.yaw);
  const nx = -dy;
  const ny = dx;
  ctx.fillStyle = "#eef2ef";
  ctx.beginPath();
  ctx.moveTo(cx + dx * length * 0.6, cy + dy * length * 0.6);
  ctx.lineTo(cx - dx * length * 0.4 + nx * half, cy - dy * length * 0.4 + ny * half);
  ctx.lineTo(cx - dx * length * 0.4 - nx * half, cy - dy * length * 0.4 - ny * half);
  ctx.closePath();
  ctx.fill();
}

function drawSlamScaleBar(ctx, view, width, height) {
  const metres = view.scale >= 40 ? 1 : 5;
  const x = 12;
  const y = height - 14;
  ctx.strokeStyle = "#b7c3bd";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + metres * view.scale, y);
  ctx.stroke();
  ctx.fillStyle = "#b7c3bd";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${metres} m`, x, y - 5);
}

// CSS pixels to canvas pixels: the canvas is 640 x 400 in its own units but
// is laid out at the column's width.
function slamCanvasScale(canvas) {
  const rect = canvas.getBoundingClientRect();
  return { x: rect.width ? canvas.width / rect.width : 1, y: rect.height ? canvas.height / rect.height : 1 };
}

function wireSlamPointer(canvas) {
  let dragging = null;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = { x: event.clientX, y: event.clientY };
    if (typeof canvas.setPointerCapture === "function" && event.pointerId !== undefined) canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const k = slamCanvasScale(canvas);
    slamModel = slamReduce(slamModel, { type: "drag", dx: (event.clientX - dragging.x) * k.x, dy: (event.clientY - dragging.y) * k.y });
    dragging = { x: event.clientX, y: event.clientY };
    renderSlamPanel();
  });
  const release = () => { dragging = null; };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (event) => {
    // preventDefault first regardless: this listener is registered
    // non-passive precisely so the page can stop the canvas from scrolling,
    // and a horizontal scroll (deltaY === 0) still needs that even though it
    // is not a zoom.
    if (typeof event.preventDefault === "function") event.preventDefault();
    if (!event.deltaY) return;
    const rect = canvas.getBoundingClientRect();
    const k = slamCanvasScale(canvas);
    const factor = event.deltaY < 0 ? SLAM_ZOOM_STEP : 1 / SLAM_ZOOM_STEP;
    slamModel = slamReduce(slamModel, { type: "wheel", factor, atX: (event.clientX - rect.left) * k.x, atY: (event.clientY - rect.top) * k.y });
    renderSlamPanel();
  }, { passive: false });
}

function renderSlamPanel() {
  if (!slamEls) return;
  slamEls.slamState.textContent = slamStateText(slamModel);
  slamEls.slamStats.textContent = slamStatsText(slamModel);
  slamEls.slamReadout.textContent = slamReadoutText(slamModel);
  slamEls.slamReason.textContent = slamModel.reason || "";
  slamEls.slamFollow.checked = slamModel.view.follow;
  slamEls.slamHide.textContent = slamModel.hidden ? "Show" : "Hide";
  slamEls.slamBody.classList.toggle("hidden", slamModel.hidden);
  if (!slamModel.hidden) drawSlam(slamEls.slamCanvas.getContext("2d"), slamModel, slamEls.slamCanvas.width, slamEls.slamCanvas.height);
}

function startSlamPanel(els) {
  slamEls = els;
  slamModel = slamReduce(slamModel, { type: "resize", width: els.slamCanvas.width, height: els.slamCanvas.height });
  // Seeded before the first refreshSlam(), so a page opened in a background
  // tab does not spend its first poll before visibilitychange has ever fired.
  slamModel = slamReduce(slamModel, { type: "tabHidden", hidden: Boolean(document.hidden) });
  wireSlamPointer(els.slamCanvas);
  els.slamFollow.addEventListener("change", () => {
    slamModel = slamReduce(slamModel, { type: "follow", on: els.slamFollow.checked });
    renderSlamPanel();
  });
  els.slamReset.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "reset" });
    renderSlamPanel();
  });
  els.slamZoomIn.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "zoom", factor: SLAM_ZOOM_STEP });
    renderSlamPanel();
  });
  els.slamZoomOut.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "zoom", factor: 1 / SLAM_ZOOM_STEP });
    renderSlamPanel();
  });
  els.slamHide.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "hidden", hidden: !slamModel.hidden });
    renderSlamPanel();
    if (!slamModel.hidden) refreshSlam();
  });
  document.addEventListener("visibilitychange", () => {
    slamModel = slamReduce(slamModel, { type: "tabHidden", hidden: Boolean(document.hidden) });
    if (!slamModel.tabHidden) refreshSlam();
  });
  setInterval(refreshSlam, SLAM_POLL_MS);
  renderSlamPanel();
  refreshSlam();
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
