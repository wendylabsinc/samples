// The camera streams versus the control channel.
//
// Field incident, 2026-08-06, Safari: the page holds one MJPEG connection per
// tile, and when reconnectFeed re-pointed a tile's src Safari kept the OLD
// multipart connection alive alongside the new one. Each periodic refresh
// added a zombie socket until the browser's per-origin connection limit was
// exhausted, at which point every drive, gamepad and status fetch queued
// behind dead streams and aborted at FETCH_TIMEOUT_MS. The operator saw the
// car freeze ~30-60 s into every session; the car itself was healthy and
// answering other clients the whole time.
//
// Two behaviors close that hole, and this file pins both:
//
//   1. reconnectFeed aborts the old stream (src = "") and opens the new one
//      on a FRESH element, so the old connection is released rather than
//      accumulated.
//
//   2. A circuit breaker: consecutive control-channel failures suspend every
//      stream (src = ""), freeing the connection pool for the traffic that
//      steers the car. Video must never win a socket contest against driving.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadPage } from "./harness.mjs";

async function pageWithTiles() {
  const page = loadPage();
  await page.run("refreshStatus()");
  await page.settle();
  assert.deepEqual(page.tileIds(), ["hp60c_depth", "hp60c_rgb"], "harness feeds produce two tiles");
  return page;
}

// One stream at a time =======================================================
//
// Six connections per origin is the browser's budget. Four permanent MJPEG
// streams left two sockets for three control channels (drive, gamepad,
// status), and on a link with a 1212 ms p90 those three overlap constantly:
// demand exceeded capacity, the queue never drained, and the page went
// silent mid-drive while the car sat healthy. So at most ONE tile -- the
// expanded one -- may hold a stream, and the page starts with the first
// reported feed expanded so the operator always has a live view.

test("FEEDS: only the first-reported feed streams on load; the rest are disconnected", async () => {
  const page = await pageWithTiles();
  assert.equal(page.state.expandedFeed, "hp60c_depth", "the first feed the car reports is expanded on load");
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg\?r=/);
  assert.equal(page.tile("hp60c_rgb").img.src, "", "a tile that is not expanded rents no socket");
});

test("FEEDS: expanding another feed moves the single stream to it", async () => {
  const page = await pageWithTiles();
  page.run("setExpandedFeed('hp60c_rgb')");
  assert.equal(page.tile("hp60c_depth").img.src, "", "the previous stream is closed, not accumulated");
  assert.match(page.tile("hp60c_rgb").img.src, /frame_hp60c_rgb\.jpg\?r=/);
});

test("FEEDS: collapsing to the gallery closes the last stream", async () => {
  const page = await pageWithTiles();
  page.run("toggleExpandedFeed('hp60c_depth')");
  assert.equal(page.state.expandedFeed, null);
  for (const id of page.tileIds()) {
    assert.equal(page.tile(id).img.src, "", `${id} holds no stream in gallery view`);
  }
});

test("FEEDS: the periodic refresh never opens a stream on a non-expanded tile", async () => {
  const page = await pageWithTiles();
  page.run("reconnectFeed('hp60c_rgb')");
  page.run("reconnectNextFeed()");
  page.run("reconnectNextFeed()");
  assert.equal(page.tile("hp60c_rgb").img.src, "", "only the expanded feed may hold a stream");
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg/);
});

test("FEEDS: a loaded frame schedules the next request with a fresh cache buster", async () => {
  const page = await pageWithTiles();
  const first = page.tile("hp60c_depth").img.src;
  assert.match(first, /frame_hp60c_depth\.jpg\?r=/, "the tile polls finite frames, it never opens a stream");

  page.tile("hp60c_depth").img.dispatch("load", {});
  await page.settle();

  const second = page.tile("hp60c_depth").img.src;
  assert.match(second, /frame_hp60c_depth\.jpg\?r=/);
  assert.notEqual(second, first, "each frame is its own completing request; nothing can pin a connection");
});

test("FEEDS: a frame request that never answers is retried after the stall bound", async () => {
  const page = await pageWithTiles();
  const stalled = page.tile("hp60c_depth").img.src;

  // No load, no error: the request just hangs. The stall timer is a long
  // timeout, which the harness holds like a fetch timeout; expiring it is
  // the bound elapsing.
  page.expireFetchTimeouts();
  await page.settle();

  const retried = page.tile("hp60c_depth").img.src;
  assert.match(retried, /frame_hp60c_depth\.jpg\?r=/);
  assert.notEqual(retried, stalled, "a hung frame request must not end the poll loop");
});

test("FEEDS: consecutive drive failures suspend every stream", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/drive");
  for (let i = 0; i < 3; i += 1) {
    await page.run("sendDrive({ enabled: true, linear_x: 0.2, steering_y: 0, angular_z: 0 })");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, true, "three straight control failures mean the pool is starving; the streams give their sockets back");
  for (const id of page.tileIds()) {
    assert.equal(page.tile(id).img.src, "", `${id} stream is closed while suspended`);
  }
});

test("FEEDS: status-poll failures count toward suspension too", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/status");
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshStatus()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, true);
});

test("FEEDS: a control success resets the failure streak", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/drive");
  for (let i = 0; i < 2; i += 1) {
    await page.run("sendDrive({ enabled: true, linear_x: 0.2, steering_y: 0, angular_z: 0 })");
    await page.settle();
  }
  page.fake.failing.delete("/api/drive");
  await page.run("sendDrive({ enabled: true, linear_x: 0.2, steering_y: 0, angular_z: 0 })");
  await page.settle();
  page.fake.failing.add("/api/drive");
  for (let i = 0; i < 2; i += 1) {
    await page.run("sendDrive({ enabled: true, linear_x: 0.2, steering_y: 0, angular_z: 0 })");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, false, "two failures, a success, two failures is a flaky link, not a starved pool");
});

test("FEEDS: reconnect is a no-op while suspended, so nothing reopens a socket behind the breaker", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/status");
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshStatus()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, true);
  page.run("reconnectFeed('hp60c_depth')");
  assert.equal(page.tile("hp60c_depth").img.src, "", "the periodic refresh and error handlers must not undo the suspension");
});

test("FEEDS: streams come back staggered once control succeeds and the pause elapses", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/status");
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshStatus()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, true);

  page.fake.failing.delete("/api/status");
  await page.run("refreshStatus()");
  await page.settle();

  // The resume delay is a long timer, which the harness holds like a fetch
  // timeout; expiring it is the pause elapsing.
  page.expireFetchTimeouts();
  await page.settle();

  assert.equal(page.state.feedsSuspended, false);
  assert.match(page.tile("hp60c_depth").img.src, /\.jpg\?r=/, "the expanded stream reopened after the pause");
  assert.equal(page.tile("hp60c_rgb").img.src, "", "a non-expanded tile stays disconnected even after resume");
});

test("FEEDS: while control keeps failing the streams stay dark after the pause", async () => {
  const page = await pageWithTiles();
  page.fake.failing.add("/api/status");
  for (let i = 0; i < 4; i += 1) {
    await page.run("refreshStatus()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, true);

  page.expireFetchTimeouts();
  await page.settle();

  assert.equal(page.state.feedsSuspended, true, "a page that still cannot reach the car has no business spending sockets on video");
  for (const id of page.tileIds()) {
    assert.equal(page.tile(id).img.src, "");
  }
});
