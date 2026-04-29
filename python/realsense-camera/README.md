# RealSense Camera Sample

Live multi-stream viewer for an Intel RealSense **D415** depth camera.

* `./server` — Python (uv, 3.14) server that owns the librealsense
  pipeline and exposes one MJPEG endpoint per stream.
* `./` (this directory) — Vite + React + Tailwind v4 + shadcn/ui frontend
  that drops each MJPEG endpoint into a `<Card>` tile.

The D415 has two IR imagers (a left imager and a right imager) used for
active-stereo depth. The four UI tiles map directly to the four streams
the device exposes:

| Tile             | RealSense stream                         |
| ---------------- | ---------------------------------------- |
| Color Stream     | `rs.stream.color` (BGR8)                 |
| Left IR Stream   | `rs.stream.infrared` index `1` (Y8)      |
| Right IR Stream  | `rs.stream.infrared` index `2` (Y8)      |
| Depth Stream     | `rs.stream.depth` (Z16) → colorized      |

## Quick start

In two terminals:

```bash
# Terminal 1 — Python server (port 8000)
cd server
uv sync
uv run python main.py

# Terminal 2 — frontend (port 5454)
npm install
npm run dev -- --port 5454
```

Open <http://localhost:5454/>, pick resolution + FPS, hit **Start**.
The Vite dev server proxies `/stream`, `/config`, and `/health` to the
Python server, so the browser uses same-origin paths only.

## Layout behaviour

* All four checkboxes on by default → 2 × 2 grid.
* Two streams enabled → side-by-side.
* Exactly one enabled → that stream goes fullscreen (no quadrant).
* None enabled → empty-state hint.

## Files of interest

* `src/App.tsx` — UI, stream toggles, MJPEG `<img>` wiring.
* `vite.config.ts` — proxy config.
* `server/main.py` — librealsense pipeline + MJPEG endpoints.
* `server/README.md` — server-specific docs and Python 3.14 caveat.

## Hardware notes

Targeted at the D415 specifically. Other RealSense devices that expose
color + two IR + depth (D435, D435i, D455, …) should also work; if the
camera exposes only a single IR imager, the right-IR tile simply will
not receive frames.
