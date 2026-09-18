#!/usr/bin/env python3
"""Render <out>/maps/latest/map.pgm with the slam trajectory (blue, start
green, end red) and the odometry (orange, placed with the final map->odom
correction) from <out>/traj.json, to <out>/map_overlay.png. Needs Pillow
(in the .venv). Used by scripts/slam_offline_check.sh."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main(out_dir: str) -> None:
    out = Path(out_dir)
    latest = out / "maps" / "latest"
    meta = {}
    for line in (latest / "map.yaml").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    res = float(meta["resolution"])
    origin = json.loads(meta["origin"])
    img = Image.open(latest / "map.pgm").convert("RGB")
    w, h = img.size
    scale = max(1, 1600 // max(w, h))
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    def to_px(x, y):
        return (x - origin[0]) / res * scale, (h - (y - origin[1]) / res) * scale

    traj = json.load(open(out / "traj.json"))
    mo = traj["map_odom"][-1] if traj["map_odom"] else [0, 0, 0, 0]
    c, s = math.cos(mo[3]), math.sin(mo[3])
    odom_in_map = [to_px(mo[1] + c * p[1] - s * p[2], mo[2] + s * p[1] + c * p[2]) for p in traj["odom"]]
    if len(odom_in_map) > 1:
        draw.line(odom_in_map, fill=(255, 140, 0), width=1)
    slam = [to_px(p[1], p[2]) for p in traj["slam"]]
    if len(slam) > 1:
        draw.line(slam, fill=(0, 90, 255), width=2)
    if slam:
        for (x, y), colour in ((slam[0], (0, 160, 0)), (slam[-1], (220, 0, 0))):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=colour, width=3)
    img.save(out / "map_overlay.png")
    print("saved", out / "map_overlay.png", "grid", w, "x", h, "res", res, "origin", origin)


if __name__ == "__main__":
    main(sys.argv[1])
