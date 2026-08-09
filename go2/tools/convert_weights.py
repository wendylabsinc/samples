#!/usr/bin/env python3
"""Convert an Ultralytics YOLOv8n checkpoint to safetensors.

Runs on the DEVELOPMENT MACHINE ONLY. This exists so that torch never
has to be installed in a device image: installing torch on a Jetson
needs the jetson-ai-lab wheel index plus a CUDA soname-shadowing
workaround, and that workaround needs Stagefile ops that are not
available on the branch we build with.

Usage:
    pip install -r tools/requirements-dev.txt
    python tools/convert_weights.py --output weights/
"""

import argparse
import json
from pathlib import Path

import numpy as np


def convert_state_dict(state_dict, source):
    """Convert a torch-like state dict into numpy arrays plus a manifest.

    Entries that are not tensors (epoch counters, optimizer state) are
    skipped. Everything is coerced to float32: bfloat16 weights fail to
    build on Orin due to its ARM CPU/GPU pairing, so we never emit them.
    """
    tensors = {}
    manifest = {"source": source, "tensors": {}}

    for name, value in state_dict.items():
        if not hasattr(value, "numpy"):
            continue

        array = np.ascontiguousarray(value.detach().cpu().numpy())
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        tensors[name] = array
        manifest["tensors"][name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    return tensors, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()

    from safetensors.numpy import save_file
    from ultralytics import YOLO

    model = YOLO(args.model)
    state_dict = model.model.state_dict()

    tensors, manifest = convert_state_dict(state_dict, source=args.model)

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output / "yolov8n.safetensors"))
    (args.output / "yolov8n.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
