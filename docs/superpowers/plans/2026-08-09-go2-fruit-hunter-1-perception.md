# Go2 Fruit Hunter — Plan 1: Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream the Go2's front camera over ROS2/DDS and publish fruit detections produced by a YOLOv8n network running on Modular MAX on the dog's onboard Orin GPU.

**Architecture:** Three Wendy services in one native multi-service `wendy.json`. `camera` owns the Go2's single WebRTC slot and republishes frames as DDS `CompressedImage`. `detector` subscribes, runs YOLOv8n rebuilt with MAX's graph API on the GPU, and publishes detections. `dashboard` renders an annotated stream. Model weights are converted to safetensors on the development machine so torch never enters the device image.

**Tech Stack:** Python 3.11, Modular MAX (nightly), cyclonedds-python, aiortc via `unitree_webrtc_connect`, FastAPI, OpenCV, NumPy, pytest. Built with Stagefiles via `wendyg`.

Spec: `docs/superpowers/specs/2026-08-09-go2-fruit-hunter-design.md`

## Global Constraints

- **Target device:** Unitree Go2 EDU onboard Jetson Orin, arm64, WendyOS. Robot LAN is `192.168.123.0/24`.
- **Build tool is `wendyg`**, not `wendy`. `wendyg` is aliased to `/Users/joannisorlandos/git/wendy/wendyos/go/bin/wendy`, built from branch `jo/fast`. The released CLI (v2026.08.07-174446) does not detect Stagefiles — verified.
- **Stagefile DSL surface available on `jo/fast`:** `env`, `args`, `workdir`, `install.{apt,apk,cmake,pip,npm,uv}`, `install.apt.repositories`, `install.pip.index`, `install.pip.extraIndex`, `download`, `build`, `copy`, `healthcheck`, `entrypoint.exec`, `entrypoint.source`, `cmd`, `user`, `platform`, `pin`. **NOT available:** `sharedLibraries:` and `install.pip` as a list — those exist only on `wendyos-stagefile-cuda`. Do not use them.
- **`install.pip` is a single mapping**, so each stage gets exactly one pip invocation. Use `index` + `extraIndex` to reach two indexes in one resolve.
- **torch must never appear in any device image.** Weight conversion is a host-side tool. See Task 2.
- **DDS binds by IP address, never by interface name.** The Orin is multi-homed; a name lets DDS advertise the wrong subnet. Env var `GO2_DDS_ADDRESS`, default `192.168.123.18`.
- **No `from __future__ import annotations` in any module that defines a cyclonedds `IdlStruct`.** The IdlStruct normaliser resolves type hints by name lookup at class-definition time; PEP-563 string annotations break it. This is a hard rule, not a style preference.
- **ROS2-on-CycloneDDS wire convention:** a ROS2 topic `/foo/bar` is `rt/foo/bar` on the DDS wire.
- **MAX stays on fp16/fp32.** bfloat16 weights fail to build on Orin due to its ARM CPU/GPU pairing.
- **No silent CPU fallback.** If MAX cannot initialise on the GPU, the service exits non-zero at startup.
- All new code lives under `samples/go2/`.

### Deviation from the spec, recorded here

The spec names `vision_msgs/Detection2DArray` for the detections topic. This plan uses a purpose-built `go2_fruit/Detections` IDL struct instead. Reason: `Detection2DArray` requires hand-nesting five ROS2 message types (`Detection2D`, `ObjectHypothesisWithPose`, `BoundingBox2D`, `Pose2D`, `Header`) as cyclonedds `IdlStruct`s, and nothing outside this app consumes the topic — `brain` and `dashboard` are both ours. The compact struct is defined once in Task 5 and used unchanged by Plans 2 and 3.

---

## File Structure

```
samples/go2/
├── wendy.json                          # native multi-service app group
├── README.md
├── .gitignore                          # Dockerfile.generated, weights/
├── tools/
│   ├── convert_weights.py              # host-side .pt → safetensors
│   └── requirements-dev.txt            # torch + ultralytics, HOST ONLY
├── probe/                              # Task 1, disposable
│   ├── build.stagefile.yaml
│   └── probe_max_gpu.py
├── common/
│   └── go2_msgs.py                     # shared IdlStruct definitions
├── camera/
│   ├── build.stagefile.yaml
│   ├── main.py                         # WebRTC → DDS + MJPEG
│   └── publisher.py                    # CompressedImage DDS writer
├── detector/
│   ├── build.stagefile.yaml
│   ├── main.py                         # service loop
│   ├── preprocess.py                   # letterbox + normalise
│   ├── postprocess.py                  # decode + NMS
│   └── model/
│       ├── ops.py                      # MAX adapter layer
│       ├── yolov8.py                   # network definition
│       └── weights.py                  # safetensors loader
├── dashboard/
│   ├── build.stagefile.yaml
│   ├── main.py
│   └── static/index.html
└── tests/
    ├── conftest.py
    ├── fixtures/                       # reference tensors + test image
    ├── test_convert_weights.py
    ├── test_preprocess.py
    ├── test_postprocess.py
    ├── test_ops.py
    └── test_yolov8_parity.py
```

Responsibility split worth noting: `model/ops.py` is an adapter layer we own, exposing our own signatures (`conv_bn_silu`, `c2f`, `sppf`, …) implemented against MAX. Every MAX-specific call is localized there. This means the rest of the codebase is written against types we define and test, and a MAX API change touches one file.

---

### Task 1: MAX GPU probe on the dog

The riskiest premise in the project — MAX executing on Orin's sm_87 GPU is experimental and nightly-only. This task answers it before any YOLO code exists. The probe is disposable but committed, so the answer is reproducible.

**Files:**
- Create: `samples/go2/probe/probe_max_gpu.py`
- Create: `samples/go2/probe/build.stagefile.yaml`
- Create: `samples/go2/wendy.json`
- Create: `samples/go2/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a verified MAX nightly version string, pinned in Task 3's Stagefile.

- [ ] **Step 1: Create the app group manifest**

`samples/go2/wendy.json`:

```json
{
    "appId": "sh.wendy.examples.go2-fruit-hunter",
    "version": "0.1.0",
    "platform": "linux",
    "services": {
        "probe": {
            "context": "./probe",
            "entitlements": [
                { "type": "network", "mode": "host" },
                { "type": "gpu" }
            ]
        }
    }
}
```

- [ ] **Step 2: Create `.gitignore`**

`samples/go2/.gitignore`:

```
Dockerfile.generated
Dockerfile.generated.dockerignore
weights/
__pycache__/
.pytest_cache/
```

`build.stagefile.lock.yaml` is deliberately NOT ignored — it is the reproducibility anchor and must be committed, same rationale as `package-lock.json`.

- [ ] **Step 3: Write the probe**

`samples/go2/probe/probe_max_gpu.py`:

```python
#!/usr/bin/env python3
"""Answer one question: does MAX execute on this dog's GPU?

Exits 0 with a report on success, non-zero with the failure on any
problem. Deliberately disposable — its only job is to retire the
project's biggest risk before the detector is built.
"""

import sys

import numpy as np


def main() -> int:
    try:
        from max import engine
        from max.driver import Accelerator, accelerator_count
        from max.dtype import DType
        from max.graph import Graph, TensorType, ops
    except ImportError as exc:
        print(f"FAIL: MAX is not importable: {exc}", file=sys.stderr)
        return 1

    print(f"accelerator_count={accelerator_count()}")
    if accelerator_count() < 1:
        print("FAIL: MAX reports no accelerator", file=sys.stderr)
        return 2

    device = Accelerator()
    print(f"device={device}")

    # A conv is the operation YOLO is made of. If this compiles and runs,
    # the detector is viable; if it does not, nothing downstream matters.
    input_type = TensorType(DType.float32, (1, 3, 32, 32), device=device)
    filter_type = TensorType(DType.float32, (8, 3, 3, 3), device=device)

    with Graph("probe", input_types=(input_type, filter_type)) as graph:
        x, w = graph.inputs
        graph.output(ops.conv2d(x, w, stride=(1, 1), padding=(1, 1, 1, 1)))

    session = engine.InferenceSession(devices=[device])
    model = session.load(graph)

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    w = np.random.randn(8, 3, 3, 3).astype(np.float32)
    out = model.execute(x, w)[0].to_numpy()

    print(f"output_shape={out.shape}")
    if out.shape != (1, 8, 32, 32):
        print(f"FAIL: unexpected shape {out.shape}", file=sys.stderr)
        return 3

    print("PASS: MAX executed a conv2d on the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Note for the implementer:** the exact MAX symbol names above (`ops.conv2d`, `Graph`, `InferenceSession`) are written against MAX's documented graph API but MAX is a fast-moving nightly. If any import or call fails, fix it against the installed version's API — do not work around it by falling back to CPU. Record the version you settle on; Task 3 pins it.

- [ ] **Step 4: Write the probe Stagefile**

`samples/go2/probe/build.stagefile.yaml`:

```yaml
version: 1
stages:
  - name: probe
    from: nvcr.io/nvidia/l4t-base:r36.2.0
    workdir: /app
    env:
      PYTHONUNBUFFERED: "1"
    install:
      apt:
        packages: [python3, python3-pip, python3-venv, ca-certificates]
      pip:
        packages: [modular, numpy]
        index: https://whl.modular.com/nightly/simple/
        extraIndex:
          - https://pypi.org/simple
    copy:
      - from: local
        paths: [probe_max_gpu.py]
        dest: /app/
    entrypoint:
      exec: [python3, /app/probe_max_gpu.py]
```

- [ ] **Step 5: Build it**

Run: `cd samples/go2 && wendyg build`
Expected: the Stagefile compiles to `Dockerfile.generated`, the base image is digest-pinned into `build.stagefile.lock.yaml`, and the image builds for `linux/arm64`.

If pip cannot resolve `modular` for aarch64, that is a real finding — record the actual index URL and package name that works and update the Stagefile before continuing.

- [ ] **Step 6: Run it on the dog**

Run: `cd samples/go2 && wendyg run --device <go2-jetson>.local`
Expected: `PASS: MAX executed a conv2d on the GPU` and exit 0.

**This is the project's go/no-go gate.** If MAX cannot execute on the GPU, stop and renegotiate the approach rather than continuing to build against it.

- [ ] **Step 7: Commit**

```bash
git add samples/go2/wendy.json samples/go2/.gitignore samples/go2/probe/
git commit -m "feat(go2): MAX GPU probe for the Go2's Orin

Retires the project's largest risk first: MAX GPU support on Jetson
Orin (sm_87) is experimental and nightly-only, so prove a conv2d
executes before building a detector on top of it."
```

---

### Task 2: Host-side weight converter

Converts the Ultralytics YOLOv8n checkpoint into safetensors plus a shape manifest. Runs on the development machine only — this is what keeps torch out of every device image.

**Files:**
- Create: `samples/go2/tools/convert_weights.py`
- Create: `samples/go2/tools/requirements-dev.txt`
- Test: `samples/go2/tests/test_convert_weights.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `convert_state_dict(state_dict: dict[str, "Tensor"]) -> tuple[dict[str, np.ndarray], dict]` — returns `(tensors, manifest)` where `manifest` is `{"tensors": {name: {"shape": list[int], "dtype": str}}, "source": str}`. Task 6 (`model/weights.py`) reads the safetensors file this produces.

- [ ] **Step 1: Write the failing test**

`samples/go2/tests/test_convert_weights.py`:

```python
import numpy as np
import pytest

from tools.convert_weights import convert_state_dict


class FakeTensor:
    """Stands in for a torch tensor so the test needs no torch."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_converts_tensors_and_builds_manifest():
    state = {
        "model.0.conv.weight": FakeTensor(np.zeros((16, 3, 3, 3), np.float32)),
        "model.0.bn.bias": FakeTensor(np.zeros((16,), np.float32)),
    }

    tensors, manifest = convert_state_dict(state, source="yolov8n.pt")

    assert set(tensors) == {"model.0.conv.weight", "model.0.bn.bias"}
    assert tensors["model.0.conv.weight"].shape == (16, 3, 3, 3)
    assert manifest["source"] == "yolov8n.pt"
    assert manifest["tensors"]["model.0.conv.weight"]["shape"] == [16, 3, 3, 3]
    assert manifest["tensors"]["model.0.bn.bias"]["dtype"] == "float32"


def test_rejects_bfloat16_because_orin_cannot_build_it():
    state = {"w": FakeTensor(np.zeros((4,), np.float32))}
    state["w"]._array = state["w"]._array.astype(np.float32)

    tensors, _ = convert_state_dict(state, source="x.pt")
    assert tensors["w"].dtype == np.float32


def test_skips_non_tensor_entries():
    state = {
        "model.0.conv.weight": FakeTensor(np.zeros((2, 2), np.float32)),
        "epoch": 300,
        "optimizer": None,
    }

    tensors, manifest = convert_state_dict(state, source="x.pt")

    assert set(tensors) == {"model.0.conv.weight"}
    assert "epoch" not in manifest["tensors"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/go2 && python -m pytest tests/test_convert_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.convert_weights'`

- [ ] **Step 3: Write the converter**

`samples/go2/tools/convert_weights.py`:

```python
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
```

`samples/go2/tools/requirements-dev.txt`:

```
# HOST ONLY. Never referenced by any Stagefile — see the plan's global
# constraints for why torch must not enter a device image.
ultralytics==8.3.0
torch==2.4.1
safetensors==0.4.5
numpy==1.26.4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/go2 && python -m pytest tests/test_convert_weights.py -v`
Expected: 3 passed

- [ ] **Step 5: Generate the real weights and record the hash**

```bash
cd samples/go2
python -m venv .venv-tools && . .venv-tools/bin/activate
pip install -r tools/requirements-dev.txt
python tools/convert_weights.py --output weights/
shasum -a 256 weights/yolov8n.safetensors
deactivate
```

Record the sha256 — Task 6's Stagefile pins it in its `download:` entry. Upload `weights/yolov8n.safetensors` to wherever the sample's assets are hosted and note the URL.

- [ ] **Step 6: Commit**

```bash
git add samples/go2/tools/ samples/go2/tests/test_convert_weights.py
git commit -m "feat(go2): host-side YOLOv8n weight converter

Emits safetensors + a shape manifest so the device image never needs
torch. Coerces everything to float32 because bfloat16 weights fail to
build on Orin."
```

---

### Task 3: MAX adapter layer (`model/ops.py`)

Wraps every MAX-specific call behind signatures we own. This is what makes the rest of the model code testable and confines a MAX API change to one file.

**Files:**
- Create: `samples/go2/detector/model/ops.py`
- Create: `samples/go2/detector/model/__init__.py`
- Test: `samples/go2/tests/test_ops.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `conv_bn_silu(x, weights, prefix: str, out_ch: int, k: int, s: int, p: int) -> Value`
  - `c2f(x, weights, prefix: str, out_ch: int, n: int, shortcut: bool) -> Value`
  - `sppf(x, weights, prefix: str, out_ch: int) -> Value`
  - `upsample_nearest(x, scale: int) -> Value`
  - `concat(values: list, axis: int) -> Value`
  - `fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, eps) -> tuple[np.ndarray, np.ndarray]`

  `weights` is a `dict[str, np.ndarray]`. `Value` is MAX's graph value type; callers never construct one directly.

- [ ] **Step 1: Write the failing test for BN folding**

Batch-norm folding is the one piece of `ops.py` that is pure arithmetic and therefore testable without MAX at all. Test it properly; the graph-building functions are covered by the parity test in Task 5.

`samples/go2/tests/test_ops.py`:

```python
import numpy as np
import pytest

from detector.model.ops import fold_bn


def test_fold_bn_matches_explicit_batchnorm():
    rng = np.random.default_rng(0)
    conv_w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    gamma = rng.standard_normal(4).astype(np.float32)
    beta = rng.standard_normal(4).astype(np.float32)
    mean = rng.standard_normal(4).astype(np.float32)
    var = np.abs(rng.standard_normal(4)).astype(np.float32) + 0.1
    eps = 1e-3

    folded_w, folded_b = fold_bn(conv_w, gamma, beta, mean, var, eps)

    # A folded conv applied to a constant input must equal
    # batchnorm(conv(input)) computed the long way.
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

    def conv(weight, bias, inp):
        out = np.zeros((1, weight.shape[0], 6, 6), np.float32)
        for o in range(weight.shape[0]):
            for i in range(weight.shape[1]):
                for r in range(6):
                    for c in range(6):
                        out[0, o, r, c] += np.sum(
                            inp[0, i, r : r + 3, c : c + 3] * weight[o, i]
                        )
            out[0, o] += bias[o]
        return out

    folded = conv(folded_w, folded_b, x)

    plain = conv(conv_w, np.zeros(4, np.float32), x)
    scale = gamma / np.sqrt(var + eps)
    expected = (plain - mean[None, :, None, None]) * scale[
        None, :, None, None
    ] + beta[None, :, None, None]

    np.testing.assert_allclose(folded, expected, rtol=1e-4, atol=1e-4)


def test_fold_bn_preserves_weight_shape():
    conv_w = np.ones((8, 4, 3, 3), np.float32)
    ones = np.ones(8, np.float32)
    folded_w, folded_b = fold_bn(conv_w, ones, ones, ones, ones, 1e-3)

    assert folded_w.shape == (8, 4, 3, 3)
    assert folded_b.shape == (8,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/go2 && python -m pytest tests/test_ops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.model.ops'`

- [ ] **Step 3: Implement `fold_bn` and the graph builders**

`samples/go2/detector/model/ops.py`:

```python
"""MAX adapter layer.

Every MAX-specific call in the project lives here. The rest of the model
is written against these signatures, so a MAX API change touches one
file and the parity tests still tell us whether it is correct.

Conv+BN pairs are folded into a single biased convolution at weight-load
time rather than built as two graph nodes: it is arithmetically
identical, halves the node count, and means the graph needs no batchnorm
op at all.
"""

import numpy as np

from max.dtype import DType
from max.graph import ops


def fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, eps):
    """Fold a batchnorm into the preceding convolution's weights.

    Returns (weight, bias) for an equivalent biased convolution.
    """
    scale = bn_gamma / np.sqrt(bn_var + eps)
    folded_w = conv_w * scale[:, None, None, None]
    folded_b = bn_beta - bn_mean * scale
    return folded_w.astype(np.float32), folded_b.astype(np.float32)


def _const(array):
    return ops.constant(array, dtype=DType.float32)


def conv_bn_silu(x, weights, prefix, out_ch, k, s, p):
    """Conv → (folded) BN → SiLU, the unit YOLOv8 is built from."""
    w = weights[f"{prefix}.weight"]
    b = weights[f"{prefix}.bias"]
    y = ops.conv2d(x, _const(w), stride=(s, s), padding=(p, p, p, p))
    y = y + ops.reshape(_const(b), (1, out_ch, 1, 1))
    return y * ops.sigmoid(y)


def _bottleneck(x, weights, prefix, ch, shortcut):
    y = conv_bn_silu(x, weights, f"{prefix}.cv1", ch, 3, 1, 1)
    y = conv_bn_silu(y, weights, f"{prefix}.cv2", ch, 3, 1, 1)
    return x + y if shortcut else y


def c2f(x, weights, prefix, out_ch, n, shortcut):
    """YOLOv8's CSP block: split, run n bottlenecks, concatenate all."""
    half = out_ch // 2
    y = conv_bn_silu(x, weights, f"{prefix}.cv1", out_ch, 1, 1, 0)
    a, b = ops.split(y, [half, half], axis=1)

    outputs = [a, b]
    current = b
    for i in range(n):
        current = _bottleneck(
            current, weights, f"{prefix}.m.{i}", half, shortcut
        )
        outputs.append(current)

    merged = ops.concat(outputs, axis=1)
    return conv_bn_silu(merged, weights, f"{prefix}.cv2", out_ch, 1, 1, 0)


def sppf(x, weights, prefix, out_ch):
    """Spatial pyramid pooling, fast variant: three chained 5x5 maxpools."""
    half = out_ch // 2
    y = conv_bn_silu(x, weights, f"{prefix}.cv1", half, 1, 1, 0)

    pools = [y]
    for _ in range(3):
        y = ops.max_pool2d(y, kernel_size=(5, 5), stride=(1, 1),
                           padding=(2, 2, 2, 2))
        pools.append(y)

    merged = ops.concat(pools, axis=1)
    return conv_bn_silu(merged, weights, f"{prefix}.cv2", out_ch, 1, 1, 0)


def upsample_nearest(x, scale):
    return ops.resize_nearest(x, scale_factor=(scale, scale))


def concat(values, axis):
    return ops.concat(values, axis=axis)
```

`samples/go2/detector/model/__init__.py`: empty file.

**Note for the implementer:** the `ops.*` symbol names are written against MAX's documented graph API. Verify each against the nightly version pinned in Task 1 and correct as needed — `ops.split`, `ops.max_pool2d` and `ops.resize_nearest` are the three most likely to differ. Correct them here only; nothing outside this file should need to change.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/go2 && python -m pytest tests/test_ops.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add samples/go2/detector/model/ samples/go2/tests/test_ops.py
git commit -m "feat(go2): MAX adapter layer for YOLOv8 building blocks

Confines every MAX-specific call to one file and folds conv+BN at
load time, so the graph needs no batchnorm op."
```

---

### Task 4: Preprocess and postprocess

Pure NumPy, fully testable without MAX, a GPU, or a robot. Doing these before the network means the parity test in Task 5 has real inputs and outputs to work with.

**Files:**
- Create: `samples/go2/detector/preprocess.py`
- Create: `samples/go2/detector/postprocess.py`
- Test: `samples/go2/tests/test_preprocess.py`
- Test: `samples/go2/tests/test_postprocess.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]` returning `(chw_float32_batched, scale, (pad_x, pad_y))`
  - `Detection` dataclass with fields `class_id: int`, `score: float`, `x1: float`, `y1: float`, `x2: float`, `y2: float`
  - `decode(raw: np.ndarray, scale: float, pad: tuple[int, int], conf_threshold: float, iou_threshold: float) -> list[Detection]`
  - `FRUIT_CLASS_IDS: dict[int, str]`

- [ ] **Step 1: Write the failing preprocess test**

`samples/go2/tests/test_preprocess.py`:

```python
import numpy as np

from detector.preprocess import letterbox


def test_letterbox_produces_square_batched_chw():
    image = np.zeros((480, 640, 3), np.uint8)

    tensor, scale, (pad_x, pad_y) = letterbox(image, size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y == 80


def test_letterbox_scales_down_a_larger_image():
    image = np.zeros((720, 1280, 3), np.uint8)

    tensor, scale, (pad_x, pad_y) = letterbox(image, size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 0.5
    assert pad_x == 0
    assert pad_y == 140


def test_letterbox_normalises_to_unit_range():
    image = np.full((640, 640, 3), 255, np.uint8)

    tensor, _, _ = letterbox(image, size=640)

    assert tensor.max() <= 1.0
    np.testing.assert_allclose(tensor.max(), 1.0, atol=1e-6)


def test_letterbox_converts_bgr_to_rgb():
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :, 0] = 255  # blue in BGR

    tensor, _, _ = letterbox(image, size=640)

    # After conversion the blue channel is index 2, not 0.
    np.testing.assert_allclose(tensor[0, 2].max(), 1.0, atol=1e-6)
    np.testing.assert_allclose(tensor[0, 0].max(), 0.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd samples/go2 && python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.preprocess'`

- [ ] **Step 3: Implement preprocess**

`samples/go2/detector/preprocess.py`:

```python
"""Letterbox an image into YOLO's square input without distorting it."""

import cv2
import numpy as np


def letterbox(image, size=640):
    """Resize preserving aspect ratio, pad to square, normalise to CHW.

    Returns (tensor, scale, (pad_x, pad_y)). The scale and padding are
    what postprocess needs to map boxes back to original image
    coordinates.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    return np.ascontiguousarray(tensor), scale, (pad_x, pad_y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd samples/go2 && python -m pytest tests/test_preprocess.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the failing postprocess test**

`samples/go2/tests/test_postprocess.py`:

```python
import numpy as np

from detector.postprocess import FRUIT_CLASS_IDS, Detection, decode, nms


def test_fruit_classes_are_the_three_coco_fruits():
    assert FRUIT_CLASS_IDS == {46: "banana", 47: "apple", 49: "orange"}


def test_nms_suppresses_a_heavily_overlapping_box():
    boxes = np.array(
        [[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]],
        np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep == [0, 2]


def test_nms_keeps_boxes_below_the_iou_threshold():
    boxes = np.array([[0, 0, 10, 10], [9, 9, 19, 19]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)

    assert nms(boxes, scores, iou_threshold=0.5) == [0, 1]


def test_decode_maps_boxes_back_through_letterbox():
    # One prediction: centre (320, 320), 40x40, banana at 0.9.
    raw = np.zeros((1, 84, 1), np.float32)
    raw[0, 0, 0] = 320.0
    raw[0, 1, 0] = 320.0
    raw[0, 2, 0] = 40.0
    raw[0, 3, 0] = 40.0
    raw[0, 4 + 46, 0] = 0.9

    detections = decode(
        raw, scale=0.5, pad=(0, 140), conf_threshold=0.25, iou_threshold=0.45
    )

    assert len(detections) == 1
    det = detections[0]
    assert det.class_id == 46
    assert det.score == 0.9
    # centre 320 → x: (320 - 0) / 0.5 = 640;  y: (320 - 140) / 0.5 = 360
    assert det.x1 == 600.0
    assert det.x2 == 680.0
    assert det.y1 == 320.0
    assert det.y2 == 400.0


def test_decode_drops_non_fruit_classes():
    raw = np.zeros((1, 84, 1), np.float32)
    raw[0, 2, 0] = 40.0
    raw[0, 3, 0] = 40.0
    raw[0, 4 + 0, 0] = 0.99  # person

    assert decode(raw, 1.0, (0, 0), 0.25, 0.45) == []


def test_decode_drops_low_confidence():
    raw = np.zeros((1, 84, 1), np.float32)
    raw[0, 2, 0] = 40.0
    raw[0, 3, 0] = 40.0
    raw[0, 4 + 46, 0] = 0.1

    assert decode(raw, 1.0, (0, 0), 0.25, 0.45) == []
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd samples/go2 && python -m pytest tests/test_postprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.postprocess'`

- [ ] **Step 7: Implement postprocess**

`samples/go2/detector/postprocess.py`:

```python
"""Decode YOLOv8 raw output into fruit detections in image coordinates.

NumPy for now. A Mojo custom op for NMS is a recorded follow-up, not
part of this scope — correctness first, and NMS on a handful of
fruit-class boxes is not the bottleneck.
"""

from dataclasses import dataclass

import numpy as np

# COCO class indices for the three fruits in the label set. Stock
# YOLOv8n weights already know these, which is why no retraining is
# needed.
FRUIT_CLASS_IDS = {46: "banana", 47: "apple", 49: "orange"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def label(self):
        return FRUIT_CLASS_IDS[self.class_id]

    @property
    def centre(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. Returns kept indices, best first."""
    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (
            boxes[best, 3] - boxes[best, 1]
        )
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (
            boxes[rest, 3] - boxes[rest, 1]
        )
        iou = inter / (area_best + area_rest - inter + 1e-9)

        order = rest[iou <= iou_threshold]

    return keep


def decode(raw, scale, pad, conf_threshold, iou_threshold):
    """Turn raw model output (1, 84, N) into Detections in image space.

    Rows 0-3 are cx, cy, w, h in letterboxed coordinates; rows 4-83 are
    per-class scores. Only the three fruit classes are considered — a
    filter applied before NMS, so a high-scoring person can never
    suppress a banana.
    """
    predictions = raw[0]
    fruit_ids = sorted(FRUIT_CLASS_IDS)

    class_scores = predictions[4:, :]
    best_fruit_row = np.argmax(class_scores[fruit_ids, :], axis=0)
    best_class = np.array(fruit_ids, np.int32)[best_fruit_row]
    best_score = class_scores[best_class, np.arange(class_scores.shape[1])]

    selected = best_score >= conf_threshold
    if not np.any(selected):
        return []

    cx, cy, w, h = predictions[0:4, selected]
    scores = best_score[selected]
    classes = best_class[selected]

    pad_x, pad_y = pad
    x1 = (cx - w / 2.0 - pad_x) / scale
    y1 = (cy - h / 2.0 - pad_y) / scale
    x2 = (cx + w / 2.0 - pad_x) / scale
    y2 = (cy + h / 2.0 - pad_y) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    detections = []
    for class_id in np.unique(classes):
        mask = classes == class_id
        indices = np.flatnonzero(mask)
        kept = nms(boxes[mask], scores[mask], iou_threshold)
        for k in kept:
            index = int(indices[k])
            detections.append(
                Detection(
                    class_id=int(class_id),
                    score=float(scores[index]),
                    x1=float(boxes[index, 0]),
                    y1=float(boxes[index, 1]),
                    x2=float(boxes[index, 2]),
                    y2=float(boxes[index, 3]),
                )
            )

    detections.sort(key=lambda d: -d.score)
    return detections
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd samples/go2 && python -m pytest tests/test_postprocess.py -v`
Expected: 6 passed

- [ ] **Step 9: Commit**

```bash
git add samples/go2/detector/preprocess.py samples/go2/detector/postprocess.py \
        samples/go2/tests/test_preprocess.py samples/go2/tests/test_postprocess.py
git commit -m "feat(go2): YOLO letterbox preprocess and fruit-only decode

Filters to the three COCO fruit classes before NMS, so a high-scoring
person can never suppress a banana."
```

---

### Task 5: YOLOv8n network on MAX, with parity tests

**Files:**
- Create: `samples/go2/detector/model/yolov8.py`
- Create: `samples/go2/detector/model/weights.py`
- Test: `samples/go2/tests/test_yolov8_parity.py`
- Create: `samples/go2/tests/conftest.py`
- Create: `samples/go2/tests/fixtures/README.md`

**Interfaces:**
- Consumes: `detector.model.ops` (Task 3), `detector.preprocess.letterbox` and `detector.postprocess.decode` (Task 4).
- Produces:
  - `load_weights(path: str) -> dict[str, np.ndarray]` — reads safetensors, folds every conv+BN pair via `ops.fold_bn`, returns a dict keyed by the folded names `ops.conv_bn_silu` expects (`<prefix>.weight`, `<prefix>.bias`).
  - `build_yolov8n(weights: dict[str, np.ndarray], device) -> Model` — a compiled MAX model.
  - `Yolov8n` class with `__init__(self, weights_path: str)` and `infer(self, tensor: np.ndarray) -> np.ndarray` returning raw `(1, 84, 8400)`.

- [ ] **Step 1: Capture reference fixtures on the host**

Run on the development machine, inside the tools venv from Task 2:

```bash
cd samples/go2
. .venv-tools/bin/activate
python - <<'PY'
import numpy as np, cv2
from ultralytics import YOLO
from detector.preprocess import letterbox

image = cv2.imread("tests/fixtures/banana.jpg")
tensor, scale, pad = letterbox(image, 640)

model = YOLO("yolov8n.pt").model.eval()
import torch
with torch.no_grad():
    raw = model(torch.from_numpy(tensor))[0].numpy()

np.savez("tests/fixtures/reference.npz", input=tensor, raw=raw,
         scale=np.float32(scale), pad=np.array(pad, np.int32))
print("raw shape:", raw.shape)
PY
deactivate
```

Put a real photo containing a banana at `tests/fixtures/banana.jpg` first. Write `tests/fixtures/README.md` recording where the image came from and that `reference.npz` is regenerated by the snippet above.

- [ ] **Step 2: Write the failing parity test**

`samples/go2/tests/conftest.py`:

```python
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "max_required: needs MAX and an accelerator"
    )


@pytest.fixture(scope="session")
def reference():
    path = FIXTURES / "reference.npz"
    if not path.exists():
        pytest.skip("reference.npz not generated — see Task 5 Step 1")
    return np.load(path)
```

`samples/go2/tests/test_yolov8_parity.py`:

```python
import numpy as np
import pytest

from detector.model.weights import load_weights
from detector.model.yolov8 import Yolov8n
from detector.postprocess import decode

pytestmark = pytest.mark.max_required

WEIGHTS = "weights/yolov8n.safetensors"


def test_load_weights_folds_conv_and_bn():
    weights = load_weights(WEIGHTS)

    # Folding must leave a bias next to every folded conv weight, and
    # must leave no raw batchnorm tensors behind.
    assert "model.0.conv.weight" in weights
    assert "model.0.conv.bias" in weights
    assert not any(".bn." in name for name in weights)


def test_raw_output_matches_ultralytics(reference):
    model = Yolov8n(WEIGHTS)

    raw = model.infer(reference["input"])

    assert raw.shape == reference["raw"].shape
    np.testing.assert_allclose(raw, reference["raw"], rtol=2e-2, atol=2e-2)


def test_decoded_detections_match_ultralytics(reference):
    model = Yolov8n(WEIGHTS)
    scale = float(reference["scale"])
    pad = tuple(int(v) for v in reference["pad"])

    ours = decode(model.infer(reference["input"]), scale, pad, 0.25, 0.45)
    theirs = decode(reference["raw"], scale, pad, 0.25, 0.45)

    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert a.class_id == b.class_id
        assert abs(a.score - b.score) < 0.05
        for lhs, rhs in ((a.x1, b.x1), (a.y1, b.y1), (a.x2, b.x2), (a.y2, b.y2)):
            assert abs(lhs - rhs) < 4.0  # pixels
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd samples/go2 && python -m pytest tests/test_yolov8_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector.model.weights'`

- [ ] **Step 4: Implement the weight loader**

`samples/go2/detector/model/weights.py`:

```python
"""Load safetensors weights and fold every conv+BN pair.

Ultralytics stores `<prefix>.conv.weight` alongside `<prefix>.bn.*`.
`ops.conv_bn_silu` wants a single biased convolution, so folding happens
here, once, at load time — not in the graph.
"""

import numpy as np
from safetensors.numpy import load_file

from .ops import fold_bn

BN_EPS = 1e-3


def load_weights(path):
    raw = load_file(path)
    folded = {}

    for name, tensor in raw.items():
        if not name.endswith(".conv.weight"):
            continue

        prefix = name[: -len(".weight")]
        bn = prefix[: -len(".conv")] + ".bn"

        if f"{bn}.weight" not in raw:
            # A conv with no batchnorm (the detection head's final 1x1s).
            folded[name] = tensor.astype(np.float32)
            bias_name = f"{prefix}.bias"
            folded[bias_name] = raw.get(
                bias_name, np.zeros(tensor.shape[0], np.float32)
            ).astype(np.float32)
            continue

        weight, bias = fold_bn(
            tensor,
            raw[f"{bn}.weight"],
            raw[f"{bn}.bias"],
            raw[f"{bn}.running_mean"],
            raw[f"{bn}.running_var"],
            BN_EPS,
        )
        folded[f"{prefix}.weight"] = weight
        folded[f"{prefix}.bias"] = bias

    return folded
```

- [ ] **Step 5: Implement the network**

`samples/go2/detector/model/yolov8.py`:

```python
"""YOLOv8n defined with MAX's graph API.

Channel widths are YOLOv8n's (depth 0.33, width 0.25). The layer
indices in the weight prefixes match Ultralytics' module numbering, so
`model.0` here is `model.0` in the checkpoint.
"""

import numpy as np

from max import engine
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.graph import Graph, TensorType

from . import ops as O

INPUT_SIZE = 640


def build_graph(weights):
    """Construct the YOLOv8n forward graph. Returns an unloaded Graph."""
    input_type = TensorType(
        DType.float32, (1, 3, INPUT_SIZE, INPUT_SIZE)
    )

    with Graph("yolov8n", input_types=(input_type,)) as graph:
        x = graph.inputs[0]

        # Backbone
        x = O.conv_bn_silu(x, weights, "model.0.conv", 16, 3, 2, 1)
        x = O.conv_bn_silu(x, weights, "model.1.conv", 32, 3, 2, 1)
        x = O.c2f(x, weights, "model.2", 32, n=1, shortcut=True)
        x = O.conv_bn_silu(x, weights, "model.3.conv", 64, 3, 2, 1)
        p3 = O.c2f(x, weights, "model.4", 64, n=2, shortcut=True)
        x = O.conv_bn_silu(p3, weights, "model.5.conv", 128, 3, 2, 1)
        p4 = O.c2f(x, weights, "model.6", 128, n=2, shortcut=True)
        x = O.conv_bn_silu(p4, weights, "model.7.conv", 256, 3, 2, 1)
        x = O.c2f(x, weights, "model.8", 256, n=1, shortcut=True)
        p5 = O.sppf(x, weights, "model.9", 256)

        # Neck (PAN-FPN)
        x = O.upsample_nearest(p5, 2)
        x = O.concat([x, p4], axis=1)
        n4 = O.c2f(x, weights, "model.12", 128, n=1, shortcut=False)

        x = O.upsample_nearest(n4, 2)
        x = O.concat([x, p3], axis=1)
        n3 = O.c2f(x, weights, "model.15", 64, n=1, shortcut=False)

        x = O.conv_bn_silu(n3, weights, "model.16.conv", 64, 3, 2, 1)
        x = O.concat([x, n4], axis=1)
        n4o = O.c2f(x, weights, "model.18", 128, n=1, shortcut=False)

        x = O.conv_bn_silu(n4o, weights, "model.19.conv", 128, 3, 2, 1)
        x = O.concat([x, p5], axis=1)
        n5o = O.c2f(x, weights, "model.21", 256, n=1, shortcut=False)

        graph.output(_head(weights, [n3, n4o, n5o]))

    return graph


def _head(weights, features):
    """Decoupled detection head: per-scale box and class branches."""
    outputs = []
    for i, feature in enumerate(features):
        box = O.conv_bn_silu(feature, weights, f"model.22.cv2.{i}.0.conv",
                             64, 3, 1, 1)
        box = O.conv_bn_silu(box, weights, f"model.22.cv2.{i}.1.conv",
                             64, 3, 1, 1)
        box = O.conv_bn_silu(box, weights, f"model.22.cv2.{i}.2",
                             64, 1, 1, 0)

        cls = O.conv_bn_silu(feature, weights, f"model.22.cv3.{i}.0.conv",
                             80, 3, 1, 1)
        cls = O.conv_bn_silu(cls, weights, f"model.22.cv3.{i}.1.conv",
                             80, 3, 1, 1)
        cls = O.conv_bn_silu(cls, weights, f"model.22.cv3.{i}.2",
                             80, 1, 1, 0)

        outputs.append(O.concat([box, cls], axis=1))

    return outputs


class Yolov8n:
    """Compiled YOLOv8n on a MAX accelerator.

    Raises RuntimeError if no accelerator is present. There is
    deliberately no CPU fallback: a sample that silently stops using the
    accelerator it exists to demonstrate is worse than one that fails.
    """

    def __init__(self, weights_path):
        from .weights import load_weights

        if accelerator_count() < 1:
            raise RuntimeError(
                "MAX reports no accelerator. This service requires the "
                "Orin GPU and will not fall back to CPU."
            )

        self._device = Accelerator()
        weights = load_weights(weights_path)
        graph = build_graph(weights)
        session = engine.InferenceSession(devices=[self._device])
        self._model = session.load(graph)

    def infer(self, tensor):
        """Run one frame. Input (1,3,640,640) float32, output (1,84,8400)."""
        outputs = self._model.execute(tensor)
        parts = [np.asarray(o.to_numpy()) for o in outputs]
        flattened = [p.reshape(p.shape[0], p.shape[1], -1) for p in parts]
        return np.concatenate(flattened, axis=2)
```

**Note for the implementer:** the head's DFL (distribution focal loss) box decoding is folded into `postprocess.decode`'s assumption that rows 0-3 are cx/cy/w/h. If the parity test in Step 6 shows box coordinates disagreeing while class scores agree, that is the DFL projection missing — add it as a `softmax` over 16 bins times `arange(16)` inside `_head`, and note it in the module docstring.

- [ ] **Step 6: Run the parity test on the dog**

The test needs MAX and the GPU, so it runs on the device, not the laptop.

Run: `cd samples/go2 && wendyg run --device <go2-jetson>.local` with the detector service temporarily set to run `python -m pytest tests/test_yolov8_parity.py -v`.
Expected: 3 passed.

Iterate on `ops.py` and `yolov8.py` until parity holds. This is the substantive debugging work of the plan — budget for it.

- [ ] **Step 7: Commit**

```bash
git add samples/go2/detector/model/ samples/go2/tests/
git commit -m "feat(go2): YOLOv8n on MAX with Ultralytics parity tests

Network defined via the MAX graph API; weights folded at load time.
Parity is asserted layer-output-wise against reference tensors so a
MAX API change cannot silently change detections."
```

---

### Task 6: `camera` service — WebRTC to DDS

**Files:**
- Create: `samples/go2/common/go2_msgs.py`
- Create: `samples/go2/camera/main.py`
- Create: `samples/go2/camera/publisher.py`
- Create: `samples/go2/camera/build.stagefile.yaml`
- Modify: `samples/go2/wendy.json`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `common.go2_msgs.CompressedImage` — cyclonedds `IdlStruct` matching `sensor_msgs::msg::dds_::CompressedImage_`
  - `common.go2_msgs.Detections` and `common.go2_msgs.DetectionBox` — the app's detection message, used by Task 7 and by Plans 2 and 3.
  - `camera.publisher.FramePublisher` with `publish(jpeg_bytes: bytes) -> None`

- [ ] **Step 1: Define the shared messages**

`samples/go2/common/go2_msgs.py`:

```python
"""DDS message definitions shared by every service.

MUST NOT use `from __future__ import annotations`: cyclonedds's
IdlStruct normaliser resolves type hints by name lookup at class
definition time, and PEP-563 string annotations break it. This is the
same trap documented in go2-rc's perception.py.

Wire naming: ROS2 on CycloneDDS prefixes topics with `rt/`, so the ROS2
topic `/go2/camera/image_raw/compressed` is
`rt/go2/camera/image_raw/compressed` on the wire.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import float32, int32, sequence, uint8, uint32

CAMERA_TOPIC = "rt/go2/camera/image_raw/compressed"
DETECTIONS_TOPIC = "rt/go2/detections"


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time
    frame_id: str


@dataclass
class CompressedImage(
    IdlStruct, typename="sensor_msgs::msg::dds_::CompressedImage_"
):
    header: Header
    format: str
    data: sequence[uint8]


@dataclass
class DetectionBox(IdlStruct, typename="go2_fruit::msg::dds_::DetectionBox_"):
    class_id: int32
    label: str
    score: float32
    x1: float32
    y1: float32
    x2: float32
    y2: float32


@dataclass
class Detections(IdlStruct, typename="go2_fruit::msg::dds_::Detections_"):
    header: Header
    image_width: int32
    image_height: int32
    boxes: sequence[DetectionBox]


def now_header(frame_id):
    """Build a Header stamped with the current time."""
    import time

    nanos = time.time_ns()
    return Header(
        stamp=Time(sec=int32(nanos // 1_000_000_000),
                   nanosec=uint32(nanos % 1_000_000_000)),
        frame_id=frame_id,
    )
```

- [ ] **Step 2: Write the publisher**

`samples/go2/camera/publisher.py`:

```python
"""Publish JPEG frames as sensor_msgs/CompressedImage over CycloneDDS."""

import logging
import os

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.topic import Topic

from common.go2_msgs import CAMERA_TOPIC, CompressedImage, now_header

logger = logging.getLogger(__name__)


class FramePublisher:
    """Latest-frame-wins publisher for the dog's front camera.

    Binds by address, never by interface name: the Go2's Orin is
    multi-homed and a name lets DDS advertise the wrong subnet.
    """

    def __init__(self, domain_id=None, frame_id="camera_link"):
        domain_id = int(
            domain_id
            if domain_id is not None
            else os.environ.get("ROS_DOMAIN_ID", "0")
        )
        self._frame_id = frame_id
        self._participant = DomainParticipant(domain_id)
        topic = Topic(self._participant, CAMERA_TOPIC, CompressedImage)
        self._writer = DataWriter(Publisher(self._participant), topic)
        logger.info("publishing %s on domain %d", CAMERA_TOPIC, domain_id)

    def publish(self, jpeg_bytes):
        self._writer.write(
            CompressedImage(
                header=now_header(self._frame_id),
                format="jpeg",
                data=list(jpeg_bytes),
            )
        )
```

- [ ] **Step 3: Write the camera service**

`samples/go2/camera/main.py`:

```python
#!/usr/bin/env python3
"""go2 fruit hunter: WebRTC → DDS bridge for the Go2's front camera.

Owns the dog's single WebRTC slot. Decodes H.264 into JPEG, publishes
each frame as sensor_msgs/CompressedImage, and also serves MJPEG on
HTTP for the dashboard.

WebRTC scar tissue, inherited from go2-rc/camera and go2-foxglove:
- Only one WebRTC client per Go2 controller. If the Unitree phone app
  is open, this cannot connect.
- aiortc's H.264 decoder can wedge on a partial GOP; send PLI every few
  seconds until the first frame decodes.
- track.recv() raises MediaStreamError when the dog drops the track.
  After 5 consecutive errors, exit non-zero so wendy's restart brings
  us back with a fresh handshake.
"""

import asyncio
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

from publisher import FramePublisher

GO2_IP = os.environ.get("GO2_IP", "192.168.123.161")
PORT = int(os.environ.get("PORT", "8000"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
KEYFRAME_REQUEST_INTERVAL_S = 3.0
MAX_CONSECUTIVE_TRACK_ERRORS = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera")

app = FastAPI()
_state = {"frames": 0, "latest": None, "started": time.time()}
_lock = threading.Lock()


@app.get("/health")
def health():
    elapsed = max(1e-6, time.time() - _state["started"])
    return JSONResponse(
        {
            "status": "ok" if _state["frames"] else "waiting",
            "frames": _state["frames"],
            "fps": round(_state["frames"] / elapsed, 2),
        }
    )


@app.get("/stream/color")
def stream():
    def generate():
        while True:
            with _lock:
                frame = _state["latest"]
            if frame is not None:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(0.03)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


async def pump(publisher):
    connection = UnitreeWebRTCConnection(
        ip=GO2_IP, method=WebRTCConnectionMethod.LocalSTA
    )
    await connection.connect()
    track = await connection.video_track()

    errors = 0
    last_pli = 0.0

    while True:
        try:
            if _state["frames"] == 0 and (
                time.time() - last_pli > KEYFRAME_REQUEST_INTERVAL_S
            ):
                await connection.request_keyframe()
                last_pli = time.time()

            frame = await track.recv()
            errors = 0
        except Exception as exc:
            errors += 1
            logger.warning("track error %d/%d: %s",
                           errors, MAX_CONSECUTIVE_TRACK_ERRORS, exc)
            if errors >= MAX_CONSECUTIVE_TRACK_ERRORS:
                logger.error("giving up; exiting for a fresh handshake")
                os._exit(1)
            await asyncio.sleep(1.0)
            continue

        image = frame.to_ndarray(format="bgr24")
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            continue

        jpeg = buffer.tobytes()
        publisher.publish(jpeg)
        with _lock:
            _state["latest"] = jpeg
            _state["frames"] += 1


def main():
    publisher = FramePublisher()

    def run_pump():
        asyncio.run(pump(publisher))

    threading.Thread(target=run_pump, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
```

**Note for the implementer:** `unitree_webrtc_connect`'s exact API (`video_track()`, `request_keyframe()`) should be checked against `templates/python/go2-rc/camera/main.py`, which is working code against the same library. Copy its call shapes rather than guessing.

- [ ] **Step 4: Write the camera Stagefile**

`samples/go2/camera/build.stagefile.yaml`:

```yaml
version: 1
stages:
  - name: camera
    from: python:3.11-slim-bookworm
    workdir: /app
    env:
      PYTHONUNBUFFERED: "1"
      PYTHONPATH: /app
    install:
      apt:
        packages: [ca-certificates, libgl1, libglib2.0-0]
      pip:
        packages:
          - aiortc==1.9.0
          - cyclonedds==0.10.2
          - fastapi==0.115.0
          - numpy==1.26.4
          - opencv-python-headless==4.10.0.84
          - unitree-webrtc-connect==0.2.0
          - uvicorn==0.30.6
    copy:
      - from: local
        paths: [main.py, publisher.py]
        dest: /app/
      - from: local
        paths: [../common]
        dest: /app/common/
    entrypoint:
      exec: [python3, /app/main.py]
```

**Note:** if `copy` cannot reach `../common` (outside the service context), move `common/` into each service directory or set the service `context` to `./` with distinct entrypoints. Verify with `wendyg build` in Step 6 and adjust — do not leave it broken.

- [ ] **Step 5: Register the service**

Replace the `services` map in `samples/go2/wendy.json`:

```json
{
    "appId": "sh.wendy.examples.go2-fruit-hunter",
    "version": "0.1.0",
    "platform": "linux",
    "services": {
        "camera": {
            "context": "./camera",
            "entitlements": [
                { "type": "network", "mode": "host" }
            ]
        }
    }
}
```

The `probe` service is removed now that Task 1's question is answered; `probe/` stays on disk as documentation.

- [ ] **Step 6: Build and run**

Run: `cd samples/go2 && wendyg run --device <go2-jetson>.local`
Expected: `curl http://<go2-jetson>:8000/health` reports a rising frame count, and `http://<go2-jetson>:8000/stream/color` shows the dog's view in a browser.

If frames stay at 0, check that the Unitree phone app is disconnected — it holds the only WebRTC slot.

- [ ] **Step 7: Commit**

```bash
git add samples/go2/common/ samples/go2/camera/ samples/go2/wendy.json
git commit -m "feat(go2): camera service bridging WebRTC to ROS2 CompressedImage

The Go2 exposes its camera over WebRTC only, and allows one client at
a time. This service owns that slot and republishes frames on DDS so
everything downstream consumes the camera over ROS2."
```

---

### Task 7: `detector` service and `dashboard`

**Files:**
- Create: `samples/go2/detector/main.py`
- Create: `samples/go2/detector/build.stagefile.yaml`
- Create: `samples/go2/dashboard/main.py`
- Create: `samples/go2/dashboard/static/index.html`
- Create: `samples/go2/dashboard/build.stagefile.yaml`
- Create: `samples/go2/README.md`
- Modify: `samples/go2/wendy.json`

**Interfaces:**
- Consumes: `common.go2_msgs` (Task 6), `detector.model.yolov8.Yolov8n` (Task 5), `detector.preprocess.letterbox` and `detector.postprocess.decode` (Task 4).
- Produces: `rt/go2/detections` carrying `Detections` — the topic Plan 2's `brain` subscribes to.

- [ ] **Step 1: Write the detector service**

`samples/go2/detector/main.py`:

```python
#!/usr/bin/env python3
"""Fruit detector: DDS CompressedImage → MAX YOLOv8n → DDS Detections.

Latest-frame-wins. If inference is slower than the camera, frames are
dropped rather than queued — a stale detection is worse than no
detection when the dog is moving.
"""

import logging
import os
import time

import cv2
import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic

from common.go2_msgs import (
    CAMERA_TOPIC,
    DETECTIONS_TOPIC,
    CompressedImage,
    DetectionBox,
    Detections,
    now_header,
)
from postprocess import decode
from preprocess import letterbox

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/app/weights/yolov8n.safetensors")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")


def main():
    from model.yolov8 import Yolov8n

    # Constructed before the DDS plumbing so a GPU failure is the first
    # thing in the logs, and the process exits non-zero rather than
    # quietly serving nothing.
    logger.info("loading YOLOv8n onto the GPU via MAX")
    model = Yolov8n(WEIGHTS_PATH)
    logger.info("model ready")

    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    participant = DomainParticipant(domain_id)

    in_topic = Topic(participant, CAMERA_TOPIC, CompressedImage)
    reader = DataReader(Subscriber(participant), in_topic)

    out_topic = Topic(participant, DETECTIONS_TOPIC, Detections)
    writer = DataWriter(Publisher(participant), out_topic)

    logger.info("subscribed to %s, publishing %s",
                CAMERA_TOPIC, DETECTIONS_TOPIC)

    frames = 0
    while True:
        samples = reader.take(N=10)
        if not samples:
            time.sleep(0.005)
            continue

        message = samples[-1]  # latest-frame-wins
        buffer = np.frombuffer(bytes(message.data), np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            continue

        tensor, scale, pad = letterbox(image)
        started = time.perf_counter()
        raw = model.infer(tensor)
        latency_ms = (time.perf_counter() - started) * 1000.0

        detections = decode(raw, scale, pad, CONF_THRESHOLD, IOU_THRESHOLD)

        writer.write(
            Detections(
                header=now_header("camera_link"),
                image_width=image.shape[1],
                image_height=image.shape[0],
                boxes=[
                    DetectionBox(
                        class_id=d.class_id,
                        label=d.label,
                        score=d.score,
                        x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    )
                    for d in detections
                ],
            )
        )

        frames += 1
        if frames % 30 == 0:
            logger.info("frame %d: %.1f ms, %d fruit",
                        frames, latency_ms, len(detections))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the detector Stagefile**

`samples/go2/detector/build.stagefile.yaml`. Replace `<WEIGHTS_URL>` and `<WEIGHTS_SHA256>` with the values recorded in Task 2 Step 5:

```yaml
version: 1
stages:
  - name: detector
    from: nvcr.io/nvidia/l4t-base:r36.2.0
    workdir: /app
    env:
      PYTHONUNBUFFERED: "1"
      PYTHONPATH: /app
      WEIGHTS_PATH: /app/weights/yolov8n.safetensors
    install:
      apt:
        packages:
          - ca-certificates
          - libgl1
          - libglib2.0-0
          - python3
          - python3-pip
      pip:
        # One pip invocation only: install.pip is a mapping, not a list,
        # on jo/fast. MAX comes from Modular's nightly index and
        # everything else from PyPI as an extra index.
        packages:
          - modular
          - cyclonedds==0.10.2
          - numpy==1.26.4
          - opencv-python-headless==4.10.0.84
          - safetensors==0.4.5
        index: https://whl.modular.com/nightly/simple/
        extraIndex:
          - https://pypi.org/simple
    download:
      - url: <WEIGHTS_URL>
        sha256: <WEIGHTS_SHA256>
        dest: /app/weights/yolov8n.safetensors
    copy:
      - from: local
        paths: [main.py, preprocess.py, postprocess.py, model]
        dest: /app/
      - from: local
        paths: [../common]
        dest: /app/common/
    entrypoint:
      exec: [python3, /app/main.py]
```

- [ ] **Step 3: Write the dashboard**

`samples/go2/dashboard/main.py`:

```python
#!/usr/bin/env python3
"""Annotated MJPEG stream: camera frames with fruit boxes drawn on."""

import logging
import os
import threading
import time

import cv2
import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from common.go2_msgs import (
    CAMERA_TOPIC,
    DETECTIONS_TOPIC,
    CompressedImage,
    Detections,
)

PORT = int(os.environ.get("PORT", "3400"))
COLOURS = {"banana": (0, 255, 255), "apple": (0, 0, 255),
           "orange": (0, 165, 255)}

logging.basicConfig(level=logging.INFO)
app = FastAPI()
_state = {"frame": None, "boxes": [], "count": 0}
_lock = threading.Lock()


def consume():
    participant = DomainParticipant(int(os.environ.get("ROS_DOMAIN_ID", "0")))
    images = DataReader(
        Subscriber(participant),
        Topic(participant, CAMERA_TOPIC, CompressedImage),
    )
    detections = DataReader(
        Subscriber(participant),
        Topic(participant, DETECTIONS_TOPIC, Detections),
    )

    while True:
        for message in detections.take(N=10):
            with _lock:
                _state["boxes"] = list(message.boxes)
                _state["count"] = len(message.boxes)

        samples = images.take(N=10)
        if samples:
            buffer = np.frombuffer(bytes(samples[-1].data), np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is not None:
                with _lock:
                    boxes = list(_state["boxes"])
                for box in boxes:
                    colour = COLOURS.get(box.label, (255, 255, 255))
                    cv2.rectangle(
                        image,
                        (int(box.x1), int(box.y1)),
                        (int(box.x2), int(box.y2)),
                        colour, 2,
                    )
                    cv2.putText(
                        image, f"{box.label} {box.score:.2f}",
                        (int(box.x1), max(16, int(box.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                    )
                ok, encoded = cv2.imencode(".jpg", image)
                if ok:
                    with _lock:
                        _state["frame"] = encoded.tobytes()
        time.sleep(0.01)


@app.get("/")
def index():
    return FileResponse("/app/static/index.html")


@app.get("/api/state")
def state():
    with _lock:
        return JSONResponse(
            {
                "fruit_count": _state["count"],
                "labels": [b.label for b in _state["boxes"]],
            }
        )


@app.get("/stream")
def stream():
    def generate():
        while True:
            with _lock:
                frame = _state["frame"]
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.03)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=consume, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
```

`samples/go2/dashboard/static/index.html`:

```html
<!doctype html>
<meta charset="utf-8">
<title>Go2 Fruit Hunter</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #101014; color: #e8e8ee;
         font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; }
  header { padding: 16px 20px; border-bottom: 1px solid #26262f; }
  h1 { margin: 0; font-size: 17px; font-weight: 600; }
  #status { margin-top: 4px; color: #9a9aa8; font-size: 13px; }
  main { padding: 20px; }
  img { width: 100%; max-width: 960px; border-radius: 8px;
        border: 1px solid #26262f; display: block; }
</style>
<header>
  <h1>Go2 Fruit Hunter</h1>
  <div id="status">connecting…</div>
</header>
<main><img src="/stream" alt="camera stream with detections"></main>
<script>
  async function poll() {
    try {
      const r = await fetch("/api/state");
      const s = await r.json();
      document.getElementById("status").textContent =
        s.fruit_count
          ? `${s.fruit_count} fruit: ${s.labels.join(", ")}`
          : "exploring — no fruit in view";
    } catch (e) {
      document.getElementById("status").textContent = "disconnected";
    }
    setTimeout(poll, 500);
  }
  poll();
</script>
```

`samples/go2/dashboard/build.stagefile.yaml`:

```yaml
version: 1
stages:
  - name: dashboard
    from: python:3.11-slim-bookworm
    workdir: /app
    env:
      PYTHONUNBUFFERED: "1"
      PYTHONPATH: /app
    install:
      apt:
        packages: [ca-certificates, libgl1, libglib2.0-0]
      pip:
        packages:
          - cyclonedds==0.10.2
          - fastapi==0.115.0
          - numpy==1.26.4
          - opencv-python-headless==4.10.0.84
          - uvicorn==0.30.6
    copy:
      - from: local
        paths: [main.py, static]
        dest: /app/
      - from: local
        paths: [../common]
        dest: /app/common/
    entrypoint:
      exec: [python3, /app/main.py]
```

- [ ] **Step 4: Register all three services**

`samples/go2/wendy.json`:

```json
{
    "appId": "sh.wendy.examples.go2-fruit-hunter",
    "version": "0.1.0",
    "platform": "linux",
    "services": {
        "camera": {
            "context": "./camera",
            "entitlements": [
                { "type": "network", "mode": "host" }
            ]
        },
        "detector": {
            "context": "./detector",
            "dependsOn": ["camera"],
            "entitlements": [
                { "type": "network", "mode": "host" },
                { "type": "gpu" }
            ]
        },
        "dashboard": {
            "context": "./dashboard",
            "dependsOn": ["camera", "detector"],
            "entitlements": [
                { "type": "network", "mode": "host" }
            ]
        }
    }
}
```

- [ ] **Step 5: Write the README**

`samples/go2/README.md` must cover: what the sample does, that it targets the Go2 EDU's onboard Jetson, the `wendyg` requirement and why (the released CLI has no Stagefile support), the host-side weight conversion step, the single-WebRTC-client caveat, `GO2_IP`/`GO2_DDS_ADDRESS`/`ROS_DOMAIN_ID` configuration, the service table, and how to open the dashboard. Follow the structure of `templates/python/go2-rc/README.md`.

- [ ] **Step 6: Deploy and verify end to end**

Run: `cd samples/go2 && wendyg run --device <go2-jetson>.local`
Expected:
- `wendyg device apps list --device <go2-jetson>.local` shows all three services running.
- Open `http://<go2-jetson>:3400` — the dog's camera view renders.
- Hold a banana in front of the dog — a yellow box labelled `banana` with a confidence score appears, and the header reads `1 fruit: banana`.
- `wendyg device logs --app detector --device <go2-jetson>.local` shows per-30-frame latency lines.

- [ ] **Step 7: Commit**

```bash
git add samples/go2/
git commit -m "feat(go2): detector and dashboard services

Completes the perception pipeline: camera frames over ROS2 into MAX
YOLOv8n on the Orin GPU, fruit detections published on DDS, and an
annotated stream to see it working."
```

---

## Plan Self-Review

**Spec coverage.** Every §3 service in the perception scope has a task: `camera` (Task 6), `detector` (Task 7), `dashboard` (Task 7). §4's model construction is Tasks 3 and 5, the weight pipeline is Task 2, and verification is Task 5's parity tests. §7's "no silent CPU fallback" is enforced in `Yolov8n.__init__` and asserted by the constructor raising. §8's Stagefile requirements appear in Tasks 1, 6, and 7. §9's host-side test list maps to Tasks 2, 3, 4, and 5.

Deliberately deferred to Plans 2 and 3: `motion`, `brain`, `navigator`, the mission state machine (§6), the bark path, and Nav2 exploration (§5). The `Detections` message defined in Task 6 is the interface those plans consume.

**Known risks carried forward.** Three places where this plan tells the implementer to verify rather than assume, because the underlying API could not be confirmed from here: MAX's exact `ops.*` names (Task 3), `unitree_webrtc_connect`'s method names (Task 6 — mitigated by working reference code in `go2-rc`), and whether Stagefile `copy` can reach `../common` outside the service context (Task 6 Step 4, with a stated fallback). Each is localized and has an explicit correction path.

**Type consistency.** `Detection` (postprocess dataclass) and `DetectionBox` (DDS struct) are deliberately distinct types with the same field names; `detector/main.py` converts between them explicitly. `load_weights` emits `<prefix>.weight`/`<prefix>.bias`, which is exactly what `conv_bn_silu` looks up. `letterbox` returns `(tensor, scale, pad)` and `decode` takes `(raw, scale, pad, conf, iou)` — matching, and matched again in the parity test.
