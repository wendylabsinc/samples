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


def test_coerces_non_float32_to_float32():
    state = {"w": FakeTensor(np.zeros((4,), np.float64))}

    tensors, manifest = convert_state_dict(state, source="x.pt")

    assert tensors["w"].dtype == np.float32
    assert manifest["tensors"]["w"]["dtype"] == "float32"


def test_skips_non_tensor_entries():
    state = {
        "model.0.conv.weight": FakeTensor(np.zeros((2, 2), np.float32)),
        "epoch": 300,
        "optimizer": None,
    }

    tensors, manifest = convert_state_dict(state, source="x.pt")

    assert set(tensors) == {"model.0.conv.weight"}
    assert "epoch" not in manifest["tensors"]
