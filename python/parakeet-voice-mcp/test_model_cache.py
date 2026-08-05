from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from model_cache import ensure_model


MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def _write_model_archive(path: Path) -> str:
    files = {
        "encoder.int8.onnx": b"complete encoder",
        "decoder.int8.onnx": b"complete decoder",
        "joiner.int8.onnx": b"complete joiner",
        "tokens.txt": b"<blk> 0\n",
    }
    with tarfile.open(path, "w:bz2") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(f"{MODEL_NAME}/{name}")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnsureModelTests(unittest.TestCase):
    def test_replaces_an_interrupted_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "models"
            partial = model_dir / MODEL_NAME
            partial.mkdir(parents=True)
            (partial / "encoder.int8.onnx").write_bytes(b"partial")

            archive = Path(root) / "model.tar.bz2"
            digest = _write_model_archive(archive)

            installed = ensure_model(
                model_dir=model_dir,
                model_url=archive.as_uri(),
                expected_sha256=digest,
                model_name=MODEL_NAME,
            )

            self.assertEqual(installed, model_dir / MODEL_NAME)
            self.assertEqual(
                (installed / "encoder.int8.onnx").read_bytes(), b"complete encoder"
            )
            self.assertEqual((installed / ".ready-sha256").read_text(), digest)

    def test_ready_marker_avoids_a_second_download(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "models"
            archive = Path(root) / "model.tar.bz2"
            digest = _write_model_archive(archive)
            installed = ensure_model(
                model_dir=model_dir,
                model_url=archive.as_uri(),
                expected_sha256=digest,
                model_name=MODEL_NAME,
            )

            same = ensure_model(
                model_dir=model_dir,
                model_url="file:///does/not/exist",
                expected_sha256=digest,
                model_name=MODEL_NAME,
            )

            self.assertEqual(same, installed)


if __name__ == "__main__":
    unittest.main()
