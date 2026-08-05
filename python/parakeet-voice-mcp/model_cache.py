"""Atomic, checksum-verified installation of the Parakeet model cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Union


REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
READY_MARKER = ".ready-sha256"


def _is_ready(model_path: Path, expected_sha256: str) -> bool:
    marker = model_path / READY_MARKER
    return (
        marker.is_file()
        and marker.read_text().strip() == expected_sha256
        and all((model_path / name).is_file() and (model_path / name).stat().st_size > 0
                for name in REQUIRED_FILES)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_verified(archive_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"model archive contains an unsafe path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"model archive contains an unsupported link: {member.name}")
        archive.extractall(destination)


def ensure_model(
    *,
    model_dir: Union[str, Path],
    model_url: str,
    expected_sha256: str,
    model_name: str,
) -> Path:
    """Return a complete model, repairing partial downloads atomically."""
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    installed = root / model_name
    if _is_ready(installed, expected_sha256):
        print(f"[asr] verified model already present in {installed}", flush=True)
        return installed

    print(f"[asr] downloading model: {model_url}", flush=True)
    with tempfile.NamedTemporaryFile(
        prefix=".parakeet-download-", suffix=".tar.bz2", dir=root, delete=False
    ) as temporary:
        archive_path = Path(temporary.name)
    staging = Path(tempfile.mkdtemp(prefix=".parakeet-install-", dir=root))

    try:
        urllib.request.urlretrieve(model_url, archive_path)
        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "model archive checksum mismatch: "
                f"expected {expected_sha256}, received {actual_sha256}"
            )

        _extract_verified(archive_path, staging)
        staged_model = staging / model_name
        missing = [name for name in REQUIRED_FILES
                   if not (staged_model / name).is_file()
                   or (staged_model / name).stat().st_size == 0]
        if missing:
            raise ValueError(f"model archive is incomplete: missing {', '.join(missing)}")
        (staged_model / READY_MARKER).write_text(expected_sha256)

        if installed.exists():
            shutil.rmtree(installed)
        os.replace(staged_model, installed)
    finally:
        archive_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)

    print(f"[asr] model ready in {installed}", flush=True)
    return installed
