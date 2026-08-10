"""Lightweight image-quality metrics computed from a JPEG frame.

Used inside each capture child process to produce two comparable, qualitative
numbers for the benchmark table:

- ``sharpness``  — variance of the Laplacian of the luma channel. Higher = crisper.
- ``brightness`` — mean luma (0-255).

Deliberately uses numpy + Pillow (small, reliable arm64 wheels) rather than
OpenCV. Decoding is done on a downscaled grayscale image and sampled at a low
rate by the caller, so the cost stays small.
"""
from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

# Imported lazily so the module (and the child) still load if the wheels are
# somehow missing — quality metrics just degrade to ``None``.
try:
    import numpy as np
    from PIL import Image

    _AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on runtime image
    logger.warning("image-quality deps unavailable (%s); metrics disabled", exc)
    _AVAILABLE = False

# 3x3 Laplacian kernel (used via manual convolution to avoid a scipy dependency).
_MAX_DIM = 320  # downscale longest side to bound cost


def available() -> bool:
    return _AVAILABLE


def measure(jpeg: bytes) -> tuple[float | None, float | None]:
    """Return ``(sharpness, brightness)`` for a JPEG frame, or ``(None, None)``."""
    if not _AVAILABLE:
        return None, None
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("L")
        w, h = img.size
        scale = _MAX_DIM / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        a = np.asarray(img, dtype=np.float64)
        brightness = float(a.mean())

        # Laplacian via the discrete 4-neighbour kernel, on the interior pixels.
        lap = (
            -4.0 * a[1:-1, 1:-1]
            + a[:-2, 1:-1]
            + a[2:, 1:-1]
            + a[1:-1, :-2]
            + a[1:-1, 2:]
        )
        sharpness = float(lap.var())
        return round(sharpness, 1), round(brightness, 1)
    except Exception as exc:
        logger.debug("image-quality measure failed: %s", exc)
        return None, None
