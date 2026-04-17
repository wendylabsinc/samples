from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen

import nltk
from faster_whisper.utils import download_model

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "voices-v1.0.bin"
)


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=300) as response:
        destination.write_bytes(response.read())


def main() -> None:
    whisper_model = os.environ.get("WHISPER_MODEL", "base")
    whisper_model_path = os.environ["WHISPER_MODEL_PATH"]
    kokoro_model_path = Path(os.environ["KOKORO_MODEL_PATH"])
    kokoro_voices_path = Path(os.environ["KOKORO_VOICES_PATH"])
    nltk_data_dir = os.environ["NLTK_DATA"]

    download_model(whisper_model, output_dir=whisper_model_path)
    download_file(KOKORO_MODEL_URL, kokoro_model_path)
    download_file(KOKORO_VOICES_URL, kokoro_voices_path)
    nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)


if __name__ == "__main__":
    main()
