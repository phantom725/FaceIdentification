from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "face_antispoof_minifasnetv2.onnx": "https://github.com/facenox/face-antispoof-onnx/raw/main/models/best/98.20/best_model.onnx",
}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name} ...")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    print(f"Saved {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    for filename, url in MODELS.items():
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 1024 * 1024:
            print(f"Skip {target.name}: already exists.")
            continue
        download(url, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
