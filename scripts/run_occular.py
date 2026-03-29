#!/usr/bin/env python3
"""Run Occular OCR through both detection and recognition benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES_DIR, PREDICTIONS_DIR
from src.parse_gt import parse_cvat_xml


def get_image_paths() -> list[str]:
    gt = parse_cvat_xml()
    paths = []
    for filename in sorted(gt.keys()):
        path = IMAGES_DIR / filename
        if path.exists():
            paths.append(str(path))
    return paths


def main():
    image_paths = get_image_paths()
    print(f"Dataset: {len(image_paths)} images\n")

    # Detection benchmark
    print("=" * 60)
    print("DETECTION BENCHMARK")
    print("=" * 60)
    from src.detectors.occular_detector import OccularDetector
    det = OccularDetector(onnx=True)
    det.run_and_save(image_paths)

    # Recognition benchmark
    print("\n" + "=" * 60)
    print("RECOGNITION BENCHMARK")
    print("=" * 60)
    from src.recognizers.occular_rec import OccularRecognizer
    rec = OccularRecognizer(onnx=True)
    rec.run_and_save(image_paths)

    print("\nDone! Now run compute_metrics.py and compute_recognition_metrics.py")


if __name__ == "__main__":
    main()
