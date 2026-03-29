#!/usr/bin/env python3
"""Run a single detector by name. Designed to work in isolated venvs."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name to run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    image_paths = get_image_paths()
    print(f"Dataset: {len(image_paths)} images")

    pred_path = PREDICTIONS_DIR / f"{args.model}.json"
    if pred_path.exists() and not args.force:
        print(f"[SKIP] {args.model} — predictions exist. Use --force to re-run.")
        return

    model = args.model.lower()

    if model == "tesseract":
        from src.detectors.tesseract import TesseractDetector
        det = TesseractDetector()

    elif model == "easyocr_craft":
        from src.detectors.easyocr_detector import EasyOCRDetector
        det = EasyOCRDetector()

    elif model == "surya":
        from src.detectors.surya_detector import SuryaDetector
        det = SuryaDetector()

    elif model.startswith("doctr_"):
        arch = model.replace("doctr_", "")
        from src.detectors.doctr_detector import DoctrDetector
        det = DoctrDetector(arch=arch)

    elif model in ("paddle_v5_mobile", "paddle_v3_mobile"):
        from src.detectors.paddle_detector import PaddleMobileDetector
        det = PaddleMobileDetector()

    elif model in ("paddle_v5_server", "paddle_v4_server"):
        from src.detectors.paddle_detector import PaddleServerDetector
        det = PaddleServerDetector()

    elif model in ("occular", "occular_ocr"):
        from src.detectors.occular_detector import OccularDetector
        det = OccularDetector()

    elif model in ("yandex", "yandex_vision"):
        from src.detectors.yandex_detector import YandexDetector
        det = YandexDetector()

    else:
        print(f"Unknown model: {args.model}")
        print("Available: tesseract, easyocr_craft, surya, doctr_*, paddle_v5_mobile,")
        print("           paddle_v5_server, occular, yandex_vision")
        sys.exit(1)

    print(f"Running {det.name}...")
    det.run_and_save(image_paths)


if __name__ == "__main__":
    main()
