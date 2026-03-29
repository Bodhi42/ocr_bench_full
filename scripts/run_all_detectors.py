#!/usr/bin/env python3
"""Run all detection models on the dataset and save predictions."""

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
        else:
            print(f"  [WARN] Image not found: {path}")
    return paths


def load_detectors(model_filter: list[str] | None = None) -> list:
    detectors = []

    def should_load(name):
        if model_filter is None:
            return True
        return any(f.lower() in name.lower() for f in model_filter)

    # Tesseract
    if should_load("tesseract"):
        try:
            from src.detectors.tesseract import TesseractDetector
            detectors.append(TesseractDetector())
        except Exception as e:
            print(f"  [SKIP] tesseract: {e}")

    # EasyOCR
    if should_load("easyocr"):
        try:
            from src.detectors.easyocr_detector import EasyOCRDetector
            detectors.append(EasyOCRDetector())
        except Exception as e:
            print(f"  [SKIP] easyocr: {e}")

    # Surya
    if should_load("surya"):
        try:
            from src.detectors.surya_detector import SuryaDetector
            detectors.append(SuryaDetector())
        except Exception as e:
            print(f"  [SKIP] surya: {e}")

    # doctr — all architectures
    if should_load("doctr"):
        try:
            from src.detectors.doctr_detector import get_all_doctr_detectors
            detectors.extend(get_all_doctr_detectors())
        except Exception as e:
            print(f"  [SKIP] doctr: {e}")

    # PaddleOCR
    if should_load("paddle"):
        try:
            from src.detectors.paddle_detector import PaddleMobileDetector
            detectors.append(PaddleMobileDetector())
        except Exception as e:
            print(f"  [SKIP] paddle_mobile: {e}")
        try:
            from src.detectors.paddle_detector import PaddleServerDetector
            detectors.append(PaddleServerDetector())
        except Exception as e:
            print(f"  [SKIP] paddle_server: {e}")

    # Occular OCR
    if should_load("occular"):
        try:
            from src.detectors.occular_detector import OccularDetector
            detectors.append(OccularDetector())
        except Exception as e:
            print(f"  [SKIP] occular: {e}")

    # Yandex Vision (requires yc CLI + auth)
    if should_load("yandex"):
        try:
            from src.detectors.yandex_detector import YandexDetector
            detectors.append(YandexDetector())
        except Exception as e:
            print(f"  [SKIP] yandex: {e}")

    return detectors


def main():
    parser = argparse.ArgumentParser(description="Run OCR detection benchmark")
    parser.add_argument("--models", nargs="*", help="Filter: only run these models")
    parser.add_argument("--force", action="store_true", help="Re-run even if predictions exist")
    args = parser.parse_args()

    image_paths = get_image_paths()
    print(f"Dataset: {len(image_paths)} images\n")

    detectors = load_detectors(args.models)
    print(f"Loaded {len(detectors)} detectors: {[d.name for d in detectors]}\n")

    for detector in detectors:
        pred_path = PREDICTIONS_DIR / f"{detector.name}.json"
        if pred_path.exists() and not args.force:
            print(f"  [SKIP] {detector.name} — predictions exist ({pred_path})")
            continue

        print(f"Running {detector.name}...")
        detector.run_and_save(image_paths)
        print()


if __name__ == "__main__":
    main()
