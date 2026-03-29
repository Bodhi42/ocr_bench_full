"""Run Yandex Vision OCR detection + recognition on the full dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parse_gt import parse_cvat_xml
from src.config import ANNOTATIONS_PATH, IMAGES_DIR
from src.detectors.yandex_detector import YandexDetector
from src.recognizers.yandex_rec import YandexRecognizer


def main():
    gt = parse_cvat_xml(str(ANNOTATIONS_PATH))
    image_paths = sorted(str(IMAGES_DIR / name) for name in gt)
    print(f"Dataset: {len(image_paths)} images")

    detector = YandexDetector()
    recognizer = YandexRecognizer()

    print("\n=== Running Yandex Vision detection ===")
    det_path = detector.run_and_save(image_paths)
    print(f"Detection saved to: {det_path}")

    print("\n=== Running Yandex Vision recognition ===")
    rec_path = recognizer.run_and_save(image_paths)
    print(f"Recognition saved to: {rec_path}")


if __name__ == "__main__":
    main()
