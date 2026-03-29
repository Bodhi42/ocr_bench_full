#!/usr/bin/env python3
"""Run Qwen VL detection + recognition on the full dataset via OpenRouter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parse_gt import parse_cvat_xml
from src.config import ANNOTATIONS_PATH, IMAGES_DIR
from src.detectors.qwen_detector import QwenVLDetector
from src.recognizers.qwen_rec import QwenVLRecognizer


MODELS = [
    ("qwen/qwen2.5-vl-32b-instruct", "Parasail"),
    ("qwen/qwen2.5-vl-72b-instruct", "Parasail"),
]


def main():
    gt = parse_cvat_xml(str(ANNOTATIONS_PATH))
    image_paths = sorted(str(IMAGES_DIR / name) for name in gt)
    print(f"Dataset: {len(image_paths)} images")

    # Filter models if argument given
    selected = sys.argv[1] if len(sys.argv) > 1 else None

    for model_id, provider in MODELS:
        short = model_id.split("/")[-1]
        if selected and selected not in short:
            continue

        print(f"\n{'='*60}")
        print(f"Model: {short} (provider: {provider})")
        print(f"{'='*60}")

        detector = QwenVLDetector(model_id, provider=provider)
        recognizer = QwenVLRecognizer(model_id, provider=provider)

        print("\n--- Detection ---")
        det_path = detector.run_and_save(image_paths)
        print(f"Saved: {det_path}")

        print("\n--- Recognition ---")
        rec_path = recognizer.run_and_save(image_paths)
        print(f"Saved: {rec_path}")


if __name__ == "__main__":
    main()
