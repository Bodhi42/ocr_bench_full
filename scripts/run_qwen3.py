#!/usr/bin/env python3
"""Run Qwen3 VL 32B recognition on the full dataset via OpenRouter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parse_gt import parse_cvat_xml
from src.config import ANNOTATIONS_PATH, IMAGES_DIR
from src.recognizers.qwen_rec import QwenVLRecognizer


def main():
    gt = parse_cvat_xml(str(ANNOTATIONS_PATH))
    image_paths = sorted(str(IMAGES_DIR / name) for name in gt)
    print(f"Dataset: {len(image_paths)} images")

    rec = QwenVLRecognizer("qwen/qwen3-vl-32b-instruct", provider="Parasail")
    rec_path = rec.run_and_save(image_paths)
    print(f"Saved: {rec_path}")


if __name__ == "__main__":
    main()
