#!/usr/bin/env python3
"""Run Qwen VL recognition only on the full dataset via OpenRouter."""

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

    model = sys.argv[1] if len(sys.argv) > 1 else None
    if not model:
        print("Usage: run_qwen_rec.py <model_substring>")
        return

    models = [
        ("qwen/qwen2.5-vl-32b-instruct", "Parasail"),
        ("qwen/qwen2.5-vl-72b-instruct", "Parasail"),
    ]

    for model_id, provider in models:
        if model not in model_id:
            continue
        short = model_id.split("/")[-1]
        print(f"\n=== {short} (provider: {provider}) ===")
        rec = QwenVLRecognizer(model_id, provider=provider)
        rec_path = rec.run_and_save(image_paths)
        print(f"Saved: {rec_path}")


if __name__ == "__main__":
    main()
