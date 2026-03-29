#!/usr/bin/env python3
"""Run a single recognition model and save predictions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES_DIR, PREDICTIONS_DIR
from src.parse_gt import parse_cvat_xml_with_text


def get_image_paths() -> list[str]:
    gt = parse_cvat_xml_with_text()
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

    rec_dir = PREDICTIONS_DIR / "recognition"
    rec_dir.mkdir(parents=True, exist_ok=True)
    pred_path = rec_dir / f"{args.model}.json"

    if pred_path.exists() and not args.force:
        print(f"[SKIP] {args.model} — predictions exist. Use --force to re-run.")
        return

    model = args.model.lower()

    if model == "tesseract":
        from src.recognizers.tesseract_rec import TesseractRecognizer
        rec = TesseractRecognizer()

    elif model == "easyocr":
        from src.recognizers.easyocr_rec import EasyOCRRecognizer
        rec = EasyOCRRecognizer()

    elif model == "surya":
        from src.recognizers.surya_rec import SuryaRecognizer
        rec = SuryaRecognizer()

    elif model == "paddle_eslav":
        from src.recognizers.paddle_rec import PaddleESlavRecognizer
        rec = PaddleESlavRecognizer()

    elif model == "paddle_cyrillic":
        from src.recognizers.paddle_rec import PaddleCyrillicRecognizer
        rec = PaddleCyrillicRecognizer()

    elif model == "docling":
        from src.recognizers.docling_rec import DoclingRecognizer
        rec = DoclingRecognizer()

    elif model == "dotsocr":
        from src.recognizers.dotsocr_rec import DotsOCRRecognizer
        rec = DotsOCRRecognizer()

    elif model == "occular" or model == "occular_ocr":
        from src.recognizers.occular_rec import OccularRecognizer
        rec = OccularRecognizer()

    elif model == "yandex" or model == "yandex_vision":
        from src.recognizers.yandex_rec import YandexRecognizer
        rec = YandexRecognizer()

    elif model.startswith("qwen"):
        from src.recognizers.qwen_rec import QwenVLRecognizer
        model_map = {
            "qwen2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct",
            "qwen2.5-vl-72b": "qwen/qwen2.5-vl-72b-instruct",
            "qwen3-vl-32b": "qwen/qwen3-vl-32b-instruct",
        }
        model_id = model_map.get(model)
        if not model_id:
            print(f"Unknown qwen model: {model}. Available: {list(model_map.keys())}")
            sys.exit(1)
        rec = QwenVLRecognizer(model_id, provider="Parasail")

    else:
        print(f"Unknown model: {args.model}")
        print("Available: tesseract, easyocr, surya, paddle_eslav, paddle_cyrillic,")
        print("           docling, dotsocr, occular, yandex_vision,")
        print("           qwen2.5-vl-32b, qwen2.5-vl-72b, qwen3-vl-32b")
        sys.exit(1)

    print(f"Running {rec.name}...")
    rec.run_and_save(image_paths)


if __name__ == "__main__":
    main()
