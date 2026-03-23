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

    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    print(f"Running {rec.name}...")
    rec.run_and_save(image_paths)


if __name__ == "__main__":
    main()
