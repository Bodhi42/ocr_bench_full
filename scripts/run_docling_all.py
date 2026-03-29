#!/usr/bin/env python3
"""Run all 24 Docling combinations (4 OCR engines × 6 layout models)."""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.pipeline_options import (
    TesseractOcrOptions,
    TesseractCliOcrOptions,
    EasyOcrOptions,
    RapidOcrOptions,
)
from docling.datamodel import layout_model_specs as lms

from src.parse_gt import parse_cvat_xml
from src.config import ANNOTATIONS_PATH, IMAGES_DIR, PREDICTIONS_DIR
from tqdm import tqdm

OCR_ENGINES = {
    "tesseract": TesseractOcrOptions(lang=["rus", "eng"], force_full_page_ocr=True),
    "tesseract_cli": TesseractCliOcrOptions(lang=["rus", "eng"], force_full_page_ocr=True),
    "easyocr": EasyOcrOptions(lang=["ru", "en"], force_full_page_ocr=True, use_gpu=True),
    "rapidocr": RapidOcrOptions(force_full_page_ocr=True),
}

LAYOUT_MODELS = {
    "heron": lms.DOCLING_LAYOUT_HERON,
    "heron_101": lms.DOCLING_LAYOUT_HERON_101,
    "egret_medium": lms.DOCLING_LAYOUT_EGRET_MEDIUM,
    "egret_large": lms.DOCLING_LAYOUT_EGRET_LARGE,
    "egret_xlarge": lms.DOCLING_LAYOUT_EGRET_XLARGE,
    "layout_v2": lms.DOCLING_LAYOUT_V2,
}


def run_combination(ocr_name, ocr_opts, layout_name, layout_spec, image_paths):
    combo_name = f"docling_{ocr_name}_{layout_name}"
    rec_dir = PREDICTIONS_DIR / "recognition"
    rec_dir.mkdir(parents=True, exist_ok=True)
    out_path = rec_dir / f"{combo_name}.json"

    if out_path.exists():
        data = json.loads(out_path.read_text())
        if data.get("num_images", 0) == len(image_paths):
            print(f"  SKIP {combo_name} (already done)")
            return

    fmt = ImageFormatOption()
    fmt.pipeline_options.do_ocr = True
    fmt.pipeline_options.ocr_options = ocr_opts
    fmt.pipeline_options.layout_options.model_spec = layout_spec

    converter = DocumentConverter(format_options={"image": fmt})

    predictions = {}
    start = time.time()

    for path in tqdm(image_paths, desc=combo_name):
        filename = Path(path).name
        try:
            result = converter.convert(path)
            text = result.document.export_to_text().strip()
            predictions[filename] = text
        except Exception as e:
            print(f"  [ERROR] {combo_name} failed on {filename}: {e}")
            predictions[filename] = ""

    elapsed = time.time() - start

    result = {
        "model": combo_name,
        "runtime_seconds": round(elapsed, 2),
        "num_images": len(predictions),
        "predictions": predictions,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  {combo_name}: {len(predictions)} images, {elapsed:.1f}s -> {out_path}")


def main():
    gt = parse_cvat_xml(str(ANNOTATIONS_PATH))
    image_paths = sorted(str(IMAGES_DIR / name) for name in gt)
    print(f"Dataset: {len(image_paths)} images")
    print(f"Combinations: {len(OCR_ENGINES)} OCR × {len(LAYOUT_MODELS)} layout = {len(OCR_ENGINES) * len(LAYOUT_MODELS)}")

    # Filter if argument given
    filt = sys.argv[1] if len(sys.argv) > 1 else None

    for ocr_name, ocr_opts in OCR_ENGINES.items():
        for layout_name, layout_spec in LAYOUT_MODELS.items():
            combo = f"{ocr_name}_{layout_name}"
            if filt and filt not in combo:
                continue
            print(f"\n=== {combo} ===")
            try:
                run_combination(ocr_name, ocr_opts, layout_name, layout_spec, image_paths)
            except Exception as e:
                print(f"  FATAL: {e}")


if __name__ == "__main__":
    main()
