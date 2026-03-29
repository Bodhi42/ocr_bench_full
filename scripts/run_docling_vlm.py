#!/usr/bin/env python3
"""Run Docling VLM pipeline presets on the full dataset."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.datamodel.pipeline_options import VlmPipelineOptions, VlmConvertOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

from src.parse_gt import parse_cvat_xml
from src.config import ANNOTATIONS_PATH, IMAGES_DIR, PREDICTIONS_DIR
from tqdm import tqdm


def run_preset(preset_name, image_paths, engine_override=None):
    """Run a single VLM preset on all images."""
    combo_name = f"docling_vlm_{preset_name}"
    rec_dir = PREDICTIONS_DIR / "recognition"
    rec_dir.mkdir(parents=True, exist_ok=True)
    out_path = rec_dir / f"{combo_name}.json"

    if out_path.exists():
        data = json.loads(out_path.read_text())
        if data.get("num_images", 0) == len(image_paths):
            print(f"  SKIP {combo_name} (already done)")
            return

    if engine_override:
        from docling.datamodel.vlm_engine_options import (
            ApiVlmEngineOptions, TransformersVlmEngineOptions,
            VllmVlmEngineOptions, VlmEngineType,
        )
        engine_opts_map = {
            "ollama": lambda: ApiVlmEngineOptions(engine_type=VlmEngineType.API_OLLAMA),
            "transformers": lambda: TransformersVlmEngineOptions(),
            "vllm": lambda: VllmVlmEngineOptions(),
        }
        vlm_options = VlmConvertOptions.from_preset(
            preset_name, engine_options=engine_opts_map[engine_override]()
        )
    else:
        vlm_options = VlmConvertOptions.from_preset(preset_name)

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    converter = DocumentConverter(
        format_options={
            "image": ImageFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipeline_options),
        }
    )

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

    # Filter by argument
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    engine = sys.argv[2] if len(sys.argv) > 2 else None

    PRESETS = [
        ("granite_docling", "ollama"),
        ("granite_vision", "ollama"),
        ("deepseek_ocr", "ollama"),
        ("smoldocling", "transformers"),
        ("qwen", "transformers"),
        ("got_ocr", "transformers"),
        ("dolphin", "transformers"),
        ("pixtral", "transformers"),
    ]

    for preset_name, default_engine in PRESETS:
        if filt and filt not in preset_name:
            continue
        eng = engine or default_engine
        print(f"\n=== {preset_name} (engine: {eng}) ===")
        try:
            run_preset(preset_name, image_paths, engine_override=eng)
        except Exception as e:
            print(f"  FATAL: {e}")


if __name__ == "__main__":
    main()
