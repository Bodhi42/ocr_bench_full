#!/usr/bin/env python3
"""Compute WER for all saved recognition predictions."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PREDICTIONS_DIR, RESULTS_DIR
from src.recognition_metrics import evaluate_recognition
from src.parse_gt import parse_cvat_xml_with_text


def main():
    gt_data = parse_cvat_xml_with_text()
    total_words = sum(
        len(b["text"].split()) for v in gt_data.values() for b in v["boxes"] if b["text"].strip()
    )
    print(f"Ground truth: {len(gt_data)} images, {total_words} words\n")

    rec_dir = PREDICTIONS_DIR / "recognition"
    pred_files = sorted(rec_dir.glob("*.json"))
    if not pred_files:
        print("No recognition prediction files found.")
        return

    all_results = {}

    for pred_file in pred_files:
        pred_data = json.loads(pred_file.read_text())
        model_name = pred_data["model"]
        predictions = pred_data["predictions"]

        print(f"Evaluating {model_name}...")
        metrics = evaluate_recognition(gt_data, predictions)

        all_results[model_name] = {
            "micro_wer": metrics["micro_wer"],
            "macro_wer": metrics["macro_wer"],
            "total_ref_words": metrics["total_ref_words"],
            "per_image": metrics["per_image"],
        }

        print(f"  WER (micro): {metrics['micro_wer']:.3f}  WER (macro): {metrics['macro_wer']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "recognition_metrics.json"
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
