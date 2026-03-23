#!/usr/bin/env python3
"""Compute metrics for all saved predictions and generate results JSON."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PREDICTIONS_DIR, RESULTS_DIR, IOU_THRESHOLDS, DEFAULT_IOU_THRESHOLD
from src.metrics import evaluate_model, compute_area_coverage
from src.parse_gt import parse_cvat_xml


def main():
    gt_data = parse_cvat_xml()
    print(f"Ground truth: {len(gt_data)} images, "
          f"{sum(len(v['boxes']) for v in gt_data.values())} boxes\n")

    pred_files = sorted(PREDICTIONS_DIR.glob("*.json"))
    if not pred_files:
        print("No prediction files found. Run detectors first.")
        return

    all_results = {}

    for pred_file in pred_files:
        pred_data = json.loads(pred_file.read_text())
        model_name = pred_data["model"]
        predictions = pred_data["predictions"]
        runtime = pred_data.get("runtime_seconds", 0)

        print(f"Evaluating {model_name}...")

        # Main evaluation at default threshold
        main_eval = evaluate_model(gt_data, predictions, DEFAULT_IOU_THRESHOLD)

        # IoU sweep
        iou_sweep = {}
        for threshold in np.arange(0.1, 0.95, 0.05):
            t = round(threshold, 2)
            ev = evaluate_model(gt_data, predictions, t)
            iou_sweep[str(t)] = {
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
            }

        # Area coverage per GT box
        all_coverages = []
        for filename, gt_info in gt_data.items():
            pred_boxes = predictions.get(filename, [])
            coverages = compute_area_coverage(gt_info["boxes"], pred_boxes)
            all_coverages.extend(coverages)

        all_results[model_name] = {
            "runtime_seconds": runtime,
            "total_predictions": pred_data["total_boxes"],
            "aggregate": main_eval,
            "iou_sweep": iou_sweep,
            "coverage_values": [round(c, 4) for c in all_coverages],
        }

        print(f"  P={main_eval['precision']:.3f}  R={main_eval['recall']:.3f}  "
              f"F1={main_eval['f1']:.3f}  Coverage={main_eval['mean_area_coverage']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "metrics.json"
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
