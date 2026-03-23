"""Compute detection metrics: IoU matching, precision, recall, F1."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two axis-aligned boxes."""
    x1 = max(box_a["xtl"], box_b["xtl"])
    y1 = max(box_a["ytl"], box_b["ytl"])
    x2 = min(box_a["xbr"], box_b["xbr"])
    y2 = min(box_a["ybr"], box_b["ybr"])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a["xbr"] - box_a["xtl"]) * (box_a["ybr"] - box_a["ytl"])
    area_b = (box_b["xbr"] - box_b["xtl"]) * (box_b["ybr"] - box_b["ytl"])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(gt_boxes: list[dict], pred_boxes: list[dict],
                iou_threshold: float = 0.5) -> dict:
    """Match predicted boxes to ground truth using Hungarian algorithm.

    Returns dict with tp, fp, fn counts and matched pairs with IoU values.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "matches": []}
    if len(gt_boxes) == 0:
        return {"tp": 0, "fp": len(pred_boxes), "fn": 0, "matches": []}
    if len(pred_boxes) == 0:
        return {"tp": 0, "fp": 0, "fn": len(gt_boxes), "matches": []}

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt, pred)

    # Hungarian matching (minimize cost = maximize IoU)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    matches = []
    tp = 0
    matched_gt = set()
    matched_pred = set()

    for gi, pi in zip(gt_indices, pred_indices):
        iou_val = iou_matrix[gi, pi]
        if iou_val >= iou_threshold:
            tp += 1
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append({"gt_idx": int(gi), "pred_idx": int(pi), "iou": float(iou_val)})

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    return {"tp": tp, "fp": fp, "fn": fn, "matches": matches}


def compute_area_coverage(gt_boxes: list[dict], pred_boxes: list[dict]) -> list[float]:
    """For each GT box, compute the max IoU with any prediction.

    Returns list of coverage values (one per GT box).
    """
    coverages = []
    for gt in gt_boxes:
        if not pred_boxes:
            coverages.append(0.0)
            continue
        max_iou = max(compute_iou(gt, pred) for pred in pred_boxes)
        coverages.append(max_iou)
    return coverages


def evaluate_model(gt_data: dict, predictions: dict,
                   iou_threshold: float = 0.5) -> dict:
    """Evaluate a model's predictions against ground truth.

    Args:
        gt_data: output of parse_cvat_xml()
        predictions: {filename: [boxes]}
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with per-image and aggregate metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_coverages = []
    per_image = {}

    for filename, gt_info in gt_data.items():
        gt_boxes = gt_info["boxes"]
        pred_boxes = predictions.get(filename, [])

        result = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        coverages = compute_area_coverage(gt_boxes, pred_boxes)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        all_coverages.extend(coverages)

        p = result["tp"] / (result["tp"] + result["fp"]) if (result["tp"] + result["fp"]) > 0 else 0
        r = result["tp"] / (result["tp"] + result["fn"]) if (result["tp"] + result["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        per_image[filename] = {
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mean_coverage": round(np.mean(coverages), 4) if coverages else 0,
        }

    # Micro-averaged metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_coverage = float(np.mean(all_coverages)) if all_coverages else 0

    return {
        "iou_threshold": iou_threshold,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_area_coverage": round(mean_coverage, 4),
        "per_image": per_image,
    }
