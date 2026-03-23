"""Tesseract text detector with word-to-line merging."""

import cv2
import pytesseract

from src.detectors.base import BaseDetector


def _merge_words_to_lines(word_boxes: list[dict], y_overlap_thresh: float = 0.5,
                          x_gap_thresh: float = 30) -> list[dict]:
    """Merge word-level boxes into line-level boxes.

    Words are grouped when they overlap vertically and are close horizontally.
    """
    if not word_boxes:
        return []

    sorted_boxes = sorted(word_boxes, key=lambda b: (b["ytl"], b["xtl"]))
    lines = []
    current_line = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        last = current_line[-1]

        # Check vertical overlap
        overlap_top = max(last["ytl"], box["ytl"])
        overlap_bot = min(last["ybr"], box["ybr"])
        overlap_h = max(0, overlap_bot - overlap_top)
        min_height = min(last["ybr"] - last["ytl"], box["ybr"] - box["ytl"])

        if min_height > 0 and overlap_h / min_height >= y_overlap_thresh:
            # Check horizontal gap
            gap = box["xtl"] - max(b["xbr"] for b in current_line)
            if gap < x_gap_thresh:
                current_line.append(box)
                continue

        # Flush current line
        lines.append({
            "xtl": min(b["xtl"] for b in current_line),
            "ytl": min(b["ytl"] for b in current_line),
            "xbr": max(b["xbr"] for b in current_line),
            "ybr": max(b["ybr"] for b in current_line),
        })
        current_line = [box]

    if current_line:
        lines.append({
            "xtl": min(b["xtl"] for b in current_line),
            "ytl": min(b["ytl"] for b in current_line),
            "xbr": max(b["xbr"] for b in current_line),
            "ybr": max(b["ybr"] for b in current_line),
        })

    return lines


class TesseractDetector(BaseDetector):
    name = "tesseract"

    def detect(self, image_path: str) -> list[dict]:
        image = cv2.imread(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT,
                                         config="--oem 3 --psm 3")

        word_boxes = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) < 0 or not data["text"][i].strip():
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w > 0 and h > 0:
                word_boxes.append({"xtl": x, "ytl": y, "xbr": x + w, "ybr": y + h})

        return _merge_words_to_lines(word_boxes)
