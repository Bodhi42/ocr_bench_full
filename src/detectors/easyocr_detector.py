"""EasyOCR CRAFT-based text detector."""

import easyocr

from src.detectors.base import BaseDetector


class EasyOCRDetector(BaseDetector):
    name = "easyocr_craft"

    def __init__(self):
        self.reader = easyocr.Reader(["ru", "en"], gpu=True)

    def detect(self, image_path: str) -> list[dict]:
        horizontal, free_form = self.reader.detect(image_path)

        boxes = []

        # Horizontal boxes: [[x_min, x_max, y_min, y_max], ...]
        if horizontal and len(horizontal) > 0:
            for bbox in horizontal[0]:
                x_min, x_max, y_min, y_max = bbox
                boxes.append({
                    "xtl": float(x_min),
                    "ytl": float(y_min),
                    "xbr": float(x_max),
                    "ybr": float(y_max),
                })

        # Free-form boxes: list of polygons -> convert to axis-aligned
        if free_form and len(free_form) > 0:
            for polygon in free_form[0]:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                boxes.append({
                    "xtl": float(min(xs)),
                    "ytl": float(min(ys)),
                    "xbr": float(max(xs)),
                    "ybr": float(max(ys)),
                })

        return boxes
