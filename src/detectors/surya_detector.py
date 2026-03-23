"""Surya text detector."""

from PIL import Image
from surya.detection import DetectionPredictor

from src.detectors.base import BaseDetector


class SuryaDetector(BaseDetector):
    name = "surya"

    def __init__(self):
        self.predictor = DetectionPredictor()

    def detect(self, image_path: str) -> list[dict]:
        image = Image.open(image_path)
        results = self.predictor([image])

        boxes = []
        for text_line in results[0].bboxes:
            bbox = text_line.bbox
            boxes.append({
                "xtl": float(bbox[0]),
                "ytl": float(bbox[1]),
                "xbr": float(bbox[2]),
                "ybr": float(bbox[3]),
            })

        return boxes
