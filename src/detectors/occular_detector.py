"""Occular OCR (DBNet) text detector."""

from ocr_skel import OCRPipeline

from src.detectors.base import BaseDetector


class OccularDetector(BaseDetector):
    name = "occular_ocr"

    def __init__(self, onnx: bool = True):
        self.pipeline = OCRPipeline(onnx=onnx)

    def detect(self, image_path: str) -> list[dict]:
        results = self.pipeline.process_image(image_path)
        boxes = []
        for r in results:
            quad = r["quad"]  # [[x,y], [x,y], [x,y], [x,y]]
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            boxes.append({
                "xtl": float(min(xs)),
                "ytl": float(min(ys)),
                "xbr": float(max(xs)),
                "ybr": float(max(ys)),
            })
        return boxes
