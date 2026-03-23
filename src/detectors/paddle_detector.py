"""PaddleOCR PP-OCRv5 text detectors — mobile and server variants."""

import os
import numpy as np

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

from src.detectors.base import BaseDetector


def _parse_v5_result(result) -> list[dict]:
    """Parse PaddleOCR v5 detection result into standardized boxes."""
    boxes = []
    results = list(result)
    if not results:
        return boxes
    r = results[0]
    polys = r.get("dt_polys", [])
    for poly in polys:
        pts = np.array(poly)
        boxes.append({
            "xtl": float(pts[:, 0].min()),
            "ytl": float(pts[:, 1].min()),
            "xbr": float(pts[:, 0].max()),
            "ybr": float(pts[:, 1].max()),
        })
    return boxes


class PaddleMobileDetector(BaseDetector):
    name = "paddle_v5_mobile"

    def __init__(self):
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def detect(self, image_path: str) -> list[dict]:
        result = self.ocr.predict(image_path)
        return _parse_v5_result(result)


class PaddleServerDetector(BaseDetector):
    name = "paddle_v5_server"

    def __init__(self):
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def detect(self, image_path: str) -> list[dict]:
        result = self.ocr.predict(image_path)
        return _parse_v5_result(result)
