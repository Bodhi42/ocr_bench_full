"""PaddleOCR full-page text recognizers — eslav and cyrillic."""

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

from src.recognizers.base import BaseRecognizer


def _parse_result(result) -> str:
    """Extract sorted text from PaddleOCR result."""
    results = list(result)
    if not results:
        return ""
    r = results[0]
    texts = r.get("rec_texts", [])
    polys = r.get("dt_polys", [])

    lines = []
    for i, poly in enumerate(polys):
        y_min = min(p[1] for p in poly)
        x_min = min(p[0] for p in poly)
        text = texts[i] if i < len(texts) else ""
        lines.append((y_min, x_min, text))

    lines.sort()
    return " ".join(line[2] for line in lines)


class PaddleESlavRecognizer(BaseRecognizer):
    name = "paddle_eslav"

    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ru",
        )

    def recognize(self, image_path: str) -> str:
        return _parse_result(self.ocr.predict(image_path))


class PaddleCyrillicRecognizer(BaseRecognizer):
    name = "paddle_cyrillic"

    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="rs_cyrillic",
        )

    def recognize(self, image_path: str) -> str:
        return _parse_result(self.ocr.predict(image_path))
