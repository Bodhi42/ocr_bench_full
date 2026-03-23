"""EasyOCR full-page text recognizer."""

import easyocr

from src.recognizers.base import BaseRecognizer


class EasyOCRRecognizer(BaseRecognizer):
    name = "easyocr"

    def __init__(self):
        self.reader = easyocr.Reader(["ru", "en"], gpu=True)

    def recognize(self, image_path: str) -> str:
        results = self.reader.readtext(image_path)
        # Sort by vertical position, then horizontal
        results.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
        lines = [r[1] for r in results]
        return " ".join(lines)
