"""Occular OCR full-page text recognizer."""

from ocr_skel import ocr

from src.recognizers.base import BaseRecognizer


class OccularRecognizer(BaseRecognizer):
    name = "occular_ocr"

    def __init__(self, onnx: bool = True):
        self._onnx = onnx

    def recognize(self, image_path: str) -> str:
        text = ocr(image_path, onnx=self._onnx)
        return text if isinstance(text, str) else " ".join(text)
