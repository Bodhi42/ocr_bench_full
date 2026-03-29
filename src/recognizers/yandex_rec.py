"""Yandex Vision OCR text recognizer."""

from src.recognizers.base import BaseRecognizer
from src.yandex_vision import call_vision_api, extract_lines


class YandexRecognizer(BaseRecognizer):
    name = "yandex_vision"

    def recognize(self, image_path: str) -> str:
        result = call_vision_api(image_path)
        lines = extract_lines(result)
        # Sort by vertical position, then horizontal
        lines.sort(key=lambda l: (l["ytl"], l["xtl"]))
        return " ".join(l["text"] for l in lines)
