"""Yandex Vision OCR text detector."""

from src.detectors.base import BaseDetector
from src.yandex_vision import call_vision_api, extract_lines


class YandexDetector(BaseDetector):
    name = "yandex_vision"

    def detect(self, image_path: str) -> list[dict]:
        result = call_vision_api(image_path)
        lines = extract_lines(result)
        return [{"xtl": l["xtl"], "ytl": l["ytl"], "xbr": l["xbr"], "ybr": l["ybr"]} for l in lines]
