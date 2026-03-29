"""Qwen VL document detector via OpenRouter."""

from src.detectors.base import BaseDetector
from src.qwen_vl import call_qwen_vl, extract_boxes


class QwenVLDetector(BaseDetector):
    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        self.provider = provider
        self.name = model.split("/")[-1]

    def detect(self, image_path: str) -> list[dict]:
        result = call_qwen_vl(image_path, self.model, self.provider)
        return extract_boxes(result)
