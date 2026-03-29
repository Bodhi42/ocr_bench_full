"""Qwen VL document recognizer via OpenRouter."""

from src.recognizers.base import BaseRecognizer
from src.qwen_vl import call_qwen_vl, extract_text


class QwenVLRecognizer(BaseRecognizer):
    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        self.provider = provider
        self.name = model.split("/")[-1]

    def recognize(self, image_path: str) -> str:
        result = call_qwen_vl(image_path, self.model, self.provider)
        return extract_text(result)
