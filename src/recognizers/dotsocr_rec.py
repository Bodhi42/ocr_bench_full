"""dots.ocr (VLM-based) full-page text recognizer via vLLM server."""

import base64
import io
import math
import requests
from PIL import Image

from src.recognizers.base import BaseRecognizer

VLLM_URL = "http://localhost:8899/v1/chat/completions"
IMAGE_FACTOR = 28
MAX_PIXELS = 11289600
MIN_PIXELS = 3136


def _smart_resize(width: int, height: int) -> tuple[int, int]:
    """Resize dimensions to be factor-aligned and within pixel bounds."""
    w = max(IMAGE_FACTOR, round(width / IMAGE_FACTOR) * IMAGE_FACTOR)
    h = max(IMAGE_FACTOR, round(height / IMAGE_FACTOR) * IMAGE_FACTOR)

    if w * h > MAX_PIXELS:
        beta = math.sqrt((w * h) / MAX_PIXELS)
        w = int(math.floor(w / beta / IMAGE_FACTOR) * IMAGE_FACTOR)
        h = int(math.floor(h / beta / IMAGE_FACTOR) * IMAGE_FACTOR)

    if w * h < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (w * h))
        w = int(math.ceil(w * beta / IMAGE_FACTOR) * IMAGE_FACTOR)
        h = int(math.ceil(h * beta / IMAGE_FACTOR) * IMAGE_FACTOR)

    return w, h


class DotsOCRRecognizer(BaseRecognizer):
    name = "dotsocr"

    def recognize(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        new_w, new_h = _smart_resize(img.width, img.height)
        img = img.resize((new_w, new_h))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = "<|img|><|imgpad|><|endofimg|>Extract the text content from this image."

        response = requests.post(
            VLLM_URL,
            json={
                "model": "dotsocr",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0.1,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        # Post-process: truncate at repetition loops
        words = text.split()
        for i in range(min(len(words), 200), len(words)):
            chunk = " ".join(words[i - 3:i])
            if " ".join(words[i:]).startswith(chunk):
                text = " ".join(words[:i])
                break

        return text
