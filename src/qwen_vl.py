"""Shared Qwen VL OCR client via OpenRouter API.

Sends images to Qwen VL models with JSON schema for structured bbox+text output.
Caches results to avoid redundant API calls.
"""

import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path

from PIL import Image

from src.config import PREDICTIONS_DIR

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

_MAX_LONG_SIDE = 2000

_DOCUMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "description": "Bounding box [x1, y1, x2, y2] in pixel coordinates",
                        "items": {"type": "integer"},
                    },
                    "label": {
                        "type": "string",
                        "description": "Document element label",
                    },
                    "text": {
                        "type": "string",
                        "description": "Extracted text content from the detected area",
                    },
                },
                "required": ["bbox_2d", "label", "text"],
            },
        }
    },
    "required": ["objects"],
}

_PROMPT = (
    "Detect every individual text line in this document image. "
    "Do NOT group lines together — each line must be a separate object. "
    "For each text line provide:\n"
    "1. label: 'text_line'\n"
    "2. bbox_2d: [x1, y1, x2, y2] bounding box around that single line.\n"
    "3. text: the exact text content of that line, preserving all characters exactly as shown.\n"
    "Do NOT censor, mask, or replace any text. Transcribe everything verbatim.\n"
)


def _resize_image(image_path: str) -> tuple[str, tuple[int, int]]:
    """Resize image so long side <= _MAX_LONG_SIDE, return base64 + original size."""
    img = Image.open(image_path)
    orig_size = img.size  # (w, h)

    long_side = max(img.size)
    if long_side > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / long_side
        new_w = int(img.size[0] * scale)
        new_h = int(img.size[1] * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, orig_size, img.size


def call_qwen_vl(image_path: str, model: str, provider: str | None = None) -> dict:
    """Call Qwen VL via OpenRouter. Returns parsed JSON with objects list.

    Results are cached on disk per model.
    """
    cache_dir = PREDICTIONS_DIR / f"qwen_cache_{model.split('/')[-1]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = Path(image_path).stem
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    img_b64, orig_size, resized = _resize_image(image_path)

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "document_ocr",
                "strict": True,
                "schema": _DOCUMENT_JSON_SCHEMA,
            },
        },
        "max_tokens": 16384,
        "temperature": 0.1,
    }

    if provider:
        body["provider"] = {"order": [provider]}

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        _API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            resp_data = json.loads(resp.read())
            break
        except Exception as e:
            if attempt == 2:
                raise
            err_body = ""
            if hasattr(e, "read"):
                err_body = e.read().decode()
            print(f"    Retry {attempt+1}: {e} {err_body[:200]}")
            time.sleep(2 ** (attempt + 1))

    content = resp_data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    # Scale bbox coordinates back to original image size if resized
    if resized != orig_size:
        scale_x = orig_size[0] / resized[0]
        scale_y = orig_size[1] / resized[1]
        for obj in parsed.get("objects", []):
            bbox = obj.get("bbox_2d", [])
            if len(bbox) == 4:
                obj["bbox_2d"] = [
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y),
                ]

    result = {
        "objects": parsed.get("objects", []),
        "orig_size": list(orig_size),
        "resized": list(resized),
        "usage": resp_data.get("usage", {}),
    }
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def extract_boxes(api_result: dict) -> list[dict]:
    """Extract bounding boxes in benchmark format."""
    boxes = []
    for obj in api_result.get("objects", []):
        bbox = obj.get("bbox_2d", [])
        if len(bbox) == 4:
            boxes.append({
                "xtl": float(bbox[0]),
                "ytl": float(bbox[1]),
                "xbr": float(bbox[2]),
                "ybr": float(bbox[3]),
            })
    return boxes


def extract_text(api_result: dict) -> str:
    """Extract full page text, sorted by position."""
    objects = api_result.get("objects", [])
    # Sort by y then x
    objects_sorted = sorted(objects, key=lambda o: (
        o.get("bbox_2d", [0, 0])[1],
        o.get("bbox_2d", [0, 0])[0],
    ))
    texts = [o.get("text", "") for o in objects_sorted if o.get("text", "").strip()]
    return " ".join(texts)
