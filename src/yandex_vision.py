"""Shared Yandex Vision OCR API client.

Caches results so detection and recognition share a single API call per image.
"""

import base64
import io
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

from src.config import PREDICTIONS_DIR

_CACHE_DIR = PREDICTIONS_DIR / "yandex_cache"
_API_URL = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
_FOLDER_ID = os.environ.get("YC_FOLDER_ID", "b1g4ft3qvdafhejcl7p3")


def _get_iam_token() -> str:
    """Get a fresh IAM token via yc CLI."""
    result = subprocess.run(
        ["/home/david/yandex-cloud/bin/yc", "iam", "create-token"],
        capture_output=True, text=True, timeout=30,
    )
    token = result.stdout.strip()
    if not token:
        raise RuntimeError(f"Failed to get IAM token: {result.stderr}")
    return token


_iam_token: str | None = None
_token_time: float = 0


def _token() -> str:
    global _iam_token, _token_time
    # Refresh every 30 minutes (tokens last 12h but be safe)
    if _iam_token is None or (time.time() - _token_time) > 1800:
        _iam_token = _get_iam_token()
        _token_time = time.time()
    return _iam_token


def call_vision_api(image_path: str) -> dict:
    """Call Yandex Vision API for a single image. Returns raw API response.

    Results are cached on disk to avoid redundant API calls.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = Path(image_path).stem
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    with open(image_path, "rb") as f:
        raw = f.read()

    # Yandex Vision limit is ~1.5MB per request body.
    # Re-encode large images as JPEG to stay under the limit.
    if len(raw) > 1_000_000:
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        raw = buf.getvalue()

    img_b64 = base64.b64encode(raw).decode()

    body = json.dumps({
        "folderId": _FOLDER_ID,
        "analyzeSpecs": [{
            "content": img_b64,
            "features": [{
                "type": "TEXT_DETECTION",
                "textDetectionConfig": {"languageCodes": ["ru", "en"]},
            }],
        }],
    }).encode()

    req = urllib.request.Request(
        _API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            result = json.loads(resp.read())
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            # Token might have expired
            global _iam_token
            _iam_token = None
            req.remove_header("Authorization")
            req.add_header("Authorization", f"Bearer {_token()}")

    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def extract_lines(api_result: dict) -> list[dict]:
    """Extract line-level bounding boxes and text from API response.

    Returns list of {"xtl", "ytl", "xbr", "ybr", "text"}.
    """
    lines = []
    try:
        pages = api_result["results"][0]["results"][0]["textDetection"]["pages"]
    except (KeyError, IndexError):
        return lines

    for page in pages:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                vertices = line.get("boundingBox", {}).get("vertices", [])
                if len(vertices) < 4:
                    continue
                xs = [int(v.get("x", 0)) for v in vertices]
                ys = [int(v.get("y", 0)) for v in vertices]
                words = [w["text"] for w in line.get("words", []) if "text" in w]
                lines.append({
                    "xtl": float(min(xs)),
                    "ytl": float(min(ys)),
                    "xbr": float(max(xs)),
                    "ybr": float(max(ys)),
                    "text": " ".join(words),
                })
    return lines
