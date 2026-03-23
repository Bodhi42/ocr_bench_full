"""Abstract base class for text detectors."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from src.config import PREDICTIONS_DIR


class BaseDetector(ABC):
    """Every detector subclass must set `name` and implement `detect()`."""

    name: str = "base"

    @abstractmethod
    def detect(self, image_path: str) -> list[dict]:
        """Run detection on a single image.

        Returns:
            List of {"xtl": float, "ytl": float, "xbr": float, "ybr": float}
        """
        ...

    def detect_dataset(self, image_paths: list[str]) -> dict:
        """Run detection on all images and return predictions dict."""
        predictions = {}
        for path in tqdm(image_paths, desc=self.name):
            filename = Path(path).name
            try:
                boxes = self.detect(path)
                predictions[filename] = boxes
            except Exception as e:
                print(f"  [ERROR] {self.name} failed on {filename}: {e}")
                predictions[filename] = []
        return predictions

    def run_and_save(self, image_paths: list[str]) -> Path:
        """Run detection on all images, measure time, save to JSON."""
        start = time.time()
        predictions = self.detect_dataset(image_paths)
        elapsed = time.time() - start

        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PREDICTIONS_DIR / f"{self.name}.json"

        result = {
            "model": self.name,
            "runtime_seconds": round(elapsed, 2),
            "num_images": len(predictions),
            "total_boxes": sum(len(b) for b in predictions.values()),
            "predictions": predictions,
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  {self.name}: {result['total_boxes']} boxes, {elapsed:.1f}s -> {output_path}")
        return output_path
