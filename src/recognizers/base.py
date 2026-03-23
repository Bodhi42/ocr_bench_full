"""Abstract base class for text recognizers."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from src.config import PREDICTIONS_DIR


class BaseRecognizer(ABC):
    """Every recognizer subclass must set `name` and implement `recognize()`."""

    name: str = "base"

    @abstractmethod
    def recognize(self, image_path: str) -> str:
        """Run full-page OCR on an image. Return recognized text."""
        ...

    def recognize_dataset(self, image_paths: list[str]) -> dict:
        """Run recognition on all images."""
        predictions = {}
        for path in tqdm(image_paths, desc=self.name):
            filename = Path(path).name
            try:
                text = self.recognize(path)
                predictions[filename] = text
            except Exception as e:
                print(f"  [ERROR] {self.name} failed on {filename}: {e}")
                predictions[filename] = ""
        return predictions

    def run_and_save(self, image_paths: list[str]) -> Path:
        """Run recognition, measure time, save to JSON."""
        start = time.time()
        predictions = self.recognize_dataset(image_paths)
        elapsed = time.time() - start

        rec_dir = PREDICTIONS_DIR / "recognition"
        rec_dir.mkdir(parents=True, exist_ok=True)
        output_path = rec_dir / f"{self.name}.json"

        result = {
            "model": self.name,
            "runtime_seconds": round(elapsed, 2),
            "num_images": len(predictions),
            "predictions": predictions,
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"  {self.name}: {len(predictions)} images, {elapsed:.1f}s -> {output_path}")
        return output_path
