"""Surya 0.13.1 full-page text recognizer."""

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from src.recognizers.base import BaseRecognizer


class SuryaRecognizer(BaseRecognizer):
    name = "surya"

    def __init__(self):
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()

    def recognize(self, image_path: str) -> str:
        image = Image.open(image_path)
        results = self.rec_predictor([image], [["ru", "en"]], self.det_predictor)

        lines = []
        for text_line in results[0].text_lines:
            lines.append((text_line.bbox[1], text_line.bbox[0], text_line.text))

        # Sort by y then x
        lines.sort()
        return " ".join(line[2] for line in lines)
