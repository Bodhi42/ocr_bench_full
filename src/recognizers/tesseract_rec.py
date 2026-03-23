"""Tesseract full-page text recognizer."""

import cv2
import pytesseract

from src.recognizers.base import BaseRecognizer


class TesseractRecognizer(BaseRecognizer):
    name = "tesseract"

    def recognize(self, image_path: str) -> str:
        image = cv2.imread(image_path)
        text = pytesseract.image_to_string(image, lang="rus+eng", config="--oem 3 --psm 3")
        return text.strip()
