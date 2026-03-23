"""Docling (IBM) full-page text recognizer with Tesseract OCR engine."""

import os
os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/5/tessdata")

from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.pipeline_options import TesseractOcrOptions

from src.recognizers.base import BaseRecognizer


class DoclingRecognizer(BaseRecognizer):
    name = "docling"

    def __init__(self):
        fmt = ImageFormatOption()
        fmt.pipeline_options.do_ocr = True
        fmt.pipeline_options.ocr_options = TesseractOcrOptions(
            lang=["rus", "eng"],
            force_full_page_ocr=True,
        )
        self.converter = DocumentConverter(
            format_options={"image": fmt},
        )

    def recognize(self, image_path: str) -> str:
        result = self.converter.convert(image_path)
        return result.document.export_to_text().strip()
