"""doctr detection models."""

import numpy as np
from PIL import Image
from doctr.models import detection_predictor

from src.detectors.base import BaseDetector


class DoctrDetector(BaseDetector):
    """Generic doctr detector wrapper."""

    def __init__(self, arch: str = "db_resnet50"):
        self.arch = arch
        self.name = f"doctr_{arch}"
        self.model = detection_predictor(arch=arch, pretrained=True)

    def detect(self, image_path: str) -> list[dict]:
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        result = self.model([image])

        boxes = []
        # result[0]["words"] is ndarray of shape (N, 5): [xtl, ytl, xbr, ybr, conf]
        # coordinates are relative [0, 1]
        words = result[0]["words"]
        for row in words:
            boxes.append({
                "xtl": float(row[0] * w),
                "ytl": float(row[1] * h),
                "xbr": float(row[2] * w),
                "ybr": float(row[3] * h),
            })

        return boxes


DOCTR_ARCHITECTURES = [
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]


def get_all_doctr_detectors() -> list[DoctrDetector]:
    """Create detector instances for all available doctr architectures."""
    detectors = []
    for arch in DOCTR_ARCHITECTURES:
        try:
            detectors.append(DoctrDetector(arch=arch))
        except Exception as e:
            print(f"  [WARN] Could not load doctr {arch}: {e}")
    return detectors
