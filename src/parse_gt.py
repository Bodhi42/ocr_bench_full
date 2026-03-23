"""Parse CVAT XML annotations into a standardized format."""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.config import ANNOTATIONS_PATH


def parse_cvat_xml(xml_path: Path = ANNOTATIONS_PATH) -> dict:
    """Parse CVAT 1.1 XML and return ground truth bounding boxes.

    Returns:
        {
            "img_0001.jpg": {
                "width": 1024,
                "height": 678,
                "boxes": [
                    {"xtl": 403.0, "ytl": 30.0, "xbr": 588.0, "ybr": 48.0},
                    ...
                ]
            },
            ...
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ground_truth = {}
    for image_el in root.findall("image"):
        name = image_el.get("name")
        width = int(image_el.get("width"))
        height = int(image_el.get("height"))

        boxes = []
        for box_el in image_el.findall("box"):
            boxes.append({
                "xtl": float(box_el.get("xtl")),
                "ytl": float(box_el.get("ytl")),
                "xbr": float(box_el.get("xbr")),
                "ybr": float(box_el.get("ybr")),
            })

        ground_truth[name] = {
            "width": width,
            "height": height,
            "boxes": boxes,
        }

    return ground_truth


def parse_cvat_xml_with_text(xml_path: Path = ANNOTATIONS_PATH) -> dict:
    """Parse CVAT XML and return ground truth with transcriptions.

    Returns:
        {
            "img_0001.jpg": {
                "width": 1024,
                "height": 678,
                "boxes": [
                    {
                        "xtl": 403.0, "ytl": 30.0, "xbr": 588.0, "ybr": 48.0,
                        "text": "77 А А 8648"
                    },
                    ...
                ]
            },
            ...
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ground_truth = {}
    for image_el in root.findall("image"):
        name = image_el.get("name")
        width = int(image_el.get("width"))
        height = int(image_el.get("height"))

        boxes = []
        for box_el in image_el.findall("box"):
            attr_el = box_el.find("attribute")
            text = attr_el.text if attr_el is not None and attr_el.text else ""
            boxes.append({
                "xtl": float(box_el.get("xtl")),
                "ytl": float(box_el.get("ytl")),
                "xbr": float(box_el.get("xbr")),
                "ybr": float(box_el.get("ybr")),
                "text": text,
            })

        ground_truth[name] = {
            "width": width,
            "height": height,
            "boxes": boxes,
        }

    return ground_truth
