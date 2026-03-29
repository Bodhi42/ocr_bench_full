#!/usr/bin/env python3
"""Generate CVAT XML ground truth annotations from Yandex Vision OCR results."""

import sys
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES_DIR, DATA_DIR
from src.yandex_vision import call_vision_api, extract_lines


def main():
    image_paths = sorted(IMAGES_DIR.iterdir())
    print(f"Generating GT from Yandex Vision for {len(image_paths)} images")

    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    # Minimal meta block
    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "name").text = "OCR Dataset (Yandex Vision GT)"
    SubElement(task, "size").text = str(len(image_paths))

    total_boxes = 0
    for idx, img_path in enumerate(image_paths):
        im = Image.open(img_path)
        w, h = im.size

        image_el = SubElement(root, "image",
                              id=str(idx),
                              name=img_path.name,
                              width=str(w),
                              height=str(h))

        result = call_vision_api(str(img_path))
        lines = extract_lines(result)

        for line in lines:
            box_el = SubElement(image_el, "box",
                                label="Text",
                                source="yandex_vision",
                                occluded="0",
                                xtl=f"{line['xtl']:.2f}",
                                ytl=f"{line['ytl']:.2f}",
                                xbr=f"{line['xbr']:.2f}",
                                ybr=f"{line['ybr']:.2f}",
                                z_order="0")
            attr_el = SubElement(box_el, "attribute", name="transcription")
            attr_el.text = line["text"]
            total_boxes += 1

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(image_paths)} images processed")

    indent(root, space="  ")
    tree = ElementTree(root)

    output_path = DATA_DIR / "annotations" / "annotations_yandex.xml"
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
    print(f"\nDone: {total_boxes} boxes across {len(image_paths)} images")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
