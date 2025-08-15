# File: data/raw/dataset/create_flickr30k_json.py

import os
import random
import json
import xml.etree.ElementTree as ET

# Paths (script lives in data/raw/dataset/)
CONTEXT_FILE = "context.txt"
ANNOTATION_DIR = "annotation"
NUM_IMAGES = 3000
OUTPUT_JSON = "flickr30k_3000_random.json"

def load_captions(context_file):
    """Load captions into {image: [caption, …]}."""
    captions = {}
    with open(context_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            img, cap = parts
            img = img.strip()
            cap = cap.strip().strip('"')
            captions.setdefault(img, []).append(cap)
    return captions

def parse_annotation(xml_path):
    """
    Parse one XML annotation. Returns:
      - objects: list of {"name", "xmin","ymin","xmax","ymax"}
      - image_info: {"width","height","depth"}
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image size
        image_info = {}
        size = root.find("size")
        if size is not None:
            w = size.find("width"); h = size.find("height"); d = size.find("depth")
            image_info = {
                "width": int(w.text) if w is not None else None,
                "height": int(h.text) if h is not None else None,
                "depth": int(d.text) if d is not None else None
            }

        objects = []
        for i, obj in enumerate(root.findall("object")):
            # Skip objects marked scene/no box
            if obj.find("nobndbox") is not None and obj.find("nobndbox").text == "1":
                continue
            b = obj.find("bndbox")
            if b is None:
                # warn and skip malformed
                print(f"[WARN] No <bndbox> in {xml_path} object index {i}; skipping.")
                continue
            name = obj.find("name").text if obj.find("name") is not None else "unknown"
            xmin = b.find("xmin"); ymin = b.find("ymin")
            xmax = b.find("xmax"); ymax = b.find("ymax")
            if None in (xmin, ymin, xmax, ymax):
                print(f"[WARN] Incomplete box coords in {xml_path} object {i}; skipping.")
                continue
            objects.append({
                "name": name,
                "xmin": int(xmin.text),
                "ymin": int(ymin.text),
                "xmax": int(xmax.text),
                "ymax": int(ymax.text),
            })
        return objects, image_info

    except ET.ParseError as e:
        print(f"[WARN] XML parse error in {xml_path}: {e}")
        return [], {}
    except Exception as e:
        print(f"[WARN] Error processing {xml_path}: {e}")
        return [], {}

def main():
    captions_dict = load_captions(CONTEXT_FILE)
    # Filter to images with both captions and XML
    available = [
        img for img in captions_dict
        if os.path.exists(os.path.join(ANNOTATION_DIR, img.replace(".jpg", ".xml")))
    ]
    print(f"Total images with captions & XML: {len(available)}")

    chosen = available if len(available) < NUM_IMAGES else random.sample(available, NUM_IMAGES)
    print(f"Processing {len(chosen)} images...")

    output = []
    for idx, img in enumerate(chosen, 1):
        xml_path = os.path.join(ANNOTATION_DIR, img.replace(".jpg", ".xml"))
        objects, info = parse_annotation(xml_path)
        output.append({
            "image": img,
            "captions": captions_dict[img],
            "objects": objects,
            "image_info": info
        })
        if idx % 50 == 0:
            print(f"  ↳ {idx}/{len(chosen)} processed")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2)
    print(f"Saved {len(output)} entries to {OUTPUT_JSON}")

    # Print summary stats
    total_objs = sum(len(e["objects"]) for e in output)
    total_caps = sum(len(e["captions"]) for e in output)
    print("Summary:")
    print(f"  – Total objects: {total_objs}")
    print(f"  – Total captions: {total_caps}")
    print(f"  – Avg objects/image: {total_objs/len(output):.2f}")
    print(f"  – Avg captions/image: {total_caps/len(output):.2f}")

if __name__ == "__main__":
    main()
