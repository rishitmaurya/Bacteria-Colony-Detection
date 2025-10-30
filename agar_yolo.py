#!/usr/bin/env python3
"""
prepare_and_train_AGAR.py
-------------------------
1. Collect all .jpg/.json pairs from AGAR_demo.
2. Convert JSON annotation files to YOLO format.
3. Create a YOLO-ready dataset folder: images + annot_YOLO.
4. Call train_yolov8.py to fine-tune a YOLOv8 model.

Usage:
python prepare_and_train_AGAR.py --agar_dir ./AGAR_demo --train_script ./train_yolov8.py
"""

import json
import os
import shutil
from pathlib import Path
import subprocess
import argparse
import sys
import cv2


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--agar_dir", required=True, help="Path to AGAR_demo root directory")
    p.add_argument("--train_script", required=True, help="Path to train_yolov8.py file")
    p.add_argument("--output_dir", default="./AGAR_YOLO_ready", help="Path for processed YOLO dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def convert_json_to_yolo(json_path, yolo_txt_path, img_width, img_height):
    """
    Convert AGAR JSON annotation to YOLO format.
    Expects keys: x, y, width, height under each object in 'labels'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "labels" not in data:
        print(f"  Skipped: {json_path} — unexpected JSON format.")
        return

    lines = []
    for obj in data["labels"]:
        if not all(k in obj for k in ("x", "y", "width", "height")):
            continue

        x_center = (obj["x"] + obj["width"] / 2) / img_width
        y_center = (obj["y"] + obj["height"] / 2) / img_height
        w_norm = obj["width"] / img_width
        h_norm = obj["height"] / img_height

        class_id = 0  # single class
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    if not lines:
        print(f"  Skipped: {json_path} — no valid objects.")
        return

    with open(yolo_txt_path, "w") as f:
        f.write("\n".join(lines))


def prepare_yolo_dataset(agar_dir, output_dir):
    """Convert all .json files and create YOLO-ready dataset structure."""
    agar_dir = Path(agar_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "annot_YOLO").mkdir(parents=True, exist_ok=True)

    image_paths = list(agar_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images...")

    for img_path in image_paths:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue

        dst_img = output_dir / img_path.name
        shutil.copy(img_path, dst_img)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipped: {img_path} — unreadable image.")
            continue

        h, w = img.shape[:2]
        dst_lbl = output_dir / "annot_YOLO" / (img_path.stem + ".txt")
        convert_json_to_yolo(json_path, dst_lbl, w, h)

    print(f"YOLO dataset prepared at: {output_dir}")
    return output_dir


def run_training(train_script, dataset_dir, epochs, batch, imgsz):
    """Execute YOLO training script as a subprocess."""
    python_exec = sys.executable
    cmd = [
        python_exec,
        str(train_script),
        "--dataset_dir", str(dataset_dir),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--imgsz", str(imgsz)
    ]
    print("Running training command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = parse_args()
    yolo_ready_dir = prepare_yolo_dataset(args.agar_dir, args.output_dir)
    run_training(args.train_script, yolo_ready_dir, args.epochs, args.batch, args.imgsz)
