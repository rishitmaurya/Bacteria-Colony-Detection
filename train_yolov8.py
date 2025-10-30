#!/usr/bin/env python3
"""
train_yolov8.py
Prepare YOLO-style dataset from your provided folder and fine-tune a YOLOv8 model.

Assumptions:
- Root dataset folder contains images (spXX_imgYY.jpg) in its root.
- YOLO-format .txt label files are in `annot_YOLO/` with matching basenames:
  e.g. sp01_img01.jpg  <->  annot_YOLO/sp01_img01.txt
- All labels correspond to a single class ("colony").

Usage:
python train_yolov8.py --dataset_dir /path/to/22022540 --output_dir ./yolo_runs --epochs 50
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import yaml

# Training uses ultralytics YOLOv8 API
# pip install ultralytics
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="Path to dataset root")
    p.add_argument("--output_dir", default="./yolo_runs", help="Where to save training outputs")
    p.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--pretrained", default="yolov8l.pt", help="Pretrained YOLOv8 weights (yolov8n.pt/yolov8s.pt etc.)")
    return p.parse_args()

def collect_image_label_pairs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    images = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    label_dir = dataset_dir / "annot_YOLO"
    pairs = []
    for img in images:
        lbl = label_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            # skip image if no matching label; you can change behavior if desired
            continue
    return pairs

def prepare_yolo_structure(pairs, out_base, val_split=0.2, seed=42):
    random.seed(seed)
    out_base = Path(out_base)
    # Remove existing prepared dataset if present
    prepared = out_base / "dataset_prepared"
    if prepared.exists():
        shutil.rmtree(prepared)
    # create folders
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (prepared / sub).mkdir(parents=True, exist_ok=True)

    # shuffle and split
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_split)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def copy_pairs(pairs_list, split):
        for img_path, lbl_path in pairs_list:
            dst_img = prepared / f"images/{split}" / img_path.name
            dst_lbl = prepared / f"labels/{split}" / (lbl_path.name)
            # copy files
            shutil.copy(img_path, dst_img)
            shutil.copy(lbl_path, dst_lbl)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    return prepared, len(train_pairs), len(val_pairs)

def write_data_yaml(prepared_dir, out_yaml_path):
    # single class: colony
    data = {
        "path": str(prepared_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 24,                 # previous 1 for single class 
        "names": ["colony"]*24       # all class names will be colony 
    }
    with open(out_yaml_path, "w") as f:
        yaml.dump(data, f)
    return out_yaml_path

def main():
    args = parse_args()
    pairs = collect_image_label_pairs(args.dataset_dir)
    if len(pairs) == 0:
        raise RuntimeError("No image-label pairs found. Check dataset_dir and annot_YOLO folder.")

    print(f"Found {len(pairs)} labeled images. Preparing dataset and splitting (val={args.val_split})...")

    prepared_dir, n_train, n_val = prepare_yolo_structure(pairs, args.output_dir, val_split=args.val_split, seed=args.seed)
    print(f"Prepared dataset at {prepared_dir}. Train: {n_train}, Val: {n_val}")

    data_yaml = Path(args.output_dir) / "data.yaml"
    write_data_yaml(prepared_dir, data_yaml)
    print(f"Wrote dataset YAML to {data_yaml}")

    # Train using Ultralytics YOLOv8
    print("Starting training with YOLOv8...")
    model = YOLO(args.pretrained)  # load pretrained weights
    # training; results will be saved under ./yolo_runs (Ultralytics creates runs automatically)
    model.train(data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=args.output_dir,
                name="yolov8_finetune_colony",
                exist_ok=True,
                workers = 16,
                device = 'cuda'
                )

    print("Training finished. Check the output directory for weights and logs.")

if __name__ == "__main__":
    main()
