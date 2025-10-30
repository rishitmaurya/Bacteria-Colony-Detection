#!/usr/bin/env python3
"""
infer_and_count.py
Load trained YOLOv8 model, run inference on an input image (or folder), draw detections
as bounding boxes or plus-sign markers, and print/save the total colony count.

Usage examples:
python infer_and_count.py --model ./yolo_runs/yolov8_finetune_colony/weights/best.pt --image /path/to/sp01_img01.jpg --out ./out.jpg --mode plus
python infer_and_count.py --model ./best.pt --image_folder /images_to_process/ --out_dir ./outputs --mode box
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import math
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained YOLOv8 weights (best.pt)")
    p.add_argument("--image", help="Path to single image")
    p.add_argument("--image_folder", help="Path to folder of images (optional)")
    p.add_argument("--out", help="Output path for single image", default="./output.jpg")
    p.add_argument("--out_dir", help="Output directory for folder mode", default="./outputs")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--mode", choices=["box", "plus"], default="plus", help="Drawing mode: 'box' or 'plus'")
    return p.parse_args()

def draw_plus(img, center, size=8, thickness=3):
    x, y = center
    # horizontal line
    cv2.line(img, (x - size, y), (x + size, y), (0, 255, 255), thickness)
    # vertical line
    cv2.line(img, (x, y - size), (x, y + size), (0, 255, 255), thickness)

def draw_box(img, xyxy, thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    # small center dot
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cv2.circle(img, (cx, cy), radius=2, color=(0, 255, 0), thickness=-1)

def process_single_image(model, image_path, out_path, conf_thresh=0.25, iou=0.25, mode="plus"):  # conf_thresh = 0.25 previous, iou = 0.45
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    # Run prediction
    results = model.predict(source=str(image_path), conf=conf_thresh, iou=iou, verbose=False)

    # Ultralytics returns results list; work with first (image) result
    res = results[0]

    # Boxes may be in res.boxes.xyxy, confidences in res.boxes.conf
    boxes = []
    try:
        xyxy = res.boxes.xyxy.cpu().numpy()  # shape (n,4)
        confs = res.boxes.conf.cpu().numpy()
        # classes may be res.boxes.cls
    except Exception:
        # Fallback: try .boxes.xyxy directly as ndarray
        try:
            xyxy = np.array(res.boxes.xyxy)
            confs = np.array(res.boxes.conf)
        except Exception:
            xyxy = np.zeros((0,4))
            confs = np.zeros((0,))

    # filter by conf (model.predict already used conf, but double-check)
    keep_idx = [i for i, c in enumerate(confs) if c >= conf_thresh]
    xyxy = xyxy[keep_idx] if len(keep_idx) > 0 else np.zeros((0,4))

    # draw detections
    for box in xyxy:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if mode == "plus":
            draw_plus(img, (cx, cy), size=max(6, int((x2-x1)/8)), thickness=5)
        else:
            draw_box(img, box, thickness=2)

    count = xyxy.shape[0]
    print(f"{image_path.name}: detected colonies = {count}")

    # Add text overlay with count
    overlay_text = f"Count: {count}"
    cv2.putText(img, overlay_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,0), 5, cv2.LINE_AA)

    # Ensure output dir exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

def main():
    args = parse_args()
    model = YOLO(args.model)

    if args.image:
        process_single_image(model, Path(args.image), Path(args.out), conf_thresh=args.conf, iou=args.iou, mode=args.mode)
    elif args.image_folder:
        input_dir = Path(args.image_folder)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not images:
            print("No images found in folder.")
            return
        for img_path in images:
            out_path = out_dir / img_path.name
            process_single_image(model, img_path, out_path, conf_thresh=args.conf, iou=args.iou, mode=args.mode)
    else:
        print("Specify --image or --image_folder")
        return

if __name__ == "__main__":
    main()
