# Bacterial Colony Detection and Counting using YOLOv8

This project detects and counts bacterial colonies in petri dish images using a fine-tuned YOLOv8 model.  
It supports both single-image and batch (folder) inference, visualizes detected colonies using bounding boxes or plus markers,  
and overlays the total colony count on each output image.

---

## Project Structure

```
bacteria_colony_count/
│
├── images/                     # Input images (e.g., sp01_img01.jpg)
├── annot_YOLO/                 # YOLO-format annotations (.txt)
├── annot_COCO.json             # COCO-format annotations
├── annot_tab.csv / .tsv        # Tabular annotation files
├── annot_VOC_XML.zip           # VOC-format annotations (archived)
│
├── yolov8_finetune_colony/     # YOLOv8 training folder (runs/train)
│   └── weights/
│       └── best.pt             # Best trained weights
│
├── infer_and_count.py          # Inference and counting script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Installation and Setup

### 1. Create and activate a virtual environment

```bash
## Python = 3.10.0
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies from requirements.txt

```bash
pip install -r requirements.txt
```
---

## Training the YOLOv8 Model

If you already have a trained `best.pt` file, you can skip this section.

1. Prepare your dataset in YOLO format:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```
2. Train the model:
   ```bash
   yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 name=yolov8_finetune_colony
   ```

3. The best weights will be saved at:
   ```
   runs/detect/yolov8_finetune_colony/weights/best.pt
   ```

---

## Inference and Counting

The script `infer_and_count.py` performs colony detection and overlays the colony count.

### Single Image Mode

```bash
python infer_and_count.py   --model ./yolo_runs/yolov8_finetune_colony/weights/best.pt   --image ./images/sp01_img01.jpg   --out ./output/sp01_img01_out.jpg   --mode plus
```

### Folder Mode (Batch Processing)

```bash
python infer_and_count.py   --model ./yolo_runs/yolov8_finetune_colony/weights/best.pt   --image_folder ./images/   --out_dir ./outputs/   --mode box
```

### Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--model` | Path to YOLOv8 weights (`.pt`) | Required |
| `--image` | Path to a single image | - |
| `--image_folder` | Folder containing images | - |
| `--out` | Output path for single image | `./output.jpg` |
| `--out_dir` | Output directory for batch mode | `./outputs` |
| `--conf` | Confidence threshold | `0.25` |
| `--iou` | IoU threshold for NMS | `0.45` |
| `--mode` | Drawing mode: `plus` (cross marker) or `box` | `plus` |

---

## Output Description

Each output image includes:
- A yellow '+' marker or green bounding box for each detected colony.
- A text overlay at the top-left corner showing the total detected count, for example:
  ```
  Count: 437
  ```

---

## Detection Cap Configuration

By default, YOLOv8 limits detections to 300 objects per image (`max_det=300`).  
This project overrides that limit to allow higher counts:

```python
results = model.predict(
    source=image_path,
    conf=conf_thresh,
    iou=iou,
    verbose=False,
    max_det=10000   # increased limit for dense colonies
)
```

If you have images with thousands of colonies, you can safely increase `max_det` to 20000 or higher.

---

## Troubleshooting

| Issue | Possible Fix |
|--------|---------------|
| Output always capped at 300 detections | Increase `max_det` as shown above |
| No output images | Check input and output paths |
| Model not found error | Ensure the path to `--model` is correct and file exists |
| Drawing mode unclear | Try `--mode box` if `plus` markers are not visible enough |
| CUDA not used | Ensure CUDA toolkit and compatible PyTorch GPU version are installed |

---

## GitHub Usage

### Clone the Repo

```bash
git clone https://github.com/your-username/bacteria-colony-detection.git
cd bacteria-colony-detection
```
---

## Example End-to-End Command

```bash
python infer_and_count.py   --model ./best.pt   --image_folder ./test_images/   --out_dir ./results/   --mode plus   --conf 0.3   --iou 0.45
```

This command:
1. Loads the trained YOLOv8 model.  
2. Runs inference on all images in the specified folder.  
3. Saves the annotated results with colony counts in the output directory.  
4. Prints the count for each image in the terminal.

```bash
python train_yolov8.py --dataset_dir ./path/to/dataset --output_dir ./yolo_runs --epochs 50 --batch 8 --imgsz 640
```
The command
1. Train a YOLOv8 model using the images and annotations found inside `./path/to/dataset`.
2. Split the dataset into training and validation sets, automatically organizing them into the required YOLO directory structure.
3. Run training for 50 epochs with a batch size of 8 and an input image size of 640×640 pixels.
4. Save all training outputs (weights, logs, results) into the folder `./yolo_runs`.

---

## Notes

- For large or dense colony images, prefer `--mode plus` and increase `max_det` to avoid missed detections.
- GPU acceleration is automatically used if CUDA is available.
- Use `--conf` and `--iou` to fine-tune detection sensitivity.

---


