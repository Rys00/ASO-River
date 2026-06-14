# How to run ASO-River

End-to-end pipeline: **U-Net water segmentation** → **R-CNN object detection** on the [LaRS](https://lars-dataset.github.io/) river scene dataset.

---

## Requirements

- **Python 3.11 or 3.12** (see `pyproject.toml`; 3.13+ is not supported)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **GPU (optional):** CUDA-capable NVIDIA GPU. The project installs PyTorch with CUDA 11.8 wheels on Linux/Windows. If your GPU is too old for the installed PyTorch build, the program falls back to CPU automatically.
- **Display:** OpenCV windows for `pipeline` and `test` modes (needs a graphical session or X forwarding).
- **LaRS dataset** for training and validation modes (see below).
- **Weights:** pre-trained segmentation and detection checkpoints (see [Model files](#model-files)).

---

## Setup

From the project root:

```bash
cd ASO-River
uv sync
```

Optional: create a `.env` file with your Weights & Biases API key if you plan to train and log metrics:

```bash
WANDB_API_KEY=your_key_here
```

Training works without W&B, but `train_detect` expects it to be configured.

---

## Dataset layout

Place the LaRS v1.0.0 dataset under:

```
data/lars_v1.0.0/
  images/
    train/images/
    val/images/
    test/images/
  annotations/
    train/
    val/
    test/
```

The first run loads images from disk, resizes them to the configured shape (default **1024×576**), and writes caches under `data/lars_v1.0.0/cached/`. Later runs reuse the cache.

---

## Model files

The full pipeline needs two checkpoints:


| Stage                    | Default path                            | Purpose                                 |
| ------------------------ | --------------------------------------- | --------------------------------------- |
| Segmentation (U-Net)     | `segmentation/models/best.pth`          | Water / sky / obstacle mask             |
| Detection (Faster R-CNN) | `detection/models/lars_faster_rcnn.pth` | Bounding boxes for LaRS “Thing” classes |


Set paths in `main.py`:

```python
model_name = "segmentation/models/best.pth"
DETECTOR = DetectorSettings(
    kind="faster_rcnn",
    backbone="resnet50",
    checkpoint="lars_faster_rcnn.pth",  # also checks detection/models/
)
```

Train the U-Net separately with `segmentation/unet_main.py` (see [README.md](README.md)).

---

## Main entry point: `main.py`

All pipeline modes are selected by editing the `mode` variable at the top of `main.py`, then running:

```bash
uv run main.py
```

### Modes


| `mode`           | Command flow                 | Description                                                                                                                                                                            |
| ---------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"pipeline"`     | `show_pipeline()`            | Run segmentation + detection on the **val** split. Shows an OpenCV viewer with water overlay and boxes. Prints per-image P/R/F1, box IoU, water IoU, and a dataset summary at the end. |
| `"test"`         | `show_test()`                | Run on images in `test_data/` (no LaRS split required). Writes COCO-format `test_data/detections.json`, then opens the viewer.                                                         |
| `"train_detect"` | `train_detector_model()`     | Fine-tune the detector on LaRS train/val. Saves `final_detector.<random>.pth` and logs to W&B.                                                                                         |
| `"tune_thresh"`  | `tune_detection_threshold()` | Sweep score thresholds on val (no viewer). Prints the threshold with best global F1; copy it into `detector_score_thresh`.                                                             |


Example — run the validation pipeline:

```python
mode = "pipeline"
```

```bash
uv run main.py
```

Example — export detections from custom images:

1. Put `.jpg` / `.png` files in `test_data/`.
2. Set `mode = "test"`.
3. Run `uv run main.py`.

### Viewer controls

- **Any key** (except below): next image  
- `**q` or Esc**: quit

---

## Fast R-CNN (alternative detector)

The default `main.py` uses **Faster R-CNN** (ResNet-50 FPN).

For **Fast R-CNN** with Selective Search proposals (ResNet-18):

```bash
uv run detection/main_with_fastrcnn.py
```

This script imports `main.py` but overrides `DETECTOR` to use `detection/models/detector-18v3.pth`. Use the same `mode` values as in `main.py`.

---

## Configuration reference

Edit constants at the top of `main.py`:

### Image / segmentation


| Variable     | Default                        | Notes                                    |
| ------------ | ------------------------------ | ---------------------------------------- |
| `shape`      | `(1024, 576)`                  | Input resolution (`div=2` → half size)   |
| `hsv`        | `False`                        | Use HSV instead of BGR                   |
| `model_name` | `segmentation/models/best.pth` | U-Net weights                            |
| `features`   | `(16, 16, 32, 32, 64, 128)`    | Must match the loaded U-Net architecture |


### Detection inference


| Variable                      | Default | Notes                                          |
| ----------------------------- | ------- | ---------------------------------------------- |
| `detector_score_thresh`       | `0.35`  | Minimum confidence for displayed detections    |
| `detector_roi_score_thresh`   | `0.05`  | RoI head internal threshold (before NMS)       |
| `detector_nms_thresh`         | `0.45`  | Non-maximum suppression IoU threshold          |
| `detector_detections_per_img` | `150`   | Max boxes per image after NMS                  |
| `detector_water_filter`       | `True`  | Drop boxes that do not overlap predicted water |
| `detector_water_dilate_px`    | `30`    | Dilate water mask before filtering (pixels)    |


Metrics (P/R/F1, mAP) use **raw** detector output. The viewer and COCO export use **water-filtered** boxes.

### Detection training (`train_detect`)


| Variable                           | Default | Notes                                              |
| ---------------------------------- | ------- | -------------------------------------------------- |
| `detector_epochs`                  | `20`    | Training epochs                                    |
| `detector_batch_size`              | `4`     | Batch size                                         |
| `detector_class_weighted_training` | `True`  | Inverse-frequency class weights + weighted sampler |


### Other


| Variable        | Default     | Notes                  |
| --------------- | ----------- | ---------------------- |
| `test_data_dir` | `test_data` | Folder for `test` mode |
| `pipeline_fps`  | `2`         | Viewer frame rate      |


---

## Tuning detection without retraining

1. Set `mode = "tune_thresh"` and run `uv run main.py`.
2. Note the best threshold printed at the end.
3. Set `detector_score_thresh` to that value.
4. Optionally adjust `detector_water_dilate_px`, `detector_nms_thresh`, or `detector_score_thresh` manually and re-run `pipeline` or `test`.

---

## Output files


| Path                            | When            | Content                     |
| ------------------------------- | --------------- | --------------------------- |
| `test_data/detections.json`     | `test` mode     | COCO-style detection export |
| `final_detector.<id>.pth`       | `train_detect`  | Trained detector weights    |
| `data/lars_v1.0.0/cached/*.npy` | First data load | Cached images and masks     |


---

## Segmentation-only workflow

To train or evaluate the U-Net without detection, use `segmentation/unet_main.py`:

```bash
# Train
uv run segmentation/unet_main.py --mode train --batch-size 16 --epochs 50

# View predictions
uv run segmentation/unet_main.py --mode show --model-name best.pth
```

See [README.md](README.md) for full U-Net and mean-shift options.

---

## Troubleshooting

`**FileNotFoundError: Checkpoint not found**`  
Place weights under `detection/models/` or set an absolute path in `DETECTOR.checkpoint` / `model_name`.

`**Folder not found: data/lars_v1.0.0**`  
Download and extract LaRS into `data/lars_v1.0.0/` (see [Dataset layout](#dataset-layout)).

**CUDA errors / wrong GPU arch**  
The program may print a warning and use CPU. Reinstall a PyTorch build that matches your GPU, or run on CPU (slower).

**OpenCV window does not appear (Linux/Wayland)**  
Try `export QT_QPA_PLATFORM=xcb` before running.

**Stale cache / wrong metrics**  
Delete `data/lars_v1.0.0/cached/` and rerun; caches are rebuilt with filename sidecars for correct GT alignment.

**Selective Search slow on first Fast R-CNN run**  
Proposals are computed once per split and cached next to the LaRS cache; subsequent runs are faster.