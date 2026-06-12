# ASO 26L - Dokumentacja projektowa

### Zespół:

- Zofia Czyżewska
- Mateusz Ogniewski
- Mateusz Wawrzyniak

### Etapy:

- [Etap 1](/etap1.md)

![VII](./vii.png)

## Usage

### UNet Segmentation (`unet_main.py`)

This is the main script for training and evaluating the UNet model.

- **Train a model from scratch**:
  ```bash
  uv run segmentation/unet_main.py --mode train --batch-size 16 --epochs 50
  ```
- **Fine-tune a model**:
  ```bash
  uv run segmentation/unet_main.py --mode train --preload best.pth --epochs 20 --lr 1e-5
  ```
- **Show model predictions**:
  ```bash
  uv run segmentation/unet_main.py --mode show --model-name best.pth
  ```
- **Calculate Mean IoU**:
  ```bash
  uv run segmentation/unet_main.py --mode mean_iou --model-name best.pth
  ```
- **Browse dataset samples**:
  ```bash
  uv run segmentation/unet_main.py --mode sample
  ```

**Key Arguments:**

- `--mode`: `train`, `show` (default), `sample`, `mean_iou`
- `--model-name`: Filename of the model in `segmentation/models/` that should be used for evaluation
- `--preload`: Filename of the model in `segmentation/models/` that should be used to preload the weights of the trained model
- `--hsv`: Use HSV color space instead of RGB
- `--features`: List of feature channels for the UNet architecture (e.g., `16 32 64 128`) for training

### Mean-Shift Segmentation (`means_shift_main.py`)

Traditional computer vision approach using Mean-Shift and K-Means.

- **Run evaluation and visualization**:
  ```bash
  uv run segmentation/means_shift_main.py --mode show
  ```
- **Calculate Mean IoU**:
  ```bash
  uv run segmentation/means_shift_main.py --mode mean_iou
  ```
- **Run hyperparameter optimization**:
  ```bash
  uv run segmentation/means_shift_main.py --mode optimize
  ```

**Key Arguments:**

- `--mode`: `show` (default), `mean_iou`, `optimize`
- `--sp`: Spatial window radius (default: 32)
- `--sr`: Color window radius (default: 128)
- `--dim`: Processing dimension (default: 256)
