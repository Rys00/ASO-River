import argparse
import os
import torch
import cv2
import wandb
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports from scripts in subfolders
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lars import load_split, cycle_images, show_img
from segmentation.unet import UNet
from segmentation.unet_segmentation import KorniaSegmentationAugment, prepare_dataloaders, train
import numpy as np

# Fix for Qt and Wayland on Linux systems.
os.environ["QT_QPA_PLATFORM"] = "xcb"


def detect_features(model_path):
    """Detects UNet feature configuration from a saved state_dict."""
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Cannot detect features or load model.")
        return None
    state_dict = torch.load(model_path, map_location="cpu")
    features = []
    i = 0
    while True:
        # Each DoubleConv in downs has net.0 as its first Conv2d
        key = f"downs.{i}.net.0.weight"
        if key in state_dict:
            features.append(state_dict[key].shape[0])
            i += 1
        else:
            break
    detected = tuple(features)
    if detected:
        print(f"Detected model features from .pth: {detected}")
    return detected


def load_model(model_path, device, num_channels, num_classes, default_features, **kwargs):
    """Detects features, initializes UNet, and loads state_dict if path exists."""
    detected = None
    if model_path:
        detected = detect_features(model_path)
    else:
        print("No model path provided, using default features: ", default_features)

    features = detected if detected else tuple(default_features)
    model = UNet(in_channels=num_channels, out_channels=num_classes, features=features, **kwargs).to(device)

    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model, features


def get_args():
    parser = argparse.ArgumentParser(description="ASO River Segmentation")
    parser.add_argument("--mode", choices=["train", "show", "sample", "mean_iou"], default="show", help="Operation mode")
    parser.add_argument("--models-dir", type=str, default="segmentation/models", help="Global folder for models")
    parser.add_argument("--model-name", type=str, default="best.pth", help="Model filename (relative to models-dir)")
    parser.add_argument("--hsv", action="store_true", help="Use HSV color space")
    parser.add_argument("--h-bilateral", type=int, default=None)
    parser.add_argument("--s-bilateral", type=int, default=None)
    parser.add_argument("--v-bilateral", type=int, default=None)
    parser.add_argument("--div", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--features", type=int, nargs="+", default=[16, 16, 32, 32, 64, 128])
    parser.add_argument("--augment", action="store_true", default=True, help="Enable training augmentation")
    parser.add_argument("--weighted-loss", action="store_true", default=True, help="Use class-weighted loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--preload", type=str, default=None, help="Preload model from this filename (relative to models-dir)")
    return parser.parse_args()


def show_sample_images(args, shape):
    x, y, _ = load_split("train", hsv=args.hsv, reshape=shape, h_bilateral=args.h_bilateral, s_bilateral=args.s_bilateral, v_bilateral=args.v_bilateral)
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    cycle_images(x, y, window_name="Samples", fps=5, highlight_water=None, hsv=args.hsv, hsv_sliders=True, show_hsv_channels=True)


def train_unet(args, shape, random_chars):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky

    gpu_augment = None
    if args.augment and device.type == "cuda":
        gpu_augment = KorniaSegmentationAugment(
            hflip_p=0.5,
            jitter_p=0.8,
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
            noise_p=0.2,
            noise_std=0.01,
            input_is_bgr=not args.hsv,
            input_is_hsv=args.hsv,
        ).to(device)

    dataloaders = prepare_dataloaders(batch_size=args.batch_size, reshape=shape, hsv=args.hsv, h_bilateral=args.h_bilateral, s_bilateral=args.s_bilateral, v_bilateral=args.v_bilateral)

    lr = args.lr
    preload_path = os.path.join(args.models_dir, args.preload) if args.preload else None

    model, features = load_model(
        model_path=preload_path, device=device, num_channels=num_channels, num_classes=num_classes, default_features=args.features, dropout_p=0.1, dropout_min_features=64, dropout_bottleneck_p=0.2
    )

    if args.preload:
        lr = 1e-5  # Use a lower learning rate when fine-tuning from a pretrained model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if args.weighted_loss:
        train_masks = dataloaders["train"].dataset.tensors[1]
        valid = train_masks != 255
        class_counts = torch.bincount(train_masks[valid].reshape(-1), minlength=num_classes).float()
        class_weights = class_counts.sum() / (num_classes * class_counts.clamp_min(1.0))
        class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
        class_weights = class_weights.clamp(0.1, 10.0)
        print(f"Class counts: {class_counts.tolist()} | CE weights: {class_weights.tolist()}")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    model_save_base = f"model{str(features).replace(' ', '')}{'-aug' if args.augment else ''}{'-hsv' if args.hsv else ''}{'-wl' if args.weighted_loss else ''}"

    print(f"Training on {device} for {args.epochs} epochs...")

    train(
        wandb_project="ASO-Segmentation",
        wandb_run_name=model_save_base,
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        num_classes=num_classes,
        ignore_index=255,
        gpu_augment=gpu_augment,
        random_chars=random_chars,
        save_dir=args.models_dir,
    )

    os.makedirs(args.models_dir, exist_ok=True)
    final_path = os.path.join(args.models_dir, f"final_model.{random_chars}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")


def show_predictions(args, shape):
    x, y, _ = load_split("val", hsv=args.hsv, reshape=shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3
    num_classes = 3

    model_path = os.path.join(args.models_dir, args.model_name)
    model, _ = load_model(model_path=model_path, device=device, num_channels=num_channels, num_classes=num_classes, default_features=args.features)

    model.eval()

    fps = 2
    for i in range(len(x)):
        xc = x[i : i + 1].to(device)
        with torch.no_grad():
            output = model(xc)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            show_img(x[i], pred_mask, wrong=((pred_mask == 1) != (y[i] == 1)) & (y[i] != 255), hsv=args.hsv)

        key = cv2.waitKey(int(1000 / fps))
        if key == 27:
            break
    cv2.destroyAllWindows()


def mean_iou(args, shape):
    x, y, _ = load_split("val", hsv=args.hsv, reshape=shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3
    num_classes = 3

    model_path = os.path.join(args.models_dir, args.model_name)
    model, _ = load_model(model_path=model_path, device=device, num_channels=num_channels, num_classes=num_classes, default_features=args.features)

    model.eval()

    ious = []
    batch_size = args.batch_size
    for i in range(0, len(x), batch_size):
        xc = x[i : i + batch_size].to(device)
        with torch.no_grad():
            output = model(xc)
            pred_mask = torch.argmax(output, dim=1)
            true_mask = y[i : i + batch_size].to(device)
            pred_mask[true_mask == 255] = 255  # Ensure ignore_index is respected in IoU calculation

            # Intersection and Union for class 1 (Water)
            intersection = torch.sum((pred_mask == 1) & (true_mask == 1), dim=(1, 2)).float()
            union = torch.sum((pred_mask == 1) | (true_mask == 1), dim=(1, 2)).float()

            # Handle cases with no water (IoU = 1.0 if both empty, else intersection/union)
            iou = torch.where(union > 0, intersection / union, torch.ones_like(union))

            ious.extend(iou.cpu().numpy())
            processed = min(i + batch_size, len(x))
            print(f"Image {processed:03}/{len(x)} ({(processed/len(x))*100:.2f}%) - IoU: {iou.mean():.4f}, mean: {np.mean(ious):.4f}", end="\r")

    mean_iou_val = np.mean(ious)
    print(f"Mean IoU across validation set: {mean_iou_val:.4f}                           ")
    return mean_iou_val


def main():
    load_dotenv(dotenv_path=".env", override=False)
    args = get_args()

    shape = (1024 // args.div, 576 // args.div)
    random_chars = "".join([chr(c) for c in torch.randint(ord("A"), ord("Z") + 1, (5,), dtype=torch.uint8).cpu().numpy().tolist()])

    if args.mode == "train":
        train_unet(args, shape, random_chars)
    elif args.mode == "show":
        show_predictions(args, shape)
    elif args.mode == "sample":
        show_sample_images(args, shape)
    elif args.mode == "mean_iou":
        mean_iou(args, shape)


if __name__ == "__main__":
    main()
