import argparse
import os
import torch
import cv2
import wandb
from dotenv import load_dotenv
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path to allow absolute imports from scripts in subfolders
project_root = str(Path(__file__).resolve().parent.parent)
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lars import load_split
from detection.detection import (
    build_fast_rcnn,
    build_faster_rcnn,
    prepare_detection_dataloaders,
    train_detector,
    detect_objects,
    draw_detections,
    load_detector_state_dict,
    load_gt_boxes_for_filename,
    configure_detector_inference,
    compute_detection_class_weights,
    sweep_score_thresholds,
)

# Fix for Qt and Wayland on Linux systems.
os.environ["QT_QPA_PLATFORM"] = "xcb"


def get_device() -> torch.device:
    """Select appropriate device, preventing crashes from unsupported GPUs."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    archs = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    if archs and (major * 10 + minor < min(archs)):
        print(f"WARNING: {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not supported " f"by this PyTorch build (needs sm_{min(archs)}+); falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def get_args():
    """Build and parse command line settings for the detection tool."""
    parser = argparse.ArgumentParser(description="ASO River Object Detection Tool")
    parser.add_argument("--mode", choices=["train", "show", "sample", "tune_thresh"], default="show", help="Operation mode")
    parser.add_argument("--detector-kind", choices=["faster_rcnn", "fast_rcnn"], default="faster_rcnn", help="Detector kind")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone network architecture")
    parser.add_argument("--checkpoint", type=str, default="lars_faster_rcnn.pth", help="Checkpoint filename/path")
    parser.add_argument("--max-proposals", type=int, default=2000, help="Maximum region proposals limit")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--class-weighted-training", action="store_true", default=True, help="Use class weighted sampler")
    parser.add_argument("--no-class-weighted-training", action="store_false", dest="class_weighted_training", help="Disable class weighted sampler")
    parser.add_argument("--score-thresh", type=float, default=0.35, help="Score threshold for filtering detections")
    parser.add_argument("--roi-score-thresh", type=float, default=0.05, help="RoI score threshold")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="NMS score threshold")
    parser.add_argument("--detections-per-img", type=int, default=150, help="Maximum detections per image")
    parser.add_argument("--low-score-thresh", type=float, default=0.05, help="Lower score threshold for tuning sweeps")
    parser.add_argument("--div", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--seed", type=int, default=48, help="Random seed for reproducibility")
    return parser.parse_args()


def load_detector_model(args, device) -> torch.nn.Module:
    """Initialize detector configuration and load state dict checkpoint cleanly."""
    if args.detector_kind == "faster_rcnn":
        model = build_faster_rcnn(pretrained=False, backbone_name=args.backbone).to(device)
    else:
        model = build_fast_rcnn(pretrained=False, backbone_name=args.backbone).to(device)

    checkpoint = args.checkpoint
    if not os.path.isfile(checkpoint):
        alt = os.path.join("detection", "models", os.path.basename(checkpoint))
        if os.path.isfile(alt):
            checkpoint = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (also tried {alt})")

    load_detector_state_dict(model, checkpoint, map_location=device)
    configure_detector_inference(
        model,
        nms_thresh=args.nms_thresh,
        detections_per_img=args.detections_per_img,
        roi_score_thresh=args.roi_score_thresh,
    )
    model.eval()
    return model


def train_detector_model(args, shape, random_chars):
    """Load train features dataset and fine-tune detector architectures on W&B."""
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataloaders = prepare_detection_dataloaders(
        batch_size=args.batch_size,
        reshape=shape,
        max_proposals=args.max_proposals,
        weighted_sampler=args.class_weighted_training,
    )

    if args.detector_kind == "faster_rcnn":
        model = build_faster_rcnn(pretrained=True, backbone_name=args.backbone).to(device)
        model_type = "Faster R-CNN"
    else:
        model = build_fast_rcnn(pretrained=True, backbone_name=args.backbone, trainable_backbone_layers=3).to(device)
        model_type = "Fast R-CNN"

    class_weights = compute_detection_class_weights(dataloaders["train"].dataset) if args.class_weighted_training else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"Training {model_type} on {device} for {args.epochs} epochs...")
    train_detector(
        wandb_project="ASO-Detection",
        wandb_run_name=args.checkpoint.replace(".pth", ""),
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        random_chars=random_chars,
        class_weights=class_weights,
    )
    torch.save(model.state_dict(), f"final_detector.{random_chars}.pth")


def show_predictions(args, shape):
    """Show bboxes predicted by model over validation split images in CV2 loop."""
    x, _, _, names = load_split("val", hsv=False, reshape=shape, inc_semantic=False, return_filenames=True)
    device = get_device()
    model = load_detector_model(args, device)

    fps = 2
    for i in range(len(x)):
        boxes, labels, scores = detect_objects(model, x[i], device, score_thresh=args.score_thresh, max_proposals=args.max_proposals)

        frame = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
        frame = draw_detections(frame, boxes, labels, scores)

        cv2.imshow("Detector Predictions Only", frame)
        print(f"[{i + 1}/{len(x)}] {names[i]} - detected {len(boxes)} object(s). Press q/Esc to quit.")

        key = cv2.waitKey(int(1000 / fps))
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def show_sample_images(args, shape):
    """Show ground truth bboxes over train split images in CV2 loop."""
    x, _, _, names = load_split("train", hsv=False, reshape=shape, inc_semantic=False, return_filenames=True)

    fps = 2
    for i in range(len(x)):
        gt = load_gt_boxes_for_filename(names[i], shape, splits=("train",))
        frame = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()

        if gt is not None:
            gt_boxes, gt_labels = gt
            gt_scores = torch.ones(len(gt_boxes))
            frame = draw_detections(frame, gt_boxes, gt_labels, gt_scores)

        cv2.imshow("Ground Truth Bboxes", frame)
        print(f"[{i + 1}/{len(x)}] {names[i]} - GT labels. Press q/Esc to quit.")

        key = cv2.waitKey(int(1000 / fps))
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def tune_detection_threshold(args, shape):
    """Grid sweep candidate confidence settings on the evaluation subset validation splits."""
    x, _, _, names = load_split("val", hsv=False, reshape=shape, inc_semantic=False, return_filenames=True)
    device = get_device()
    model = load_detector_model(args, device)

    raw_preds, gts = [], []
    print(f"Running detection on {len(x)} val images at score>={args.low_score_thresh}...")
    for i, name in enumerate(names):
        boxes, labels, scores = detect_objects(model, x[i], device, score_thresh=args.low_score_thresh, max_proposals=args.max_proposals)
        gt = load_gt_boxes_for_filename(name, shape, splits=("val",))
        if gt is None:
            continue
        raw_preds.append({"boxes": boxes, "labels": labels, "scores": scores})
        gts.append({"boxes": gt[0], "labels": gt[1]})
        print(f"  [{i + 1}/{len(x)}] {name}", end="\r")
    print()

    best_thresh, _ = sweep_score_thresholds(raw_preds, gts)
    print(f"Set detector_score_thresh = {best_thresh:.2f} in main.py")


def main():
    load_dotenv(dotenv_path=".env", override=False)
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    args = get_args()
    torch.manual_seed(args.seed)

    shape = (1024 // args.div, 576 // args.div)
    random_chars = "".join([chr(c) for c in torch.randint(ord("A"), ord("Z") + 1, (5,), dtype=torch.uint8).cpu().numpy().tolist()])

    mode_map = {
        "train": lambda: train_detector_model(args, shape, random_chars),
        "show": lambda: show_predictions(args, shape),
        "sample": lambda: show_sample_images(args, shape),
        "tune_thresh": lambda: tune_detection_threshold(args, shape),
    }

    if args.mode in mode_map:
        mode_map[args.mode]()


if __name__ == "__main__":
    main()
