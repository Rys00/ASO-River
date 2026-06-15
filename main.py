from lars import load_split
from segmentation.unet import UNet
import numpy as np
import torch
from detection.detection import (
    build_fast_rcnn,
    build_faster_rcnn,
    detect_objects,
    filter_detections_by_water,
    draw_detections,
    save_coco_detections_json,
    load_detector_state_dict,
    load_gt_boxes_for_filename,
    load_gt_semantic_for_filename,
    compute_per_image_detection_stats,
    compute_dataset_detection_metrics,
    water_segmentation_iou,
    format_image_metrics_line,
    format_dataset_metrics_summary,
    configure_detector_inference,
)
import cv2
import os
import argparse
from dotenv import load_dotenv
from typing import NamedTuple

# ==============================================================================
# Pipeline Config & Constants
# ==============================================================================


class Config:
    """Namespace for default pipeline constants and configuration parameters."""

    HSV = False
    DIV = 1
    FEATURES = (16, 16, 32, 32, 64, 128)
    BATCH_SIZE = 8

    # Inference tuning defaults
    DET_SCORE_THRESH = 0.35
    DET_ROI_SCORE_THRESH = 0.05
    DET_NMS_THRESH = 0.45
    DET_DETECTIONS_PER_IMG = 150
    DET_WATER_FILTER = True
    DET_WATER_DILATE_PX = 30

    # General configuration
    TEST_DATA_DIR = "test_data"
    PIPELINE_FPS = 2
    PIPELINE_WATER_COLOR = (224, 167, 41)  # BGR
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageInference(NamedTuple):
    """Container grouping all pipeline prediction outputs for evaluation or rendering."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    water_mask: torch.Tensor
    metric_boxes: torch.Tensor
    metric_labels: torch.Tensor
    metric_scores: torch.Tensor


class DetectorSettings(NamedTuple):
    """Detector engine initialization requirements configuration."""

    kind: str  # "faster_rcnn" or "fast_rcnn"
    backbone: str
    checkpoint: str
    max_proposals: int = 2000


# Default detector configuration (overrideable dynamically by external scripts)
DETECTOR = DetectorSettings(
    kind="faster_rcnn",
    backbone="resnet50",
    checkpoint="lars_faster_rcnn.pth",
)


# ==============================================================================
# Diagnostic & Initialization Helpers
# ==============================================================================


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
    """Build and parse command line settings for the pipeline executable."""
    parser = argparse.ArgumentParser(description="ASO River Segmentation & Detection Pipeline")
    parser.add_argument("--mode", choices=["pipeline", "test", "metrics"], default="pipeline", help="Operation mode")
    parser.add_argument("--hsv", action="store_true", default=Config.HSV, help="Use HSV color space")
    parser.add_argument("--div", type=int, default=Config.DIV, help="Image downscale factor")
    parser.add_argument("--features", type=int, nargs="+", default=list(Config.FEATURES), help="UNet features")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size for validation metrics calculation")

    parser.add_argument("--seg-model-path", type=str, default="segmentation/models/best.pth", help="UNet model checkpoint path")

    # Inference tuning
    parser.add_argument("--detector-score-thresh", type=float, default=Config.DET_SCORE_THRESH, help="Score threshold for detector filtering")
    parser.add_argument("--detector-roi-score-thresh", type=float, default=Config.DET_ROI_SCORE_THRESH, help="RoI score threshold")
    parser.add_argument("--detector-nms-thresh", type=float, default=Config.DET_NMS_THRESH, help="NMS threshold during post-processing")
    parser.add_argument("--detector-detections-per-img", type=int, default=Config.DET_DETECTIONS_PER_IMG, help="Maximum detections per image")
    parser.add_argument("--detector-water-filter", action="store_true", default=Config.DET_WATER_FILTER, help="Filter out non-water bounding boxes")
    parser.add_argument("--detector-no-water-filter", action="store_false", dest="detector_water_filter", help="Disable water mask filtering")
    parser.add_argument("--detector-water-dilate-px", type=int, default=Config.DET_WATER_DILATE_PX, help="Dilation size in px for predicted water mask")

    parser.add_argument("--test-data-dir", type=str, default=Config.TEST_DATA_DIR, help="Directory containing test images")
    parser.add_argument("--pipeline-fps", type=int, default=Config.PIPELINE_FPS, help="Frames per second for pipeline/test viewer")
    parser.add_argument("--pipeline-water-color", type=int, nargs=3, default=list(Config.PIPELINE_WATER_COLOR), help="Overlay color for water mask (BGR list)")

    # Detector specific settings
    parser.add_argument("--detector-kind", choices=["faster_rcnn", "fast_rcnn"], default=DETECTOR.kind, help="Detector neural net flavor")
    parser.add_argument("--detector-backbone", type=str, default=DETECTOR.backbone, help="Detector backbone network architecture name")
    parser.add_argument("--detector-checkpoint", type=str, default=DETECTOR.checkpoint, help="Filename/path for the saved detector checkpoint")
    parser.add_argument("--detector-max-proposals", type=int, default=DETECTOR.max_proposals, help="Maximum region proposal limit")

    parser.add_argument("--seed", type=int, default=48, help="Random seed for reproducibility")

    return parser.parse_args()


# ==============================================================================
# Model Loading & Data Logic
# ==============================================================================


def load_pipeline_models(args, device: torch.device) -> tuple[UNet, torch.nn.Module]:
    """Load structural UNet and chosen Bounding Box Detector models elegantly."""
    seg_model = UNet(in_channels=3, out_channels=3, features=tuple(args.features)).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_path, map_location=device))
    seg_model.eval()

    if args.detector_kind == "faster_rcnn":
        det_model = build_faster_rcnn(pretrained=False, backbone_name=args.detector_backbone).to(device)
        model_type = "Faster R-CNN"
    else:
        det_model = build_fast_rcnn(pretrained=False, backbone_name=args.detector_backbone).to(device)
        model_type = "Fast R-CNN"

    checkpoint = args.detector_checkpoint
    if not os.path.isfile(checkpoint):
        alt = os.path.join("detection", "models", os.path.basename(checkpoint))
        if os.path.isfile(alt):
            checkpoint = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (also tried {alt})")

    load_detector_state_dict(det_model, checkpoint, map_location=device)
    configure_detector_inference(
        det_model,
        nms_thresh=args.detector_nms_thresh,
        detections_per_img=args.detector_detections_per_img,
        roi_score_thresh=args.detector_roi_score_thresh,
    )
    det_model.eval()

    print(f"Loaded {model_type} ({args.detector_backbone}) from {checkpoint} " f"[score>={args.detector_score_thresh}, nms={args.detector_nms_thresh}, water_dilate={args.detector_water_dilate_px}px]")
    return seg_model, det_model


def load_images_from_folder(folder: str, reshape: tuple[int, int], hsv: bool = False) -> tuple[torch.Tensor, list[str]]:
    """Recursively fetch images from a directory, transforming format to BGR float tensors [0, 1]."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    images, names = [], []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() not in Config.IMAGE_EXTENSIONS:
            continue
        path = os.path.join(folder, name)
        raw = cv2.imread(path)
        if raw is None:
            print(f"Warning: could not read {path}, skipping")
            continue
        img = cv2.resize(raw, reshape)
        if hsv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images.append(torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.0))
        names.append(name)

    if not images:
        raise FileNotFoundError(f"No readable images found in {folder}")
    return torch.stack(images), names


# ==============================================================================
# Core Inference & Rendering Drivers
# ==============================================================================


def infer_on_image(image: torch.Tensor, seg_model: UNet, det_model: torch.nn.Module, device: torch.device, args) -> ImageInference:
    """Run full joint predictions, isolating display boxes (water-filtered) from evaluation targets."""
    xc = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = torch.argmax(seg_model(xc), dim=1).squeeze(0).cpu()
    water_mask = pred_mask == 1

    metric_boxes, metric_labels, metric_scores = detect_objects(det_model, image, device, score_thresh=args.detector_score_thresh, max_proposals=args.detector_max_proposals)
    boxes, labels, scores = metric_boxes, metric_labels, metric_scores
    if args.detector_water_filter:
        boxes, labels, scores = filter_detections_by_water(boxes, labels, scores, water_mask, dilate_px=args.detector_water_dilate_px)
    return ImageInference(boxes, labels, scores, water_mask, metric_boxes, metric_labels, metric_scores)


def render_pipeline_frame(image: torch.Tensor, inf: ImageInference, water_color: tuple[int, int, int]) -> np.ndarray:
    """Draw professional segmentation overlay and bounding boxes onto high-quality frame."""
    frame = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
    overlay = np.zeros_like(frame)
    overlay[inf.water_mask.numpy()] = water_color
    frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
    return draw_detections(frame, inf.boxes, inf.labels, inf.scores)


# ==============================================================================
# Pipeline Execution Modes
# ==============================================================================


def run_pipeline_viewer(
    images: torch.Tensor,
    args,
    names: list[str] | None = None,
    window_name: str = "Pipeline: segmentation + detection",
    models: tuple[UNet, torch.nn.Module] | None = None,
    precomputed: list[ImageInference] | None = None,
    ground_truth: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    gt_semantic: list[torch.Tensor | None] | None = None,
):
    """Run interactive graphical loop showcasing water filters overlay and predicted targets."""
    device = get_device()
    seg_model, det_model = models if models is not None else load_pipeline_models(args, device)
    n = len(images)
    names = names or [f"image {i + 1}/{n}" for i in range(n)]
    fps = args.pipeline_fps

    dataset_preds, dataset_gts = [], []
    box_ious, water_ious = [], []

    for i in range(n):
        inf = precomputed[i] if precomputed is not None else infer_on_image(images[i], seg_model, det_model, device, args)
        frame = render_pipeline_frame(images[i], inf, tuple(args.pipeline_water_color))
        cv2.imshow(window_name, frame)

        title = names[i]
        gt = ground_truth[i] if ground_truth is not None else None
        sem = gt_semantic[i] if gt_semantic is not None else None

        water_iou = water_segmentation_iou(inf.water_mask, sem) if sem is not None else None
        stats_str = ""
        box_iou = None

        if gt is not None:
            stats = compute_per_image_detection_stats(inf.boxes, inf.labels, inf.scores, gt[0], gt[1])
            box_iou = float(stats.get("box_iou")) if stats.get("box_iou") is not None else None
            stats_str = f" — {format_image_metrics_line(stats, water_iou)}"
        elif water_iou is not None:
            stats_str = f" — water IoU={water_iou:.3f}"

        print(f"[{i + 1}/{n}] {title}{stats_str} — press q/Esc to quit, any other key for next")

        if box_iou is not None:
            box_ious.append(box_iou)
        if water_iou is not None:
            water_ious.append(water_iou)

        if gt is not None:
            dataset_preds.append(
                {
                    "boxes": inf.boxes.detach().cpu(),
                    "labels": inf.labels.detach().cpu(),
                    "scores": inf.scores.detach().cpu(),
                }
            )
            dataset_gts.append({"boxes": gt[0].detach().cpu(), "labels": gt[1].detach().cpu()})

        key = cv2.waitKey(int(1000 / fps))
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    if dataset_preds or box_ious or water_ious:
        print(format_dataset_metrics_summary(compute_dataset_detection_metrics(dataset_preds, dataset_gts, box_ious, water_ious)))


def select_split_and_load_viewer(args, shape):
    """Segment and detect targets on selected validation dataset, launching GUI pipeline viewer."""
    x, y, _, names = load_split("val", hsv=args.hsv, reshape=shape, inc_semantic=True, return_filenames=True)
    gts = [load_gt_boxes_for_filename(name, shape, splits=("val",)) for name in names]
    run_pipeline_viewer(x, args, names=names, ground_truth=gts, gt_semantic=[y[i] for i in range(len(x))])


def show_test(args, shape):
    """Process high-resolution custom evaluation dataset, saving detections in COCO formatted structure."""
    images, names = load_images_from_folder(args.test_data_dir, reshape=shape, hsv=args.hsv)
    print(f"Loaded {len(images)} images from {args.test_data_dir} (shape {tuple(images.shape)})")

    device = get_device()
    models = load_pipeline_models(args, device)

    precomputed, ground_truth, gt_semantic = [], [], []
    for i, name in enumerate(names):
        inf = infer_on_image(images[i], *models, device, args)
        precomputed.append(inf)

        gt = load_gt_boxes_for_filename(name, shape)
        ground_truth.append(gt)

        sem = load_gt_semantic_for_filename(name, shape)
        gt_semantic.append(sem)

        water_iou = water_segmentation_iou(inf.water_mask, sem) if sem is not None else None
        stats_str = ""
        if gt is not None:
            stats = compute_per_image_detection_stats(inf.boxes, inf.labels, inf.scores, gt[0], gt[1])
            stats_str = f" — {format_image_metrics_line(stats, water_iou)}"
        elif water_iou is not None:
            stats_str = f" — water IoU={water_iou:.3f}"
        print(f"Inferred [{i + 1}/{len(images)}] {name}{stats_str}")

    detections_path = os.path.join(args.test_data_dir, "detections.json")
    save_coco_detections_json(detections_path, names, [inf.boxes for inf in precomputed], [inf.labels for inf in precomputed], [inf.scores for inf in precomputed], shape)
    print(f"Wrote {sum(len(inf.boxes) for inf in precomputed)} detection(s) to {detections_path}")

    run_pipeline_viewer(images, args, names=names, window_name="Test: segmentation + detection", models=models, precomputed=precomputed, ground_truth=ground_truth, gt_semantic=gt_semantic)


def calculate_metrics(args, shape):
    """Compute overall performance values of networks on validation dataset in batch configuration."""
    x, y, _, names = load_split("val", hsv=args.hsv, reshape=shape, inc_semantic=True, return_filenames=True)
    device = get_device()
    seg_model, det_model = load_pipeline_models(args, device)

    dataset_preds, dataset_gts = [], []
    box_ious, water_ious = [], []

    batch_size = args.batch_size
    print(f"Calculating mean bbox IoU, F1, and mean water IoU on {len(x)} val images in batches of {batch_size}...")
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)

        with torch.no_grad():
            pred_masks = torch.argmax(seg_model(xb), dim=1)
            pred_masks_for_iou = pred_masks.clone()
            pred_masks_for_iou[yb == 255] = 255

            intersection = torch.sum((pred_masks_for_iou == 1) & (yb == 1), dim=(1, 2)).float()
            union = torch.sum((pred_masks_for_iou == 1) | (yb == 1), dim=(1, 2)).float()
            iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
            water_ious.extend(iou.cpu().numpy().tolist())

        for j in range(len(xb)):
            idx = i + j
            metric_boxes, metric_labels, metric_scores = detect_objects(det_model, x[idx], device, score_thresh=args.detector_score_thresh, max_proposals=args.detector_max_proposals)

            # Apply water filtration to computed metrics if active
            if args.detector_water_filter:
                metric_boxes, metric_labels, metric_scores = filter_detections_by_water(metric_boxes, metric_labels, metric_scores, pred_masks[j] == 1, dilate_px=args.detector_water_dilate_px)

            gt = load_gt_boxes_for_filename(names[idx], shape, splits=("val",))
            if gt is not None:
                dataset_preds.append(
                    {
                        "boxes": metric_boxes.detach().cpu(),
                        "labels": metric_labels.detach().cpu(),
                        "scores": metric_scores.detach().cpu(),
                    }
                )
                dataset_gts.append({"boxes": gt[0].detach().cpu(), "labels": gt[1].detach().cpu()})

                stats = compute_per_image_detection_stats(metric_boxes, metric_labels, metric_scores, gt[0], gt[1])
                box_iou_val = stats.get("box_iou")
                if box_iou_val is not None:
                    box_ious.append(float(box_iou_val))

        processed = min(i + batch_size, len(x))
        print(f"Image {processed:03}/{len(x)} ({(processed/len(x))*100:.2f}%) - water IoU batch mean: {iou.mean():.4f}, global mean: {np.mean(water_ious):.4f}", end="\r")
    print()

    metrics = compute_dataset_detection_metrics(dataset_preds, dataset_gts, box_ious, water_ious)
    print("\n" + "=" * 50)
    print("                EVALUATION SUMMARY                ")
    print("=" * 50)
    print(format_dataset_metrics_summary(metrics))
    print("-" * 50)
    print(f"Mean Bounding Box IoU:       {metrics.get('mean_box_iou'):.4f}" if metrics.get("mean_box_iou") is not None else "Mean Bounding Box IoU:       n/a")
    print(f"Detection F1 Score (IoU=0.5): {metrics.get('f1'):.4f}" if metrics.get("f1") is not None else "Detection F1 Score:          n/a")
    print(f"Mean Water Segmentation IoU: {metrics.get('mean_water_iou'):.4f}" if metrics.get("mean_water_iou") is not None else "Mean Water Segmentation IoU: n/a")
    print("=" * 50)


# ==============================================================================
# Executable Entrypoint
# ==============================================================================


def main():
    load_dotenv(dotenv_path=".env", override=False)

    args = get_args()
    torch.manual_seed(args.seed)

    shape = (1024 // args.div, 576 // args.div)

    mode_map = {
        "pipeline": lambda: select_split_and_load_viewer(args, shape),
        "test": lambda: show_test(args, shape),
        "metrics": lambda: calculate_metrics(args, shape),
    }

    if args.mode in mode_map:
        mode_map[args.mode]()


if __name__ == "__main__":
    main()
