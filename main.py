from lars import load_split, cycle_images
from segmentation.unet import UNet
import numpy as np
import torch
from detection.detection import (
    build_fast_rcnn,
    build_faster_rcnn,
    prepare_detection_dataloaders,
    train_detector,
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
    compute_detection_class_weights,
    sweep_score_thresholds,
)
import cv2
import os
import wandb
from dotenv import load_dotenv
from typing import NamedTuple

# mode = "train"
# mode = "show"
# mode = "train_detect"
mode = "pipeline"
# mode = "test"
hsv = False
h_bilateral = None
s_bilateral = None
v_bilateral = None
div = 1
shape = (1024 // div, 576 // div)

# features = (32, 32, 64, 64, 128, 256)
features = (16, 16, 32, 32, 64, 128)

augment_train = True
weighted_loss = True
batch_size = 8

seed = 48
torch.manual_seed(seed)
preload_from = None
preload_from = "best.pth"
random_chars = "".join([chr(c) for c in torch.randint(ord("A"), ord("Z") + 1, (5,), dtype=torch.uint8).cpu().numpy().tolist()])
model_name = f"model{str(features).replace(' ', '')}{'-aug' if augment_train else ''}{'-hsv(' + str(h_bilateral) + ';' + str(s_bilateral) + ';' + str(v_bilateral) + ')' if hsv else ''}{'-wl' if weighted_loss else ''}.pth"
model_name = "segmentation/models/best.pth"  # Override for showing predictions from a specific model.

detector_batch_size = 4
detector_epochs = 20
detector_class_weighted_training = True

# Inference tuning (no retraining required)
detector_score_thresh = 0.35
detector_roi_score_thresh = 0.05
detector_nms_thresh = 0.45
detector_detections_per_img = 150
detector_water_filter = True
detector_water_dilate_px = 30
detector_low_score_thresh = 0.05  # for threshold sweep / raw metric collection

test_data_dir = "test_data"
pipeline_fps = 2
pipeline_water_color = (224, 167, 41)  # BGR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageInference(NamedTuple):
    """Pipeline outputs: filtered boxes for display, raw boxes for detection metrics."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    water_mask: torch.Tensor
    metric_boxes: torch.Tensor
    metric_labels: torch.Tensor
    metric_scores: torch.Tensor


class DetectorSettings(NamedTuple):
    """Detector backend for the segmentation + detection pipeline."""

    kind: str  # "faster_rcnn" (default) or "fast_rcnn"
    backbone: str
    checkpoint: str
    max_proposals: int = 2000


# Faster R-CNN is the default. Use `uv run detection/main_with_fastrcnn.py` for Fast R-CNN.
DETECTOR = DetectorSettings(
    kind="faster_rcnn",
    backbone="resnet50",
    checkpoint="lars_faster_rcnn.pth",
)


def get_device() -> torch.device:
    """Pick cuda only if this PyTorch build actually has kernels for the GPU.

    torch.cuda.is_available() returns True even for GPUs the build does not
    support (e.g. Pascal sm_61 with cu13x wheels), which then crashes on the
    first conv with "unable to find an engine". Fall back to CPU instead.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    archs = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    if archs and major * 10 + minor < min(archs):
        print(f"WARNING: {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not supported by " f"this PyTorch build (needs sm_{min(archs)}+); falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def show_sample_images():
    x, y, _ = load_split("train", hsv=hsv, reshape=shape, h_bilateral=h_bilateral, s_bilateral=s_bilateral, v_bilateral=v_bilateral)
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    # Real-time HSV smoothing via sliders is handled inside show_img/cycle_images.
    cycle_images(x, y, window_name="Samples", fps=5, highlight_water=None, hsv=hsv, hsv_sliders=True, show_hsv_channels=True)


def train_detector_model():
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataloaders = prepare_detection_dataloaders(
        batch_size=detector_batch_size,
        reshape=shape,
        max_proposals=DETECTOR.max_proposals,
        weighted_sampler=detector_class_weighted_training,
    )

    if DETECTOR.kind == "faster_rcnn":
        model = build_faster_rcnn(pretrained=True, backbone_name=DETECTOR.backbone).to(device)
        model_type = "Faster R-CNN"
    else:
        model = build_fast_rcnn(pretrained=True, backbone_name=DETECTOR.backbone, trainable_backbone_layers=3).to(device)
        model_type = "Fast R-CNN"

    class_weights = None
    if detector_class_weighted_training:
        class_weights = compute_detection_class_weights(dataloaders["train"].dataset)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=detector_epochs, eta_min=1e-6)
    print(f"Training {model_type} on {device} for {detector_epochs} epochs...")

    train_detector(
        wandb_project="ASO-Detection",
        wandb_run_name=DETECTOR.checkpoint.replace(".pth", ""),
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=detector_epochs,
        random_chars=random_chars,
        class_weights=class_weights,
    )
    torch.save(model.state_dict(), f"final_detector.{random_chars}.pth")


def resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    alt = os.path.join("detection", "models", os.path.basename(path))
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(f"Checkpoint not found: {path} (also tried {alt})")


def load_pipeline_models(device: torch.device) -> tuple[UNet, torch.nn.Module]:
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)

    seg_model = UNet(in_channels=num_channels, out_channels=num_classes, features=features).to(device)
    seg_model.load_state_dict(torch.load(model_name, map_location=device))
    seg_model.eval()

    if DETECTOR.kind == "faster_rcnn":
        det_model = build_faster_rcnn(pretrained=False, backbone_name=DETECTOR.backbone).to(device)
        model_type = "Faster R-CNN"
    else:
        det_model = build_fast_rcnn(pretrained=False, backbone_name=DETECTOR.backbone).to(device)
        model_type = "Fast R-CNN"

    checkpoint = resolve_checkpoint(DETECTOR.checkpoint)
    load_detector_state_dict(det_model, checkpoint, map_location=device)
    configure_detector_inference(
        det_model,
        nms_thresh=detector_nms_thresh,
        detections_per_img=detector_detections_per_img,
        roi_score_thresh=detector_roi_score_thresh,
    )
    det_model.eval()
    print(
        f"Loaded {model_type} ({DETECTOR.backbone}) from {checkpoint} "
        f"[score>={detector_score_thresh}, nms={detector_nms_thresh}, water_dilate={detector_water_dilate_px}px]"
    )
    return seg_model, det_model


def load_images_from_folder(
    folder: str,
    reshape: tuple[int, int] = shape,
    hsv: bool = hsv,
) -> tuple[torch.Tensor, list[str]]:
    """Load images from a folder as BGR float tensors [N, 3, H, W] in [0, 1]."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    images: list[torch.Tensor] = []
    names: list[str] = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
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


def infer_on_image(
    image: torch.Tensor,
    seg_model: UNet,
    det_model: torch.nn.Module,
    device: torch.device,
) -> ImageInference:
    """Run segmentation + detection; return display boxes (water-filtered) and raw metric boxes."""
    xc = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = torch.argmax(seg_model(xc), dim=1).squeeze(0).cpu()
    water_mask = pred_mask == 1

    metric_boxes, metric_labels, metric_scores = detect_objects(
        det_model,
        image,
        device,
        score_thresh=detector_score_thresh,
        max_proposals=DETECTOR.max_proposals,
    )
    boxes, labels, scores = metric_boxes, metric_labels, metric_scores
    if detector_water_filter:
        boxes, labels, scores = filter_detections_by_water(
            boxes, labels, scores, water_mask, dilate_px=detector_water_dilate_px
        )
    return ImageInference(
        boxes, labels, scores, water_mask,
        metric_boxes, metric_labels, metric_scores,
    )


def render_pipeline_frame(
    image: torch.Tensor,
    seg_model: UNet,
    det_model: torch.nn.Module,
    device: torch.device,
    water_color: tuple[int, int, int] = pipeline_water_color,
    water_mask: torch.Tensor | None = None,
    boxes: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> np.ndarray:
    """Segment + detect on one (C, H, W) BGR image; return annotated BGR uint8 frame."""
    if water_mask is None or boxes is None or labels is None or scores is None:
        inference = infer_on_image(image, seg_model, det_model, device)
        boxes, labels, scores, water_mask = (
            inference.boxes, inference.labels, inference.scores, inference.water_mask
        )

    frame = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
    overlay = np.zeros_like(frame)
    overlay[water_mask.numpy()] = water_color
    frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
    return draw_detections(frame, boxes, labels, scores)


def _metrics_line_for_image(
    inference: ImageInference,
    ground_truth: tuple[torch.Tensor, torch.Tensor] | None,
    gt_semantic: torch.Tensor | None,
) -> tuple[str, float | None, float | None]:
    """Return (terminal suffix, per-image box IoU, per-image water IoU) for aggregation."""
    water_iou = water_segmentation_iou(inference.water_mask, gt_semantic) if gt_semantic is not None else None
    if ground_truth is None:
        if water_iou is None:
            return "", None, None
        return f" — water IoU={water_iou:.3f}", None, water_iou
    gt_boxes, gt_labels = ground_truth
    stats = compute_per_image_detection_stats(
        inference.metric_boxes,
        inference.metric_labels,
        inference.metric_scores,
        gt_boxes,
        gt_labels,
    )
    box_iou_val = stats.get("box_iou")
    box_iou_out = float(box_iou_val) if isinstance(box_iou_val, float) else None
    return " — " + format_image_metrics_line(stats, water_iou), box_iou_out, water_iou


def run_pipeline_viewer(
    images: torch.Tensor,
    names: list[str] | None = None,
    window_name: str = "Pipeline: segmentation + detection",
    fps: int = pipeline_fps,
    models: tuple[UNet, torch.nn.Module] | None = None,
    precomputed: list[ImageInference] | None = None,
    ground_truth: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    gt_semantic: list[torch.Tensor | None] | None = None,
):
    device = get_device()
    if models is None:
        seg_model, det_model = load_pipeline_models(device)
    else:
        seg_model, det_model = models
    n = len(images)
    if names is None:
        names = [f"image {i + 1}/{n}" for i in range(n)]

    dataset_preds: list[dict[str, torch.Tensor]] = []
    dataset_gts: list[dict[str, torch.Tensor]] = []
    box_ious: list[float] = []
    water_ious: list[float] = []

    for i in range(n):
        if precomputed is not None:
            inference = precomputed[i]
        else:
            inference = infer_on_image(images[i], seg_model, det_model, device)
        frame = render_pipeline_frame(
            images[i], seg_model, det_model, device,
            water_mask=inference.water_mask,
            boxes=inference.boxes,
            labels=inference.labels,
            scores=inference.scores,
        )
        cv2.imshow(window_name, frame)
        title = names[i]
        gt = ground_truth[i] if ground_truth is not None else None
        sem = gt_semantic[i] if gt_semantic is not None else None
        metrics, box_iou, water_iou = _metrics_line_for_image(inference, gt, sem)
        print(f"[{i + 1}/{n}] {title}{metrics} — press q/Esc to quit, any other key for next")
        if box_iou is not None:
            box_ious.append(box_iou)
        if water_iou is not None:
            water_ious.append(water_iou)
        if gt is not None:
            gt_boxes, gt_labels = gt
            dataset_preds.append(
                {
                    "boxes": inference.metric_boxes.detach().cpu(),
                    "labels": inference.metric_labels.detach().cpu(),
                    "scores": inference.metric_scores.detach().cpu(),
                }
            )
            dataset_gts.append(
                {"boxes": gt_boxes.detach().cpu(), "labels": gt_labels.detach().cpu()}
            )
        key = cv2.waitKey(int(1000 / fps))
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()

    if dataset_preds or box_ious or water_ious:
        print(
            format_dataset_metrics_summary(
                compute_dataset_detection_metrics(dataset_preds, dataset_gts, box_ious, water_ious)
            )
        )


def tune_detection_threshold():
    """Sweep score thresholds on val split (no viewer) and print best F1."""
    x, _, _, names = load_split("val", hsv=hsv, reshape=shape, inc_semantic=False, return_filenames=True)
    device = get_device()
    seg_model, det_model = load_pipeline_models(device)

    raw_preds: list[dict[str, torch.Tensor]] = []
    gts: list[dict[str, torch.Tensor]] = []
    print(f"Running detection on {len(x)} val images at score>={detector_low_score_thresh}...")
    for i in range(len(x)):
        boxes, labels, scores = detect_objects(
            det_model,
            x[i],
            device,
            score_thresh=detector_low_score_thresh,
            max_proposals=DETECTOR.max_proposals,
        )
        gt = load_gt_boxes_for_filename(names[i], shape, splits=("val",))
        if gt is None:
            continue
        gt_boxes, gt_labels = gt
        raw_preds.append({"boxes": boxes, "labels": labels, "scores": scores})
        gts.append({"boxes": gt_boxes, "labels": gt_labels})
        print(f"  [{i + 1}/{len(x)}] {names[i]}", end="\r")
    print()

    best_thresh, _ = sweep_score_thresholds(raw_preds, gts)
    print(f"Set detector_score_thresh = {best_thresh:.2f} in main.py")


def show_pipeline():
    """Full pipeline from etap1.md 4.1: image -> U-Net segmentation -> detection."""
    x, y, _, names = load_split("val", hsv=hsv, reshape=shape, inc_semantic=True, return_filenames=True)
    print(f"Images shape: {x.shape}")
    ground_truth: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for name in names:
        gt = load_gt_boxes_for_filename(name, shape, splits=("val",))
        ground_truth.append(gt)
    run_pipeline_viewer(
        x,
        names=names,
        ground_truth=ground_truth,
        gt_semantic=[y[i] for i in range(len(x))],
    )


def show_test():
    """Run segmentation + detection on images in test_data_dir."""
    images, names = load_images_from_folder(test_data_dir, reshape=shape, hsv=hsv)
    print(f"Loaded {len(images)} images from {test_data_dir} (shape {tuple(images.shape)})")

    device = get_device()
    seg_model, det_model = load_pipeline_models(device)

    all_boxes: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    precomputed: list[ImageInference] = []
    ground_truth: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    gt_semantic: list[torch.Tensor | None] = []
    for i in range(len(images)):
        inference = infer_on_image(images[i], seg_model, det_model, device)
        all_boxes.append(inference.boxes)
        all_labels.append(inference.labels)
        all_scores.append(inference.scores)
        precomputed.append(inference)
        gt = load_gt_boxes_for_filename(names[i], shape)
        ground_truth.append(gt)
        gt_semantic.append(load_gt_semantic_for_filename(names[i], shape))
        metrics, _, _ = _metrics_line_for_image(inference, gt, gt_semantic[-1])
        print(f"Inferred [{i + 1}/{len(images)}] {names[i]}{metrics}")

    detections_path = os.path.join(test_data_dir, "detections.json")
    save_coco_detections_json(detections_path, names, all_boxes, all_labels, all_scores, shape)
    total = sum(len(b) for b in all_boxes)
    print(f"Wrote {total} detection(s) to {detections_path}")

    run_pipeline_viewer(
        images,
        names=names,
        window_name="Test: segmentation + detection",
        models=(seg_model, det_model),
        precomputed=precomputed,
        ground_truth=ground_truth,
        gt_semantic=gt_semantic,
    )


def main():
    load_dotenv(dotenv_path=".env", override=False)
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    # show_sample_images()
    # evaluate()
    if mode == "train_detect":
        train_detector_model()
    elif mode == "pipeline":
        show_pipeline()
    elif mode == "test":
        show_test()
    elif mode == "tune_thresh":
        tune_detection_threshold()


if __name__ == "__main__":
    main()
