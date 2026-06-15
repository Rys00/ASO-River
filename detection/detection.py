"""Object detection stage of the pipeline (etap1.md, section 4.1).

Implements the `Obraz po segmentacji -> Fast R-CNN -> bounding boxy + etykiety`
module as a *Fast* R-CNN (not Faster): region proposals come from an external
Selective Search (the "SS" from the doc) instead of a learned RPN. The network
itself is a ResNet-FPN feature extractor + RoIAlign + two-layer MLP head with
a classifier and box regressor (torchvision building blocks), fine-tuned on the
LaRS "Thing" classes (boats, row boats, paddle boards, buoys, swimmers, animals,
floats, other). At inference time the water mask produced by the U-Net
segmentation stage is used to suppress detections that do not touch the water
region, reducing false positives outside the river area.

Selective Search is deterministic and slow, so proposals are computed once per
split and cached next to the LaRS image cache.
"""

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign, box_iou
import wandb

from lars import load_split, PANOPTIC, bar, dir as DATASET_DIR, data_paths

# LaRS "Thing" classes become detector classes 1..N; 0 is reserved for background.
THING_IDS: list[int] = sorted(k for k, v in PANOPTIC.items() if v["type"] == "Thing")
THING_ID_TO_LABEL: dict[int, int] = {tid: i + 1 for i, tid in enumerate(THING_IDS)}
LABEL_NAMES: dict[int, str] = {i + 1: PANOPTIC[tid]["name"] for i, tid in enumerate(THING_IDS)}
LABEL_TO_CATEGORY: dict[int, int] = {label: tid for label, tid in enumerate(THING_IDS, start=1)}
NUM_DETECTION_CLASSES = len(THING_IDS) + 1  # + background

# Standard Faster/Fast R-CNN bbox regression weights for (dx, dy, dw, dh).
DEFAULT_BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def boxes_from_panoptic(pan: np.ndarray, min_box_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-instance bounding boxes from a LaRS panoptic mask.

    The mask is read by cv2 in BGR order; LaRS stores the class id in the R
    channel and the instance id as G * 256 + B.

    Returns (boxes, labels) where boxes are (N, 4) xyxy float32 and labels are
    (N,) int64 detector labels (1..len(THING_IDS)).
    """
    cls = pan[..., 2].astype(np.int64)
    inst = pan[..., 1].astype(np.int64) * 256 + pan[..., 0].astype(np.int64)

    thing = np.isin(cls, THING_IDS)
    ys, xs = np.nonzero(thing)
    boxes: list[list[float]] = []
    labels: list[int] = []
    if ys.size:
        # Pack (class, instance) into one key so a single np.unique pass finds all instances.
        keys = cls[ys, xs] * (1 << 20) + inst[ys, xs]
        for key in np.unique(keys):
            sel = keys == key
            x0, x1 = int(xs[sel].min()), int(xs[sel].max())
            y0, y1 = int(ys[sel].min()), int(ys[sel].max())
            if (x1 - x0 + 1) < min_box_size or (y1 - y0 + 1) < min_box_size:
                continue
            boxes.append([x0, y0, x1 + 1, y1 + 1])
            labels.append(THING_ID_TO_LABEL[int(key >> 20)])

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Selective Search region proposals (the "SS" stage of Fast R-CNN)
# ---------------------------------------------------------------------------


def selective_search_proposals(
    image_bgr: np.ndarray,
    max_proposals: int = 2000,
    fast: bool = True,
    downscale_width: int = 512,
    min_box_size: int = 8,
) -> np.ndarray:
    """Generate region proposals for a BGR uint8 image via Selective Search.

    Selective Search is run on a downscaled copy for speed and the boxes are
    scaled back to the original resolution. Returns (N, 4) xyxy float32.
    Requires opencv-contrib (cv2.ximgproc).
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, downscale_width / w)
    small = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale)))) if scale < 1.0 else image_bgr

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small)
    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()  # (N, 4) as x, y, w, h in the downscaled image

    if rects is None or len(rects) == 0:
        return np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)

    rects = rects.astype(np.float32) / scale
    boxes = np.stack(
        [rects[:, 0], rects[:, 1], rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]],
        axis=1,
    )
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    keep = (boxes[:, 2] - boxes[:, 0] >= min_box_size) & (boxes[:, 3] - boxes[:, 1] >= min_box_size)
    boxes = boxes[keep][:max_proposals]
    if len(boxes) == 0:
        boxes = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
    return np.ascontiguousarray(boxes, dtype=np.float32)


def compute_split_proposals(
    split: str,
    images: torch.Tensor,
    max_proposals: int = 2000,
    fast: bool = True,
    num_workers: int = 8,
) -> list[torch.Tensor]:
    """Selective Search proposals for every image of a split, cached on disk."""
    h, w = int(images.shape[2]), int(images.shape[3])
    cached_dir = os.path.join(DATASET_DIR, "cached")
    os.makedirs(cached_dir, exist_ok=True)
    cached = os.path.join(
        cached_dir,
        f"{split}_proposals_ss{'fast' if fast else 'quality'}_{max_proposals}_{w}x{h}.npz",
    )
    if os.path.exists(cached):
        print(f"Loading cached proposals from {cached}")
        data = np.load(cached)
        return [torch.from_numpy(data[f"p{i}"]) for i in range(len(images))]

    n = len(images)

    def one(i: int) -> np.ndarray:
        img = (images[i] * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        return selective_search_proposals(np.ascontiguousarray(img), max_proposals=max_proposals, fast=fast)

    proposals: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, boxes in enumerate(ex.map(one, range(n))):
            print(f"Selective Search [{split}] {bar(i + 1, n)}", end="\r")
            proposals.append(boxes)
    print()
    np.savez_compressed(cached, **{f"p{i}": b for i, b in enumerate(proposals)})
    print(f"Saved proposals to {cached}")
    return [torch.from_numpy(b) for b in proposals]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LarsDetectionDataset(Dataset):
    """LaRS images + Selective Search proposals + box targets from panoptic masks.

    Loads images lazily from disk to avoid memory issues with large datasets.
    """

    def __init__(
        self,
        split: str,
        reshape: tuple[int, int] = (1024, 576),
        min_box_size: int = 4,
        max_proposals: int = 2000,
    ):
        self.split = split
        self.reshape = reshape
        images_dir = data_paths[split]["images"]
        self.image_files = sorted(os.listdir(images_dir))
        self.num_images = len(self.image_files)

        # Build annotation directories
        self.dirs = {
            "images": images_dir,
            "panoptic": os.path.join(data_paths[split]["annotations"], "panoptic_masks"),
        }

        # Pre-compute targets from panoptic masks
        self.targets: list[dict[str, torch.Tensor]] = []
        print(f"Computing targets from panoptic masks...")
        for i, name in enumerate(self.image_files):
            print(f"Extracting {split} boxes {bar(i + 1, len(self.image_files))}", end="\r")
            mask_path = os.path.join(self.dirs["panoptic"], name.replace(".jpg", ".png"))
            panoptic = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if panoptic is None:
                raise FileNotFoundError(mask_path)
            panoptic = cv2.resize(panoptic, reshape, interpolation=cv2.INTER_NEAREST)
            b, l = boxes_from_panoptic(panoptic, min_box_size=min_box_size)
            self.targets.append({"boxes": torch.from_numpy(b), "labels": torch.from_numpy(l)})
        print()

        # Pre-compute proposals via selective search (cached)
        self.proposals = self._compute_proposals(max_proposals)

    def _compute_proposals(self, max_proposals: int = 2000) -> list[torch.Tensor]:
        """Selective Search proposals for every image of a split, cached on disk."""
        h, w = self.reshape[1], self.reshape[0]
        cached_dir = os.path.join(os.path.dirname(self.dirs["images"]), "..", "cached")
        os.makedirs(cached_dir, exist_ok=True)
        cached = os.path.join(
            cached_dir,
            f"{self.split}_proposals_ss_fast_{max_proposals}_{w}x{h}.npz",
        )

        if os.path.exists(cached):
            print(f"Loading cached proposals from {cached}")
            data = np.load(cached)
            return [torch.from_numpy(data[f"p{i}"]) for i in range(len(self.image_files))]

        print(f"Computing Selective Search proposals for {self.split} split...")

        def compute_one(i: int) -> np.ndarray:
            name = self.image_files[i]
            img_path = os.path.join(self.dirs["images"], name)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(img_path)
            img = cv2.resize(img, self.reshape)
            return selective_search_proposals(img, max_proposals=max_proposals, fast=True)

        proposals: list[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, boxes in enumerate(ex.map(compute_one, range(len(self.image_files)))):
                print(f"Selective Search [{self.split}] {bar(i + 1, len(self.image_files))}", end="\r")
                proposals.append(boxes)
        print()

        np.savez_compressed(cached, **{f"p{i}": b for i, b in enumerate(proposals)})
        print(f"Saved proposals to {cached}")
        return [torch.from_numpy(b) for b in proposals]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # Load image on-demand from disk
        name = self.image_files[idx]
        img_path = os.path.join(self.dirs["images"], name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, self.reshape)

        # Convert to tensor and normalize
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float().div_(255.0)

        # The pretrained backbone expects RGB; our images are BGR.
        img = img[[2, 1, 0], :, :]

        t = self.targets[idx]
        target = {
            "boxes": t["boxes"].clone(),
            "labels": t["labels"].clone(),
            "image_id": torch.tensor(idx),
        }
        return img, self.proposals[idx].clone(), target


def detection_collate(batch):
    return tuple(zip(*batch))


def prepare_detection_dataloaders(
    batch_size: int = 4,
    reshape: tuple[int, int] = (1024, 576),
    min_box_size: int = 4,
    max_proposals: int = 2000,
    weighted_sampler: bool = False,
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    for split in ["train", "val"]:
        print(f"Loading {split} detection data...")
        dataset = LarsDetectionDataset(split, reshape=reshape, min_box_size=min_box_size, max_proposals=max_proposals)
        sampler = None
        shuffle = split == "train"
        if split == "train" and weighted_sampler:
            class_weights = compute_detection_class_weights(dataset)
            sample_weights = build_detection_sample_weights(dataset, class_weights)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
            print(f"Using WeightedRandomSampler (class weights: {[round(w, 2) for w in class_weights.tolist()]})")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=(split == "train"),
            num_workers=2,
            pin_memory=True,
            collate_fn=detection_collate,
        )
    return dataloaders


# ---------------------------------------------------------------------------
# Fast R-CNN model
# ---------------------------------------------------------------------------


class BoxRegressor(nn.Module):
    """Per-class box refinement head: predicts (dx, dy, dw, dh) for each RoI."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        return self.bbox_pred(x)


class RoIPredictor(nn.Module):
    """RoI classification + box regression head compatible with torchvision RoIHeads."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.box_regressor = BoxRegressor(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        return self.cls_score(x), self.box_regressor(x)


def _remap_detector_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Support checkpoints saved with a flat box_predictor.bbox_pred head."""
    old = "roi_heads.box_predictor.bbox_pred."
    new = "roi_heads.box_predictor.box_regressor.bbox_pred."
    return {new + k[len(old) :] if k.startswith(old) else k: v for k, v in state_dict.items()}


def load_detector_state_dict(model: nn.Module, path: str, map_location=None) -> None:
    """Load a detector checkpoint, including older flat bbox_pred key layouts."""
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_remap_detector_state_dict(state))


class FastRCNN(nn.Module):
    """Fast R-CNN: backbone + RoIAlign + MLP head over externally supplied proposals.

    Unlike torchvision's FasterRCNN there is no RPN; `forward` takes the
    Selective Search proposals alongside the images. Images are expected as
    same-sized RGB floats in [0, 1] with boxes in pixel coordinates (no
    internal resizing), which matches our fixed-resolution LaRS tensors.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(backbone.out_channels * resolution**2, representation_size)
        box_predictor = RoIPredictor(representation_size, num_classes)
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=DEFAULT_BBOX_REG_WEIGHTS,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )
        self.register_buffer("pixel_mean", torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(self.IMAGENET_STD).view(3, 1, 1), persistent=False)

    def forward(
        self,
        images: list[torch.Tensor],
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        """Returns a loss dict in training mode, a list of detection dicts in eval mode."""
        image_shapes = [tuple(img.shape[-2:]) for img in images]
        x = torch.stack([(img - self.pixel_mean) / self.pixel_std for img in images])
        features = self.backbone(x)
        detections, losses = self.roi_heads(features, list(proposals), image_shapes, targets)
        return losses if self.training else detections


def build_fast_rcnn(
    num_classes: int = NUM_DETECTION_CLASSES,
    pretrained: bool = True,
    backbone_name: str = "resnet18",
    trainable_backbone_layers: int = 3,
) -> FastRCNN:
    """Build a Fast R-CNN. backbone_name: resnet18 (default), resnet34, resnet50, ..."""
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT" if pretrained else None,
        trainable_layers=trainable_backbone_layers,
    )
    model = FastRCNN(backbone, num_classes)
    model.backbone_name = backbone_name
    return model


def build_faster_rcnn(
    num_classes: int = NUM_DETECTION_CLASSES,
    pretrained: bool = True,
    backbone_name: str = "resnet50",
    trainable_backbone_layers: int = 3,
) -> FasterRCNN:
    """Build a Faster R-CNN with RPN using a configurable backbone."""
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT" if pretrained else None,
        trainable_layers=trainable_backbone_layers,
    )
    model = FasterRCNN(backbone, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = RoIPredictor(in_features, num_classes)
    model.backbone_name = backbone_name
    return model


def _uses_external_proposals(model: nn.Module) -> bool:
    """True for our Fast R-CNN (Selective Search); False for torchvision Faster R-CNN (RPN)."""
    return isinstance(model, FastRCNN)


def _forward_detector(
    model: nn.Module,
    images: list[torch.Tensor],
    proposals: list[torch.Tensor] | None = None,
    targets: list[dict[str, torch.Tensor]] | None = None,
):
    if _uses_external_proposals(model):
        return model(images, proposals or [], targets)
    return model(images, targets)


def compute_detection_class_weights(dataset: LarsDetectionDataset) -> torch.Tensor:
    """Inverse-frequency class weights for RoI classification (background weight = 1)."""
    label_counts: Counter[int] = Counter()
    for target in dataset.targets:
        for label in target["labels"].tolist():
            label_counts[int(label)] += 1
    print("Instance counts per class:", dict(sorted(label_counts.items())))
    total = sum(label_counts.values())
    class_weights = torch.ones(NUM_DETECTION_CLASSES, dtype=torch.float32)
    for label, count in label_counts.items():
        class_weights[label] = total / (len(label_counts) * count)
    return class_weights


def build_detection_sample_weights(dataset: LarsDetectionDataset, class_weights: torch.Tensor) -> list[float]:
    """Per-image sampler weights: max class weight among objects in the image."""
    sample_weights: list[float] = []
    for target in dataset.targets:
        labels = target["labels"]
        if len(labels) == 0:
            sample_weights.append(1.0)
        else:
            sample_weights.append(max(float(class_weights[int(label)]) for label in labels))
    return sample_weights


@contextmanager
def weighted_detection_loss(class_weights: torch.Tensor):
    """Patch torchvision fastrcnn_loss to use inverse-frequency class weights."""
    import torch.nn.functional as F
    import torchvision.models.detection.roi_heads as roi_heads_module

    original = roi_heads_module.fastrcnn_loss
    weights = class_weights

    def patched_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels_cat, weight=weights.to(class_logits.device))
        sampled_pos = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos]
        box_regression = box_regression.reshape(box_regression.size(0), box_regression.size(-1) // 4, 4)
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos, labels_pos],
            regression_targets_cat[sampled_pos],
            beta=1.0 / 9,
            reduction="sum",
        )
        box_loss = box_loss / max(labels_cat.numel(), 1)
        return classification_loss, box_loss

    roi_heads_module.fastrcnn_loss = patched_fastrcnn_loss
    try:
        yield
    finally:
        roi_heads_module.fastrcnn_loss = original


def configure_detector_inference(
    model: nn.Module,
    *,
    nms_thresh: float = 0.45,
    detections_per_img: int = 150,
    roi_score_thresh: float = 0.05,
) -> None:
    """Tune RoI post-processing (no retraining required)."""
    if isinstance(model, (FasterRCNN, FastRCNN)):
        model.roi_heads.nms_thresh = nms_thresh
        model.roi_heads.score_thresh = roi_score_thresh
        model.roi_heads.detections_per_img = detections_per_img


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_detector(
    wandb_project: str,
    wandb_run_name: str,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    optimizer: optim.Optimizer,
    device: torch.device,
    lr_scheduler: optim.lr_scheduler._LRScheduler | None = None,
    epochs: int = 10,
    use_amp: bool | None = None,
    random_chars: str = "",
    class_weights: torch.Tensor | None = None,
):
    def get_lr() -> float | None:
        if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
            lr_list = lr_scheduler.get_last_lr()
            return float(lr_list[0]) if lr_list else None
        return optimizer.param_groups[0].get("lr", None)

    amp_enabled = (device.type == "cuda") if use_amp is None else bool(use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric_map = MeanAveragePrecision(box_format="xyxy")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"mAP metric unavailable ({e}); validation will track loss only.")
        metric_map = None

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "epochs": epochs,
            "device": device.type,
            "optimizer": optimizer.__class__.__name__,
            "lr": get_lr(),
            "wd": optimizer.param_groups[0].get("weight_decay", 0.0),
            "batch_size": dataloaders["train"].batch_size,
            "num_classes": NUM_DETECTION_CLASSES,
            "amp": amp_enabled,
            "class_weighted_loss": class_weights is not None,
            "model": (
                f"fast_rcnn_{getattr(model, 'backbone_name', 'resnet')}_fpn_selective_search" if _uses_external_proposals(model) else f"faster_rcnn_{getattr(model, 'backbone_name', 'resnet')}_fpn"
            ),
        },
    )

    best_score: float | None = None
    loss_ctx = weighted_detection_loss(class_weights.to(device)) if class_weights is not None else nullcontext()

    try:
        with loss_ctx:
            for epoch in range(1, epochs + 1):
                # --- Training ---
                model.train()
                loader = dataloaders["train"]
                steps = max(1, len(loader))
                loss_sum = 0.0
                for step, (images, proposals, targets) in enumerate(loader):
                    print(f"\033[94m(Epoch {epoch:02}/{epochs} [train]){bar(step + 1, steps)}\033[0m", end="\r")
                    images = [img.to(device, non_blocking=True) for img in images]
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    prop_dev = [p.to(device, non_blocking=True) for p in proposals] if _uses_external_proposals(model) else None

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                        loss_dict = _forward_detector(model, images, prop_dev, targets)
                        loss = sum(loss_dict.values())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_sum += float(loss.detach())

                train_loss = loss_sum / steps
                current_lr = get_lr()
                print(f"\033[94m(Epoch {epoch:02}/{epochs} [train]) Loss: {train_loss:.3f} lr: {current_lr*10000:.4f}e-4\033[0m         ")

                # --- Validation (mAP) ---
                model.eval()
                loader = dataloaders["val"]
                steps = max(1, len(loader))
                if metric_map is not None:
                    metric_map.reset()
                with torch.no_grad():
                    for step, (images, proposals, targets) in enumerate(loader):
                        print(f"\033[96m(Epoch {epoch:02}/{epochs} [val  ]){bar(step + 1, steps)}\033[0m", end="\r")
                        images = [img.to(device, non_blocking=True) for img in images]
                        prop_dev = [p.to(device, non_blocking=True) for p in proposals] if _uses_external_proposals(model) else None
                        outputs = _forward_detector(model, images, prop_dev)
                        if metric_map is not None:
                            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
                            gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
                            metric_map.update(preds, gts)

                log: dict[str, float | None] = {"train/loss": train_loss, "train/lr": current_lr}
                score = -train_loss  # Fallback selection criterion if mAP is unavailable.
                map_str = ""
                if metric_map is not None:
                    map_res = metric_map.compute()
                    log["val/map"] = float(map_res["map"])
                    log["val/map_50"] = float(map_res["map_50"])
                    log["val/map_75"] = float(map_res["map_75"])
                    score = float(map_res["map"])
                    map_str = f" mAP: {score:.3f} mAP@50: {float(map_res['map_50']):.3f}"

                extra = ""
                if best_score is None or score > best_score:
                    best_score = score
                    extra = " (best so far, saving model...)"
                    torch.save(model.state_dict(), f"{wandb_run_name}.{random_chars}.best.pth")
                print(f"\033[96m(Epoch {epoch:02}/{epochs} [val  ]){map_str}{extra}\033[0m         ")

                wandb.log(log, step=epoch)

                if lr_scheduler is not None:
                    lr_scheduler.step()
    finally:
        wandb.finish()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def detect_objects(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    score_thresh: float = 0.5,
    max_proposals: int = 2000,
    proposals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run object detection on a single (C, H, W) BGR float image in [0, 1].

    Fast R-CNN uses Selective Search proposals; Faster R-CNN runs the built-in RPN.
    Returns (boxes xyxy, labels, scores) on CPU, filtered by score_thresh.
    """
    model.eval()
    rgb = image[[2, 1, 0], :, :].to(device)

    if _uses_external_proposals(model):
        if proposals is None:
            bgr_u8 = np.ascontiguousarray((image * 255.0).to(torch.uint8).permute(1, 2, 0).numpy())
            proposals = torch.from_numpy(selective_search_proposals(bgr_u8, max_proposals=max_proposals)).to(device)
        else:
            proposals = proposals.to(device)
        output = model([rgb], [proposals])[0]
    else:
        output = model([rgb])[0]

    keep = output["scores"] >= score_thresh
    return output["boxes"][keep].cpu(), output["labels"][keep].cpu(), output["scores"][keep].cpu()


def filter_detections_by_water(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    water_mask: torch.Tensor | np.ndarray,
    dilate_px: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only detections whose box intersects the (dilated) water mask.

    This ties the detection stage to the segmentation stage: objects detected
    far away from the water surface (e.g. on land or in the sky region) are
    discarded as false positives. Dilation keeps objects that sit on the
    waterline (boats moored at a pier, signs on the bank edge, swimmers
    partially occluding the water boundary).
    """
    mask = water_mask.detach().cpu().numpy() if isinstance(water_mask, torch.Tensor) else np.asarray(water_mask)
    mask = mask.astype(np.uint8)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kernel)

    h, w = mask.shape
    keep = torch.zeros(len(boxes), dtype=torch.bool)
    for i, box in enumerate(boxes):
        x0 = max(0, min(w - 1, int(box[0])))
        y0 = max(0, min(h - 1, int(box[1])))
        x1 = max(x0 + 1, min(w, int(box[2])))
        y1 = max(y0 + 1, min(h, int(box[3])))

        # Intersects the dilated water mask
        keep[i] = bool(mask[y0:y1, x0:x1].any())
    return boxes[keep], labels[keep], scores[keep]


def draw_detections(
    frame: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> np.ndarray:
    """Draw labelled bounding boxes onto a BGR uint8 frame (in place) and return it."""
    rng = np.random.default_rng(7)
    colors = {label: tuple(int(c) for c in rng.integers(64, 256, size=3)) for label in LABEL_NAMES}
    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = (int(v) for v in box)
        label = int(label)
        color = colors.get(label, (255, 255, 255))
        name = LABEL_NAMES.get(label, f"cls{label}")
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        text = f"{name} {float(score):.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y0 - 4 if y0 - th - baseline - 4 >= 0 else y1 + th + baseline + 4
        cv2.rectangle(frame, (x0, ty - th - baseline), (x0 + tw, ty + baseline), color, -1)
        cv2.putText(frame, text, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def xyxy_to_coco_bbox(box: torch.Tensor | np.ndarray | list[float]) -> list[float]:
    """Convert an xyxy box to COCO [x, y, width, height]."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0, y0, x1 - x0, y1 - y0]


def coco_thing_categories() -> list[dict]:
    """LaRS Thing categories in COCO dataset format."""
    return [{"id": tid, "name": PANOPTIC[tid]["name"], "supercategory": PANOPTIC[tid]["supercategory"]} for tid in THING_IDS]


def build_coco_detections(
    file_names: list[str],
    all_boxes: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    all_scores: list[torch.Tensor],
    image_size: tuple[int, int],
) -> dict:
    """Build a self-contained COCO-style JSON dict from batched detector outputs.

    Boxes are expected in xyxy format at the same resolution as image_size.
    category_id values use LaRS panoptic ids (11, 12, ...), not detector labels.
    """
    width, height = image_size
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for image_id, (file_name, boxes, labels, scores) in enumerate(zip(file_names, all_boxes, all_labels, all_scores), start=1):
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for box, label, score in zip(boxes, labels, scores):
            bbox = xyxy_to_coco_bbox(box)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": LABEL_TO_CATEGORY[int(label)],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
            ann_id += 1

    return {"images": images, "categories": coco_thing_categories(), "annotations": annotations}


def save_coco_detections_json(
    path: str,
    file_names: list[str],
    all_boxes: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    all_scores: list[torch.Tensor],
    image_size: tuple[int, int],
) -> None:
    """Write detector outputs to a COCO-style detections.json file."""
    payload = build_coco_detections(file_names, all_boxes, all_labels, all_scores, image_size)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def load_gt_boxes_for_filename(
    filename: str,
    reshape: tuple[int, int],
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load LaRS Thing boxes (xyxy) for an image file name, if annotations exist."""
    stem = os.path.splitext(filename)[0]
    for split in splits:
        mask_path = os.path.join(data_paths[split]["annotations"], "panoptic_masks", f"{stem}.png")
        if not os.path.isfile(mask_path):
            continue
        panoptic = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if panoptic is None:
            continue
        panoptic = cv2.resize(panoptic, reshape, interpolation=cv2.INTER_NEAREST)
        boxes, labels = boxes_from_panoptic(panoptic)
        return torch.from_numpy(boxes), torch.from_numpy(labels)
    return None


def load_gt_semantic_for_filename(
    filename: str,
    reshape: tuple[int, int],
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> torch.Tensor | None:
    """Load LaRS semantic mask for an image file name, if annotations exist."""
    stem = os.path.splitext(filename)[0]
    for split in splits:
        mask_path = os.path.join(data_paths[split]["annotations"], "semantic_masks", f"{stem}.png")
        if not os.path.isfile(mask_path):
            continue
        semantic = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if semantic is None:
            continue
        return torch.from_numpy(cv2.resize(semantic, reshape, interpolation=cv2.INTER_NEAREST))
    return None


def water_segmentation_iou(pred_water_mask: torch.Tensor, gt_semantic: torch.Tensor, water_class: int = 1) -> float:
    """IoU for the water class between predicted and ground-truth semantic masks, excluding ignore index (255)."""
    # Exclude ignore index 255 identically to unet_main.py
    valid = gt_semantic != 255
    pred = pred_water_mask.bool() & valid.to(pred_water_mask.device)
    gt = (gt_semantic == water_class) & valid
    intersection = (pred & gt.to(pred.device)).sum().float()
    union = (pred | gt.to(pred.device)).sum().float()
    if union == 0:
        return 1.0
    return float((intersection / union).item())


def mean_matched_box_iou(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> float | None:
    """Class-agnostic mean IoU over GT boxes via greedy highest-IoU matching.

    Each GT box is matched to at most one prediction (and vice versa) by
    descending IoU, regardless of predicted class. Unmatched GT boxes count as 0.
    """
    if gt_boxes.numel() == 0:
        return None
    if pred_boxes.numel() == 0:
        return 0.0

    ious = box_iou(pred_boxes, gt_boxes)
    pairs = [(float(ious[pred_idx, gt_idx]), pred_idx, gt_idx) for pred_idx in range(len(pred_boxes)) for gt_idx in range(len(gt_boxes))]
    pairs.sort(key=lambda item: item[0], reverse=True)

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matched_by_gt = [0.0] * len(gt_boxes)
    for iou, pred_idx, gt_idx in pairs:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matched_by_gt[gt_idx] = iou
    return sum(matched_by_gt) / len(gt_boxes)


def match_detections_coco(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy COCO-style TP/FP/FN matching at a single IoU threshold."""
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    order = torch.argsort(pred_scores, descending=True)
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    tp = fp = 0
    for pred_idx in order.tolist():
        best_iou = 0.0
        best_gt = -1
        for gt_idx in range(n_gt):
            if gt_idx in matched_gt or int(pred_labels[pred_idx]) != int(gt_labels[gt_idx]):
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
    fn = n_gt - len(matched_gt)
    return tp, fp, fn


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_per_image_detection_stats(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Per-image detection stats meaningful on a single image (not mAP)."""
    tp, fp, fn = match_detections_coco(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=iou_threshold)
    precision, recall, f1 = _prf1(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "box_iou": mean_matched_box_iou(pred_boxes, gt_boxes),
    }


def compute_dataset_detection_metrics(
    preds: list[dict[str, torch.Tensor]],
    gts: list[dict[str, torch.Tensor]],
    box_ious: list[float] | None = None,
    water_ious: list[float] | None = None,
) -> dict[str, float | torch.Tensor | int | None]:
    """Dataset-level mAP and mean IoU over evaluated images."""
    metrics: dict[str, float | torch.Tensor | int | None] = {
        "num_images": len(preds),
        "map": None,
        "map_50": None,
        "map_75": None,
        "map_per_class": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "mean_box_iou": (sum(box_ious) / len(box_ious)) if box_ious else None,
        "num_box_iou": len(box_ious) if box_ious else 0,
        "mean_water_iou": (sum(water_ious) / len(water_ious)) if water_ious else None,
        "num_water_iou": len(water_ious) if water_ious else 0,
    }
    if not preds:
        return metrics

    total_tp = total_fp = total_fn = 0
    for pred, gt in zip(preds, gts):
        tp, fp, fn = match_detections_coco(
            pred["boxes"],
            pred["labels"],
            pred["scores"],
            gt["boxes"],
            gt["labels"],
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision, recall, f1 = _prf1(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(preds, gts)
        result = metric.compute()
        metrics.update(
            {
                "map": float(result["map"]),
                "map_50": float(result["map_50"]),
                "map_75": float(result["map_75"]),
                "map_per_class": result["map_per_class"].detach().cpu(),
            }
        )
    except (ImportError, ModuleNotFoundError):
        pass
    return metrics


def sweep_score_thresholds(
    raw_preds: list[dict[str, torch.Tensor]],
    gts: list[dict[str, torch.Tensor]],
    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> tuple[float, dict[str, float | int | None]]:
    """Pick score threshold that maximizes global F1 on precomputed low-threshold preds."""
    best_thresh = thresholds[0]
    best_metrics: dict[str, float | int | None] = {}
    best_f1 = -1.0
    print(f"{'thresh':>6}  {'F1':>6}  {'P':>6}  {'R':>6}  {'mAP@50':>7}  {'box IoU':>7}")
    for thresh in thresholds:
        filtered_preds: list[dict[str, torch.Tensor]] = []
        box_ious: list[float] = []
        for pred, gt in zip(raw_preds, gts):
            keep = pred["scores"] >= thresh
            boxes = pred["boxes"][keep]
            labels = pred["labels"][keep]
            scores = pred["scores"][keep]
            filtered_preds.append({"boxes": boxes, "labels": labels, "scores": scores})
            iou = mean_matched_box_iou(boxes, gt["boxes"])
            if iou is not None:
                box_ious.append(iou)
        metrics = compute_dataset_detection_metrics(filtered_preds, gts, box_ious, None)
        f1 = float(metrics.get("f1") or 0.0)
        print(
            f"{thresh:6.2f}  {f1:6.3f}  {float(metrics.get('precision') or 0):6.3f}  "
            f"{float(metrics.get('recall') or 0):6.3f}  "
            f"{_format_metric_value(metrics.get('map_50') if isinstance(metrics.get('map_50'), float) else None):>7}  "
            f"{(sum(box_ious) / len(box_ious) if box_ious else 0.0):7.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = metrics
    print(f"\nBest threshold by global F1: {best_thresh:.2f} (F1={best_f1:.3f})")
    return best_thresh, best_metrics


def _format_metric_value(value: float | None, undefined_label: str = "undef") -> str:
    """Format a scalar metric; torchmetrics uses -1 when a metric is undefined."""
    if value is None:
        return "n/a"
    if value < 0:
        return undefined_label
    return f"{value:.3f}"


def format_image_metrics_line(
    stats: dict[str, float | int | None],
    water_iou: float | None = None,
) -> str:
    """Format per-image detection stats for terminal output (P/R/F1, not mAP)."""

    def fmt_iou(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    parts = [
        f"TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}",
        f"P={stats['precision']:.3f}",
        f"R={stats['recall']:.3f}",
        f"F1={stats['f1']:.3f}",
        f"box IoU={fmt_iou(stats.get('box_iou'))}",
    ]
    if water_iou is not None:
        parts.append(f"water IoU={water_iou:.3f}")
    return " | ".join(parts)


def format_dataset_metrics_summary(metrics: dict[str, float | torch.Tensor | int | None]) -> str:
    """Format dataset-level mAP and mean IoU summary."""
    n = metrics.get("num_images", 0)
    n_box = int(metrics.get("num_box_iou") or 0)
    n_water = int(metrics.get("num_water_iou") or 0)
    if not n and not n_water:
        return "No images with ground truth — dataset metrics skipped."

    lines: list[str] = []
    if n:
        lines.append(f"Dataset detection metrics ({n} image(s) with GT boxes, raw detector before water filter):")
        lines.append(f"  mAP={_format_metric_value(metrics.get('map'))} " f"mAP@50={_format_metric_value(metrics.get('map_50'))} " f"mAP@75={_format_metric_value(metrics.get('map_75'))}")
        if metrics.get("f1") is not None:
            lines.append(
                f"  Global @ IoU 0.5: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                f"P={float(metrics['precision']):.3f} R={float(metrics['recall']):.3f} "
                f"F1={float(metrics['f1']):.3f}"
            )
        map_per_class = metrics.get("map_per_class")
        if isinstance(map_per_class, torch.Tensor) and map_per_class.dim() > 0:
            per_class = map_per_class.flatten().tolist()
            missing_classes = [LABEL_NAMES[label_idx] for label_idx, ap in enumerate(per_class) if label_idx in LABEL_NAMES and ap < 0]
            if missing_classes:
                lines.append(f"  Classes with no GT in evaluated set: {', '.join(missing_classes)} (AP=undef)")

    iou_parts: list[str] = []
    if metrics.get("mean_box_iou") is not None:
        iou_parts.append(f"mean box IoU={float(metrics['mean_box_iou']):.3f} ({n_box} image(s))")
    if metrics.get("mean_water_iou") is not None:
        iou_parts.append(f"mean water IoU={float(metrics['mean_water_iou']):.3f} ({n_water} image(s))")
    if iou_parts:
        lines.append("Dataset IoU: " + " | ".join(iou_parts))

    return "\n".join(lines)
