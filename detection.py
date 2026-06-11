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

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
import wandb

from lars import load_split, PANOPTIC, bar, dir as DATASET_DIR

# LaRS "Thing" classes become detector classes 1..N; 0 is reserved for background.
THING_IDS: list[int] = sorted(k for k, v in PANOPTIC.items() if v["type"] == "Thing")
THING_ID_TO_LABEL: dict[int, int] = {tid: i + 1 for i, tid in enumerate(THING_IDS)}
LABEL_NAMES: dict[int, str] = {i + 1: PANOPTIC[tid]["name"] for i, tid in enumerate(THING_IDS)}
NUM_DETECTION_CLASSES = len(THING_IDS) + 1  # + background


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
    """LaRS images + Selective Search proposals + box targets from panoptic masks."""

    def __init__(
        self,
        split: str,
        reshape: tuple[int, int] = (1024, 576),
        min_box_size: int = 4,
        max_proposals: int = 2000,
    ):
        images, _, panoptic = load_split(split, inc_semantic=False, inc_panoptic=True, reshape=reshape)
        self.images = images  # float32 in [0, 1], BGR, NCHW
        self.targets: list[dict[str, torch.Tensor]] = []
        n = len(images)
        for i in range(n):
            print(f"Extracting {split} boxes {bar(i + 1, n)}", end="\r")
            b, l = boxes_from_panoptic(panoptic[i].numpy(), min_box_size=min_box_size)
            self.targets.append({"boxes": torch.from_numpy(b), "labels": torch.from_numpy(l)})
        print()
        self.proposals = compute_split_proposals(split, images, max_proposals=max_proposals)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # The pretrained backbone expects RGB; our cached tensors are BGR.
        img = self.images[idx][[2, 1, 0], :, :]
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
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    for split in ["train", "val"]:
        print(f"Loading {split} detection data...")
        dataset = LarsDetectionDataset(split, reshape=reshape, min_box_size=min_box_size, max_proposals=max_proposals)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=2,
            pin_memory=True,
            collate_fn=detection_collate,
        )
    return dataloaders


# ---------------------------------------------------------------------------
# Fast R-CNN model
# ---------------------------------------------------------------------------


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
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
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
            "model": f"fast_rcnn_{getattr(model, 'backbone_name', 'resnet')}_fpn_selective_search",
        },
    )

    best_score: float | None = None

    try:
        for epoch in range(1, epochs + 1):
            # --- Training ---
            model.train()
            loader = dataloaders["train"]
            steps = max(1, len(loader))
            loss_sum = 0.0
            for step, (images, proposals, targets) in enumerate(loader):
                print(f"\033[94m(Epoch {epoch:02}/{epochs} [train]){bar(step + 1, steps)}\033[0m", end="\r")
                images = [img.to(device, non_blocking=True) for img in images]
                proposals = [p.to(device, non_blocking=True) for p in proposals]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    loss_dict = model(images, proposals, targets)
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
                    proposals = [p.to(device, non_blocking=True) for p in proposals]
                    outputs = model(images, proposals)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Selective Search + Fast R-CNN on a single (C, H, W) BGR float image in [0, 1].

    Returns (boxes xyxy, labels, scores) on CPU, filtered by score_thresh.
    """
    model.eval()
    bgr_u8 = np.ascontiguousarray((image * 255.0).to(torch.uint8).permute(1, 2, 0).numpy())
    proposals = torch.from_numpy(selective_search_proposals(bgr_u8, max_proposals=max_proposals)).to(device)
    rgb = image[[2, 1, 0], :, :].to(device)
    output = model([rgb], [proposals])[0]
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
