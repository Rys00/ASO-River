from torch.utils.data import TensorDataset, DataLoader
from lars import load_split, bar
from torch import nn, optim
import torch
import torchmetrics
import wandb
import kornia.augmentation as K
import os


class KorniaSegmentationAugment(nn.Module):
    def __init__(
        self,
        hflip_p: float = 0.5,
        jitter_p: float = 0.8,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.10,
        hue: float = 0.02,
        noise_p: float = 0.2,
        noise_std: float = 0.01,
        input_is_bgr: bool = False,
        input_is_hsv: bool = False,
    ):
        super().__init__()
        self.input_is_bgr = bool(input_is_bgr)
        self.input_is_hsv = bool(input_is_hsv)
        if self.input_is_hsv and self.input_is_bgr:
            raise ValueError("input_is_hsv and input_is_bgr cannot both be True")

        self.jitter_p = float(jitter_p)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        # Keep masks out of photometric transforms for speed and to avoid
        # unnecessary dtype conversions.
        self.geom = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=hflip_p),
            data_keys=["input", "mask"],
            same_on_batch=False,
        )
        # Kornia ColorJitter assumes RGB ordering, so we only use it for RGB/BGR.
        self.rgb_jitter = K.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=jitter_p,
        )
        self.noise = K.RandomGaussianNoise(mean=0.0, std=noise_std, p=noise_p)

    def _hsv_jitter(self, images: torch.Tensor) -> torch.Tensor:
        """HSV-native photometric jitter.

        Assumes OpenCV HSV scaling that was normalized by /255:
        - H in [0, 179/255]
        - S in [0, 1]
        - V in [0, 1]
        """
        if self.jitter_p <= 0.0:
            return images

        b = images.size(0)
        device = images.device
        mask = (torch.rand(b, device=device) < self.jitter_p).view(b, 1, 1, 1)
        if not bool(mask.any()):
            return images

        h, s, v = images[:, 0:1], images[:, 1:2], images[:, 2:3]

        # Hue wrap range under OpenCV HSV after normalization by 255.
        h_max = 179.0 / 255.0

        if self.hue > 0.0:
            dh = (torch.rand(b, device=device) * 2.0 - 1.0).view(b, 1, 1, 1) * (self.hue * h_max)
            h2 = torch.remainder(h + dh, h_max)
            h = torch.where(mask, h2, h)

        if self.saturation > 0.0:
            s_fac = (1.0 + (torch.rand(b, device=device) * 2.0 - 1.0) * self.saturation).view(b, 1, 1, 1)
            s2 = (s * s_fac).clamp(0.0, 1.0)
            s = torch.where(mask, s2, s)

        # Brightness and contrast applied on V.
        v_out = v
        if self.brightness > 0.0:
            b_fac = (1.0 + (torch.rand(b, device=device) * 2.0 - 1.0) * self.brightness).view(b, 1, 1, 1)
            v_out = v_out * b_fac
        if self.contrast > 0.0:
            c_fac = (1.0 + (torch.rand(b, device=device) * 2.0 - 1.0) * self.contrast).view(b, 1, 1, 1)
            v_out = (v_out - 0.5) * c_fac + 0.5
        v_out = v_out.clamp(0.0, 1.0)
        v = torch.where(mask, v_out, v)

        return torch.cat((h, s, v), dim=1)

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images as (B,C,H,W), got {tuple(images.shape)}")
        if masks.ndim != 3:
            raise ValueError(f"Expected masks as (B,H,W), got {tuple(masks.shape)}")
        if images.size(1) != 3:
            raise ValueError(f"Expected 3-channel images, got C={int(images.size(1))}")

        masks_dtype = masks.dtype

        # Kornia expects masks as (B,1,H,W). RandomHorizontalFlip is exact.
        masks_4d = masks.unsqueeze(1)
        images, masks_4d = self.geom(images, masks_4d)

        if self.input_is_hsv:
            images = self._hsv_jitter(images)
            images = self.noise(images)
        else:
            # Kornia's ColorJitter assumes RGB ordering. If our tensors are BGR,
            # swap to RGB for photometric aug and swap back.
            if self.input_is_bgr:
                images = images[:, [2, 1, 0], :, :]

            images = self.rgb_jitter(images)
            images = self.noise(images)

            if self.input_is_bgr:
                images = images[:, [2, 1, 0], :, :]

        images = images.clamp(0.0, 1.0)

        masks = masks_4d.squeeze(1)
        if masks.dtype != masks_dtype:
            # Should be unnecessary for flips, but keep it robust.
            if masks.dtype.is_floating_point:
                masks = masks.round().to(dtype=masks_dtype)
            else:
                masks = masks.to(dtype=masks_dtype)

        return images, masks


def prepare_dataloaders(
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    reshape: tuple[int, int] = (1024, 576),
    hsv: bool = False,
    h_bilateral: int | None = None,
    s_bilateral: int | None = None,
    v_bilateral: int | None = None,
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}

    for split in ["train", "val"]:
        print(f"Loading {split} data...")
        images, masks, _ = load_split(split, reshape=reshape, hsv=hsv, h_bilateral=h_bilateral, s_bilateral=s_bilateral, v_bilateral=v_bilateral)
        images = images.float()
        masks = masks.long()
        print(f"Preparing dataloader for {split} split...")
        dataset = TensorDataset(images, masks)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=2,
            pin_memory=True,
        )
    return dataloaders


def train(
    wandb_project: str,
    wandb_run_name: str,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lr_scheduler: optim.lr_scheduler._LRScheduler | None,
    epochs: int = 10,
    num_classes: int = 3,
    ignore_index: int | None = 255,
    gpu_augment: nn.Module | None = None,
    use_amp: bool | None = None,
    random_chars: str = "",
    save_dir: str = ".",
):
    def get_lr() -> float | None:
        if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
            lr_list = lr_scheduler.get_last_lr()
            return float(lr_list[0]) if lr_list else None
        return optimizer.param_groups[0].get("lr", None)

    amp_enabled = (device.type == "cuda") if use_amp is None else bool(use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "epochs": epochs,
            "device": device.type,
            "optimizer": optimizer.__class__.__name__,
            "lr": get_lr(),
            "wd": optimizer.param_groups[0].get("weight_decay", 0.0),
            "criterion": criterion.__class__.__name__,
            "batch_size": dataloaders["train"].batch_size,
            "resolution": dataloaders["train"].dataset.tensors[0].shape[2:],
            "dropout_p": model.dropout_p,
            "dropout_min_features": model.dropout_min_features,
            "dropout_bottleneck_p": model.dropout_bottleneck_p,
            "gpu_augment": gpu_augment is not None,
            "amp": amp_enabled,
        },
    )

    metric_loss = torchmetrics.aggregation.MeanMetric().to(device)
    if ignore_index is None:
        metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    else:
        metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)

    # Per-class IoU (Jaccard). This is typically more informative than accuracy for
    # imbalanced segmentation classes (e.g., small obstacles).
    if ignore_index is None:
        metric_iou = torchmetrics.classification.JaccardIndex(task="multiclass", num_classes=num_classes, average=None).to(device)
    else:
        metric_iou = torchmetrics.classification.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        ).to(device)

    best_loss: float | None = None

    try:
        for epoch in range(1, epochs + 1):
            for phase in ("train", "val"):
                is_train = phase == "train"
                model.train() if is_train else model.eval()

                loader = dataloaders[phase]
                steps = max(1, len(loader))
                metric_loss.reset()
                metric_acc.reset()
                metric_iou.reset()

                for step, (images, masks) in enumerate(loader):
                    # blue terminal color for training, cyan for validation
                    color_code = "\033[94m" if is_train else "\033[96m"
                    print(f"{color_code}(Epoch {epoch:02}/{epochs} [{phase:5}]){bar(step + 1, steps)}\033[0m", end="\r")
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    if is_train and gpu_augment is not None and device.type == "cuda":
                        # Augmentation is a data operation; keep it out of autograd.
                        with torch.no_grad():
                            images, masks = gpu_augment(images, masks)

                    if is_train:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(is_train):
                        # AMP speeds up large UNet-like models significantly on CUDA.
                        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                            logits = model(images)
                            preds = logits.argmax(dim=1)
                            loss = criterion(logits, masks)

                        metric_loss.update(loss)
                        metric_acc.update(preds, masks)
                        metric_iou.update(preds, masks)

                        if is_train:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                mean_loss = float(metric_loss.compute().detach().cpu())
                acc = float(metric_acc.compute().detach().cpu())
                iou_per_class = metric_iou.compute().detach().cpu()
                current_lr = get_lr()

                iou_metrics = {f"{phase}/iou_class_{i}": float(iou_per_class[i]) for i in range(num_classes)}
                iou_metrics[f"{phase}/iou_mean"] = float(torch.nanmean(iou_per_class).item())

                wandb.log(
                    {
                        f"{phase}/loss": mean_loss,
                        f"{phase}/accuracy": acc,
                        f"{phase}/lr": current_lr,
                        **iou_metrics,
                    },
                    step=epoch,
                )

                extra = ""
                if phase == "val" and (best_loss is None or mean_loss < best_loss):
                    extra = " (best so far, saving model...)"
                    best_loss = mean_loss
                print(f"{color_code}(Epoch {epoch:02}/{epochs} [{phase:5}]) Loss: {mean_loss:.3f} Accuracy: {acc:.3f} lr: {current_lr*10000:.4f}e-4{extra}\033[0m         ")
                if extra:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{wandb_run_name}.{random_chars}.best.pth"))

            if lr_scheduler is not None:
                lr_scheduler.step()
    finally:
        wandb.finish()
