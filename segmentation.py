from torch.utils.data import TensorDataset, DataLoader
from lars import load_split, bar
from torch import nn, optim
import torch
import torchmetrics
import wandb
from torch.utils.data import Dataset
import torchvision.transforms as TV


class SegmentationAugment:
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
    ):
        self.hflip_p = hflip_p
        self.jitter_p = jitter_p
        self.noise_p = noise_p
        self.noise_std = noise_std
        self.jitter = TV.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: (C,H,W) float in [0,1]
        # mask: (H,W) long
        if torch.rand(()) < self.hflip_p:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if torch.rand(()) < self.jitter_p:
            image = self.jitter(image)

        if torch.rand(()) < self.noise_p:
            image = (image + torch.randn_like(image) * self.noise_std).clamp(0.0, 1.0)

        return image, mask


class AugmentedSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, augment: SegmentationAugment | None = None):
        self.images = images
        self.masks = masks
        self.augment = augment
        # Compatibility with TensorDataset users.
        self.tensors = (images, masks)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return image, mask


def prepare_dataloaders(
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    augment_train: bool = True,
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    train_augment = SegmentationAugment() if augment_train else None
    for split in ["train", "val"]:
        print(f"Loading {split} data...")
        images, masks, _ = load_split(split)
        images = images.float()
        masks = masks.long()
        print(f"Preparing dataloader for {split} split...")
        if split == "train" and train_augment is not None:
            dataset = AugmentedSegmentationDataset(images, masks, augment=train_augment)
        else:
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
):
    def get_lr() -> float | None:
        if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
            lr_list = lr_scheduler.get_last_lr()
            return float(lr_list[0]) if lr_list else None
        return optimizer.param_groups[0].get("lr", None)

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

                    if is_train:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(is_train):
                        logits = model(images)
                        preds = logits.argmax(dim=1)
                        loss = criterion(logits, masks)

                        metric_loss.update(loss)
                        metric_acc.update(preds, masks)
                        metric_iou.update(preds, masks)

                        if is_train:
                            loss.backward()
                            optimizer.step()

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
                print(f"{color_code}(Epoch {epoch:02}/{epochs} [{phase:5}]) Loss: {mean_loss:.3f} Accuracy: {acc:.3f} lr: {current_lr*10000:.4f}e-4{extra}\033[0m                           ")
                if extra:
                    torch.save(model.state_dict(), "best_model.pth")

            if lr_scheduler is not None:
                lr_scheduler.step()
    finally:
        wandb.finish()
