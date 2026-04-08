from torch.utils.data import TensorDataset, DataLoader
from lars import load_split, bar
from torch import nn, optim
import torch
import torchmetrics
import wandb


def prepare_dataloaders(batch_size: int = 4, shuffle: bool = True, drop_last: bool = True) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    for split in ["train", "val"]:
        print(f"Loading {split} data...")
        images, masks, _ = load_split(split)
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
        },
    )

    metric_loss = torchmetrics.aggregation.MeanMetric().to(device)
    if ignore_index is None:
        metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    else:
        metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)

    best_acc: float | None = None

    try:
        for epoch in range(1, epochs + 1):
            for phase in ("train", "val"):
                is_train = phase == "train"
                model.train() if is_train else model.eval()

                loader = dataloaders[phase]
                steps = max(1, len(loader))
                metric_loss.reset()
                metric_acc.reset()

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

                        if is_train:
                            loss.backward()
                            optimizer.step()

                mean_loss = float(metric_loss.compute().detach().cpu())
                acc = float(metric_acc.compute().detach().cpu())
                current_lr = get_lr()

                wandb.log(
                    {
                        f"{phase}/loss": mean_loss,
                        f"{phase}/accuracy": acc,
                        f"{phase}/lr": current_lr,
                    },
                    step=epoch,
                )

                extra = ""
                if phase == "val" and (best_acc is None or acc > best_acc):
                    extra = " (best so far, saving model...)"
                    best_acc = acc
                print(f"{color_code}(Epoch {epoch:02}/{epochs} [{phase:5}]) Loss: {mean_loss:.3f} Accuracy: {acc:.3f} lr: {current_lr*10000:.4f}e-4{extra}\033[0m                           ")
                if extra:
                    torch.save(model.state_dict(), "best_model.pth")

            if lr_scheduler is not None:
                lr_scheduler.step()
    finally:
        wandb.finish()
