from torch.utils.data import TensorDataset, DataLoader
from lars import load_split, bar
from torch import nn, optim
import torch


def prepare_dataloaders(batch_size: int = 4, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    dataloaders = {}
    for split in ["train", "val"]:
        print(f"Loading {split} data...")
        images, masks, _ = load_split(split)
        images = images.float()
        masks = masks.long()
        print(f"Preparing dataloader for {split} split...")
        dataset = TensorDataset(images, masks)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=2,
            pin_memory=True,
        )  # pick what fits your GPU  # often useful for stable batch sizes
        dataloaders[split] = loader
    return dataloaders


def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
):
    model.train()
    print("Starting training...")
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, (images, masks) in enumerate(dataloaders["train"]):
            print(f"Epoch [{epoch+1}/{epochs}]: {'train':>10} {bar(step, len(dataloaders['train']))}", end="\r")
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast_ctx():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / max(1, grad_accum_steps)

            scaler.scale(loss).backward()
            running_loss += loss.item() * max(1, grad_accum_steps)

            if (step + 1) % max(1, grad_accum_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if grad_accum_steps > 1 and (len(dataloaders["train"]) % grad_accum_steps) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = running_loss / max(1, len(dataloaders["train"]) / grad_accum_steps)
        model.eval()
        avg_val_loss = 0.0
        for step, (images, masks) in enumerate(dataloaders["val"]):
            print(f"Epoch [{epoch+1}/{epochs}]: {'val':>10} {bar(step, len(dataloaders['val']))}", end="\r")
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad(), autocast_ctx():
                outputs = model(images)
                loss = criterion(outputs, masks)
                avg_val_loss += loss.item() * images.size(0)

        avg_val_loss = avg_val_loss / len(dataloaders["val"].dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Train loss: {avg_train_loss:.4f} Val loss: {avg_val_loss:.4f}                             ")
