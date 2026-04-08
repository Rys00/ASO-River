from lars import load_split, cycle_images, show_img
from unet import UNet
import torch
from segmentation import prepare_dataloaders, train
import cv2
import os
import wandb
from dotenv import load_dotenv


def show_sample_images():
    x, y, _ = load_split("train")
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    cycle_images(x, y, fps=30)


features = (16, 32, 16, 32, 64, 128)
model_name = f"model{str(features).replace(' ', '')}-wl-aug.pth"


def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)
    dataloaders = prepare_dataloaders(batch_size=8)
    model = UNet(in_channels=num_channels, out_channels=num_classes, features=features, dropout_p=0.1, dropout_min_features=32, dropout_bottleneck_p=0.2).to(device)
    # Good, standard defaults for semantic segmentation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # Class-balanced loss weighting (helps when obstacles are rare).
    train_masks = dataloaders["train"].dataset.tensors[1]
    valid = train_masks != 255
    class_counts = torch.bincount(train_masks[valid].reshape(-1), minlength=num_classes).float()
    class_weights = class_counts.sum() / (num_classes * class_counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
    class_weights = class_weights.clamp(0.1, 10.0)
    print(f"Class counts: {class_counts.tolist()} | CE weights: {class_weights.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=None)  # class_weights.to(device))
    epochs = 20
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train(
        wandb_project="ASO-Segmentation",
        wandb_run_name=f"{features}-features-unet-augmented",
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=epochs,
        num_classes=num_classes,
        ignore_index=255,
    )
    # save the model
    torch.save(model.state_dict(), "final_model.pth")


def show_predictions():
    x, y, _ = load_split("train")
    print(f"Images shape: {x.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)
    model = UNet(in_channels=num_channels, out_channels=num_classes, features=features).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    fps = 2
    for i in range(len(x)):
        xc = x[i : i + 1].to(device)
        with torch.no_grad():
            output = model(xc)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            show_img(x[i], pred_mask, wrong=((pred_mask == 1) != (y[i] == 1)) & (y[i] != 255))
        cv2.waitKey(int(1000 / fps))
    cv2.destroyAllWindows()


def main():
    load_dotenv(dotenv_path=".env", override=False)
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    # show_sample_images()
    train_unet()
    # evaluate()
    # show_predictions()


if __name__ == "__main__":
    main()
