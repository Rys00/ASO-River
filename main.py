from lars import load_split, cycle_images, show_img
from unet import UNet
import torch
from segmentation import KorniaSegmentationAugment, prepare_dataloaders, train
import cv2
import os
import wandb
from dotenv import load_dotenv

# mode = "train"
mode = "show"
hsv = False
h_bilateral = None
s_bilateral = None
v_bilateral = None
div = 1
shape = (1024 // div, 576 // div)

features = (16, 16, 32, 32, 64, 128)

augment_train = True
weighted_loss = True
batch_size = 16

seed = 45
torch.manual_seed(seed)
random_chars = "".join([chr(c) for c in torch.randint(ord("A"), ord("Z") + 1, (5,), dtype=torch.uint8).cpu().numpy().tolist()])
model_name = f"model{str(features).replace(' ', '')}{'-aug' if augment_train else ''}{'-hsv(' + str(h_bilateral) + ';' + str(s_bilateral) + ';' + str(v_bilateral) + ')' if hsv else ''}{'-wl' if weighted_loss else ''}.pth"


def show_sample_images():
    x, y, _ = load_split("train", hsv=hsv, reshape=shape, h_bilateral=h_bilateral, s_bilateral=s_bilateral, v_bilateral=v_bilateral)
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    # Real-time HSV smoothing via sliders is handled inside show_img/cycle_images.
    cycle_images(x, y, window_name="Samples", fps=5, highlight_water=None, hsv=hsv, hsv_sliders=True, show_hsv_channels=True)


def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)

    gpu_augment = None
    if augment_train and device.type == "cuda":
        gpu_augment = KorniaSegmentationAugment(
            hflip_p=0.5,
            jitter_p=0.8,
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
            noise_p=0.2,
            noise_std=0.01,
            input_is_bgr=not hsv,
            input_is_hsv=hsv,
        ).to(device)

    dataloaders = prepare_dataloaders(batch_size=batch_size, reshape=shape, hsv=hsv, h_bilateral=h_bilateral, s_bilateral=s_bilateral, v_bilateral=v_bilateral)
    model = UNet(in_channels=num_channels, out_channels=num_classes, features=features, dropout_p=0.1, dropout_min_features=64, dropout_bottleneck_p=0.2).to(device)
    # Good, standard defaults for semantic segmentation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Class-balanced loss weighting (helps when obstacles are rare).
    train_masks = dataloaders["train"].dataset.tensors[1]
    if weighted_loss:
        valid = train_masks != 255
        class_counts = torch.bincount(train_masks[valid].reshape(-1), minlength=num_classes).float()
        class_weights = class_counts.sum() / (num_classes * class_counts.clamp_min(1.0))
        class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
        class_weights = class_weights.clamp(0.1, 10.0)
        print(f"Class counts: {class_counts.tolist()} | CE weights: {class_weights.tolist()}")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=class_weights.to(device)) if weighted_loss else torch.nn.CrossEntropyLoss(ignore_index=255, weight=None)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=None)
    epochs = 40
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"Training on {device} for {epochs} epochs with model {model_name}...")

    train(
        wandb_project="ASO-Segmentation",
        wandb_run_name=model_name.replace(".pth", ""),
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=epochs,
        num_classes=num_classes,
        ignore_index=255,
        gpu_augment=gpu_augment,
        random_chars=random_chars,
    )
    # save the model
    torch.save(model.state_dict(), f"final_model.{random_chars}.pth")


def show_predictions():
    x, y, _ = load_split("train", hsv=hsv)
    print(f"Images shape: {x.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)
    model = UNet(in_channels=num_channels, out_channels=num_classes, features=features).to(device)
    filename = model_name.replace(".pth", f"{'.' + random_chars if random_chars else ''}.best.pth")
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    fps = 2
    for i in range(len(x)):
        xc = x[i : i + 1].to(device)
        with torch.no_grad():
            output = model(xc)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            show_img(x[i], pred_mask, wrong=((pred_mask == 1) != (y[i] == 1)) & (y[i] != 255), hsv=hsv)
        cv2.waitKey(int(1000 / fps))
    cv2.destroyAllWindows()


def main():
    load_dotenv(dotenv_path=".env", override=False)
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    # show_sample_images()
    # evaluate()
    if mode == "train":
        train_unet()
    elif mode == "show":
        show_predictions()


if __name__ == "__main__":
    main()
