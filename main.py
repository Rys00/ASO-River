from lars import load_split, cycle_images
from unet import UNet
import torch


def main():
    x, y, _ = load_split("train", limit=None, reshape=(1024, 576), inc_semantic=True, inc_panoptic=False, save=True)
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    cycle_images(x, y, fps=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(in_channels=3, out_channels=2, features=(64, 128, 256, 512))
    model.to(device)
    x = x[:1].to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
