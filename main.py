from lars import load_split, cycle_images, show_img
from unet import UNet
import torch
from segmentation import prepare_dataloaders, train
import cv2


def show_sample_images():
    x, y, _ = load_split("train")
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    cycle_images(x, y, fps=30)


def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)
    dataloaders = prepare_dataloaders(batch_size=4)
    model = UNet(in_channels=num_channels, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    train(model, dataloaders, optimizer, criterion, device)
    # save the model
    torch.save(model.state_dict(), "unet_model.pth")


def show_predictions():
    x, y, _ = load_split("train")
    print(f"Images shape: {x.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)
    model = UNet(in_channels=num_channels, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
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
    # show_sample_images()
    train_unet()
    # evaluate()
    # show_predictions()


if __name__ == "__main__":
    main()
