from lars import load_split, cycle_images
from segmentation.unet import UNet
import numpy as np
import torch
from detection import (
    build_fast_rcnn,
    prepare_detection_dataloaders,
    train_detector,
    detect_objects,
    filter_detections_by_water,
    draw_detections,
)
import cv2
import os
import wandb
from dotenv import load_dotenv

# mode = "train"
# mode = "show"
mode = "train_detect"
# mode = "pipeline"
hsv = False
h_bilateral = None
s_bilateral = None
v_bilateral = None
div = 1
shape = (1024 // div, 576 // div)

# features = (32, 32, 64, 64, 128, 256)
features = (16, 16, 32, 32, 64, 128)

augment_train = True
weighted_loss = True
batch_size = 8

seed = 48
torch.manual_seed(seed)
preload_from = None
preload_from = "best.pth"
random_chars = "".join([chr(c) for c in torch.randint(ord("A"), ord("Z") + 1, (5,), dtype=torch.uint8).cpu().numpy().tolist()])
model_name = f"model{str(features).replace(' ', '')}{'-aug' if augment_train else ''}{'-hsv(' + str(h_bilateral) + ';' + str(s_bilateral) + ';' + str(v_bilateral) + ')' if hsv else ''}{'-wl' if weighted_loss else ''}.pth"
model_name = "best.pth"  # Override for showing predictions from a specific model.

# Detection (Fast R-CNN) settings.
detector_backbone = "resnet18"  # resnet18 / resnet34 / resnet50
detector_batch_size = 4
detector_epochs = 15
detector_name = "detector.pth"
detector_score_thresh = 0.5
detector_max_proposals = 2000  # Selective Search proposals per image (Fast R-CNN has no RPN).
detector_water_filter = True  # Drop detections that do not touch the predicted water mask.


def get_device() -> torch.device:
    """Pick cuda only if this PyTorch build actually has kernels for the GPU.

    torch.cuda.is_available() returns True even for GPUs the build does not
    support (e.g. Pascal sm_61 with cu13x wheels), which then crashes on the
    first conv with "unable to find an engine". Fall back to CPU instead.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    archs = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    if archs and major * 10 + minor < min(archs):
        print(f"WARNING: {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not supported by " f"this PyTorch build (needs sm_{min(archs)}+); falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def show_sample_images():
    x, y, _ = load_split("train", hsv=hsv, reshape=shape, h_bilateral=h_bilateral, s_bilateral=s_bilateral, v_bilateral=v_bilateral)
    print(f"Images shape: {x.shape}")
    print(f"Semantic masks shape: {y.shape}")
    # Real-time HSV smoothing via sliders is handled inside show_img/cycle_images.
    cycle_images(x, y, window_name="Samples", fps=5, highlight_water=None, hsv=hsv, hsv_sliders=True, show_hsv_channels=True)


def train_fast_rcnn():
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataloaders = prepare_detection_dataloaders(batch_size=detector_batch_size, reshape=shape, max_proposals=detector_max_proposals)
    model = build_fast_rcnn(pretrained=True, backbone_name=detector_backbone, trainable_backbone_layers=3).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=detector_epochs, eta_min=1e-6)
    print(f"Training Fast R-CNN on {device} for {detector_epochs} epochs...")

    train_detector(
        wandb_project="ASO-Detection",
        wandb_run_name=detector_name.replace(".pth", ""),
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        epochs=detector_epochs,
        random_chars=random_chars,
    )
    torch.save(model.state_dict(), f"final_detector.{random_chars}.pth")


def show_pipeline():
    """Full pipeline from etap1.md 4.1: image -> U-Net segmentation -> Fast R-CNN detection."""
    x, _, _ = load_split("val", hsv=hsv)
    print(f"Images shape: {x.shape}")
    device = get_device()
    num_channels = 3  # BGR images
    num_classes = 3  # Obstacles, Water, Sky (255 is ignored via ignore_index)

    seg_model = UNet(in_channels=num_channels, out_channels=num_classes, features=features).to(device)
    seg_model.load_state_dict(torch.load(model_name, map_location=device))
    seg_model.eval()

    det_model = build_fast_rcnn(pretrained=False, backbone_name=detector_backbone).to(device)
    det_model.load_state_dict(torch.load(detector_name, map_location=device))
    det_model.eval()

    fps = 2
    water_color = (224, 167, 41)  # BGR
    for i in range(len(x)):
        xc = x[i : i + 1].to(device)
        with torch.no_grad():
            pred_mask = torch.argmax(seg_model(xc), dim=1).squeeze(0).cpu()
        water_mask = pred_mask == 1

        boxes, labels, scores = detect_objects(det_model, x[i], device, score_thresh=detector_score_thresh)
        if detector_water_filter:
            boxes, labels, scores = filter_detections_by_water(boxes, labels, scores, water_mask)

        frame = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
        overlay = np.zeros_like(frame)
        overlay[water_mask.numpy()] = water_color
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
        frame = draw_detections(frame, boxes, labels, scores)
        cv2.imshow("Pipeline: segmentation + detection", frame)
        key = cv2.waitKey(int(1000 / fps))
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def main():
    load_dotenv(dotenv_path=".env", override=False)
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    # show_sample_images()
    # evaluate()
    if mode == "train_detect":
        train_fast_rcnn()
    elif mode == "pipeline":
        show_pipeline()


if __name__ == "__main__":
    main()
