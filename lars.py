import cv2
import os
from typing import Literal
import numpy as np
import torch

dir = os.path.join(os.getcwd(), "data", "lars_v1.0.0")
data_paths = {}
for split in ["train", "val", "test"]:
    data_paths[split] = {
        "images": os.path.join(dir, "images", split, "images"),
        "annotations": os.path.join(dir, "annotations", split),
    }

PANOPTIC = {
    1: {"name": "Static Obstacle", "type": "Stuff", "supercategory": "obstacle"},
    3: {"name": "Water", "type": "Stuff", "supercategory": "water"},
    5: {"name": "Sky", "type": "Stuff", "supercategory": "sky"},
    11: {"name": "Boat/ship", "type": "Thing", "supercategory": "obstacle"},
    12: {"name": "Row boats", "type": "Thing", "supercategory": "obstacle"},
    13: {"name": "Paddle board", "type": "Thing", "supercategory": "obstacle"},
    14: {"name": "Buoy", "type": "Thing", "supercategory": "obstacle"},
    15: {"name": "Swimmer", "type": "Thing", "supercategory": "obstacle"},
    16: {"name": "Animal", "type": "Thing", "supercategory": "obstacle"},
    17: {"name": "Float", "type": "Thing", "supercategory": "obstacle"},
    19: {"name": "Other", "type": "Thing", "supercategory": "obstacle"},
}

SEMANTIC: dict[int, str] = {
    0: "Obstacles",
    1: "Water",
    2: "Sky",
    255: "Ignore",
}


def bar(i: int, n: int, size: int = 40, prc: bool = True) -> str:
    if n == 0:
        n = 1
        i = 1
    if i > n:
        i = n
    chars = "-▏▎▍▌▋▊▉█"
    bar = f"▕{chars[-1] * int(i/n*size):40}▎".replace(" ", chars[0])
    partial = chars[int(i / n * size % 1 * (len(chars) - 1))]
    bar = bar.replace(f"▕{chars[0]}", f"▕{partial}")
    bar = bar.replace(f"{chars[-1]}{chars[0]}", f"{chars[-1]}{partial}")
    if prc:
        bar += f" {i/n:.2%}"
    return bar


def load_split(
    split: Literal["train", "val", "test"], limit: int = None, inc_semantic: bool = True, inc_panoptic: bool = False, reshape: tuple[int, int] = (1024, 576), save=True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a split of the LARS dataset.

    Args:
        split: Which split to load ("train", "val", or "test").
        limit: Maximum number of samples to load (default: None, which loads all).
        inc_semantic: Whether to include semantic masks (default: True).
        inc_panoptic: Whether to include panoptic masks (default: False).
        reshape: Tuple of (width, height) to resize images and masks to (default: (1024, 576)).

    Returns:
        A tuple containing:
        - images: A tensor of shape (N, C, H, W) containing the loaded images.
        - semantic: A tensor of shape (N, H, W) containing the semantic masks (or None if inc_semantic is False).
        - panoptic: A tensor of shape (N, H, W, C) containing the panoptic masks (or None if inc_panoptic is False).
    """
    reshape = (int(reshape[0]), int(reshape[1]))
    cached_dir = os.path.join(dir, "cached")
    os.makedirs(cached_dir, exist_ok=True)
    cached_img = os.path.join(cached_dir, f"{split}_images.npy")
    cached_semantic = os.path.join(cached_dir, f"{split}_semantic.npy")
    cached_panoptic = os.path.join(cached_dir, f"{split}_panoptic.npy")
    is_cached_img = os.path.exists(cached_img)
    is_cached_semantic = os.path.exists(cached_semantic)
    is_cached_panoptic = os.path.exists(cached_panoptic)
    images_dir = data_paths[split]["images"]
    panoptic_dir = os.path.join(data_paths[split]["annotations"], "panoptic_masks")
    semantic_dir = os.path.join(data_paths[split]["annotations"], "semantic_masks")
    images = []
    if is_cached_img:
        print(f"Loading cached images from {cached_img}")
        images = torch.from_numpy(np.load(cached_img))
    semantic = [] if inc_semantic else None
    if inc_semantic and is_cached_semantic:
        print(f"Loading cached semantic masks from {cached_semantic}")
        semantic = torch.from_numpy(np.load(cached_semantic))
    panoptic = [] if inc_panoptic else None
    if inc_panoptic and is_cached_panoptic:
        print(f"Loading cached panoptic masks from {cached_panoptic}")
        panoptic = torch.from_numpy(np.load(cached_panoptic))
    if not (not is_cached_img or (inc_semantic and not is_cached_semantic) or (inc_panoptic and not is_cached_panoptic)):
        print(f"All data for {split} split is cached. Skipping loading from disk.")
        return images, semantic, panoptic
    files = os.listdir(images_dir)
    end = len(files) if limit is None else min(len(files), limit)
    for i, img_name in enumerate(files):
        print(f"Loading {split} sample {i+1:04}/{end:04} {bar(i+1, end)}", end="\r")
        if limit is not None and i >= limit:
            break
        img_path = os.path.join(images_dir, img_name)
        panoptic_path = os.path.join(panoptic_dir, img_name.replace(".jpg", ".png"))
        semantic_path = os.path.join(semantic_dir, img_name.replace(".jpg", ".png"))
        if not is_cached_img:
            images.append(cv2.resize(cv2.imread(img_path), reshape))
        if inc_semantic and not is_cached_semantic:
            semantic.append(cv2.resize(cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED), reshape, interpolation=cv2.INTER_NEAREST))
        if inc_panoptic and not is_cached_panoptic:
            panoptic.append(cv2.resize(cv2.imread(panoptic_path, cv2.IMREAD_UNCHANGED), reshape, interpolation=cv2.INTER_NEAREST))
    print()
    end = int(inc_semantic and not is_cached_semantic) + int(inc_panoptic and not is_cached_panoptic) + int(not is_cached_img)
    i = 0
    print(f"Converting samples to tensors {bar(i, end)}", end="\r")
    if not is_cached_img:
        images = np.array(images).astype(np.float32) / 255.0  # Numpy array of shape (N, H, W, C)
        images = images.transpose(0, 3, 1, 2)  # Convert to shape (N, C, H, W)
        images = torch.from_numpy(images)  # Convert to PyTorch tensor
        i += 1
        print(f"Converting samples to tensors {bar(i, end)}", end="\r")
    if inc_semantic and not is_cached_semantic:
        semantic = np.array(semantic)
        semantic = torch.from_numpy(semantic)  # Convert to PyTorch tensor
        i += 1
        print(f"Converting samples to tensors {bar(i, end)}", end="\r")
    if inc_panoptic and not is_cached_panoptic:
        panoptic = np.array(panoptic)
        panoptic = torch.from_numpy(panoptic)  # Convert to PyTorch tensor
        i += 1
        print(f"Converting samples to tensors {bar(i, end)}", end="\r")
    print()
    if save:
        i = 0
        print(f"Saving tensors to disk {bar(i, end)}", end="\r")
        np.save(cached_img, images.detach().cpu().numpy())
        i += 1
        print(f"Saving tensors to disk {bar(i, end)}", end="\r")
        if inc_semantic:
            np.save(cached_semantic, semantic.detach().cpu().numpy())
            i += 1
            print(f"Saving tensors to disk {bar(i, end)}", end="\r")
        if inc_panoptic:
            np.save(cached_panoptic, panoptic.detach().cpu().numpy())
            i += 1
            print(f"Saving tensors to disk {bar(i, end)}", end="\r")
    print(f"\nSuccessfully loaded {len(images)} samples from {split} split.")

    return images, semantic, panoptic


def show_img(img: torch.Tensor, semantic: torch.Tensor, window_name: str = "Image", highlight_water: tuple[int, int, int] | None = (41, 167, 224), wrong: torch.Tensor | None = None):
    """
    Show an image with its semantic mask overlaid.

    Args:
        img: A tensor of shape (C, H, W) containing the image to show.
        semantic: A tensor of shape (H, W) containing the semantic mask for the image.
        window_name: The name of the window to show the image in (default: "Image").
        highlight_water: The color to highlight water areas in RGB format (default: (41, 167, 224)). If None, water areas will not be highlighted.
    """
    img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    if highlight_water is not None:
        water_id = [k for k, v in SEMANTIC.items() if v == "Water"][0]
        water = semantic == water_id
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[water] = highlight_water[::-1]  # Convert RGB to BGR for OpenCV
        img = cv2.addWeighted(img, 1.0, overlay, 0.75, 0)
    if wrong is not None:
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[wrong] = (0, 0, 255)  # Red color for wrong predictions
        img = cv2.addWeighted(img, 1.0, overlay, 0.75, 0)
    cv2.imshow(window_name, img)


def cycle_images(images: torch.Tensor, semantic: torch.Tensor, window_name: str = "Image", fps: int = 30, highlight_water: tuple[int, int, int] | None = (41, 167, 224)):
    """
    Cycle through a list of images, showing each one for a short period of time.

    Args:
        images: A tensor of shape (N, C, H, W) containing the images to show.
        semantic: A tensor of shape (N, H, W) containing the semantic masks for the images.
        window_name: The name of the window to show the images in (default: "Image").
        fps: The frames per second to cycle through the images at (default: 30).
        highlight_water: The color to highlight water areas in RGB format (default: (41, 167, 224)). If None, water areas will not be highlighted.
    """
    for i in range(len(images)):
        show_img(images[i], semantic[i], window_name, highlight_water)
        cv2.waitKey(int(1000 / fps))
    cv2.destroyAllWindows()
