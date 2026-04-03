import cv2
import os
from typing import Literal
import numpy as np

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
    split: Literal["train", "val", "test"], limit: int = None, inc_semantic: bool = True, inc_panoptic: bool = False, reshape: tuple[int, int] = (1920, 1080), save=True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a split of the LARS dataset.

    Args:
        split: Which split to load ("train", "val", or "test").
        limit: Maximum number of samples to load (default: None, which loads all).
        inc_semantic: Whether to include semantic masks (default: True).
        inc_panoptic: Whether to include panoptic masks (default: False).

    Returns:
        A tuple of (images, semantic_masks, panoptic_masks), where each is a numpy array.
    """
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
        images = np.load(cached_img)
    semantic = [] if inc_semantic else None
    if inc_semantic and is_cached_semantic:
        print(f"Loading cached semantic masks from {cached_semantic}")
        semantic = np.load(cached_semantic)
    panoptic = [] if inc_panoptic else None
    if inc_panoptic and is_cached_panoptic:
        print(f"Loading cached panoptic masks from {cached_panoptic}")
        panoptic = np.load(cached_panoptic)
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
    print(f"Converting samples to numpy arrays {bar(i, end)}", end="\r")
    if not is_cached_img:
        images = np.array(images)
        i += 1
        print(f"Converting samples to numpy arrays {bar(i, end)}", end="\r")
    if inc_semantic and not is_cached_semantic:
        semantic = np.array(semantic)
        i += 1
        print(f"Converting samples to numpy arrays {bar(i, end)}", end="\r")
    if inc_panoptic and not is_cached_panoptic:
        panoptic = np.array(panoptic)
        i += 1
        print(f"Converting samples to numpy arrays {bar(i, end)}", end="\r")
    print()
    if save:
        i = 0
        print(f"Saving numpy arrays to disk {bar(i, end)}", end="\r")
        np.save(cached_img, images)
        i += 1
        print(f"Saving numpy arrays to disk {bar(i, end)}", end="\r")
        if inc_semantic:
            np.save(cached_semantic, semantic)
            i += 1
            print(f"Saving numpy arrays to disk {bar(i, end)}", end="\r")
        if inc_panoptic:
            np.save(cached_panoptic, panoptic)
            i += 1
            print(f"Saving numpy arrays to disk {bar(i, end)}", end="\r")
    print(f"\nSuccessfully loaded {len(images)} samples from {split} split.")

    return images, semantic, panoptic
