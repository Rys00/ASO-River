import cv2
import os
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def bar(i: int, n: int, size: int = 41, prc: bool = True) -> str:
    if n == 0:
        n = 1
        i = 1
    if i > n:
        i = n
    chars = "-▏▎▍▌▋▊▉█"
    bar = f"▕{chars[-1] * int(i/n*size):41}▎".replace(" ", chars[0])
    partial = chars[int(i / n * size % 1 * (len(chars) - 1))]
    bar = bar.replace(f"▕{chars[0]}", f"▕{partial}")
    bar = bar.replace(f"{chars[-1]}{chars[0]}", f"{chars[-1]}{partial}")
    if prc:
        bar += f" {i/n:.2%}"
    return bar


def load_split(
    split: Literal["train", "val", "test"],
    limit: int = None,
    inc_semantic: bool = True,
    inc_panoptic: bool = False,
    reshape: tuple[int, int] = (1024, 576),
    save=True,
    hsv=False,
    num_workers: int = None,
    save_workers: int = None,
    return_filenames: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[str]
]:
    """
    Load a split of the LARS dataset with caching support.

    Cached tensors are aligned with sorted image file names. Stale caches built
    before filename tracking (or with a different ordering) are rebuilt automatically.
    """
    reshape = (int(reshape[0]), int(reshape[1]))
    cached_dir = os.path.join(dir, "cached")
    os.makedirs(cached_dir, exist_ok=True)
    images_dir = data_paths[split]["images"]
    expected_files = sorted(os.listdir(images_dir))
    if limit is not None:
        expected_files = expected_files[:limit]

    filenames_path = os.path.join(cached_dir, f"{split}_filenames.npy")

    # Cache paths and status
    cache_info = {
        "images": {"path": os.path.join(cached_dir, f"{split}_images.npy")},
        "semantic": {"path": os.path.join(cached_dir, f"{split}_semantic.npy"), "enabled": inc_semantic},
        "panoptic": {"path": os.path.join(cached_dir, f"{split}_panoptic.npy"), "enabled": inc_panoptic},
    }
    for k, v in cache_info.items():
        v["exists"] = os.path.exists(v["path"])

    cache_order_ok = False
    if cache_info["images"]["exists"] and os.path.exists(filenames_path):
        cached_files = np.load(filenames_path, allow_pickle=True).tolist()
        cache_order_ok = cached_files == expected_files
    if cache_info["images"]["exists"] and not cache_order_ok:
        print(
            f"Stale {split} cache (filename order mismatch or missing sidecar); "
            "reloading from disk..."
        )
        for k in cache_info:
            cache_info[k]["exists"] = False

    # Result containers
    results = {k: None for k in cache_info}
    filenames: list[str] = expected_files

    # 1. Load available caches
    if cache_info["images"]["exists"]:
        print(f"Loading cached images from {cache_info['images']['path']}")
        results["images"] = torch.from_numpy(np.load(cache_info["images"]["path"])).float().div_(255.0)
        filenames = np.load(filenames_path, allow_pickle=True).tolist()

    for k in ["semantic", "panoptic"]:
        if cache_info[k]["enabled"] and cache_info[k]["exists"]:
            print(f"Loading cached {k} masks from {cache_info[k]['path']}")
            results[k] = torch.from_numpy(np.load(cache_info[k]["path"]))

    # Check if anything else needs loading from disk
    missing = [k for k, v in cache_info.items() if (k == "images" or v["enabled"]) and not v["exists"]]
    if not missing:
        print(f"All requested data for {split} split is cached.")
        if return_filenames:
            return results["images"], results["semantic"], results["panoptic"], filenames
        return results["images"], results["semantic"], results["panoptic"]

    # 2. Load missing data from disk
    files = expected_files
    n_files = len(files)

    # Pre-allocate arrays to save memory and time
    disk_data = {}
    if "images" in missing:
        disk_data["images"] = np.empty((n_files, 3, reshape[1], reshape[0]), dtype=np.uint8)
    if "semantic" in missing:
        disk_data["semantic"] = np.empty((n_files, reshape[1], reshape[0]), dtype=np.uint8)
    if "panoptic" in missing:
        disk_data["panoptic"] = np.empty((n_files, reshape[1], reshape[0]), dtype=np.uint16)

    dirs = {
        "images": images_dir,
        "semantic": os.path.join(data_paths[split]["annotations"], "semantic_masks"),
        "panoptic": os.path.join(data_paths[split]["annotations"], "panoptic_masks"),
    }

    def load_one(idx_name):
        idx, name = idx_name
        if "images" in missing:
            raw = cv2.imread(os.path.join(dirs["images"], name))
            if raw is None:
                raise FileNotFoundError(name)
            img = cv2.resize(raw, reshape)
            if hsv:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Parallel Transpose: HWC -> CHW
            disk_data["images"][idx] = img.transpose(2, 0, 1)

        for k in ["semantic", "panoptic"]:
            if k in missing:
                mask_path = os.path.join(dirs[k], name.replace(".jpg", ".png"))
                raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise FileNotFoundError(mask_path)
                disk_data[k][idx] = cv2.resize(raw, reshape, interpolation=cv2.INTER_NEAREST)

    print(f"Loading {n_files} samples from disk...")
    # Parallel load
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, _ in enumerate(ex.map(load_one, enumerate(files))):
            print(f"Progress: {i+1:04}/{n_files:04} {bar(i+1, n_files)}", end="\r")
    print()

    # 3. Update results
    print("Converting loaded data to tensors...", end="\r")
    if "images" in missing:
        results["images"] = torch.from_numpy(disk_data["images"]).float().div_(255.0)

    for k in ["semantic", "panoptic"]:
        if k in missing:
            results[k] = torch.from_numpy(disk_data[k])
    print("Created all data tensors. " + " " * 50)

    # 4. Save new caches
    if save:
        save_tasks = []
        if "images" in missing:
            save_tasks.append((cache_info["images"]["path"], disk_data["images"]))
        for k in ["semantic", "panoptic"]:
            if k in missing:
                save_tasks.append((cache_info[k]["path"], disk_data[k]))
        if "images" in missing:
            save_tasks.append((filenames_path, np.array(files, dtype=object)))

        if save_tasks:
            print(f"Saving {len(save_tasks)} components to disk...")
            with ThreadPoolExecutor(max_workers=save_workers or 1) as ex:
                list(ex.map(lambda t: np.save(t[0], t[1]), save_tasks))

    print(f"Successfully loaded {n_files} samples.")
    if return_filenames:
        return results["images"], results["semantic"], results["panoptic"], files
    return results["images"], results["semantic"], results["panoptic"]


def show_img(
    img: torch.Tensor,
    semantic: torch.Tensor,
    window_name: str = "Image",
    highlight_water: tuple[int, int, int] | None = (41, 167, 224),
    wrong: torch.Tensor | None = None,
    hsv: bool = False,
):
    """
    Show an image with its semantic mask overlaid.

    Args:
        img: A tensor of shape (C, H, W) containing the image to show.
        semantic: A tensor of shape (H, W) containing the semantic mask for the image.
        window_name: The name of the window to show the image in.
        highlight_water: The color to highlight water areas in RGB format.
        wrong: Optional mask of misclassified pixels to highlight in red.
        hsv: Whether the input image is in HSV color space.
    """
    # Convert NCHW [0, 1] float tensor to HWC [0, 255] uint8 numpy
    frame = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    semantic_np = semantic.detach().cpu().numpy() if isinstance(semantic, torch.Tensor) else np.asarray(semantic)

    if hsv:
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    if highlight_water is not None:
        water_id = [k for k, v in SEMANTIC.items() if v == "Water"][0]
        water_mask = semantic_np == water_id
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[water_mask] = highlight_water[::-1]  # RGB to BGR
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)

    if wrong is not None:
        wrong_np = wrong.detach().cpu().numpy() if isinstance(wrong, torch.Tensor) else np.asarray(wrong)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[wrong_np.astype(bool)] = (0, 0, 255)  # Red for errors
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.75, 0)

    cv2.imshow(window_name, frame)


def cycle_images(
    images: torch.Tensor,
    semantic: torch.Tensor,
    window_name: str = "Image",
    fps: float = 30,
    highlight_water: tuple[int, int, int] | None = (41, 167, 224),
    hsv: bool = False,
):
    """
    Cycle through a list of images.
    """
    wait_ms = int(1000.0 / float(fps)) if fps > 0 else 0
    for i in range(len(images)):
        show_img(
            images[i],
            semantic[i],
            window_name=window_name,
            highlight_water=highlight_water,
            hsv=hsv,
        )
        if cv2.waitKey(wait_ms) in (27, ord("q")):
            break
    cv2.destroyAllWindows()
