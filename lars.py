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
    h_bilateral: int | None = None,
    s_bilateral: int | None = None,
    v_bilateral: int | None = None,
    num_workers: int = 16,
    save_workers: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a split of the LARS dataset.

    Args:
        split: Which split to load ("train", "val", or "test").
        limit: Maximum number of samples to load (default: None, which loads all).
        inc_semantic: Whether to include semantic masks (default: True).
        inc_panoptic: Whether to include panoptic masks (default: False).
        reshape: Tuple of (width, height) to resize images and masks to (default: (1024, 576)).
        save: Whether to save the loaded tensors to disk for faster loading next time (default: True).
        hsv: Whether to convert images to HSV color space (default: False).

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
    images_cache_u8 = None
    if is_cached_img:
        print(f"Loading cached images from {cached_img}")
        images = torch.from_numpy(np.load(cached_img)).to(torch.float32).div_(255.0)
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
    files = files[:end]

    need_img = not is_cached_img
    need_sem = inc_semantic and not is_cached_semantic
    need_pan = inc_panoptic and not is_cached_panoptic

    def load_one(img_name: str):
        img_out = None
        sem_out = None
        pan_out = None
        img_path = os.path.join(images_dir, img_name)
        panoptic_path = os.path.join(panoptic_dir, img_name.replace(".jpg", ".png"))
        semantic_path = os.path.join(semantic_dir, img_name.replace(".jpg", ".png"))

        if need_img:
            raw = cv2.imread(img_path)
            if raw is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            img_local = cv2.resize(raw, reshape)
            if hsv:
                img_local = cv2.cvtColor(img_local, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img_local)
                if h_bilateral is not None and h_bilateral > 1:
                    h = cv2.bilateralFilter(h, d=int(h_bilateral), sigmaColor=float(h_bilateral) * 2.0, sigmaSpace=float(h_bilateral) * 2.0)
                if s_bilateral is not None and s_bilateral > 1:
                    s = cv2.bilateralFilter(s, d=int(s_bilateral), sigmaColor=float(s_bilateral) * 2.0, sigmaSpace=float(s_bilateral) * 2.0)
                if v_bilateral is not None and v_bilateral > 1:
                    v = cv2.bilateralFilter(v, d=int(v_bilateral), sigmaColor=float(v_bilateral) * 2.0, sigmaSpace=float(v_bilateral) * 2.0)
                img_local = cv2.merge((h, s, v))
            img_out = img_local

        if need_sem:
            sem_raw = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            if sem_raw is None:
                raise FileNotFoundError(f"Failed to read semantic mask: {semantic_path}")
            sem_out = cv2.resize(sem_raw, reshape, interpolation=cv2.INTER_NEAREST)

        if need_pan:
            pan_raw = cv2.imread(panoptic_path, cv2.IMREAD_UNCHANGED)
            if pan_raw is None:
                raise FileNotFoundError(f"Failed to read panoptic mask: {panoptic_path}")
            pan_out = cv2.resize(pan_raw, reshape, interpolation=cv2.INTER_NEAREST)

        return img_out, sem_out, pan_out

    use_threads = int(num_workers) if num_workers is not None else 0
    if use_threads > 0 and (need_img or need_sem or need_pan):
        with ThreadPoolExecutor(max_workers=use_threads) as ex:
            for i, (img_out, sem_out, pan_out) in enumerate(ex.map(load_one, files)):
                print(f"Loading {split} sample {i+1:04}/{end:04} {bar(i+1, end)}", end="\r")
                if need_img:
                    images.append(img_out)
                if need_sem:
                    semantic.append(sem_out)
                if need_pan:
                    panoptic.append(pan_out)
    else:
        for i, img_name in enumerate(files):
            print(f"Loading {split} sample {i+1:04}/{end:04} {bar(i+1, end)}", end="\r")
            img_out, sem_out, pan_out = load_one(img_name)
            if need_img:
                images.append(img_out)
            if need_sem:
                semantic.append(sem_out)
            if need_pan:
                panoptic.append(pan_out)
    print()
    end = int(inc_semantic and not is_cached_semantic) + int(inc_panoptic and not is_cached_panoptic) + int(not is_cached_img)
    i = 0
    print(f"Converting samples to tensors {bar(i, end)}", end="\r")
    if not is_cached_img:
        # Cache in compact uint8 NCHW, but return float32 in [0, 1].
        images_cache_u8 = np.asarray(images, dtype=np.uint8).transpose(0, 3, 1, 2)  # NCHW
        images = torch.from_numpy(images_cache_u8).to(torch.float32).div_(255.0)
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
        save_items: list[tuple[str, np.ndarray]] = []
        if not is_cached_img and images_cache_u8 is not None:
            save_items.append((cached_img, images_cache_u8))
        if inc_semantic and not is_cached_semantic:
            save_items.append((cached_semantic, semantic.detach().cpu().numpy()))
        if inc_panoptic and not is_cached_panoptic:
            save_items.append((cached_panoptic, panoptic.detach().cpu().numpy()))

        save_end = max(1, len(save_items))
        i = 0
        print(f"Saving tensors to disk {bar(i, save_end)}", end="\r")

        if save_items:
            use_save_threads = int(save_workers) if save_workers is not None else 0
            if use_save_threads and use_save_threads > 0 and len(save_items) > 1:
                with ThreadPoolExecutor(max_workers=use_save_threads) as ex:
                    futures = [ex.submit(np.save, path, arr) for path, arr in save_items]
                    for fut in as_completed(futures):
                        fut.result()
                        i += 1
                        print(f"Saving tensors to disk {bar(i, save_end)}", end="\r")
            else:
                for path, arr in save_items:
                    np.save(path, arr)
                    i += 1
                    print(f"Saving tensors to disk {bar(i, save_end)}", end="\r")
    print(f"\nSuccessfully loaded {len(images)} samples from {split} split.")

    return images, semantic, panoptic


def show_img(
    img: torch.Tensor,
    semantic: torch.Tensor,
    window_name: str = "Image",
    highlight_water: tuple[int, int, int] | None = (41, 167, 224),
    wrong: torch.Tensor | None = None,
    hsv: bool = False,
    hsv_smooth_ksize: tuple[int, int, int] | None = None,
    hsv_sliders: bool = False,
    hsv_sliders_max_ksize: int = 51,
    show_hsv_channels: bool = False,
    hold_ms: int | None = None,
    refresh_ms: int = 15,
):
    """
    Show an image with its semantic mask overlaid.

    Args:
        img: A tensor of shape (C, H, W) containing the image to show.
        semantic: A tensor of shape (H, W) containing the semantic mask for the image.
        window_name: The name of the window to show the image in (default: "Image").
        highlight_water: The color to highlight water areas in RGB format (default: (41, 167, 224)). If None, water areas will not be highlighted.
    """
    base = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    semantic_np = semantic.detach().cpu().numpy() if isinstance(semantic, torch.Tensor) else np.asarray(semantic)
    wrong_np = (wrong.detach().cpu().numpy() if isinstance(wrong, torch.Tensor) else np.asarray(wrong)) if wrong is not None else None

    water_mask = None
    if highlight_water is not None:
        water_id = [k for k, v in SEMANTIC.items() if v == "Water"][0]
        water_mask = semantic_np == water_id

    def to_odd_ksize(v: int, max_ksize: int) -> int:
        max_ksize = int(max_ksize)
        max_ksize = max_ksize if (max_ksize % 2 == 1) else (max_ksize - 1)
        max_ksize = max(1, max_ksize)
        if v <= 0:
            return 1
        k = int(round((v / 255.0) * max_ksize))
        k = max(1, min(max_ksize, k))
        if k % 2 == 0:
            k = k + 1 if k < max_ksize else k - 1
        return max(1, k)

    def ensure_hsv_trackbars():
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.getTrackbarPos("H", window_name)
            cv2.getTrackbarPos("S", window_name)
            cv2.getTrackbarPos("V", window_name)
        except cv2.error:
            cv2.createTrackbar("H", window_name, 50, 255, lambda _: None)
            cv2.createTrackbar("S", window_name, 60, 255, lambda _: None)
            cv2.createTrackbar("V", window_name, 30, 255, lambda _: None)

    def current_smoothing_from_sliders() -> tuple[int, int, int] | None:
        try:
            h_val = cv2.getTrackbarPos("H", window_name)
            s_val = cv2.getTrackbarPos("S", window_name)
            v_val = cv2.getTrackbarPos("V", window_name)
        except cv2.error:
            return None
        return (
            to_odd_ksize(h_val, hsv_sliders_max_ksize),
            to_odd_ksize(s_val, hsv_sliders_max_ksize),
            to_odd_ksize(v_val, hsv_sliders_max_ksize),
        )

    def render_frame(smooth_ksize: tuple[int, int, int] | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        frame = base
        h_ch = s_ch = v_ch = None
        if hsv and smooth_ksize is not None:
            kh, ks, kv = smooth_ksize
            h, s, v = cv2.split(frame)
            if kh > 1:
                # Bilateral filter preserves edges better than Gaussian blur.
                h = cv2.bilateralFilter(h, d=int(kh), sigmaColor=float(kh) * 2.0, sigmaSpace=float(kh) * 2.0)
            if ks > 1:
                s = cv2.bilateralFilter(s, d=int(ks), sigmaColor=float(ks) * 2.0, sigmaSpace=float(ks) * 2.0)
            if kv > 1:
                v = cv2.bilateralFilter(v, d=int(kv), sigmaColor=float(kv) * 2.0, sigmaSpace=float(kv) * 2.0)
            h_ch, s_ch, v_ch = h, s, v
            frame = cv2.merge((h, s, v))
        elif hsv:
            h, s, v = cv2.split(frame)
            h_ch, s_ch, v_ch = h, s, v
        if hsv:
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        if highlight_water is not None and water_mask is not None:
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[water_mask] = highlight_water[::-1]  # Convert RGB to BGR for OpenCV
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.75, 0)
        if wrong_np is not None:
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[wrong_np.astype(bool)] = (0, 0, 255)
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.75, 0)
        return frame, h_ch, s_ch, v_ch

    if hsv_sliders and hsv:
        ensure_hsv_trackbars()

    def smoothing_now() -> tuple[int, int, int] | None:
        if hsv_sliders and hsv:
            return current_smoothing_from_sliders()
        return hsv_smooth_ksize

    def show_channels(h_ch: np.ndarray | None, s_ch: np.ndarray | None, v_ch: np.ndarray | None):
        if not (show_hsv_channels and hsv):
            return
        if h_ch is None or s_ch is None or v_ch is None:
            return
        # OpenCV HSV uses H in [0, 179]. Scale to full grayscale for visualization.
        h_view = (h_ch.astype(np.float32) * (255.0 / 179.0)).clip(0, 255).astype(np.uint8)
        cv2.imshow(f"{window_name} - H", h_view)
        cv2.imshow(f"{window_name} - S", s_ch)
        cv2.imshow(f"{window_name} - V", v_ch)

    if hold_ms is None:
        frame, h_ch, s_ch, v_ch = render_frame(smoothing_now())
        cv2.imshow(window_name, frame)
        show_channels(h_ch, s_ch, v_ch)
        return

    # Re-render the same image for hold_ms, so slider changes update it live.
    t0 = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    while True:
        frame, h_ch, s_ch, v_ch = render_frame(smoothing_now())
        cv2.imshow(window_name, frame)
        show_channels(h_ch, s_ch, v_ch)
        key = cv2.waitKey(max(1, int(refresh_ms)))
        if key in (27, ord("q")):
            break
        elapsed_ms = (cv2.getTickCount() - t0) * 1000.0 / freq
        if elapsed_ms >= hold_ms:
            break


def cycle_images(
    images: torch.Tensor,
    semantic: torch.Tensor,
    window_name: str = "Image",
    fps: float = 30,
    highlight_water: tuple[int, int, int] | None = (41, 167, 224),
    hsv: bool = False,
    hsv_smooth_ksize: tuple[int, int, int] | None = None,
    hsv_smooth_ksize_fn=None,
    hsv_sliders: bool = False,
    hsv_sliders_max_ksize: int = 51,
    show_hsv_channels: bool = False,
    refresh_ms: int = 15,
):
    """
    Cycle through a list of images, showing each one for a short period of time.

    Args:
        images: A tensor of shape (N, C, H, W) containing the images to show.
        semantic: A tensor of shape (N, H, W) containing the semantic masks for the images.
        window_name: The name of the window to show the images in (default: "Image").
        fps: The frames per second to cycle through the images at (default: 30).
        highlight_water: The color to highlight water areas in RGB format (default: (41, 167, 224)). If None, water areas will not be highlighted.
    """
    wait_ms = int(1000.0 / float(fps)) if fps and float(fps) > 0 else 0
    for i in range(len(images)):
        current_ksize = hsv_smooth_ksize_fn() if hsv_smooth_ksize_fn is not None else hsv_smooth_ksize
        show_img(
            images[i],
            semantic[i],
            window_name=window_name,
            highlight_water=highlight_water,
            hsv=hsv,
            hsv_smooth_ksize=current_ksize,
            hsv_sliders=hsv_sliders,
            hsv_sliders_max_ksize=hsv_sliders_max_ksize,
            show_hsv_channels=show_hsv_channels,
            hold_ms=wait_ms,
            refresh_ms=refresh_ms,
        )
    cv2.destroyAllWindows()
