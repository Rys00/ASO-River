import os
import sys
from pathlib import Path

# Fix for Qt and Wayland on Linux systems.
# opencv-python's bundled Qt doesn't support Wayland natively.
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path to allow absolute imports from scripts in subfolders
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from lars import load_split, show_img


def segment_water(image, process_dim=256, sp=32, sr=128, k=3):
    """
    Segments water from an image using Mean-Shift and K-Means.

    Parameters:
    - process_dim: Max dimension to resize to for speed. Smaller = faster but less precise edges.
    - sp: Spatial window radius for Mean-Shift. Higher = smoother spatial regions.
    - sr: Color window radius for Mean-Shift. Higher = groups a wider range of colors together.
    - k: Number of internal clusters. 3 is recommended to separate Sky, Water, and Land.
    """
    original_h, original_w = image.shape[:2]

    # 1. SPEED UP: Downscale the image for processing
    scale = process_dim / max(original_h, original_w)
    if scale < 1.0:
        proc_img = cv2.resize(image, (int(original_w * scale), int(original_h * scale)))
    else:
        proc_img = image.copy()

    h, w = proc_img.shape[:2]

    # 2. FILTER: Apply Mean-Shift to smooth textures (like waves)
    shifted = cv2.pyrMeanShiftFiltering(proc_img, sp=sp, sr=sr)

    # 3. CLUSTER: Flatten and apply K-Means
    pixel_values = shifted.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    labels2d = labels.reshape(h, w)

    # 4. HEURISTIC: Differentiate Sky from Water
    # We score each cluster based on its "Blueness" AND its vertical position (Y-axis).
    # Sky has low Y (top of image). Water has high Y (bottom of image).
    scores = []
    for i in range(k):
        b, g, r = centers[i]

        # Find the average vertical position (Y-coordinate) of this cluster
        y_indices = np.where(labels2d == i)[0]
        avg_y = np.mean(y_indices) if len(y_indices) > 0 else 0

        # Normalize Y position (0.0 to 1.0, where 1.0 is the very bottom)
        normalized_y = avg_y / h

        # Score = (Blue + Green / 2 - Red) * How low it is in the image
        # This heavily penalizes the sky while rewarding the water
        blue_dominance = int(b) + int(g) / 2 - int(r)
        water_score = blue_dominance * normalized_y
        scores.append(water_score)

    # The cluster with the highest score is our water
    water_label = np.argmax(scores)

    # 5. MASK CREATION: 2-Class Output
    segmentation_map = (labels2d == water_label).astype(np.uint8)

    # Upscale the mask back to the original image resolution
    if scale < 1.0:
        segmentation_map = cv2.resize(segmentation_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return segmentation_map


def process_image(img_tensor, mask_numpy, process_dim=256, sp=32, sr=128):
    image_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    water_mask = segment_water(image_np, process_dim=process_dim, sp=sp, sr=sr, k=3)
    smooth_mask = cv2.medianBlur(water_mask * 255, 5) // 255
    iou = np.sum((smooth_mask == 1) & (mask_numpy == 1)) / np.sum((smooth_mask == 1) | (mask_numpy == 1))
    return iou, smooth_mask


def show_predictions():
    x, y, _ = load_split("val", hsv=False)
    y = y.cpu().numpy()  # Move to CPU and convert to NumPy for easier processing
    print(f"Images shape: {x.shape}")
    fps = 2
    for i in range(len(x)):
        img = x[i]
        iou, water_mask = process_image(img, y[i])
        wrong = ((water_mask == 1) != (y[i] == 1)) & (y[i] != 255)
        show_img(img, water_mask, wrong=wrong, hsv=False)
        print(f"Image {i} - IoU: {iou:.4f}")
        key = cv2.waitKey(int(1000 / fps))
        if key == 27:  # ESC key to exit early
            break
    cv2.destroyAllWindows()


def mean_iou(process_dim=256, sp=32, sr=128, x=None, y=None):
    if x is None or y is None:
        x, y, _ = load_split("val", hsv=False)
        y = y.cpu().numpy()  # Move to CPU and convert to NumPy for easier processing
        print(f"Evaluating Means-Shift segmentation on {len(x)} validation images using multiple workers...")
    ious = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, x[i], y[i], process_dim, sp, sr) for i in range(len(x))]
        for future in as_completed(futures):
            iou, _ = future.result()
            ious.append(iou)
            mean_iou_val = np.mean(ious)
            print(f"Image {len(ious):03}/{len(x)} ({(len(ious))/len(x)*100:.2f}%) - IoU: {iou:.4f}, mean: {mean_iou_val:.4f}", end="\r")
    mean_iou_val = np.mean(ious)
    print(f"Mean IoU across validation set: {mean_iou_val:.4f}                                       ")
    return mean_iou_val


def optimize_hyperparameters():
    x, y, _ = load_split("val", hsv=False)
    y = y.cpu().numpy()  # Move to CPU and convert to NumPy for easier processing
    best_iou = 0
    best_params = None
    sps = [
        # 16,
        32,
        # 64,
        # 128,
    ]
    srs = [
        # 32,
        # 64,
        128,
        # 256,
    ]
    for sp in sps:
        for sr in srs:
            print(f"Testing SP={sp}, SR={sr}...")
            mean_iou_val = mean_iou(process_dim=256, sp=sp, sr=sr, x=x, y=y)
            if mean_iou_val > best_iou:
                best_iou = mean_iou_val
                best_params = (sp, sr)
    print(f"Best Hyperparameters - SP: {best_params[0]}, SR: {best_params[1]} with Mean IoU: {best_iou:.4f}")


if __name__ == "__main__":
    # mean_iou()
    show_predictions()
    # optimize_hyperparameters()
