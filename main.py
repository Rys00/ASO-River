from lars import load_split, cycle_images
import cv2
import numpy as np


def main():
    images, semantic, panoptic = load_split("train", limit=None, reshape=(1920 / 2, 1080 / 2), inc_semantic=True, inc_panoptic=True, save=True)
    print(f"Images shape: {images.shape}")
    print(f"Semantic shape: {semantic.shape}")
    print(f"Panoptic shape: {panoptic.shape}")
    cycle_images(images, semantic, fps=30)


if __name__ == "__main__":
    main()
