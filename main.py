from lars import load_split, SEMANTIC
import cv2
import numpy as np


def main():
    images, semantic, _ = load_split("train", limit=None)
    print(f"Images shape: {images.shape}")
    print(f"Semantic shape: {semantic.shape}")
    mode = "automatic"  # "manual"
    fps = 30
    for i in range(len(images)):
        img = images[i]
        # find where semantic mask is water
        # find water id
        water_id = [k for k, v in SEMANTIC.items() if v == "Water"][0]
        water = semantic[i] == water_id
        # blue transparent overlay for water
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[water] = [255, 0, 0]
        img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        cv2.imshow("Image", img)
        if mode == "manual":
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(1000 / fps))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
