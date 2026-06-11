"""Rearrange extracted LaRS archives into the layout expected by lars.py.

The official LaRS downloads (images and annotations zips) extract into plain
train/ val/ test/ folders. This script moves their contents into:

    data/lars_v1.0.0/images/{split}/images/*.jpg
    data/lars_v1.0.0/annotations/{split}/semantic_masks/*.png
    data/lars_v1.0.0/annotations/{split}/panoptic_masks/*.png

Run it from the repo root after extracting one or both zips:

    uv run prepare_lars.py                  # looks for ./train ./val ./test
    uv run prepare_lars.py path/to/extract  # splits live somewhere else

It moves (renames) files, so it is fast and leaves nothing behind. Already
existing destination folders are left untouched and reported, so it is safe to
re-run, e.g. after extracting the annotations zip later.
"""

import argparse
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")

# What we look for inside each split folder and where it should end up,
# relative to the dataset root.
CONTENT_DESTS = {
    "images": "images/{split}/images",
    "semantic_masks": "annotations/{split}/semantic_masks",
    "panoptic_masks": "annotations/{split}/panoptic_masks",
}
# Loose metadata files worth keeping next to their content.
FILE_DESTS = {
    "image_list.txt": "images/{split}",
    "panoptic_annotations.json": "annotations/{split}",
    "image_annotations.json": "annotations/{split}",
}


def move(src: Path, dst: Path) -> bool:
    if dst.exists():
        print(f"  SKIP {dst} (already exists)")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  {src} -> {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=".", help="Directory containing the extracted train/ val/ test folders (default: current directory).")
    parser.add_argument("--dest", default="data/lars_v1.0.0", help="Dataset root expected by lars.py (default: data/lars_v1.0.0).")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    moved = 0

    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"{split}: not found in {source}, skipping")
            continue
        print(f"{split}:")

        for name, dst_template in CONTENT_DESTS.items():
            src = split_dir / name
            if src.is_dir():
                moved += move(src, dest / dst_template.format(split=split))

        for name, dst_template in FILE_DESTS.items():
            src = split_dir / name
            if src.is_file():
                moved += move(src, dest / dst_template.format(split=split) / name)

        leftovers = list(split_dir.iterdir())
        if leftovers:
            print(f"  NOTE: leaving {len(leftovers)} unrecognized item(s) in {split_dir}: {', '.join(p.name for p in leftovers[:5])}")
        else:
            split_dir.rmdir()
            print(f"  removed empty {split_dir}")

    print(f"\nDone, moved {moved} item(s) into {dest}.")

    # Quick sanity report against what lars.py will look for.
    print("\nDataset status:")
    for split in SPLITS:
        img = dest / "images" / split / "images"
        sem = dest / "annotations" / split / "semantic_masks"
        pan = dest / "annotations" / split / "panoptic_masks"
        n = lambda p: len(list(p.iterdir())) if p.is_dir() else 0
        print(f"  {split:5}: images={n(img):4}  semantic_masks={n(sem):4}  panoptic_masks={n(pan):4}")


if __name__ == "__main__":
    main()
