"""Run the ASO pipeline with Fast R-CNN (Selective Search proposals).

The default entry point `main.py` uses Faster R-CNN. Only this script switches
the detector to Fast R-CNN:

    uv run detection/main_with_fastrcnn.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)

# Running this file adds detection/ to sys.path, which shadows the detection package
# (Python would import detection.py as top-level "detection" instead of the package).
sys.path = [p for p in sys.path if p not in {SCRIPT_DIR, str(ROOT)}]
sys.path.insert(0, str(ROOT))

import main

main.DETECTOR = main.DetectorSettings(
    kind="fast_rcnn",
    backbone="resnet18",
    checkpoint="detection/models/detector-18v3.pth",
    max_proposals=2000,
)

if __name__ == "__main__":
    main.main()
