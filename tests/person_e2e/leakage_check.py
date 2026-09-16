"""Per-face change ratio between original fixtures and Detailer outputs.

Run ON THE SERVER (needs ultralytics):
  python tests/person_e2e/leakage_check.py ORIGINAL.png DETAILED.png [ORIGINAL2.png DETAILED2.png ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

COMFY = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
FACE_MODEL = COMFY / "models/ultralytics/bbox/face_yolov8m.pt"
DIFF_THRESHOLD = 25


def main(argv: list[str]) -> None:
    if len(argv) < 2 or len(argv) % 2:
        raise SystemExit(__doc__)
    model = YOLO(str(FACE_MODEL))
    for orig_path, detailed_path in zip(argv[0::2], argv[1::2]):
        orig = Image.open(orig_path).convert("RGB")
        detailed = Image.open(detailed_path).convert("RGB")
        a = np.asarray(orig).astype(np.int16)
        b = np.asarray(detailed).astype(np.int16)
        changed = np.abs(a - b).mean(axis=2) > DIFF_THRESHOLD
        faces = model(orig, conf=0.5, verbose=False)[0].boxes.xyxy.tolist()
        print(f"\n## {Path(detailed_path).name} (whole image changed {changed.mean():.1%})")
        for x1, y1, x2, y2 in sorted(faces, key=lambda box: box[0] + box[2]):
            x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
            print(f"  face center_x={(x1 + x2) // 2:>5} box=({x1},{y1},{x2},{y2}) changed={changed[y1:y2, x1:x2].mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
