"""Calibrate Person Detector exclusion defaults on v1 + v2 fixtures (realistic segm + anime person + head models).

Run ON THE SERVER from the synced repo root:
  COMFY_ROOT=/path/to/ComfyUI python3 tests/person_e2e/calibrate_exclusions.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMFY = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
sys.path[:0] = [str(COMFY), str(ROOT / "modules")]

import numpy as np
import torch
from impact import person_core as pc
from impact import person_vlm as pv
from PIL import Image
from ultralytics import YOLO

MODELS = COMFY / "models" / "ultralytics"
OUT = ROOT / "tests" / "person_e2e" / "out"
SETS = [("v1", ROOT / "tests" / "person_fixtures"), ("v2", ROOT / "tests" / "person_fixtures_v2")]
THRESHOLD = 0.5


def detect(model, pil):
    result = model(pil, conf=THRESHOLD, verbose=False)[0]
    return [pc.Detection(tuple(int(v) for v in box), float(conf))
            for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]


def main() -> None:
    body_model = YOLO(str(MODELS / "segm" / "person_yolov8m-seg.pt"))
    anime_model = YOLO(str(MODELS / "bbox" / "person_detect_v1.1_m.pt"))
    head_model = YOLO(str(MODELS / "bbox" / "head_detect_v2.0_s_yv11.pt"))
    face_model = YOLO(str(MODELS / "bbox" / "face_yolov8m.pt"))
    OUT.mkdir(parents=True, exist_ok=True)
    for set_name, folder in SETS:
        truth = json.loads((folder / "truth.json").read_text())
        for fx in truth["fixtures"]:
            pil = Image.open(folder / fx["file"]).convert("RGB")
            w, h = pil.size
            arr = np.asarray(pil).astype(np.float32) / 255.0
            gray = arr.mean(axis=2) * 255.0
            base, extra, heads = detect(body_model, pil), detect(anime_model, pil), detect(head_model, pil)
            order = pc.merge_person_detections(extra, base, heads)
            people = [extra[i] if source == "primary" else base[i] for source, i in order]
            cands = pc.associate(people, detect(face_model, pil), w, h)
            pc.assign_heads(cands, heads)
            for c in cands:
                if c.face_box is not None:
                    x1, y1, x2, y2 = c.face_box
                    c.sharpness = pc.laplacian_variance(gray[y1:y2, x1:x2])
            largest_face = max((pc.face_short_side(c.face_box) for c in cands if c.face_box is not None), default=0)
            largest_body = max((pc.box_area(c.person_box) for c in cands), default=0)
            largest_head = max((pc.face_short_side(c.head_box) for c in cands if c.head_box is not None), default=0)

            print(f"\n## {set_name}/{fx['name']}: {len(cands)} candidates, truth {len(fx['people'])} people")
            print("   k | anchor_x | face_px | head_px | area_ratio | rel_scale | sharpness | synthetic")
            boxes = []
            for k, c in enumerate(sorted(cands, key=pc.anchor), start=1):
                rel = pc.relative_scale(c, largest_face, largest_body, largest_head)
                sharp = "-" if c.sharpness is None else f"{c.sharpness:.0f}"
                print(f"  {k:>2} | {pc.anchor(c)[0]:>8.0f} | {pc.face_short_side(c.face_box):>7} | "
                      f"{pc.face_short_side(c.head_box):>7} | {c.area_ratio:>10.4f} | {rel:>9.3f} | {sharp:>9} | {c.person_det is None}")
                boxes.append((c.person_box, f"k{k}", pv.palette(k - 1), pc.anchor_box(c)))
            preview = pv.draw_boxes(torch.from_numpy(arr).unsqueeze(0), boxes)
            Image.fromarray((preview[0].numpy() * 255).astype(np.uint8)).save(OUT / f"calib_{set_name}_{fx['name']}.png")


if __name__ == "__main__":
    main()
