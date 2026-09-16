"""Spike: measure pick_ids / yes_probability accuracy on the Krea 2 encoder, outside ComfyUI.

Run ON THE SERVER from the synced repo root (needs ~6GB free VRAM):
  COMFY_ROOT=/path/to/ComfyUI python3 tests/person_e2e/spike_vlm.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMFY = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(ROOT / "modules"))

import comfy.sd
import numpy as np
import torch
from impact import person_core as pc
from impact import person_vlm as pv
from PIL import Image
from ultralytics import YOLO

FIXTURES = ROOT / "tests" / "person_fixtures"
OUT = ROOT / "tests" / "person_e2e" / "out"
CLIP_PATH = COMFY / "models" / "clip" / "qwen3vl_4b_fp8_scaled.safetensors"
PERSON_MODEL = COMFY / "models" / "ultralytics" / "segm" / "person_yolov8m-seg.pt"


def detect_people(pil: Image.Image) -> list[pc.PersonCandidate]:
    result = YOLO(str(PERSON_MODEL))(pil, conf=0.3, verbose=False)[0]
    dets = [pc.Detection(tuple(int(v) for v in xyxy), float(conf))
            for xyxy, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
    cands = pc.associate(dets, [], pil.width, pil.height)
    pc.apply_exclusions(cands, min_person_ratio=0.02, min_face_size=0, min_sharpness=0.0, min_relative_size=0.25)
    return pc.number_candidates(cands, "left_to_right")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clip = comfy.sd.load_clip(ckpt_paths=[str(CLIP_PATH)], clip_type=comfy.sd.CLIPType.KREA2)
    truth = json.loads((FIXTURES / "truth.json").read_text())

    index_ok = index_total = pick_ok = pick_total = verify_ok = verify_total = 0
    for fx in truth["fixtures"]:
        pil = Image.open(FIXTURES / fx["file"]).convert("RGB")
        image = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)
        people = detect_people(pil)
        print(f"\n## {fx['name']}: detected {len(people)} numbered people (truth {len(fx['people'])})")
        marked = pv.draw_boxes(image, [(p.person_box, str(p.index), pv.palette(p.index - 1)) for p in people])
        Image.fromarray((marked[0].numpy() * 255).astype(np.uint8)).save(OUT / f"spike_{fx['name']}.png")

        for case in fx["cases"]:
            ids, _ = pc.parse_index_expr(case["index"], len(people))
            chosen = [people[i - 1] for i in ids]
            needs_vlm = case["gender"] != "any" or case["description"].strip()
            if not needs_vlm:
                got = [p.index for p in chosen]
                index_total += 1
                index_ok += got == case["expected"]
                print(f"  [index] {case['name']}: got {got} expected {case['expected']}")
                continue

            condition = pv.build_condition(case["gender"], case["description"])
            t0 = time.time()
            pick_total += 1
            try:
                got, dropped, raw = pv.pick_ids(clip, image, [(p.index, p.person_box) for p in people],
                                                [p.index for p in chosen], condition)
                dt = time.time() - t0
                pick_ok += sorted(got) == sorted(case["expected"])
                print(f"  [pick ] {case['name']}: got {got} expected {case['expected']} dropped={dropped} ({dt:.1f}s) raw={raw!r}")

                for p in chosen:
                    x1, y1, x2, y2 = p.person_box
                    verify_total += 1
                    try:
                        prob = pv.yes_probability(clip, image[:, y1:y2, x1:x2, :], condition)
                        correct = (prob >= 0.5) == (p.index in case["expected"])
                        verify_ok += correct
                        print(f"      verify #{p.index}: P(yes)={prob:.3f} {'ok' if correct else 'WRONG'}")
                    except pv.VLMError as e:
                        print(f"      verify #{p.index}: VLMError: {e}")
            except pv.VLMError as e:
                # Not a code crash: the VLM's answer failed to parse / referenced an invalid id.
                # Count it as a pick failure and keep going so the full spike still completes.
                dt = time.time() - t0
                print(f"  [pick ] {case['name']}: VLMError ({dt:.1f}s): {e}")

    def pct(a, b):
        return f"{a}/{b} = {100.0 * a / b:.0f}%" if b else "n/a"

    print(f"\nINDEX  {pct(index_ok, index_total)}")
    print(f"PICK   {pct(pick_ok, pick_total)}")
    print(f"VERIFY {pct(verify_ok, verify_total)}")


if __name__ == "__main__":
    main()
