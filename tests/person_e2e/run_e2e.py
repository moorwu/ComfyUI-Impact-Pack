"""End-to-end: Person Detector -> Person Selector (-> Detailer) through the ComfyUI API.

Usage (from the local repo root):
  python3 tests/person_e2e/run_e2e.py                                      # all v1 cases
  python3 tests/person_e2e/run_e2e.py --fixtures tests/person_fixtures_v2  # all v2 cases
  python3 tests/person_e2e/run_e2e.py --verify                             # VLM cases with verify enabled
  python3 tests/person_e2e/run_e2e.py --detailer                          # v1 demo: real_01 / red jacket, face SEGS
  python3 tests/person_e2e/run_e2e.py --fixtures tests/person_fixtures_v2 \
      --detailer "real_02:denim jacket man" --segs person --prompt "a person wearing a bright pink outfit"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_http

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "person_fixtures"
OUT = ROOT / "tests" / "person_e2e" / "out"

CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
UNET = "krea2_turbo_int8_convrot.safetensors"
VAE = "qwen_image_vae.safetensors"
FACE_PROMPT = "close-up portrait of a face with bright blue eyes and a big open smile, detailed skin"
SEGS_SLOT = {"face": 0, "person": 1}
MIN_PERSON_RATIO = 0.015
MIN_RELATIVE_SIZE = 0.31
MIN_SHARPNESS = 0.0


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def expected_positions(fx: dict, case: dict) -> list[int]:
    if "expected" in case:
        return sorted(case["expected"])
    if "expect_pos" in case:
        return sorted(case["expect_pos"])
    keys = [p["key"] for p in fx["people"]]
    positions = []
    for key in case["expect"]:
        if key not in keys:
            raise ValueError(f"unknown person key {key!r} in case {case.get('name')!r}")
        positions.append(keys.index(key) + 1)
    return sorted(positions)


def case_ok(ids: list[int], face_count: int, fx: dict, case: dict) -> tuple[bool, list[int], str]:
    exp = expected_positions(fx, case)
    reasons = []
    if ids != exp:
        reasons.append(f"ids {ids} != {exp}")
    if "expect_face" in case:
        want = len(exp) if case["expect_face"] else 0
        if face_count != want:
            reasons.append(f"face_SEGS count {face_count} != {want}")
    return not reasons, exp, "; ".join(reasons)


def selection_workflow(image_name: str, case: dict, verify: bool, prefix: str, sam: str | None = None) -> dict:
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "segm/person_yolov8m-seg.pt"}},
        "3": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}},
        "5": {"class_type": "ImpactPersonDetectorSEGS", "inputs": {
            "image": ["1", 0], "person_detector": ["2", 0], "face_detector": ["3", 0],
            "person_segm_detector": ["2", 1], "extra_person_detector": ["21", 0], "head_detector": ["22", 0],
            "threshold": 0.5, "min_person_ratio": MIN_PERSON_RATIO, "min_relative_size": MIN_RELATIVE_SIZE,
            "min_face_size": 24, "min_sharpness": MIN_SHARPNESS,
            "sort_by": "left_to_right", "crop_factor": 3.0, "drop_size": 10}},
        "6": {"class_type": "ImpactPersonSelector", "inputs": {
            "persons": ["5", 0], "image": ["1", 0], "index": case["index"], "gender": case["gender"],
            "description": case["description"], "verify": verify, "verify_threshold": 0.5, "seed": 0, "clip": ["4", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 4], "filename_prefix": prefix}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["5", 1], "filename_prefix": prefix + "_detect"}},
        "9": {"class_type": "SEGSPreview", "inputs": {"segs": ["6", 0], "alpha_mode": True, "min_alpha": 0.2}},
        "21": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/person_detect_v1.1_m.pt"}},
        "22": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/head_detect_v2.0_s_yv11.pt"}},
    }
    if sam:
        wf["20"] = {"class_type": "SAMLoader", "inputs": {"model_name": sam, "device_mode": "AUTO"}}
        wf["5"]["inputs"]["sam_model"] = ["20", 0]
    return wf


def detailer_workflow(image_name: str, case: dict, prefix: str, segs: str = "face", prompt: str = FACE_PROMPT,
                      sam: str | None = None) -> dict:
    wf = selection_workflow(image_name, case, False, prefix, sam)
    wf.update({
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "12": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["12", 0]}},
        "14": {"class_type": "DetailerForEach", "inputs": {
            "image": ["1", 0], "segs": ["6", SEGS_SLOT[segs]], "model": ["10", 0], "clip": ["4", 0], "vae": ["11", 0],
            "guide_size": 512, "guide_size_for": True, "max_size": 1024, "seed": 0, "steps": 8, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["13", 0],
            "denoise": 0.6, "feather": 5, "noise_mask": True, "force_inpaint": True, "wildcard": "", "cycle": 1}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix + "_detailed"}},
    })
    return wf


def parse_debug(outputs: dict) -> tuple[list[int], int, str]:
    text = outputs["6"]["text"][0]
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].startswith("selected="):
        raise RuntimeError(f"unexpected debug_text: {text!r}")
    ids = [int(x) for x in lines[0][len("selected="):].split(",") if x]
    m = re.match(r"numbered=(\d+)", lines[1])
    if m is None:
        raise RuntimeError(f"unexpected debug_text: {text!r}")
    return ids, int(m.group(1)), text


def face_count(outputs: dict) -> int:
    return len(outputs.get("9", {}).get("images", []))


def save_images(outputs: dict, node_ids: list[str]) -> list[Path]:
    saved = []
    for nid in node_ids:
        for img in outputs.get(nid, {}).get("images", []):
            saved.append(comfy_http.download(img["filename"], img["subfolder"], img["type"], OUT / img["filename"]))
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--detailer", nargs="?", const="real_01:red jacket", help="FIXTURE:CASE name")
    ap.add_argument("--segs", choices=sorted(SEGS_SLOT), default="face")
    ap.add_argument("--prompt", default=FACE_PROMPT)
    ap.add_argument("--only")
    ap.add_argument("--sam", help="SAM model file, e.g. sam_vit_b_01ec64.pth")
    args = ap.parse_args()

    fixtures = args.fixtures.resolve()
    truth = json.loads((fixtures / "truth.json").read_text())
    tag = "" if fixtures == FIXTURES else f"{fixtures.name}_"

    if args.detailer:
        fx_name, case_name = args.detailer.split(":", 1)
        fx = next(f for f in truth["fixtures"] if f["name"] == fx_name)
        case = next(c for c in fx["cases"] if c["name"] == case_name)
        name = comfy_http.upload_image(fixtures / fx["file"])
        prefix = f"person_e2e_{tag}detailer_{fx_name}_{slug(case_name)}_{args.segs}" + ("_sam" if args.sam else "")
        outputs = comfy_http.wait(comfy_http.queue(detailer_workflow(name, case, prefix, args.segs, args.prompt, args.sam)))
        _, _, text = parse_debug(outputs)
        print(text)
        for path in save_images(outputs, ["7", "8", "15"]):
            print(f"saved {path}")
        return

    idx_ok = idx_total = vlm_ok = vlm_total = 0
    for fx in truth["fixtures"]:
        if args.only and fx["name"] != args.only:
            continue
        name = comfy_http.upload_image(fixtures / fx["file"])
        for case in fx["cases"]:
            needs_vlm = case["gender"] != "any" or case["description"].strip() != ""
            if args.verify and not needs_vlm:
                continue
            prefix = f"person_e2e_{tag}{fx['name']}_{slug(case['name'])}" + ("_verify" if args.verify else "") \
                + ("_sam" if args.sam else "")
            outputs = comfy_http.wait(comfy_http.queue(selection_workflow(name, case, args.verify, prefix, args.sam)))
            ids, numbered, text = parse_debug(outputs)
            faces = face_count(outputs)
            ok, exp, why = case_ok(ids, faces, fx, case)
            if needs_vlm:
                vlm_total += 1
                vlm_ok += ok
            else:
                idx_total += 1
                idx_ok += ok
            warn = "" if numbered == len(fx["people"]) else f" [numbered={numbered}, truth={len(fx['people'])}]"
            print(f"[{'vlm' if needs_vlm else 'idx'}] {fx['name']} / {case['name']}: got {ids} faces={faces} "
                  f"expected {exp} {'ok' if ok else 'WRONG (' + why + ')'}{warn}")
            if not ok:
                print("    " + text.replace("\n", "\n    "))
            save_images(outputs, ["7", "8"])

    def pct(a, b):
        return f"{a}/{b} = {100.0 * a / b:.0f}%" if b else "n/a"

    print(f"\nINDEX {pct(idx_ok, idx_total)}")
    print(f"VLM   {pct(vlm_ok, vlm_total)}")


if __name__ == "__main__":
    main()
