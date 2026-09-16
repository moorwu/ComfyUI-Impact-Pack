"""Generate group-photo fixtures with Krea 2 Turbo on the ComfyUI server.

Usage: python3 tests/person_e2e/gen_fixtures.py [--fixtures DIR] [--only NAME] [--force] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_http as comfy_api

FIXTURES = Path(__file__).resolve().parents[1] / "person_fixtures"

UNET = "krea2_turbo_int8_convrot.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def build_workflow(prompt: str, seed: int, prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1536, "height": 864, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, help="override seed (use with --only)")
    args = ap.parse_args()

    fixtures = args.fixtures.resolve()
    truth = json.loads((fixtures / "truth.json").read_text())
    for fx in truth["fixtures"]:
        if args.only and fx["name"] != args.only:
            continue
        dest = fixtures / fx["file"]
        if dest.exists() and not args.force:
            print(f"skip {fx['name']} (exists)")
            continue
        seed = args.seed if args.seed is not None else fx["seed"]
        pid = comfy_api.queue(build_workflow(fx["prompt"], seed, f"person_fixture_{fixtures.name}_{fx['name']}"))
        outputs = comfy_api.wait(pid)
        img = outputs["9"]["images"][0]
        comfy_api.download(img["filename"], img["subfolder"], img["type"], dest)
        print(f"{fx['name']}: seed={seed} -> {dest}")


if __name__ == "__main__":
    main()
