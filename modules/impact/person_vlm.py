"""Qwen3-VL helpers for Person Selector, driven through the Krea 2 CLIP object (no extra model)."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from comfy import model_management
from PIL import Image, ImageDraw, ImageFont

from . import person_core

logger = logging.getLogger(__name__)


class VLMError(RuntimeError):
    pass


_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
            (70, 190, 190), (240, 50, 230), (128, 128, 0), (0, 0, 128), (170, 110, 40)]

_CLIP_REQUIRED = ("Person Selector: gender/description requires a Krea 2 (Qwen3-VL) CLIP input. "
                  "Connect a CLIPLoader with type 'krea2' (e.g. qwen3vl_4b_fp8_scaled.safetensors).")

PICK_SYSTEM = ("You are a precise visual assistant. Every person in the image is marked with a colored box "
               "and a number label placed just above the person's face (or just above the head when the face is "
               "not visible, or at the top-left corner of the box when neither is available).")
VERIFY_SYSTEM = "You are a precise visual assistant. Answer strictly with yes or no."


def palette(i: int) -> tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def cap_megapixels(image: torch.Tensor, megapixels: float) -> torch.Tensor:
    h, w = image.shape[1], image.shape[2]
    scale = math.sqrt(megapixels * 1_000_000 / max(h * w, 1))
    if scale >= 1.0:
        return image
    nh, nw = max(28, round(h * scale)), max(28, round(w * scale))
    resized = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(nh, nw), mode="bicubic", antialias=True)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def draw_boxes(image: torch.Tensor, boxes: list[tuple]) -> torch.Tensor:
    """boxes: (box, label, color) or (box, label, color, label_anchor_box); the label sits above the anchor box."""
    arr = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    line = max(2, round(min(w, h) / 400))
    font = ImageFont.load_default(size=max(18, round(min(w, h) / 25)))

    for item in boxes:
        box, label, color = item[0], item[1], item[2]
        label_box = item[3] if len(item) > 3 and item[3] is not None else box
        draw.rectangle(box, outline=color, width=line)
        if not label:
            continue
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        x = max(0, min(label_box[0], w - tw - 2 * line))
        y = label_box[1] - th - 2 * line
        if y < 0:
            y = label_box[1]
        draw.rectangle((x, y, x + tw + 2 * line, y + th + 2 * line), fill=color)
        draw.text((x + line, y + line - tb[1]), label, fill=(255, 255, 255), font=font)

    return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)


def build_condition(gender: str, description: str) -> str:
    parts = []
    if gender == "male":
        parts.append("is male (a man or a boy)")
    elif gender == "female":
        parts.append("is female (a woman or a girl)")
    description = description.strip()
    if description:
        parts.append(f"matches: {description}")
    if not parts:
        raise ValueError("Person Selector: build_condition needs a gender or a description")
    return "the person " + " and ".join(parts)


def _image_template(system: str) -> str:
    system = system.replace("{", "{{").replace("}", "}}")
    return ("<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")


def _inner_model(clip):
    if clip is None or not hasattr(clip, "generate"):
        raise VLMError(_CLIP_REQUIRED)
    cond = clip.cond_stage_model
    inner = getattr(cond, getattr(cond, "clip", ""), None)
    transformer = getattr(inner, "transformer", None)
    if transformer is None or not hasattr(transformer, "build_image_inputs"):
        raise VLMError(_CLIP_REQUIRED)
    return inner


def _generate_text(clip, image: torch.Tensor, system: str, prompt: str, max_new_tokens: int, seed: int) -> str:
    tokens = clip.tokenize(prompt, image=image, llama_template=_image_template(system))
    try:
        ids = clip.generate(tokens, do_sample=False, max_length=max_new_tokens, seed=seed)
    except Exception as e:
        raise VLMError(f"Person Selector: Qwen3-VL generation failed ({e}). "
                       f"If the vision tower cannot run with this encoder, try qwen3vl_4b_bf16.") from e
    return clip.decode(ids).strip()


def _pick_prompt(candidate_ids: list[int], condition: str, strict: bool) -> str:
    text = (f"Candidate numbers: {', '.join(str(i) for i in candidate_ids)}. "
            f"Ignore every person whose number is not in this list.\n"
            f"Select every candidate for whom this is true: {condition}.\n"
            f"Answer with JSON only, in the form {{\"ids\": [numbers]}}. Use {{\"ids\": []}} if nobody matches.")
    if strict:
        text += "\nOutput exactly one JSON object and nothing else."
    return text


def pick_ids(clip, image: torch.Tensor, people: list[tuple], candidate_ids: list[int], condition: str,
             seed: int = 0, megapixels: float = 1.0) -> tuple[list[int], list[int], str]:
    """Returns (picked candidate ids, dropped non-candidate ids the VLM mentioned, raw answers)."""
    _inner_model(clip)
    if not candidate_ids:
        return [], [], ""

    marked = draw_boxes(image[0:1], [(t[1], str(t[0]), palette(t[0] - 1), t[2] if len(t) > 2 else None) for t in people])
    marked = cap_megapixels(marked, megapixels)
    all_ids = {t[0] for t in people}
    candidates = set(candidate_ids)

    answers = []
    last_error = None
    for strict in (False, True):
        text = _generate_text(clip, marked, PICK_SYSTEM, _pick_prompt(sorted(candidates), condition, strict), 64, seed)
        answers.append(text)
        try:
            ids = person_core.parse_ids_json(text, all_ids)
        except ValueError as e:
            last_error = e
            continue
        picked = [i for i in ids if i in candidates]
        dropped = [i for i in ids if i not in candidates]
        return picked, dropped, "\n---\n".join(answers)
    raise VLMError(f"Person Selector: could not parse VLM answer after retry ({last_error}). "
                   f"Raw answers:\n" + "\n---\n".join(answers))


def _hf_tokenizer(clip):
    """Returns the underlying HF tokenizer, or None if that path isn't available on this ComfyUI build."""
    try:
        return getattr(clip.tokenizer, clip.tokenizer.clip).tokenizer
    except AttributeError:
        return None


_single_token_id_cache: dict[tuple[int, tuple[str, ...]], list[int]] = {}


def _single_token_ids(hf, words: list[str]) -> list[int]:
    key = (id(hf), tuple(words))
    cached = _single_token_id_cache.get(key)
    if cached is not None:
        return cached
    ids = set()
    for word in words:
        enc = hf(word, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.add(enc[0])
    if not ids:
        raise VLMError(f"Person Selector: tokenizer has no single-token id for {words}")
    result = sorted(ids)
    _single_token_id_cache[key] = result
    return result


def _logits_path_available(clip, inner) -> bool:
    """Whether the ComfyUI internals the manual prefill needs are present and callable.

    Used to decide up front whether to take the documented generated-yes/no degrade path, instead
    of discovering it mid-forward-pass via a caught exception (which would also swallow real
    failures such as an fp8 vision tower crash or OOM).
    """
    try:
        patcher = clip.patcher
        cond = clip.cond_stage_model
        transformer = getattr(inner, "transformer", None)
        model = getattr(transformer, "model", None)
        return all((
            callable(getattr(clip, "load_model", None)),
            hasattr(patcher, "load_device"),
            callable(getattr(cond, "reset_clip_options", None)),
            callable(getattr(cond, "set_clip_options", None)),
            callable(getattr(inner, "process_tokens", None)),
            transformer is not None,
            callable(getattr(transformer, "build_image_inputs", None)),
            callable(getattr(transformer, "init_kv_cache", None)),
            callable(getattr(transformer, "logits", None)),
            callable(getattr(model, "forward", None)),
        ))
    except AttributeError:
        return False


def _prefill_last_logits(clip, inner, tokens) -> torch.Tensor:
    clip.cond_stage_model.reset_clip_options()
    clip.load_model(tokens)
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"layer": None})
    clip.cond_stage_model.set_clip_options({"execution_device": device})

    batch = next(iter(tokens.values()))
    tokens_only = [[t[0] for t in row] for row in batch]
    transformer = inner.transformer
    with torch.no_grad(), model_management.cuda_device_context(device):
        embeds, _, _, embeds_info = inner.process_tokens(tokens_only, device)
        position_ids, visual_pos_masks, deepstack = transformer.build_image_inputs(embeds, embeds_info)
        dtype = torch.bfloat16 if model_management.should_use_bf16(device) else torch.float32
        embeds = embeds.to(dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        kv = transformer.init_kv_cache(embeds.shape[0], embeds.shape[1] + 1, device, dtype)
        x, _, _ = transformer.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=kv, input_ids=None,
                                            position_ids=position_ids, deepstack_embeds=deepstack,
                                            visual_pos_masks=visual_pos_masks, embeds_info=embeds_info)
        return transformer.logits(x)[:, -1].float()[0]


def yes_probability(clip, image: torch.Tensor, condition: str, megapixels: float = 0.5) -> float:
    inner = _inner_model(clip)
    img = cap_megapixels(image[0:1], megapixels)
    prompt = f"Is this true for the person shown: {condition}? Answer yes or no."
    hf = _hf_tokenizer(clip)

    if hf is None or not _logits_path_available(clip, inner):
        # ComfyUI internals are missing on this build: degrade to a hard yes/no answer instead of a
        # probability. This is the documented fallback, not a response to a runtime failure.
        logger.warning("[Impact Pack] Person Selector: logits path unavailable; using generated yes/no")
        answer = _generate_text(clip, img, VERIFY_SYSTEM, prompt, 3, 0).lower()
        return 1.0 if answer.startswith("yes") else 0.0

    tokens = clip.tokenize(prompt, image=img, llama_template=_image_template(VERIFY_SYSTEM))
    yes_ids = _single_token_ids(hf, ["yes", "Yes", " yes", " Yes"])
    no_ids = _single_token_ids(hf, ["no", "No", " no", " No"])

    try:
        logits = _prefill_last_logits(clip, inner, tokens)
    except Exception as e:
        # Internals are present but the forward pass itself failed (e.g. fp8 vision tower crash,
        # OOM): surface it, don't silently degrade to a hard 0.0/1.0.
        raise VLMError(f"Person Selector: Qwen3-VL verify failed ({e}). "
                       f"If the vision tower cannot run with this encoder, try qwen3vl_4b_bf16.") from e

    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.sigmoid(yes - no))
