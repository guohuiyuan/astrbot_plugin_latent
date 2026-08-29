from __future__ import annotations

import json
import os
import struct
import zlib
from typing import Any


def parse_png_metadata(data: bytes) -> dict[str, Any]:
    """Extract embedded metadata from a PNG.

    Latent.moe images produced by ComfyUI carry the workflow as a ``tEXt``
    chunk named ``prompt``. From that JSON we can recover the model, cfg scale,
    strength (denoise), sampler, scheduler, steps and seed without inventing
    values that the REST API does not expose.
    """
    info: dict[str, Any] = {}
    if not data or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return info

    i = 8
    prompt_json: dict[str, Any] | None = None
    n = len(data)
    while i + 8 <= n:
        length = struct.unpack(">I", data[i : i + 4])[0]
        if length < 0 or i + 8 + length > n:
            break
        ctype = data[i + 4 : i + 8]
        chunk = data[i + 8 : i + 8 + length]
        if ctype == b"tEXt":
            key, _, value = chunk.partition(b"\x00")
            key_s = key.decode("latin-1", "replace")
            if key_s == "prompt":
                try:
                    prompt_json = json.loads(value.decode("utf-8", "replace"))
                except ValueError:
                    prompt_json = None
            elif key_s not in info:
                info[key_s] = value.decode("latin-1", "replace")
        elif ctype == b"zTXt":
            key, _, rest = chunk.partition(b"\x00")
            try:
                info[key.decode("latin-1", "replace")] = zlib.decompress(rest[1:]).decode(
                    "utf-8", "replace"
                )
            except Exception:
                pass
        elif ctype == b"iTXt":
            key, _, rest = chunk.partition(b"\x00")
            # rest = compression-flag(1) + compression-method(1) + language-tag
            # + translated-keyword + text, each terminated by a NUL before the
            # free-form value.
            if len(rest) < 2:
                continue
            comp_flag = rest[0]
            fields = rest[2:].split(b"\x00", 2)
            if len(fields) < 2:
                continue
            value = fields[2] if len(fields) > 2 else b""
            if comp_flag == 1:
                try:
                    value = zlib.decompress(value)
                except Exception:
                    value = b""
            try:
                info[key.decode("latin-1", "replace")] = value.decode("utf-8", "replace")
            except Exception:
                pass
        i += 12 + length

    if prompt_json:
        _merge_comfyui_params(prompt_json, info)
    return info


def _merge_comfyui_params(workflow: dict[str, Any], info: dict[str, Any]) -> None:
    sampler_inputs: dict[str, Any] | None = None
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if class_type == "UNETLoader":
            name = inputs.get("unet_name")
            if name:
                base = os.path.basename(str(name))
                if base.endswith(".safetensors"):
                    base = base[: -len(".safetensors")]
                info.setdefault("model", base)
        elif class_type == "KSampler":
            sampler_inputs = inputs
            for key in ("cfg", "steps", "sampler_name", "scheduler", "seed", "denoise"):
                if inputs.get(key) is not None:
                    info.setdefault(key, inputs[key])

    if sampler_inputs:
        positive = _resolve_clip_text(workflow, sampler_inputs.get("positive"))
        negative = _resolve_clip_text(workflow, sampler_inputs.get("negative"))
        if positive and "prompt" not in info:
            info["prompt"] = positive
        if negative and "negative_prompt" not in info:
            info["negative_prompt"] = negative

    # Normalise aliases so callers can rely on one set of keys.
    if "sampler_name" in info and "sampler" not in info:
        info["sampler"] = info["sampler_name"]
    if "denoise" in info and "strength" not in info:
        info["strength"] = info["denoise"]


def _resolve_clip_text(
    workflow: dict[str, Any],
    ref: Any,
    depth: int = 0,
) -> str | None:
    """Resolve a ComfyUI node reference to the text of a CLIPTextEncode node.

    Handles the common ``[node_id, slot]`` reference directly and follows a few
    levels through conditioning/sampler merge nodes before giving up.
    """
    if depth > 6 or not isinstance(ref, list) or not ref:
        return None
    node = workflow.get(str(ref[0]))
    if not isinstance(node, dict):
        return None
    if node.get("class_type") == "CLIPTextEncode":
        text = node.get("inputs", {}).get("text")
        return str(text) if text else None
    inputs = node.get("inputs", {})
    if isinstance(inputs, dict):
        for value in inputs.values():
            if isinstance(value, list) and value:
                sub = _resolve_clip_text(workflow, value, depth + 1)
                if sub:
                    return sub
    return None
