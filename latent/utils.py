from __future__ import annotations

import json
import re
from typing import Any

from .api import normalize_tags


VALID_RESOLUTIONS = {"square", "portrait", "landscape"}
VALID_SAMPLERS = {
    "euler",
    "euler_ancestral",
    "dpmpp_2s_ancestral",
    "dpmpp_2m",
    "dpmpp_sde",
    "dpmpp_2m_sde",
    "ddim",
}
VALID_SCHEDULERS = {"karras", "beta", "normal", "simple", "exponential"}
VALID_SIZES = {"thumb", "preview", "original"}
VALID_SOURCES = {"novelai", "sd-webui", "comfyui", "invokeai"}


OPTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"--steps\s+(\d+)"), "steps"),
    (re.compile(r"--resolution\s+(\w+)"), "resolution"),
    (re.compile(r"--sampler\s+(\w+)"), "sampler"),
    (re.compile(r"--scheduler\s+(\w+)"), "scheduler"),
    (re.compile(r"--negative\s+(.+?)(?=\s--|\s*$)"), "negative"),
    (re.compile(r"--seed\s+(\d+)"), "seed"),
    (re.compile(r"--rank\s+(\d+)"), "rank"),
    (re.compile(r"--size\s+(\w+)"), "size"),
    (re.compile(r"--source\s+(\w+)"), "source"),
    (re.compile(r"--model\s+(.+?)(?=\s--|\s*$)"), "model"),
    (re.compile(r"--count\s+(\d+)"), "count"),
]


def parse_options(text: str) -> tuple[str, dict[str, Any]]:
    """Extract ``--key value`` options from a prompt/description string."""
    options: dict[str, Any] = {}
    for pattern, key in OPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            options[key] = match.group(1).strip().strip('"').strip("'")
            text = text[: match.start()] + " " + text[match.end() :]
    text = re.sub(r"\s{2,}", " ", text).strip().strip(",").strip()
    return text, options


def strip_command(text: str, names: set[str]) -> str:
    """Remove a leading command token if it is one of ``names``."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    head = parts[0].lstrip("/").lower()
    if head in names:
        return parts[1].strip() if len(parts) > 1 else ""
    return text


def resolve_enum(value: Any, choices: set[str], default: str | None) -> str | None:
    """Return ``value`` when it is a member of ``choices``, else ``default``."""
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    return raw if raw in choices else default


def to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def try_nonnegative_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def extract_tag_text(text: str) -> str:
    """Extract a clean comma-separated tag string from an LLM response."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"```(?:danbooru|text|json)?\s*", "", cleaned)
    cleaned = cleaned.strip("`\n ")
    if cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return normalize_tags([str(item) for item in parsed if str(item).strip()])
        except (ValueError, TypeError):
            pass
    candidates = [line.strip() for line in cleaned.splitlines() if line.strip()]
    best = max(candidates, key=lambda line: line.count(","), default=cleaned)
    best = re.sub(r"^(?:danbooru\s+)?tags?\s*[:\-]\s*", "", best, flags=re.IGNORECASE)
    return normalize_tags(best)
