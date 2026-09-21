"""Small helpers shared across stages."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM reply.

    Models often wrap JSON in code fences or add a sentence before it, so we look
    inside fences first and then fall back to the outermost ``{...}`` span.
    """
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for chunk in candidates:
        start, end = chunk.find("{"), chunk.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            value = json.loads(chunk[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
