"""Small helpers shared across stages."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DECODER = json.JSONDecoder()


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM reply.

    Models often wrap JSON in code fences or add sentences around it (which may contain
    braces of their own), so we look inside fences first, then try to decode an object
    at each "{" in turn and return the first one that parses.
    """
    chunks = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for chunk in chunks:
        for start in (m.start() for m in re.finditer(r"\{", chunk)):
            try:
                value, _ = _DECODER.raw_decode(chunk, start)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None
