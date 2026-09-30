"""Tolerant JSON extraction from model output (vendored, stdlib-only)."""
from __future__ import annotations

import json
import re


class LLMDecompositionError(Exception):
    """Raised when a model response contains no usable JSON."""


def _extract_json(text: str):
    """Parse a complete JSON value, or the first object or array in a response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        pass
    start = next((i for i, ch in enumerate(text) if ch in "[{"), None)
    if start is None:
        raise LLMDecompositionError("no JSON found in model output")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, RecursionError) as e:
        raise LLMDecompositionError(f"malformed JSON: {e}") from e
    return value
