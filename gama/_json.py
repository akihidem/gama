"""Tolerant JSON extraction from model output (vendored, stdlib-only)."""
from __future__ import annotations

import json
import re


class LLMDecompositionError(Exception):
    """Raised when a model response contains no usable JSON."""


def _extract_json(text: str):
    """Pull the first JSON value out of a model response (tolerates ``` fences)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = next((i for i, ch in enumerate(text) if ch in "[{"), None)
    if start is None:
        raise LLMDecompositionError("no JSON found in model output")
    opench = text[start]
    closech = "]" if opench == "[" else "}"
    depth = 0
    in_str = esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == opench:
            depth += 1
        elif ch == closech:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except Exception as e:
                    raise LLMDecompositionError(f"malformed JSON: {e}")
    raise LLMDecompositionError("unbalanced JSON in model output")
