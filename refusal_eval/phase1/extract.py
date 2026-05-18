"""Structured-output extraction for Phase 1 analysis answers."""

from __future__ import annotations

import json
import re

JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _validate_candidate(candidate: object) -> dict[str, str] | None:
    if not isinstance(candidate, dict):
        return None
    location = candidate.get("location")
    vuln_class = candidate.get("class")
    if not isinstance(location, dict) or not isinstance(vuln_class, str):
        return None
    file_path = location.get("file")
    symbol = location.get("symbol")
    if not isinstance(file_path, str) or not isinstance(symbol, str):
        return None
    file_path = file_path.strip()
    symbol = symbol.strip()
    vuln_class = vuln_class.strip()
    if not file_path or not symbol or not vuln_class:
        return None
    return {"file": file_path, "symbol": symbol, "class": vuln_class}


def _find_last_json_object(text: str) -> dict[str, str] | None:
    decoder = json.JSONDecoder()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        validated = _validate_candidate(candidate)
        if validated is not None:
            return validated
    return None


def extract_analysis_output(text: str) -> dict[str, str] | None:
    """Extract the structured Phase 1 answer from assistant text."""
    fenced = JSON_FENCE_RE.findall(text or "")
    for block in reversed(fenced):
        try:
            candidate = json.loads(block)
        except json.JSONDecodeError:
            continue
        validated = _validate_candidate(candidate)
        if validated is not None:
            return validated
    return _find_last_json_object(text or "")
