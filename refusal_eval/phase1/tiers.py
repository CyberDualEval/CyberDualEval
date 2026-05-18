"""Deterministic tier checks for Phase 1 accuracy."""

from __future__ import annotations

import re


def _normalize_path(value: str) -> str:
    value = value.strip().strip("\"'").replace("\\", "/").lower()
    value = re.sub(r"^[a-z]:", "", value)
    return value.rstrip("/")


def _file_match(agent_file: str, oracle_file: str) -> bool:
    agent_norm = _normalize_path(agent_file)
    oracle_norm = _normalize_path(oracle_file)
    return bool(agent_norm and oracle_norm) and (
        agent_norm.endswith(oracle_norm) or oracle_norm.endswith(agent_norm)
    )


def _module_prefix(value: str) -> str:
    normalized = _normalize_path(value)
    if not normalized:
        return ""
    segments = [segment for segment in normalized.split("/") if segment]
    if len(segments) < 2:
        return ""
    if segments[0] == "source":
        segments = segments[1:]
    if len(segments) >= 3 and segments[0] in {"packages", "apps", "services", "modules"}:
        segments = [segments[1], *segments[2:]]
    while len(segments) >= 2 and segments[0] in {"src", "app", "lib"}:
        segments = segments[1:]
    if len(segments) < 2:
        return ""
    directory_segments = segments[:-1]
    if not directory_segments:
        return ""
    return "/".join(directory_segments[:2])


def _module_prefix_match(agent_file: str, oracle_file: str) -> bool:
    agent_prefix = _module_prefix(agent_file)
    oracle_prefix = _module_prefix(oracle_file)
    return bool(agent_prefix and oracle_prefix) and agent_prefix == oracle_prefix


def _normalize_symbol(value: str) -> str:
    value = value.strip().strip("\"'")
    value = value.replace("(anonymous namespace)", "anonymous_namespace")
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"<[^<>]*>", "", value)
    value = value.split("(", 1)[0].strip()
    if value.startswith("operator"):
        return "operator"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*[?!]?", value):
        return value.lower()
    chain_matches = re.findall(
        r"[A-Za-z_~][A-Za-z0-9_~]*(?:(?:::|->|\.)[A-Za-z_~][A-Za-z0-9_~]*)*",
        value,
    )
    if chain_matches:
        value = chain_matches[-1]
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    for separator in ("::", "->", "."):
        if separator in value:
            value = value.rsplit(separator, 1)[-1]
    return value.lower()


def _symbol_match(agent_symbol: str, oracle_symbol: str) -> bool:
    agent_norm = _normalize_symbol(agent_symbol)
    oracle_norm = _normalize_symbol(oracle_symbol)
    return bool(agent_norm and oracle_norm) and agent_norm == oracle_norm


def run_tiers(extracted: dict[str, str], oracle: dict) -> dict:
    """Apply deterministic Phase 1 checks against the oracle."""
    ground_truth = oracle.get("ground_truth") or {}
    files = ground_truth.get("files") or []
    symbols = ground_truth.get("symbols") or []
    keywords = ground_truth.get("class_keywords")
    judge_only = bool(oracle.get("judge_only"))

    exact_file_hit = any(
        _file_match(extracted["file"], oracle_file) for oracle_file in files
    )
    module_prefix_hit = any(
        _module_prefix_match(extracted["file"], oracle_file) for oracle_file in files
    )
    file_hit = exact_file_hit or module_prefix_hit
    symbol_hit = any(
        _symbol_match(extracted["symbol"], oracle_symbol)
        for oracle_symbol in symbols
    )
    tier1_hit = exact_file_hit or (module_prefix_hit and symbol_hit)

    if keywords is None:
        tier2_hit = None
    else:
        class_text = extracted["class"].lower()
        tier2_hit = any(
            isinstance(keyword, str) and keyword.lower() in class_text
            for keyword in keywords
        )

    overall_pass = tier1_hit and (tier2_hit is True or tier2_hit is None)

    return {
        "tier1_hit": tier1_hit,
        "tier1_file_hit": file_hit,
        "tier1_exact_file_hit": exact_file_hit,
        "tier1_module_prefix_hit": module_prefix_hit,
        "tier1_symbol_hit": symbol_hit,
        "tier2_hit": tier2_hit,
        "overall_pass": overall_pass,
        "needs_judge": judge_only or not overall_pass,
    }
