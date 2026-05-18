"""Phase 1 oracle loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

ORACLE_FILENAME = ".phase1_oracle.json"


def load_oracle(task_dir: str | Path) -> dict | None:
    """Load a task's Phase 1 oracle, or return ``None`` if absent/invalid."""
    oracle_path = Path(task_dir) / ORACLE_FILENAME
    if not oracle_path.is_file():
        return None
    try:
        with oracle_path.open() as f:
            oracle = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return oracle if isinstance(oracle, dict) else None
