"""Register model pricing for cost tracking in Inspect evaluations.

Loads pricing data from a bundled LiteLLM model-pricing JSON and registers
costs for all providers so Inspect can compute ``total_cost`` for runs,
including AWS Bedrock-backed models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from inspect_ai.model._model_info import ModelCost, ModelInfo, set_model_cost, set_model_info

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_LITELLM_JSON = _DATA_DIR / "litellm_model_prices.json"

# Manual cost overrides ($/1M tokens) for models not in the bundled data.
_MANUAL_COSTS: dict[str, ModelCost] = {
    "openai/gpt-4o-mini": ModelCost(
        input=0.15, output=0.60, input_cache_write=0.15, input_cache_read=0.075
    ),
    "openai/gpt-4o": ModelCost(
        input=2.50, output=10.00, input_cache_write=2.50, input_cache_read=1.25
    ),
}


def _register_model(model_key: str, entry: dict, provider: str) -> None:
    """Register a single model's cost with Inspect."""
    input_cost_per_token = entry.get("input_cost_per_token")
    output_cost_per_token = entry.get("output_cost_per_token")
    if input_cost_per_token is None or output_cost_per_token is None:
        return

    input_cache_write = entry.get("cache_creation_input_token_cost") or 0
    input_cache_read = entry.get("cache_read_input_token_cost") or 0

    cost = ModelCost(
        input=input_cost_per_token * 1_000_000,
        output=output_cost_per_token * 1_000_000,
        input_cache_write=input_cache_write * 1_000_000,
        input_cache_read=input_cache_read * 1_000_000,
    )

    inspect_name = model_key if "/" in model_key else f"{provider}/{model_key}"

    try:
        set_model_cost(inspect_name, cost)
    except (ValueError, AttributeError):
        max_input = entry.get("max_input_tokens")
        max_output = entry.get("max_tokens") or entry.get("max_output_tokens")
        info = ModelInfo(cost=cost)
        if max_input is not None:
            info = info.model_copy(update={"context_length": max_input})
        if max_output is not None:
            info = info.model_copy(update={"output_tokens": max_output})
        set_model_info(inspect_name, info)


def register_all_costs() -> None:
    """Register pricing for all known models from the bundled LiteLLM data."""
    for name, cost in _MANUAL_COSTS.items():
        try:
            set_model_cost(name, cost)
        except (ValueError, AttributeError):
            set_model_info(name, ModelInfo(cost=cost))

    if not _LITELLM_JSON.is_file():
        logger.warning(
            "LiteLLM pricing data not found at %s. Model costs will not be tracked.",
            _LITELLM_JSON,
        )
        return

    try:
        data = json.loads(_LITELLM_JSON.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load LiteLLM pricing data: %s", exc)
        return

    count = 0
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        litellm_provider = entry.get("litellm_provider", "")
        if not isinstance(litellm_provider, str) or not litellm_provider:
            continue

        provider = litellm_provider.split("_")[0]
        _register_model(key, entry, provider)
        count += 1

    logger.debug("Registered costs for %d models from LiteLLM data", count)
