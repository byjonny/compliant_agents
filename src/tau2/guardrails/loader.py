"""
Build a GuardrailMiddleware instance from a JSON config dict or file.

JSON format
-----------
Null (default, no-op):
    {"type": "null"}

Sequential (guards evaluated in order, first failure wins):
    {
      "type": "sequential",
      "guards": [
        {"type": "flight_status"},
        {"type": "cancellation_eligibility"},
        {
          "type": "llm_policy",
          "llm": "gpt-4.1-mini",
          "tool_names_filter": ["cancel_reservation"],
          "history_window": 10
        }
      ]
    }

Guard types
-----------
  "flight_status"          → FlightStatusGuard       (no extra args)
  "cancellation_eligibility" → CancellationEligibilityGuard (no extra args)
  "llm_policy"             → LLMPolicyGuard
      llm               (required) — model string, e.g. "gpt-4.1-mini"
      llm_args          (optional) — dict of extra kwargs for generate()
      tool_names_filter (optional) — list of tool names to check; null = all
      history_window    (optional) — number of recent messages (default 10)
"""

import json
from pathlib import Path
from typing import Union

from tau2.guardrails.guard import Guard
from tau2.guardrails.guards.cancellation_guard import CancellationEligibilityGuard
from tau2.guardrails.guards.flight_status_guard import FlightStatusGuard
from tau2.guardrails.guards.llm_policy_guard import LLMPolicyGuard
from tau2.guardrails.middleware import (
    GuardrailMiddleware,
    NullGuardrailMiddleware,
    SequentialGuardrailMiddleware,
)

_GUARD_REGISTRY: dict[str, type[Guard]] = {
    "flight_status": FlightStatusGuard,
    "cancellation_eligibility": CancellationEligibilityGuard,
    "llm_policy": LLMPolicyGuard,
}

_MIDDLEWARE_REGISTRY: dict[str, type[GuardrailMiddleware]] = {
    "null": NullGuardrailMiddleware,
    "sequential": SequentialGuardrailMiddleware,
}


def _build_guard(cfg: dict) -> Guard:
    guard_type = cfg.get("type")
    cls = _GUARD_REGISTRY.get(guard_type)
    if cls is None:
        known = ", ".join(sorted(_GUARD_REGISTRY))
        raise ValueError(
            f"Unknown guard type '{guard_type}'. Known types: {known}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return cls(**kwargs)


def build_middleware_from_config(cfg: dict) -> GuardrailMiddleware:
    """Construct a GuardrailMiddleware from a plain-dict config."""
    middleware_type = cfg.get("type", "null")

    if middleware_type == "null":
        return NullGuardrailMiddleware()

    if middleware_type == "sequential":
        guards = [_build_guard(g) for g in cfg.get("guards", [])]
        return SequentialGuardrailMiddleware(guards=guards)

    known = ", ".join(sorted(_MIDDLEWARE_REGISTRY))
    raise ValueError(
        f"Unknown middleware type '{middleware_type}'. Known types: {known}"
    )


def load_middleware_from_file(path: Union[str, Path]) -> GuardrailMiddleware:
    """Load a GuardrailMiddleware from a JSON config file."""
    with open(path) as f:
        cfg = json.load(f)
    return build_middleware_from_config(cfg)
