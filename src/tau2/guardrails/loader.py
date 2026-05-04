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
          "type": "llm_guard",
          "tool_name": "cancel_reservation",
          "mappings_file": "policy_tool_mapper/output/airline-mappings-gpt-4.1-mini.json",
          "llm": "claude-haiku-4-5-20251001",
          "history_window": 10
        }
      ]
    }

Guard types
-----------
  "flight_status"            → FlightStatusGuard       (no extra args)
  "cancellation_eligibility" → CancellationEligibilityGuard (no extra args)
  "llm_guard"                → LLMGuard  (per-tool; policy passages loaded from mappings file)
      tool_name       (required) — tool name this guard is responsible for
      mappings_file   (required) — path to policy_tool_mapper output JSON
                                   resolved relative to CWD when running tau2
      llm             (optional) — model string, default "claude-haiku-4-5-20251001"
      llm_args        (optional) — dict of extra kwargs for litellm.completion()
      template_path   (optional) — path to a custom Jinja2 prompt template
      history_window  (optional) — number of recent messages to include (default 10)
  "llm_policy"               → LLMPolicyGuard (legacy; fetches full policy from env)
      llm               (required) — model string, e.g. "gpt-4.1-mini"
      llm_args          (optional) — dict of extra kwargs for generate()
      tool_names_filter (optional) — list of tool names to check; null = all
      history_window    (optional) — number of recent messages (default 10)
"""

import json
from pathlib import Path
from typing import Union

from tau2.guardrails.guard import Guard, PolicyPassage
from tau2.guardrails.guards.cancellation_guard import CancellationEligibilityGuard
from tau2.guardrails.guards.flight_status_guard import FlightStatusGuard
from tau2.guardrails.guards.llm_policy_guard import LLMGuard, LLMPolicyGuard
from tau2.guardrails.middleware import (
    GuardrailMiddleware,
    NullGuardrailMiddleware,
    SequentialGuardrailMiddleware,
)

_SIMPLE_GUARD_REGISTRY: dict[str, type[Guard]] = {
    "flight_status": FlightStatusGuard,
    "cancellation_eligibility": CancellationEligibilityGuard,
    "llm_policy": LLMPolicyGuard,
}


def _build_llm_guard(cfg: dict) -> LLMGuard:
    """Construct an LLMGuard by loading policy passages from the mappings file."""
    tool_name = cfg["tool_name"]

    mappings_path = Path(cfg["mappings_file"])
    if not mappings_path.is_absolute():
        mappings_path = Path.cwd() / mappings_path

    with open(mappings_path) as f:
        mappings_data = json.load(f)

    entry = next(
        (m for m in mappings_data.get("mappings", []) if m["tool_id"] == tool_name),
        None,
    )
    if entry is None:
        available = [m["tool_id"] for m in mappings_data.get("mappings", [])]
        raise ValueError(
            f"Tool '{tool_name}' not found in '{mappings_path}'. "
            f"Available tool IDs: {available}"
        )

    passages = [
        PolicyPassage(
            id=s["id"],
            text=s["text"],
            section=s.get("section"),
        )
        for s in entry.get("statements", [])
    ]

    return LLMGuard(
        tool_name=tool_name,
        tool_description=entry.get("tool_name", tool_name),
        policy_passages=passages,
        llm=cfg.get("llm", "claude-haiku-4-5-20251001"),
        llm_args=cfg.get("llm_args"),
        template_path=cfg.get("template_path"),
        history_window=cfg.get("history_window", 10),
    )


def _build_guard(cfg: dict) -> Guard:
    guard_type = cfg.get("type")

    if guard_type == "llm_guard":
        return _build_llm_guard(cfg)

    cls = _SIMPLE_GUARD_REGISTRY.get(guard_type)
    if cls is None:
        known = ", ".join(sorted([*_SIMPLE_GUARD_REGISTRY, "llm_guard"]))
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

    raise ValueError(
        f"Unknown middleware type '{middleware_type}'. Known types: null, sequential"
    )


def load_middleware_from_file(path: Union[str, Path]) -> GuardrailMiddleware:
    """Load a GuardrailMiddleware from a JSON config file."""
    with open(path) as f:
        cfg = json.load(f)
    return build_middleware_from_config(cfg)
