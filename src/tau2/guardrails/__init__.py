"""
Guardrail middleware for agent tool calls.

Guardrails are natively integrated into BaseOrchestrator. Every Orchestrator
instance has a guardrail_middleware; the default is NullGuardrailMiddleware
which passes all calls through unchanged.

To activate guardrails, specify a JSON config file:

  CLI:
    uv run tau2 run --domain airline --guardrail-config guardrail_configs/airline_defaults.json

  Python:
    from tau2.data_model.simulation import TextRunConfig
    config = TextRunConfig(
        domain="airline",
        agent="llm_agent",
        llm_agent="gpt-4.1-mini",
        guardrail_config_path="guardrail_configs/airline_defaults.json",
    )

  Or build the middleware directly:
    from tau2.guardrails import SequentialGuardrailMiddleware
    from tau2.guardrails.guards import FlightStatusGuard, CancellationEligibilityGuard
    middleware = SequentialGuardrailMiddleware(guards=[
        FlightStatusGuard(),
        CancellationEligibilityGuard(),
    ])
    orchestrator = Orchestrator(..., guardrail_middleware=middleware)

Architecture:
  GuardrailMiddleware (abstract)    — interface; implement evaluate()
    NullGuardrailMiddleware         — no-op default; every call allowed
    SequentialGuardrailMiddleware   — run guards in order, first block wins

  Guard (abstract)                  — interface; implement applies_to() + check()
    FlightStatusGuard               — verify new flights are "available"
    CancellationEligibilityGuard    — verify cancellation policy conditions
    LLMPolicyGuard                  — LLM judge using policy + conversation history

JSON config format:
  See tau2/guardrails/loader.py and guardrail_configs/*.json
"""

from tau2.guardrails.guard import Guard, GuardVerdict
from tau2.guardrails.loader import build_middleware_from_config, load_middleware_from_file
from tau2.guardrails.middleware import (
    GuardrailMiddleware,
    NullGuardrailMiddleware,
    SequentialGuardrailMiddleware,
)

__all__ = [
    "Guard",
    "GuardVerdict",
    "GuardrailMiddleware",
    "NullGuardrailMiddleware",
    "SequentialGuardrailMiddleware",
    "build_middleware_from_config",
    "load_middleware_from_file",
]
