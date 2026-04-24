import uuid
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from tau2.data_model.message import Message, ToolCall, ToolMessage
from tau2.environment.environment import Environment
from tau2.guardrails.guard import Guard, GuardVerdict


class GuardrailMiddleware(ABC):
    """
    Abstract base class for guardrail middleware.

    A GuardrailMiddleware sits between the orchestrator and the environment.
    It receives a tool call, evaluates it, and either allows it through or
    returns a structured rejection message to the agent.

    Subclass this to implement different evaluation strategies:
      - SequentialGuardrailMiddleware: evaluates guards one by one, first failure wins
      - Custom implementations: voting, parallel, domain-specific logic, etc.

    The static helper _make_rejection_message() is available to all subclasses
    to produce consistently formatted feedback messages.
    """

    @abstractmethod
    def evaluate(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> tuple[GuardVerdict, Optional[ToolMessage]]:
        """
        Evaluate a tool call and return a verdict.

        Args:
            tool_call: The tool call to evaluate.
            env:       Live environment (read-only access for guards).
            history:   Full conversation trajectory at this point.

        Returns:
            (GuardVerdict(allowed=True), None)                   — call is permitted
            (GuardVerdict(allowed=False, reason=...), ToolMessage) — call is blocked
        """
        ...

    @staticmethod
    def _make_rejection_message(
        tool_call: ToolCall, verdict: GuardVerdict
    ) -> ToolMessage:
        """
        Build a ToolMessage that delivers structured guardrail feedback to the agent.

        The message uses error=False so that:
          - It is not counted toward the orchestrator's num_errors budget
          - The agent LLM reads it as actionable policy guidance, not a crash

        Format is intentionally verbose so the agent can self-correct:
          POLICY GUARDRAIL — {guard_name}
          Tool call '{name}' was blocked before execution.
          Reason: ...
          The tool was NOT executed. ...
        """
        content = (
            f"POLICY GUARDRAIL — {verdict.guard_name}\n"
            f"Tool call '{tool_call.name}' was blocked before execution.\n\n"
            f"Reason: {verdict.reason}\n\n"
            f"The tool was NOT executed. You must resolve the policy violation "
            f"before retrying this call."
        )
        return ToolMessage(
            id=tool_call.id or str(uuid.uuid4()),
            role="tool",
            content=content,
            requestor=tool_call.requestor,
            error=False,
        )


class NullGuardrailMiddleware(GuardrailMiddleware):
    """
    No-op middleware — every tool call is allowed through unchanged.

    This is the default used by BaseOrchestrator when no guardrail config is
    provided. It adds zero overhead (no guard evaluation, no extra calls).
    """

    def evaluate(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> tuple[GuardVerdict, Optional[ToolMessage]]:
        return GuardVerdict(allowed=True, reason="null guardrail"), None


class SequentialGuardrailMiddleware(GuardrailMiddleware):
    """
    Evaluates guards in order, returning on the first block.

    This is the standard implementation: guards are tried one by one and the
    first guard that returns allowed=False wins. Subsequent guards are skipped.
    Guards that raise exceptions are logged and skipped (fail-open).

    Args:
        guards: Ordered list of Guard instances to evaluate.
    """

    def __init__(self, guards: list[Guard]):
        self.guards = guards

    def evaluate(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> tuple[GuardVerdict, Optional[ToolMessage]]:
        for guard in self.guards:
            try:
                if not guard.applies_to(tool_call):
                    continue
                verdict = guard.check(tool_call, env, history)
                verdict.guard_name = guard.name
            except Exception as exc:
                logger.warning(
                    f"Guard '{guard.name}' raised an unexpected exception for "
                    f"tool '{tool_call.name}': {exc}. Failing open (allowing call)."
                )
                continue

            if not verdict.allowed:
                rejection = self._make_rejection_message(tool_call, verdict)
                logger.info(
                    f"[GUARDRAIL BLOCKED] tool='{tool_call.name}' "
                    f"guard='{guard.name}' reason='{verdict.reason}'"
                )
                return verdict, rejection

        return GuardVerdict(allowed=True, reason="All guards passed"), None
