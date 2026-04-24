from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from tau2.data_model.message import Message, ToolCall
from tau2.environment.environment import Environment


@dataclass
class GuardVerdict:
    """Result of a single guard evaluation."""

    allowed: bool
    reason: Optional[str] = None
    guard_name: str = field(default="")


class Guard(ABC):
    """
    Abstract base class for a single guardrail check.

    A Guard inspects one ToolCall — it must not mutate environment state.
    It may call env.use_tool() for read-only lookups or env.get_policy() for
    the policy text. It also receives the full conversation history so that
    LLM-based guards can reason about prior context.

    Exceptions raised inside check() are caught by GuardrailMiddleware, which
    fails open (logs and allows the call) to prevent guard bugs from silently
    blocking all tool executions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and rejection messages."""
        ...

    @abstractmethod
    def applies_to(self, tool_call: ToolCall) -> bool:
        """
        Cheap pre-filter. Return True if this guard should run for this tool call.
        Called before check() — guards returning False are completely skipped.
        """
        ...

    @abstractmethod
    def check(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> GuardVerdict:
        """
        Evaluate the tool call and return a verdict.

        Args:
            tool_call: The tool call about to be executed.
            env:       Live environment — use for READ-ONLY lookups only.
                       Call env.use_tool(name, **kwargs) or env.get_policy().
            history:   Full conversation trajectory up to this point.
                       Useful for LLM-based guards that need conversational context
                       (e.g., the user's stated cancellation reason).

        Returns:
            GuardVerdict with allowed=True to permit the call,
            or allowed=False with a descriptive reason to block it.
        """
        ...
