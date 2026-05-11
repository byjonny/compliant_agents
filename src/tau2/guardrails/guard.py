from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.data_model.message import Message, ToolCall
from tau2.environment.environment import Environment


@dataclass
class GuardVerdict:
    """
    Result of a single guard evaluation.

    This is the public interface shared by all guards, the middleware, and the
    orchestrator. It is intentionally minimal — richer internal representations
    (e.g. GuardJudgement for LLM-based guards) must be converted to this before
    returning from check().
    """

    allowed: bool
    reason: Optional[str] = None
    guard_name: str = field(default="")


# ---------------------------------------------------------------------------
# Rich verdict models — used internally by LLM-based guards
# ---------------------------------------------------------------------------


class PolicyPassage(BaseModel):
    """A single verbatim policy passage pre-loaded for a specific tool."""

    id: str                        # e.g. "PS-001"
    text: str                      # verbatim policy text
    section: Optional[str] = None  # nearest section heading


class VerdictType(str, Enum):
    ALLOW    = "ALLOW"     # call is policy-compliant — allow execution
    DENY     = "DENY"      # call violates policy — block and give feedback
    ESCALATE = "ESCALATE"  # genuinely uncertain — block and flag for review


class PolicyRuleCheck(BaseModel):
    """Per-passage compliance assessment."""

    passage_id: str
    assessment: Literal["compliant", "violation", "unclear"]
    note: Optional[str] = None


class GuardJudgement(BaseModel):
    """
    Structured LLM output for a policy compliance decision.

    Internal to LLM-based guards. Must be converted to GuardVerdict before
    returning from check() so the middleware/orchestrator stay decoupled from
    the richer schema.
    """

    verdict: VerdictType
    reason: str = Field(
        description=(
            "One concise sentence explaining the verdict. If blocking, name the "
            "specific policy requirement that is not satisfied and the concrete "
            "tool-call issue, such as an invalid argument, missing prerequisite, "
            "or missing context. Do not include an action plan here."
        )
    )
    feedback: Optional[str] = Field(
        default=None,
        description=(
            "For DENY or ESCALATE, give the agent one actionable compliant next "
            "step, such as what to ask the user, what information to verify, "
            "which prerequisite tool to call first, or what policy-compliant "
            "alternative to offer. If the decision is final because the policy "
            "violation cannot be fixed by another conversation turn or more "
            "information, explicitly say that the agent should not retry the "
            "blocked tool call and should explain the final denial to the user "
            "instead. Leave null for ALLOW."
        ),
    )
    intent_summary: str                  # what the user is trying to achieve
    policy_check: list[PolicyRuleCheck]  # per-passage assessment


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
