from dataclasses import dataclass

from tau2.data_model.message import Message, ToolCall, ToolMessage
from tau2.guardrails.guard import Guard, GuardVerdict
from tau2.guardrails.loader import build_middleware_from_config
from tau2.guardrails.middleware import (
    NullGuardrailMiddleware,
    SequentialGuardrailMiddleware,
)
from tau2.orchestrator.orchestrator import BaseOrchestrator


class DummyGuard(Guard):
    def __init__(
        self,
        name: str,
        *,
        applies: bool = True,
        allowed: bool = True,
        reason: str = "checked",
        raises: bool = False,
    ):
        self._name = name
        self.applies = applies
        self.allowed = allowed
        self.reason = reason
        self.raises = raises
        self.applies_calls = 0
        self.check_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def applies_to(self, tool_call: ToolCall) -> bool:
        self.applies_calls += 1
        return self.applies

    def check(
        self,
        tool_call: ToolCall,
        env,
        history: list[Message],
    ) -> GuardVerdict:
        self.check_calls += 1
        if self.raises:
            raise RuntimeError("guard exploded")
        return GuardVerdict(allowed=self.allowed, reason=self.reason)


class FakeEnvironment:
    def __init__(self, *, error: bool = False):
        self.error = error
        self.calls: list[ToolCall] = []

    def get_response(self, tool_call: ToolCall) -> ToolMessage:
        self.calls.append(tool_call)
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content=f"executed:{tool_call.name}",
            requestor=tool_call.requestor,
            error=self.error,
        )


@dataclass
class DummyParticipant:
    def generate_next_message(self, *args, **kwargs):
        raise AssertionError("not used by these unit tests")


class HarnessOrchestrator(BaseOrchestrator):
    def initialize(self) -> None:
        pass

    def step(self) -> None:
        pass

    def get_trajectory(self) -> list[Message]:
        return []

    def get_messages(self) -> list[Message]:
        return []

    def _validate_mode_compatibility(self) -> None:
        pass

    def _check_termination(self) -> None:
        pass

    def _finalize(self):
        return None


def tool_call(name: str = "write_tool", *, requestor: str = "assistant") -> ToolCall:
    return ToolCall(
        id=f"{name}-1",
        name=name,
        arguments={"value": 1},
        requestor=requestor,
    )


def test_null_middleware_executes_tool_call_unchanged():
    env = FakeEnvironment()
    call = tool_call()

    verdict, rejection = NullGuardrailMiddleware().evaluate(call, env, [])
    result = env.get_response(call)

    assert verdict.allowed is True
    assert rejection is None
    assert result.content == "executed:write_tool"
    assert env.calls == [call]


def test_sequential_middleware_blocks_first_failing_guard_without_execution():
    env = FakeEnvironment()
    skipped_after_block = DummyGuard("after-block")
    middleware = SequentialGuardrailMiddleware(
        guards=[
            DummyGuard("skip-me", applies=False),
            DummyGuard("blocker", allowed=False, reason="policy violation"),
            skipped_after_block,
        ]
    )

    verdict, rejection = middleware.evaluate(tool_call(), env, [])

    assert verdict.allowed is False
    assert verdict.guard_name == "blocker"
    assert "policy violation" in rejection.content
    assert "was blocked before execution" in rejection.content
    assert env.calls == []
    assert skipped_after_block.applies_calls == 0


def test_sequential_middleware_fails_open_when_guard_raises():
    env = FakeEnvironment()
    middleware = SequentialGuardrailMiddleware(
        guards=[
            DummyGuard("broken", raises=True),
            DummyGuard("allow-after-error", allowed=True),
        ]
    )

    call = tool_call()
    verdict, rejection = middleware.evaluate(call, env, [])
    result = env.get_response(call)

    assert verdict.allowed is True
    assert rejection is None
    assert result.content == "executed:write_tool"
    assert env.calls


def test_loader_builds_null_and_sequential_middleware():
    null_middleware = build_middleware_from_config({"type": "null"})
    sequential_middleware = build_middleware_from_config(
        {"type": "sequential", "guards": [{"type": "flight_status"}]}
    )

    assert isinstance(null_middleware, NullGuardrailMiddleware)
    assert isinstance(sequential_middleware, SequentialGuardrailMiddleware)
    assert [guard.name for guard in sequential_middleware.guards] == [
        "FlightStatusGuard"
    ]


def test_orchestrator_one_line_guarded_call_preserves_block_accounting():
    env = FakeEnvironment()
    orchestrator = HarnessOrchestrator(
        domain="mock",
        agent=DummyParticipant(),
        user=DummyParticipant(),
        environment=env,
        task=None,
        guardrail_middleware=SequentialGuardrailMiddleware(
            guards=[DummyGuard("blocker", allowed=False, reason="not allowed")]
        ),
    )

    result = orchestrator._execute_tool_calls([tool_call("dangerous")])[0]

    assert env.calls == []
    assert result.error is False
    assert "not allowed" in result.content
    assert orchestrator.num_errors == 0
    assert orchestrator.guardrail_block_count == 1
    assert len(orchestrator._guardrail_events) == 1
    assert orchestrator._guardrail_events[0].tool_name == "dangerous"
    assert orchestrator._guardrail_events[0].guard_name == "blocker"


def test_orchestrator_one_line_guarded_call_preserves_error_accounting():
    env = FakeEnvironment(error=True)
    orchestrator = HarnessOrchestrator(
        domain="mock",
        agent=DummyParticipant(),
        user=DummyParticipant(),
        environment=env,
        task=None,
        guardrail_middleware=SequentialGuardrailMiddleware(
            guards=[DummyGuard("allow", allowed=True)]
        ),
    )

    result = orchestrator._execute_tool_calls([tool_call("allowed")])[0]

    assert result.error is True
    assert env.calls
    assert orchestrator.num_errors == 1
    assert orchestrator.guardrail_block_count == 0
    assert orchestrator._guardrail_events == []
