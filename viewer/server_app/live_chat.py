from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from .config import ALLOWED_DOMAINS, GUARDRAILS, REPO
from .utils import _repo_rel, _safe_guardrail_config

SRC = REPO / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_live_sessions: dict[str, "LiveChatSession"] = {}
_live_sessions_lock = threading.Lock()
LIVE_USE_CASES_PATH = REPO / "viewer" / "server_app" / "data" / "live_use_cases.json"


def _clean_snippets(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    snippets = []
    for snippet in value:
        if not isinstance(snippet, dict):
            continue
        label = str(snippet.get("label") or "").strip()
        text = str(snippet.get("text") or "").strip()
        if label and text:
            snippets.append({"label": label, "text": text})
    return snippets


def _load_live_use_cases() -> dict[str, list[dict]]:
    try:
        raw = json.loads(LIVE_USE_CASES_PATH.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    use_cases: dict[str, list[dict]] = {}
    for domain, cases in raw.items():
        if domain not in ALLOWED_DOMAINS or not isinstance(cases, list):
            continue
        clean_cases = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("id") or "").strip()
            task_id = str(case.get("task_id") or "").strip()
            title = str(case.get("title") or "").strip()
            opening = str(case.get("opening") or "").strip()
            if not case_id or not task_id or not title or not opening:
                continue
            clean_cases.append(
                {
                    "id": case_id,
                    "task_id": task_id,
                    "title": title,
                    "summary": str(case.get("summary") or "").strip(),
                    "opening": opening,
                    "snippets": _clean_snippets(case.get("snippets")),
                }
            )
        if clean_cases:
            use_cases[domain] = clean_cases[:2]
    return use_cases


def get_live_options() -> dict:
    configs = [
        _repo_rel(path)
        for path in sorted(GUARDRAILS.glob("*.json"))
        if path.is_file()
    ]
    defaults = {}
    domains = []
    use_cases = {}
    configured_use_cases = _load_live_use_cases()
    for domain in ALLOWED_DOMAINS:
        domain_use_cases = configured_use_cases.get(domain, [])
        if domain_use_cases:
            domains.append(domain)
            use_cases[domain] = domain_use_cases
        candidate = f"guardrail_configs/{domain}_llm_guard.json"
        defaults[domain] = candidate if (REPO / candidate).exists() else "guardrail_configs/null.json"
    return {
        "schema_version": 2,
        "domains": domains,
        "use_cases": use_cases,
        "use_case_source": _repo_rel(LIVE_USE_CASES_PATH),
        "guardrail_configs": configs,
        "default_guardrails": defaults,
        "null_guardrail": "guardrail_configs/null.json",
        "default_agent_llm": "gpt-4.1-mini",
        "default_guard_llm": "gpt-4.1-mini",
    }


def _make_controlled_user_class():
    from tau2.data_model.message import (
        APICompatibleMessage,
        Message,
        MultiToolMessage,
        UserMessage,
    )
    from tau2.user.user_simulator_base import (
        OUT_OF_SCOPE,
        STOP,
        TRANSFER,
        HalfDuplexUser,
        ValidUserInputMessage,
    )

    class _ControlledUserState(BaseModel):
        messages: list[APICompatibleMessage]

    class ControlledUser(HalfDuplexUser[_ControlledUserState]):
        def __init__(self, instructions: Optional[str] = None, tools: Optional[list] = None):
            super().__init__(instructions=instructions, tools=tools)
            self._observation: list[Message] = []
            self._next_action: Optional[UserMessage] = None
            self._turn_finished = threading.Event()
            self._turn_finished.set()
            self._lock = threading.Lock()

        @property
        def observation(self) -> list[Message]:
            with self._lock:
                return deepcopy(self._observation)

        @property
        def is_user_turn(self) -> bool:
            return not self._turn_finished.is_set()

        def set_action(self, action_msg: UserMessage) -> None:
            with self._lock:
                if self._turn_finished.is_set():
                    raise RuntimeError("The live chat is not waiting for user input yet.")
                self._next_action = action_msg
                self._turn_finished.set()

        def stop(
            self,
            message: Optional[ValidUserInputMessage] = None,
            state: Optional[_ControlledUserState] = None,
        ) -> None:
            history = deepcopy(state.messages) if state else []
            with self._lock:
                self._observation = history + ([message] if message else [])
                self._turn_finished.set()

        def get_init_state(
            self,
            message_history: Optional[list[Message]] = None,
        ) -> _ControlledUserState:
            return _ControlledUserState(messages=list(message_history or []))

        def generate_next_message(
            self,
            message: ValidUserInputMessage,
            state: _ControlledUserState,
        ) -> tuple[UserMessage, _ControlledUserState]:
            with self._lock:
                self._turn_finished.clear()
                if isinstance(message, MultiToolMessage):
                    state.messages.extend(message.tool_messages)
                elif message is not None:
                    state.messages.append(message)
                self._observation = deepcopy(state.messages)

            self._turn_finished.wait()

            with self._lock:
                response = self._next_action
                self._next_action = None
            if response is None:
                response = UserMessage(role="user", content=STOP)
            state.messages.append(response)
            return response, state

        @classmethod
        def is_stop(cls, message: UserMessage) -> bool:
            if message.is_tool_call() or message.content is None:
                return False
            return (
                STOP in message.content
                or TRANSFER in message.content
                or OUT_OF_SCOPE in message.content
            )

    return ControlledUser


@dataclass
class LiveChatSession:
    id: str
    domain: str
    task_id: str
    guardrail_config: str
    agent_llm: str
    guard_llm: str
    max_steps: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "starting"
    error: str | None = None
    thread: threading.Thread | None = None
    orchestrator: object | None = None
    user: object | None = None
    simulation_run: object | None = None
    done_event: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.status = "running"
            assert self.orchestrator is not None
            self.simulation_run = self.orchestrator.run()
            self.status = "done"
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
        finally:
            self.done_event.set()

    def wait_for_user_turn(self, timeout: float = 180.0) -> bool:
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.done_event.is_set():
                return True
            if self.user is not None and getattr(self.user, "is_user_turn", False):
                return True
            self.done_event.wait(timeout=0.05)
        return False

    def snapshot(self) -> dict:
        messages = []
        guardrail_events = []
        if self.orchestrator is not None:
            try:
                messages = [
                    msg.model_dump(mode="json")
                    for msg in self.orchestrator.get_trajectory()
                ]
            except Exception:
                messages = [
                    msg.model_dump(mode="json")
                    for msg in getattr(self.orchestrator, "trajectory", [])
                    if hasattr(msg, "model_dump")
                ]
            guardrail_events = [
                ev.model_dump(mode="json")
                for ev in getattr(self.orchestrator, "_guardrail_events", [])
                if hasattr(ev, "model_dump")
            ]
        if self.simulation_run is not None:
            guardrail_events = [
                ev.model_dump(mode="json")
                for ev in (getattr(self.simulation_run, "guardrail_events", None) or [])
            ]
        return {
            "id": self.id,
            "domain": self.domain,
            "task_id": self.task_id,
            "guardrail_config": self.guardrail_config,
            "agent_llm": self.agent_llm,
            "guard_llm": self.guard_llm,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "is_user_turn": bool(self.user is not None and getattr(self.user, "is_user_turn", False)),
            "messages": messages,
            "guardrail_events": guardrail_events,
            "guardrail_block_count": len(guardrail_events),
        }


def _load_task(domain: str, task_id: str):
    from tau2.run import get_tasks

    tasks = get_tasks(task_set_name=domain, task_split_name=None)
    for task in tasks:
        if str(task.id) == str(task_id):
            return task
    raise ValueError(f"Task {task_id} was not found for domain {domain}")


def _build_live_session(payload: dict) -> LiveChatSession:
    from tau2.data_model.message import UserMessage
    from tau2.guardrails.loader import load_middleware_from_file
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_agent, build_environment
    from tau2.utils.tools import parse_action_string

    domain = str(payload.get("domain") or "airline").strip()
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Unsupported live chat domain: {domain}")

    live_use_cases = _load_live_use_cases().get(domain, [])
    default_task_id = live_use_cases[0]["task_id"] if live_use_cases else "0"
    task_id = str(payload.get("task_id") or default_task_id)
    task = _load_task(domain, task_id)

    default_guardrail = f"guardrail_configs/{domain}_llm_guard.json"
    if not (REPO / default_guardrail).exists():
        default_guardrail = "guardrail_configs/null.json"
    guardrail_config = _safe_guardrail_config(payload.get("guardrail_config") or default_guardrail)

    agent_llm = str(payload.get("agent_llm") or "gpt-4.1-mini").strip()
    guard_llm = str(payload.get("guard_llm") or "").strip()
    max_steps = max(2, min(100, int(payload.get("max_steps") or 40)))

    environment = build_environment(domain)
    agent = build_agent(
        "llm_agent",
        environment,
        llm=agent_llm,
        llm_args={"temperature": 0},
        task=task,
    )
    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None

    ControlledUser = _make_controlled_user_class()
    user = ControlledUser(instructions=str(task.user_scenario), tools=user_tools)
    middleware = load_middleware_from_file(
        REPO / guardrail_config,
        llm_override=guard_llm or None,
    )

    session = LiveChatSession(
        id=uuid.uuid4().hex[:10],
        domain=domain,
        task_id=task_id,
        guardrail_config=guardrail_config,
        agent_llm=agent_llm,
        guard_llm=guard_llm,
        max_steps=max_steps,
    )
    session.user = user
    session.orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        guardrail_middleware=middleware,
    )

    prefill = str(payload.get("initial_message") or "").strip()
    if prefill:
        session._prefill_message = parse_action_string(prefill, requestor="user")
        assert isinstance(session._prefill_message, UserMessage)
    return session


def create_live_session(payload: dict) -> dict:
    session = _build_live_session(payload)
    with _live_sessions_lock:
        _live_sessions[session.id] = session
    session.start()
    session.wait_for_user_turn(timeout=30.0)
    prefill = getattr(session, "_prefill_message", None)
    if prefill is not None and session.user is not None and getattr(session.user, "is_user_turn", False):
        session.user.set_action(prefill)
        session.wait_for_user_turn(timeout=180.0)
    return session.snapshot()


def get_live_session(session_id: str) -> tuple[dict, int]:
    with _live_sessions_lock:
        session = _live_sessions.get(session_id)
    if session is None:
        return {"error": "live chat session not found"}, 404
    return session.snapshot(), 200


def send_live_message(session_id: str, payload: dict) -> tuple[dict, int]:
    from tau2.utils.tools import parse_action_string

    with _live_sessions_lock:
        session = _live_sessions.get(session_id)
    if session is None:
        return {"error": "live chat session not found"}, 404
    if session.done_event.is_set():
        return session.snapshot(), 200
    if session.user is None or not getattr(session.user, "is_user_turn", False):
        return {"error": "agent is still working", **session.snapshot()}, 409
    message = str(payload.get("message") or "").strip()
    if not message:
        return {"error": "message is required"}, 400
    try:
        session.user.set_action(parse_action_string(message, requestor="user"))
        session.wait_for_user_turn(timeout=float(payload.get("timeout") or 180.0))
    except Exception as exc:
        session.error = str(exc)
        return {"error": str(exc), **session.snapshot()}, 500
    return session.snapshot(), 200
