from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Experiment:
    # A queued tau2 evaluation run plus the UI metadata needed to display it.
    id: str
    name: str
    domain: str
    agent: str
    user: str
    guardrail_config: str
    agent_llm: str
    user_llm: str
    guard_llm: str
    num_trials: int
    max_concurrency: int
    task_split_name: str | None = None
    num_tasks: int | None = None
    task_ids: list[str] = field(default_factory=list)
    prompt_label: str = ""
    dataset_label: str = ""
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    command: list[str] = field(default_factory=list)
    result_id: str = ""
    result_path: str = ""
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None

@dataclass
class MapperExperiment:
    # A policy-tool mapper run, tracked separately from regular simulations.
    id: str
    name: str
    domain: str
    models: list[str]
    mode: str
    embed_model: str = "text-embedding-3-large"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_top_k: int = 20
    skip_mapping: bool = False
    skip_eval: bool = False
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    command: list[str] = field(default_factory=list)
    summary_path: str = ""
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None
