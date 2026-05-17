from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
if not (REPO / "data").exists() and (REPO / "tau2-bench" / "data").exists():
    REPO = REPO / "tau2-bench"

DATA = REPO / "data" / "simulations"
GUARDRAILS = REPO / "guardrail_configs"
POLICY_MAPPER_OUTPUT = REPO / "policy_tool_mapper" / "output"
EXPERIMENTS_DB = HERE / "experiments.json"
MAPPER_EXPERIMENTS_DB = HERE / "mapper_experiments.json"
PORT = int(os.environ.get("VIEWER_PORT", "8765"))
HOST = "localhost"

ALLOWED_DOMAINS = ["airline", "retail", "telecom", "banking_knowledge"]
ALLOWED_AGENTS = ["llm_agent"]
ALLOWED_USERS = ["user_simulator"]
