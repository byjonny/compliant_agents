from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
# Support both a repo-root checkout and a nested tau2-bench checkout.
if not (REPO / "data").exists() and (REPO / "tau2-bench" / "data").exists():
    REPO = REPO / "tau2-bench"

# Shared filesystem locations used by the HTTP handlers and background workers.
DATA = REPO / "data" / "simulations"
GUARDRAILS = REPO / "guardrail_configs"
POLICY_MAPPER_OUTPUT = REPO / "policy_tool_mapper" / "output"
EXPERIMENTS_DB = HERE / "experiments.json"
MAPPER_EXPERIMENTS_DB = HERE / "mapper_experiments.json"

# The viewer is intentionally local-only by default.
PORT = int(os.environ.get("VIEWER_PORT", "8765"))
HOST = "localhost"

# UI allow-lists keep form input aligned with the installed tau2 components.
ALLOWED_DOMAINS = ["airline", "retail", "telecom"]
ALLOWED_AGENTS = ["llm_agent"]
ALLOWED_USERS = ["user_simulator"]
