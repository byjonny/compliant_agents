from __future__ import annotations

import os
import signal
from pathlib import Path

from .config import GUARDRAILS, REPO

def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO))

def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.lower())
    return "_".join(part for part in cleaned.split("_") if part)[:48] or "experiment"

def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if local_bin not in path_parts:
        env["PATH"] = os.pathsep.join([local_bin, *path_parts])
    return env

def _safe_guardrail_config(value: str) -> str:
    path = (REPO / (value or "guardrail_configs/null.json")).resolve()
    root = GUARDRAILS.resolve()
    if root not in path.parents or path.suffix != ".json" or not path.exists():
        raise ValueError(f"Invalid guardrail config: {value}")
    return _repo_rel(path)

def _parse_task_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    return []

def _parse_models(value) -> list[str]:
    if isinstance(value, list):
        models = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        models = [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    else:
        models = []
    deduped = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped

def _prompt_label(config: str) -> str:
    name = Path(config).name
    if name == "null.json":
        return "No Guard"
    if "minimal" in name:
        return "Minimal Policy"
    if "full" in name:
        return "Full Policy"
    if "tool_results" in name:
        return "Tool Results"
    return "LLM Guard"

def _terminate_pid(pid: int) -> bool:
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except Exception:
        pass
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except Exception:
        return False
