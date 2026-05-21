from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .config import REPO
from .mapper import (
    _build_mapper_command,
    _compact_mapper_experiment,
    _ensure_mapper_worker,
)
from .models import MapperExperiment
from .store import (
    _mapper_experiment_lock,
    _mapper_experiment_queue,
    _mapper_experiments,
    _save_mapper_experiments_db,
)
from .utils import _parse_models

SRC = REPO / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXCLUDED_DOMAINS = {"airline", "retail", "telecom"}


def _rel(path: Path) -> str:
    # Display paths relative to the repo instead of absolute local paths.
    return str(path.resolve().relative_to(REPO))


def _safe_repo_file(value: str) -> Path:
    # Only stage files that are inside this checkout and already exist.
    path = (REPO / value).resolve()
    if REPO.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Invalid file path: {value}")
    return path


def _list_files(root: Path, patterns: tuple[str, ...]) -> list[dict]:
    # Find candidate policy/task/tool files for the domain journey form.
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    deduped = sorted(set(files))
    return [{"path": _rel(path), "name": path.name} for path in deduped]


def _domain_names() -> list[str]:
    # A domain can be discovered from either its data folder or source folder.
    names = set()
    data_root = REPO / "data" / "tau2" / "domains"
    src_root = REPO / "src" / "tau2" / "domains"
    for root in (data_root, src_root):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("__"):
                names.add(child.name)
    return sorted(name for name in names if name not in EXCLUDED_DOMAINS)


def get_domain_journey_options() -> dict:
    # Build all file choices needed to import a custom domain into the mapper.
    domains = []
    for domain in _domain_names():
        data_dir = REPO / "data" / "tau2" / "domains" / domain
        src_dir = REPO / "src" / "tau2" / "domains" / domain
        policies = _list_files(data_dir, ("*policy*.md", "*Policy*.md", "*.md"))
        tasks = _list_files(data_dir, ("tasks.json", "task_*.json", "tasks/*.json"))
        tool_files = []
        if (src_dir / "tools.py").exists():
            tool_files.append({"path": _rel(src_dir / "tools.py"), "name": "tools.py"})
        tool_files.extend(_list_files(data_dir, ("*Tools.json", "*tools.json", "*openapi*.json")))
        ground_truth = REPO / "policy_tool_mapper" / "ground_truth" / f"{domain}-ground-truth.json"
        domains.append(
            {
                "name": domain,
                "data_dir": _rel(data_dir) if data_dir.exists() else "",
                "src_dir": _rel(src_dir) if src_dir.exists() else "",
                "policies": policies,
                "tasks": tasks,
                "tools": tool_files,
                "has_ground_truth": ground_truth.exists(),
            }
        )
    return {
        "domains": domains,
        "modes": ["retrieval", "llm"],
        "default_models": ["gpt-4.1-mini"],
        "default_embed_model": "text-embedding-3-large",
        "default_ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "default_ce_top_k": 30,
        "excluded_domains": sorted(EXCLUDED_DOMAINS),
    }


def _title_domain(domain: str) -> str:
    # The mapper expects staged input filenames to start with TitleCase domains.
    return domain[0].upper() + domain[1:] if domain else domain


def _stage_policy(domain: str, source: Path) -> Path:
    # Copy policy markdown into the mapper's conventional input location.
    target = REPO / "policy_tool_mapper" / "input" / f"{_title_domain(domain)}Policy.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    return target


def _stage_tasks(domain: str, source: Path) -> Path:
    # Copy task definitions into the mapper's conventional input location.
    target = REPO / "policy_tool_mapper" / "input" / f"{_title_domain(domain)}Tasks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _stage_tools(domain: str, source: Path) -> Path:
    # Convert Python toolkits to JSON schemas, or normalize existing JSON tools.
    target = REPO / "policy_tool_mapper" / "input" / f"{_title_domain(domain)}Tools.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".py":
        sys.path.insert(0, str(REPO / "policy_tool_mapper"))
        from build_tools_json import extract_tools

        schemas = extract_tools(source)
        target.write_text(json.dumps(schemas, indent=2))
    else:
        data = json.loads(source.read_text())
        target.write_text(json.dumps(data, indent=2))
    return target


def _schedule_journey_mapper(payload: dict, *, skip_eval: bool) -> dict:
    # Create a mapper experiment as the final step of the domain journey import.
    domain = str(payload["domain"]).strip().lower()
    mode = str(payload.get("mode") or "retrieval").strip().lower()
    if mode not in {"retrieval", "llm"}:
        raise ValueError(f"Unsupported mapper mode: {mode}")
    models = _parse_models(payload.get("models") or "gpt-4.1-mini")
    if not models:
        raise ValueError("At least one model is required")
    exp = MapperExperiment(
        id=uuid.uuid4().hex[:10],
        name=str(payload.get("name") or f"{domain} journey mapper").strip(),
        domain=domain,
        models=models,
        mode=mode,
        embed_model=str(payload.get("embed_model") or "text-embedding-3-large").strip(),
        ce_model=str(payload.get("ce_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip(),
        ce_top_k=max(1, min(200, int(payload.get("ce_top_k") or 30))),
        skip_eval=skip_eval,
    )
    exp.command = _build_mapper_command(exp)
    with _mapper_experiment_lock:
        _mapper_experiments[exp.id] = exp
        _save_mapper_experiments_db()
    _ensure_mapper_worker()
    _mapper_experiment_queue.put(exp.id)
    return _compact_mapper_experiment(exp)


def start_domain_journey(payload: dict) -> dict:
    # Validate the selected domain files, stage them, then queue mapper work.
    domain = str(payload.get("domain") or "").strip().lower()
    if not domain:
        raise ValueError("domain is required")
    if domain in EXCLUDED_DOMAINS:
        raise ValueError(f"{domain} is already built in and cannot be imported again")
    if domain not in _domain_names():
        raise ValueError(f"Domain folder not found: {domain}")

    policy_file = _safe_repo_file(str(payload.get("policy_file") or ""))
    tasks_file = _safe_repo_file(str(payload.get("tasks_file") or ""))
    tools_file = _safe_repo_file(str(payload.get("tools_file") or ""))

    staged_policy = _stage_policy(domain, policy_file)
    staged_tasks = _stage_tasks(domain, tasks_file)
    staged_tools = _stage_tools(domain, tools_file)

    ground_truth = REPO / "policy_tool_mapper" / "ground_truth" / f"{domain}-ground-truth.json"
    skip_eval = not ground_truth.exists()
    mapper = _schedule_journey_mapper(payload, skip_eval=skip_eval)
    return {
        "domain": domain,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "staged": {
            "policy": _rel(staged_policy),
            "tasks": _rel(staged_tasks),
            "tools": _rel(staged_tools),
            "ground_truth": _rel(ground_truth) if ground_truth.exists() else "",
        },
        "skip_eval": skip_eval,
        "mapper_experiment": mapper,
    }
