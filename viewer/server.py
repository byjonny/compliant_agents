#!/usr/bin/env python3
"""
τ Trajectory Viewer — zero-dependency HTTP server for tau2-bench results.

Usage:
    python viewer/server.py
    open http://localhost:8765
"""

import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
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

_experiments: dict[str, "Experiment"] = {}
_experiment_queue: queue.Queue[str] = queue.Queue()
_experiment_lock = threading.Lock()
_running_processes: dict[str, subprocess.Popen] = {}
_worker_started = False
_mapper_experiments: dict[str, "MapperExperiment"] = {}
_mapper_experiment_queue: queue.Queue[str] = queue.Queue()
_mapper_experiment_lock = threading.Lock()
_mapper_running_processes: dict[str, subprocess.Popen] = {}
_mapper_worker_started = False


@dataclass
class Experiment:
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
    id: str
    name: str
    domain: str
    models: list[str]
    mode: str
    embed_model: str = "text-embedding-3-large"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_top_k: int = 20
    skip_mapping: bool = False
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    command: list[str] = field(default_factory=list)
    summary_path: str = ""
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None


def _load_experiments_db() -> dict[str, "Experiment"]:
    if not EXPERIMENTS_DB.exists():
        return {}
    try:
        raw = json.loads(EXPERIMENTS_DB.read_text())
        return {item["id"]: Experiment(**item) for item in raw if item.get("id")}
    except Exception:
        return {}


def _save_experiments_db() -> None:
    try:
        payload = [asdict(exp) for exp in _experiments.values()]
        EXPERIMENTS_DB.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _load_mapper_experiments_db() -> dict[str, "MapperExperiment"]:
    if not MAPPER_EXPERIMENTS_DB.exists():
        return {}
    try:
        raw = json.loads(MAPPER_EXPERIMENTS_DB.read_text())
        return {item["id"]: MapperExperiment(**item) for item in raw if item.get("id")}
    except Exception:
        return {}


def _save_mapper_experiments_db() -> None:
    try:
        payload = [asdict(exp) for exp in _mapper_experiments.values()]
        MAPPER_EXPERIMENTS_DB.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _experiment_for_result(result_id: str) -> Experiment | None:
    for exp in _experiments.values():
        if exp.result_id == result_id:
            return exp
    return None


_experiments.update(_load_experiments_db())
_mapper_experiments.update(_load_mapper_experiments_db())


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


def _build_command(exp: Experiment) -> list[str]:
    uv_path = Path.home() / ".local" / "bin" / "uv"
    uv = str(uv_path if uv_path.exists() else "uv")
    save_to = f"viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp.domain}_{_safe_slug(exp.name)}_{exp.id}"
    exp.result_id = save_to
    exp.result_path = f"data/simulations/{save_to}/results.json"
    cmd = [
        uv,
        "run",
        "tau2",
        "run",
        "--domain",
        exp.domain,
        "--agent",
        exp.agent,
        "--guardrail-config",
        exp.guardrail_config,
        "--agent-llm",
        exp.agent_llm,
        "--user-llm",
        exp.user_llm,
        "--num-trials",
        str(exp.num_trials),
        "--max-concurrency",
        str(exp.max_concurrency),
        "--save-to",
        exp.result_id,
    ]
    if exp.guard_llm:
        cmd.extend(["--guard-llm", exp.guard_llm])
    if exp.task_split_name:
        cmd.extend(["--task-split-name", exp.task_split_name])
    if exp.num_tasks:
        cmd.extend(["--num-tasks", str(exp.num_tasks)])
    if exp.task_ids:
        cmd.append("--task-ids")
        cmd.extend(exp.task_ids)
    return cmd


def _experiment_from_payload(payload: dict) -> Experiment:
    domain = str(payload.get("domain") or "telecom").strip()
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    agent = str(payload.get("agent") or "llm_agent").strip()
    if agent not in ALLOWED_AGENTS:
        raise ValueError(f"Unsupported agent: {agent}")

    user = str(payload.get("user") or "user_simulator").strip()
    if user not in ALLOWED_USERS:
        raise ValueError(f"Unsupported user: {user}")

    default_guardrail = f"guardrail_configs/{domain}_llm_guard.json"
    if not (REPO / default_guardrail).exists():
        default_guardrail = "guardrail_configs/null.json"
    guardrail_config = _safe_guardrail_config(payload.get("guardrail_config") or default_guardrail)

    raw_num_tasks = payload.get("num_tasks")
    num_tasks = None if raw_num_tasks in (None, "") else max(1, int(raw_num_tasks))
    num_trials = max(1, min(50, int(payload.get("num_trials") or 1)))
    max_concurrency = max(1, min(32, int(payload.get("max_concurrency") or 1)))

    exp = Experiment(
        id=uuid.uuid4().hex[:10],
        name=str(payload.get("name") or f"{domain} experiment").strip(),
        domain=domain,
        agent=agent,
        user=user,
        guardrail_config=guardrail_config,
        agent_llm=str(payload.get("agent_llm") or "gpt-4.1-mini").strip(),
        user_llm=str(payload.get("user_llm") or "gpt-5.1").strip(),
        guard_llm=str(payload.get("guard_llm") or "").strip(),
        num_trials=num_trials,
        max_concurrency=max_concurrency,
        task_split_name=str(payload.get("task_split_name") or "").strip() or None,
        num_tasks=num_tasks,
        task_ids=_parse_task_ids(payload.get("task_ids")),
        prompt_label=str(payload.get("prompt_label") or _prompt_label(guardrail_config)).strip(),
        dataset_label=str(payload.get("dataset_label") or domain).strip(),
    )
    exp.command = _build_command(exp)
    return exp


def _parse_dir_name(name: str) -> dict:
    """Extract date and domain from directory names like 20260424_125007_airline_llm_agent_..."""
    parts = name.split("_")
    raw_date = parts[0] if parts else ""
    raw_time = parts[1] if len(parts) > 1 else ""
    date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else raw_date
    time = f"{raw_time[:2]}:{raw_time[2:4]}" if len(raw_time) == 6 else ""
    domain = parts[2] if len(parts) > 2 else "unknown"
    return {"date": f"{date} {time}".strip(), "domain": domain}


def _arg_subset_matches(actual, expected) -> bool:
    if expected is None:
        return True
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _arg_subset_matches(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def _guard_block_is_expected(event: dict, task: dict | None) -> bool:
    criteria = (task or {}).get("evaluation_criteria") or {}
    predicates = criteria.get("compliance") or []
    tool_name = event.get("tool_name")
    tool_args = event.get("tool_arguments") or {}
    for predicate in predicates:
        if predicate.get("type") != "unauthorized_action":
            continue
        if predicate.get("tool_name") != tool_name:
            continue
        if _arg_subset_matches(tool_args, predicate.get("match_args")):
            return True
    return False


def _compute_guard_false_positives(run_id: str, sims: list, tasks: list) -> dict:
    task_by_id = {str(t.get("id")): t for t in tasks if t.get("id") is not None}
    total_blocks = 0
    false_positive_blocks = 0
    events = []

    for sim in sims:
        guard_events = sim.get("guardrail_events") or []
        if not guard_events:
            continue
        task_id = str(sim.get("task_id", "unknown"))
        task = task_by_id.get(task_id)
        for event in guard_events:
            total_blocks += 1
            if _guard_block_is_expected(event, task):
                continue
            false_positive_blocks += 1
            events.append(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "trial": sim.get("trial", 0),
                    "tool_name": event.get("tool_name"),
                    "tool_arguments": event.get("tool_arguments") or {},
                    "guard_name": event.get("guard_name"),
                    "reason": event.get("reason"),
                }
            )

    return {
        "guard_block_count": total_blocks,
        "guard_false_positive_count": false_positive_blocks,
        "guard_false_positive_rate": round(false_positive_blocks / total_blocks, 3)
        if total_blocks
        else 0.0,
        "guard_false_positive_events": events[:50],
    }


def _load_summary(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        info      = data.get("info") or {}
        agent     = info.get("agent_info") or {}
        user_info = info.get("user_info") or {}
        env       = info.get("environment_info") or {}
        sims      = data.get("simulations") or []
        tasks     = data.get("tasks") or []

        parsed = _parse_dir_name(path.parent.name)
        experiment = _experiment_for_result(path.parent.name)
        domain = env.get("domain_name") or env.get("domain") or parsed["domain"]
        guardrail_config = (
            info.get("guardrail_config_path")
            or (experiment.guardrail_config if experiment else "")
        )
        guard_model = info.get("guard_llm") or (experiment.guard_llm if experiment else "")

        rewards = [
            s["reward_info"]["reward"]
            for s in sims
            if (s.get("reward_info") or {}).get("reward") is not None
        ]
        avg_reward = round(sum(rewards) / len(rewards), 3) if rewards else None
        task_latencies: dict[str, list[float]] = {}
        compliance_runs = 0
        failed_compliance_runs = 0
        for s in sims:
            duration = s.get("duration")
            if duration is not None:
                task_latencies.setdefault(str(s.get("task_id", "unknown")), []).append(float(duration))

            checks = [
                c
                for c in ((s.get("reward_info") or {}).get("compliance_checks") or [])
                if not c.get("skipped")
            ]
            if checks:
                compliance_runs += 1
                if any(c.get("passed") is False for c in checks):
                    failed_compliance_runs += 1

        avg_task_latencies = [mean(values) for values in task_latencies.values() if values]
        avg_latency = round(mean(avg_task_latencies), 2) if avg_task_latencies else None
        task_latency_map = {
            task_id: round(mean(values), 2)
            for task_id, values in task_latencies.items()
            if values
        }
        policy_violation_rate = (
            round(failed_compliance_runs / compliance_runs, 3)
            if compliance_runs
            else 0.0
        )
        fp_summary = _compute_guard_false_positives(path.parent.name, sims, tasks)

        return {
            "id":              path.parent.name,
            "date":            parsed["date"],
            "model":           agent.get("llm", "unknown"),
            "user_model":      user_info.get("llm", ""),
            "guard_model":     guard_model,
            "guardrail_config": guardrail_config,
            "agent":           agent.get("implementation", ""),
            "domain":          domain,
            "num_tasks":       len(tasks),
            "num_simulations": len(sims),
            "avg_reward":      avg_reward,
            "policy_violation_rate": policy_violation_rate,
            "guard_block_count": fp_summary["guard_block_count"],
            "guard_false_positive_count": fp_summary["guard_false_positive_count"],
            "guard_false_positive_rate": fp_summary["guard_false_positive_rate"],
            "guard_false_positive_events": fp_summary["guard_false_positive_events"],
            "avg_latency":     avg_latency,
            "task_latencies":   task_latency_map,
            "display_name":    experiment.name if experiment else path.parent.name,
            "prompt_label":    experiment.prompt_label if experiment else "",
            "dataset_label":   domain,
            "max_steps":       info.get("max_steps"),
            "num_trials":      info.get("num_trials"),
        }
    except Exception:
        return None


def get_all_simulations() -> list:
    if not DATA.exists():
        return []
    results = []
    for d in sorted(DATA.iterdir(), reverse=True):
        p = d / "results.json"
        if p.exists():
            s = _load_summary(p)
            if s:
                results.append(s)
    return results


def get_analysis_runs() -> dict:
    runs = get_all_simulations()
    domains = sorted({r["domain"] for r in runs if r.get("domain")})
    models = sorted({r["model"] for r in runs if r.get("model")})
    return {
        "domains": domains,
        "models": models,
        "runs": [
            {
                "id": r["id"],
                "label": r.get("display_name") or r["id"],
                "date": r.get("date"),
                "domain": r.get("domain"),
                "model": r.get("model"),
                "user_model": r.get("user_model"),
                "guard_model": r.get("guard_model"),
                "guardrail_config": r.get("guardrail_config"),
                "has_guard": bool(r.get("guard_model"))
                or bool(r.get("guardrail_config") and not str(r.get("guardrail_config")).endswith("/null.json")),
                "avg_reward": r.get("avg_reward"),
                "policy_violation_rate": r.get("policy_violation_rate"),
                "guard_block_count": r.get("guard_block_count"),
                "guard_false_positive_count": r.get("guard_false_positive_count"),
                "guard_false_positive_rate": r.get("guard_false_positive_rate"),
                "guard_false_positive_events": r.get("guard_false_positive_events") or [],
                "avg_latency": r.get("avg_latency"),
                "task_latencies": r.get("task_latencies") or {},
                "num_tasks": r.get("num_tasks"),
                "num_simulations": r.get("num_simulations"),
            }
            for r in runs
        ],
    }


def _parse_policy_mapper_eval_name(path: Path) -> dict | None:
    name = path.name
    marker = "-eval-"
    if marker not in name or not name.endswith(".json"):
        return None
    domain, rest = name[:-5].split(marker, 1)
    if rest.endswith("-high"):
        conf = "high"
        rest = rest[:-5]
    elif rest.endswith("-all"):
        conf = "all"
        rest = rest[:-4]
    else:
        return None
    if rest.endswith("-retrieval"):
        mode = "retrieval"
        model = rest[:-10]
    else:
        mode = "llm"
        model = rest
    return {"domain": domain, "model": model, "mode": mode, "conf": conf}


def get_policy_mapper_results() -> dict:
    if not POLICY_MAPPER_OUTPUT.exists():
        return {"domains": [], "models": [], "results": []}

    results = []
    for path in sorted(POLICY_MAPPER_OUTPUT.glob("*-eval-*.json")):
        parsed = _parse_policy_mapper_eval_name(path)
        if not parsed:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        overall = data.get("overall") or {}
        metadata = data.get("metadata") or {}
        results.append({
            **parsed,
            "file": path.name,
            "path": str(path.relative_to(REPO)),
            "overall": overall,
            "macro": overall.get("macro") or {},
            "micro": overall.get("micro") or {},
            "threshold": metadata.get("threshold"),
            "total_tools": metadata.get("total_tools"),
            "timestamp": metadata.get("timestamp"),
        })

    return {
        "domains": sorted({r["domain"] for r in results}),
        "models": sorted({r["model"] for r in results}),
        "results": results,
    }


def _mapper_domains() -> list[str]:
    input_dir = REPO / "policy_tool_mapper" / "input"
    domains = set()
    for path in input_dir.glob("*Policy.md"):
        name = path.stem
        if name.endswith("Policy"):
            domains.add(name[:-6].lower())
    return sorted(domains or {"airline", "retail", "telecom"})


def get_mapper_options() -> dict:
    return {
        "domains": _mapper_domains(),
        "modes": ["retrieval", "llm"],
        "default_models": ["gpt-4.1-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"],
        "default_embed_model": "text-embedding-3-large",
        "default_ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "default_ce_top_k": 20,
    }


def _build_mapper_command(exp: MapperExperiment) -> list[str]:
    uv_path = Path.home() / ".local" / "bin" / "uv"
    uv = str(uv_path if uv_path.exists() else "uv")
    exp.summary_path = (
        "policy_tool_mapper/output/"
        f"viewer-mapper-summary-{datetime.now().strftime('%Y%m%d_%H%M%S')}-"
        f"{exp.domain}-{_safe_slug(exp.name)}-{exp.id}.json"
    )
    cmd = [
        uv,
        "run",
        "python",
        "policy_tool_mapper/run_pipeline.py",
        "--domain",
        exp.domain,
        "--models",
        *exp.models,
        "--mode",
        exp.mode,
        "--summary",
        exp.summary_path,
    ]
    if exp.mode == "retrieval":
        cmd.extend([
            "--embed-model",
            exp.embed_model,
            "--ce-model",
            exp.ce_model,
            "--ce-top-k",
            str(exp.ce_top_k),
        ])
    if exp.skip_mapping:
        cmd.append("--skip-mapping")
    return cmd


def _mapper_experiment_from_payload(payload: dict) -> MapperExperiment:
    domain = str(payload.get("domain") or "airline").strip().lower()
    if domain not in _mapper_domains():
        raise ValueError(f"Unsupported mapper domain: {domain}")

    mode = str(payload.get("mode") or "retrieval").strip().lower()
    if mode not in {"retrieval", "llm"}:
        raise ValueError(f"Unsupported mapper mode: {mode}")

    models = _parse_models(payload.get("models") or payload.get("model") or "gpt-4.1-mini")
    if not models:
        raise ValueError("At least one model is required")
    if len(models) > 12:
        raise ValueError("Please schedule at most 12 models per mapper run")

    ce_top_k = max(1, min(200, int(payload.get("ce_top_k") or 20)))
    exp = MapperExperiment(
        id=uuid.uuid4().hex[:10],
        name=str(payload.get("name") or f"{domain} mapper run").strip(),
        domain=domain,
        models=models,
        mode=mode,
        embed_model=str(payload.get("embed_model") or "text-embedding-3-large").strip(),
        ce_model=str(payload.get("ce_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip(),
        ce_top_k=ce_top_k,
        skip_mapping=bool(payload.get("skip_mapping")),
    )
    exp.command = _build_mapper_command(exp)
    return exp


def _load_mapper_summary(exp: MapperExperiment) -> dict:
    if not exp.summary_path:
        return {}
    path = REPO / exp.summary_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _compact_mapper_experiment(exp: MapperExperiment) -> dict:
    data = asdict(exp)
    logs = data.pop("logs", [])
    data["log_count"] = len(logs)
    data["last_logs"] = logs[-20:]
    return data


def _matching_mapper_experiment_pids(exp: MapperExperiment) -> list[int]:
    tokens = [token for token in [exp.id, exp.summary_path] if token]
    if not tokens:
        return []
    try:
        result = subprocess.run(
            ["ps", "axo", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if "policy_tool_mapper/run_pipeline.py" not in command:
            continue
        if not any(token in command for token in tokens):
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue
    return pids


def _cancel_mapper_experiment(exp_id: str) -> tuple[dict, int]:
    with _mapper_experiment_lock:
        exp = _mapper_experiments.get(exp_id)
        proc = _mapper_running_processes.get(exp_id)
        if not exp:
            return {"error": "mapper experiment not found"}, 404
        if exp.status in {"done", "failed", "cancelled"}:
            return _compact_mapper_experiment(exp), 200
        if exp.status == "queued":
            exp.status = "cancelled"
            exp.finished_at = datetime.now().isoformat(timespec="seconds")
            exp.error = "Cancelled before start"
            _save_mapper_experiments_db()
            return _compact_mapper_experiment(exp), 200
        exp.status = "cancelling"
        exp.error = "Cancellation requested"
        _save_mapper_experiments_db()

    killed = False
    has_tracked_live_proc = bool(proc and proc.poll() is None)
    if has_tracked_live_proc and proc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            killed = True
        except Exception:
            try:
                proc.terminate()
                killed = True
            except Exception:
                pass

    for pid in _matching_mapper_experiment_pids(exp):
        killed = _terminate_pid(pid) or killed

    with _mapper_experiment_lock:
        exp = _mapper_experiments.get(exp_id)
        if exp and exp.status == "cancelling" and not has_tracked_live_proc:
            exp.status = "cancelled"
            exp.finished_at = datetime.now().isoformat(timespec="seconds")
            exp.error = "Cancelled by user" if killed else "Cancelled; no running process was found"
            _save_mapper_experiments_db()
        return (_compact_mapper_experiment(exp) if exp else {"id": exp_id, "status": "cancelling"}), 200


def _compact_experiment(exp: Experiment) -> dict:
    data = asdict(exp)
    logs = data.pop("logs", [])
    data["log_count"] = len(logs)
    data["last_logs"] = logs[-20:]
    return data


def _matching_experiment_pids(exp: Experiment) -> list[int]:
    """Best-effort fallback for runs that outlive the in-memory process handle."""
    tokens = [token for token in [exp.id, exp.result_id] if token]
    if not tokens:
        return []
    try:
        result = subprocess.run(
            ["ps", "axo", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if "tau2 run" not in command and "uv run tau2" not in command:
            continue
        if not any(token in command for token in tokens):
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue
    return pids


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


def _cancel_experiment(exp_id: str) -> tuple[dict, int]:
    with _experiment_lock:
        exp = _experiments.get(exp_id)
        proc = _running_processes.get(exp_id)
        if not exp:
            return {"error": "experiment not found"}, 404
        if exp.status in {"done", "failed", "cancelled"}:
            return _compact_experiment(exp), 200
        if exp.status == "queued":
            exp.status = "cancelled"
            exp.finished_at = datetime.now().isoformat(timespec="seconds")
            exp.error = "Cancelled before start"
            _save_experiments_db()
            return _compact_experiment(exp), 200
        exp.status = "cancelling"
        exp.error = "Cancellation requested"
        _save_experiments_db()

    killed = False
    has_tracked_live_proc = bool(proc and proc.poll() is None)
    if has_tracked_live_proc and proc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            killed = True
        except Exception:
            try:
                proc.terminate()
                killed = True
            except Exception:
                pass

    for pid in _matching_experiment_pids(exp):
        killed = _terminate_pid(pid) or killed

    with _experiment_lock:
        exp = _experiments.get(exp_id)
        if exp and exp.status == "cancelling" and not has_tracked_live_proc:
            exp.status = "cancelled"
            exp.finished_at = datetime.now().isoformat(timespec="seconds")
            exp.error = "Cancelled by user" if killed else "Cancelled; no running process was found"
            _save_experiments_db()
        return (_compact_experiment(exp) if exp else {"id": exp_id, "status": "cancelling"}), 200


def get_budget_data() -> dict:
    """Aggregate agent_cost and user_cost across all runs, grouped by date."""
    from collections import defaultdict

    if not DATA.exists():
        return {"days": [], "series": {"agent": [], "user": [], "guard": []}, "totals": {}, "runs": []}

    daily: dict = defaultdict(lambda: {"agent": 0.0, "user": 0.0, "guard": 0.0})
    runs: list = []

    for d in sorted(DATA.iterdir()):
        p = d / "results.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            info        = data.get("info") or {}
            agent_info  = info.get("agent_info") or {}
            user_info   = info.get("user_info") or {}
            sims        = data.get("simulations") or []
            model       = agent_info.get("llm", "unknown")
            user_model  = user_info.get("llm", "")

            run_agent = run_user = 0.0
            run_date  = ""
            for sim in sims:
                ac = sim.get("agent_cost") or 0.0
                uc = sim.get("user_cost")  or 0.0
                start = sim.get("start_time", "")
                date  = start[:10] if start else d.name[:10]
                if not date:
                    continue
                daily[date]["agent"] += ac
                daily[date]["user"]  += uc
                run_agent += ac
                run_user  += uc
                if not run_date:
                    run_date = date

            if run_date:
                runs.append({
                    "date":       run_date,
                    "run_id":     d.name,
                    "model":      model,
                    "user_model": user_model,
                    "agent_cost": round(run_agent, 6),
                    "user_cost":  round(run_user, 6),
                    "guard_cost": 0.0,
                    "total":      round(run_agent + run_user, 6),
                })
        except Exception:
            continue

    days = sorted(daily)
    return {
        "days":   days,
        "series": {
            "agent": [round(daily[d]["agent"], 6) for d in days],
            "user":  [round(daily[d]["user"],  6) for d in days],
            "guard": [0.0] * len(days),   # guard LLM cost not yet tracked
        },
        "totals": {
            "agent": round(sum(daily[d]["agent"] for d in days), 6),
            "user":  round(sum(daily[d]["user"]  for d in days), 6),
            "guard": 0.0,
            "grand": round(sum(daily[d]["agent"] + daily[d]["user"] for d in days), 6),
        },
        "runs": sorted(runs, key=lambda r: r["date"]),
    }


def get_simulation(sim_id: str) -> dict | None:
    p = DATA / sim_id / "results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_experiment_options() -> dict:
    configs = [
        _repo_rel(path)
        for path in sorted(GUARDRAILS.glob("*.json"))
        if path.is_file()
    ]
    defaults = {}
    for domain in ALLOWED_DOMAINS:
        candidate = f"guardrail_configs/{domain}_llm_guard.json"
        defaults[domain] = candidate if (REPO / candidate).exists() else "guardrail_configs/null.json"
    return {
        "domains": ALLOWED_DOMAINS,
        "agents": ALLOWED_AGENTS,
        "users": ALLOWED_USERS,
        "guardrail_configs": configs,
        "default_guardrails": defaults,
    }


def _run_experiment_worker():
    while True:
        exp_id = _experiment_queue.get()
        with _experiment_lock:
            exp = _experiments.get(exp_id)
            if not exp:
                _experiment_queue.task_done()
                continue
            if exp.status == "cancelled":
                _experiment_queue.task_done()
                continue
            exp.status = "running"
            exp.started_at = datetime.now().isoformat(timespec="seconds")
            _save_experiments_db()

        result_path = DATA / exp.result_id / "results.json"
        try:
            proc = subprocess.Popen(
                exp.command,
                cwd=REPO,
                env=_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with _experiment_lock:
                _running_processes[exp_id] = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                with _experiment_lock:
                    exp.logs.append(line.rstrip())
                    exp.logs = exp.logs[-300:]

            return_code = proc.wait()
            with _experiment_lock:
                exp.finished_at = datetime.now().isoformat(timespec="seconds")
                if exp.status == "cancelling":
                    exp.status = "cancelled"
                    exp.error = "Cancelled by user"
                elif return_code == 0 and result_path.exists():
                    exp.status = "done"
                    summary = _load_summary(result_path) or {}
                    exp.metrics = {
                        "reward": summary.get("avg_reward"),
                        "policy_violation_rate": summary.get("policy_violation_rate"),
                        "latency": summary.get("avg_latency"),
                        "samples": summary.get("num_tasks"),
                        "runs": summary.get("num_simulations"),
                    }
                else:
                    exp.status = "failed"
                    exp.error = f"tau2 exited with code {return_code}"
                _save_experiments_db()
        except Exception as exc:
            with _experiment_lock:
                exp.status = "failed"
                exp.finished_at = datetime.now().isoformat(timespec="seconds")
                exp.error = str(exc)
                _save_experiments_db()
        finally:
            with _experiment_lock:
                _running_processes.pop(exp_id, None)
            _experiment_queue.task_done()


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    threading.Thread(target=_run_experiment_worker, daemon=True).start()
    _worker_started = True


def _run_mapper_experiment_worker():
    while True:
        exp_id = _mapper_experiment_queue.get()
        with _mapper_experiment_lock:
            exp = _mapper_experiments.get(exp_id)
            if not exp:
                _mapper_experiment_queue.task_done()
                continue
            if exp.status == "cancelled":
                _mapper_experiment_queue.task_done()
                continue
            exp.status = "running"
            exp.started_at = datetime.now().isoformat(timespec="seconds")
            _save_mapper_experiments_db()

        try:
            proc = subprocess.Popen(
                exp.command,
                cwd=REPO,
                env=_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with _mapper_experiment_lock:
                _mapper_running_processes[exp_id] = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                with _mapper_experiment_lock:
                    exp.logs.append(line.rstrip())
                    exp.logs = exp.logs[-400:]

            return_code = proc.wait()
            with _mapper_experiment_lock:
                exp.finished_at = datetime.now().isoformat(timespec="seconds")
                if exp.status == "cancelling":
                    exp.status = "cancelled"
                    exp.error = "Cancelled by user"
                elif return_code == 0:
                    exp.status = "done"
                    exp.metrics = _load_mapper_summary(exp)
                else:
                    exp.status = "failed"
                    exp.error = f"policy mapper exited with code {return_code}"
                _save_mapper_experiments_db()
        except Exception as exc:
            with _mapper_experiment_lock:
                exp.status = "failed"
                exp.finished_at = datetime.now().isoformat(timespec="seconds")
                exp.error = str(exc)
                _save_mapper_experiments_db()
        finally:
            with _mapper_experiment_lock:
                _mapper_running_processes.pop(exp_id, None)
            _mapper_experiment_queue.task_done()


def _ensure_mapper_worker():
    global _mapper_worker_started
    if _mapper_worker_started:
        return
    threading.Thread(target=_run_mapper_experiment_worker, daemon=True).start()
    _mapper_worker_started = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path}")

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def _send_html(self):
        body = (HERE / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send_html()
        elif path == "/api/simulations":
            self._send_json(get_all_simulations())
        elif path == "/api/experiment-options":
            self._send_json(get_experiment_options())
        elif path == "/api/analysis-runs":
            self._send_json(get_analysis_runs())
        elif path == "/api/policy-mapper-results":
            self._send_json(get_policy_mapper_results())
        elif path == "/api/mapper-options":
            self._send_json(get_mapper_options())
        elif path == "/api/mapper-experiments":
            with _mapper_experiment_lock:
                data = [_compact_mapper_experiment(exp) for exp in _mapper_experiments.values()]
            self._send_json(data)
        elif path == "/api/experiments":
            with _experiment_lock:
                data = [_compact_experiment(exp) for exp in _experiments.values()]
            self._send_json(data)
        elif path == "/api/budget":
            self._send_json(get_budget_data())
        elif path.startswith("/api/simulations/"):
            sim_id = path[len("/api/simulations/"):]
            data = get_simulation(sim_id)
            if data:
                self._send_json(data)
            else:
                self._send_json({"error": "not found"}, 404)
        elif not path.startswith("/api/") and "." not in Path(path).name:
            self._send_html()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/mapper-experiments/") and path.endswith("/cancel"):
            exp_id = path[len("/api/mapper-experiments/"):-len("/cancel")]
            data, status = _cancel_mapper_experiment(exp_id)
            self._send_json(data, status)
            return
        if path == "/api/mapper-experiments":
            try:
                exp = _mapper_experiment_from_payload(self._read_json())
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            with _mapper_experiment_lock:
                _mapper_experiments[exp.id] = exp
                _save_mapper_experiments_db()
            _ensure_mapper_worker()
            _mapper_experiment_queue.put(exp.id)
            self._send_json(_compact_mapper_experiment(exp), 201)
            return
        if path.startswith("/api/experiments/") and path.endswith("/cancel"):
            exp_id = path[len("/api/experiments/"):-len("/cancel")]
            data, status = _cancel_experiment(exp_id)
            self._send_json(data, status)
            return
        if path != "/api/experiments":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            exp = _experiment_from_payload(self._read_json())
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        with _experiment_lock:
            _experiments[exp.id] = exp
            _save_experiments_db()
        _ensure_worker()
        _experiment_queue.put(exp.id)
        self._send_json(_compact_experiment(exp), 201)


if __name__ == "__main__":
    _ensure_worker()
    _ensure_mapper_worker()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  τ Trajectory Viewer")
    print(f"  → http://{HOST}:{PORT}")
    print(f"  → data directory: {DATA}")
    print(f"  → Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
