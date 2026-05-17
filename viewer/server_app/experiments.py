from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import ALLOWED_AGENTS, ALLOWED_DOMAINS, ALLOWED_USERS, DATA, GUARDRAILS, REPO
from .models import Experiment
from .simulations import _load_summary
from .store import _experiment_lock, _experiment_queue, _experiments, _running_processes, _save_experiments_db
from .utils import _parse_task_ids, _prompt_label, _repo_rel, _safe_guardrail_config, _safe_slug, _subprocess_env, _terminate_pid

_worker_started = False

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
