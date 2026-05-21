from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import POLICY_MAPPER_OUTPUT, REPO
from .models import MapperExperiment
from .store import (
    _mapper_experiment_lock,
    _mapper_experiment_queue,
    _mapper_experiments,
    _mapper_running_processes,
    _save_mapper_experiments_db,
)
from .utils import _parse_models, _safe_slug, _subprocess_env, _terminate_pid

_mapper_worker_started = False

def _parse_policy_mapper_eval_name(path: Path) -> dict | None:
    # Decode output filenames into dimensions the UI can filter by.
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
    # Collect completed mapper evaluation JSON files for comparison charts.
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
    # Mapper domains are inferred from staged policy files.
    input_dir = REPO / "policy_tool_mapper" / "input"
    domains = set()
    for path in input_dir.glob("*Policy.md"):
        name = path.stem
        if name.endswith("Policy"):
            domains.add(name[:-6].lower())
    return sorted(domains or {"airline", "retail", "telecom"})

def get_mapper_options() -> dict:
    # Defaults mirror the policy_tool_mapper pipeline options.
    return {
        "domains": _mapper_domains(),
        "modes": ["retrieval", "llm"],
        "default_models": ["gpt-4.1-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"],
        "default_embed_model": "text-embedding-3-large",
        "default_ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "default_ce_top_k": 30,
    }

def _build_mapper_command(exp: MapperExperiment) -> list[str]:
    # Build the policy mapper pipeline command for a queued mapper run.
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
    if exp.skip_eval:
        cmd.append("--skip-eval")
    return cmd

def _mapper_experiment_from_payload(payload: dict) -> MapperExperiment:
    # Validate and normalize mapper form data before scheduling it.
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

    ce_top_k = max(1, min(200, int(payload.get("ce_top_k") or 30)))
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
        skip_eval=bool(payload.get("skip_eval")),
    )
    exp.command = _build_mapper_command(exp)
    return exp

def _load_mapper_summary(exp: MapperExperiment) -> dict:
    # The pipeline writes metrics to the summary path supplied in the command.
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
    # Keep list responses small while preserving the latest log lines.
    data = asdict(exp)
    logs = data.pop("logs", [])
    data["log_count"] = len(logs)
    data["last_logs"] = logs[-20:]
    return data

def _matching_mapper_experiment_pids(exp: MapperExperiment) -> list[int]:
    # Find pipeline processes if the in-memory Popen handle was lost.
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
    # Cancellation mirrors normal experiments but targets mapper pipeline jobs.
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

def _run_mapper_experiment_worker():
    # Single daemon worker for mapper experiments.
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
            # Capture pipeline stdout so the browser can show live progress.
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
                # Successful mapper runs load their summary JSON as metrics.
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

def _reconcile_mapper_experiments_on_startup() -> None:
    """Repair mapper statuses for work that lost its in-memory worker state."""
    changed = False
    now = datetime.now().isoformat(timespec="seconds")
    with _mapper_experiment_lock:
        for exp in _mapper_experiments.values():
            summary_exists = bool(exp.summary_path and (REPO / exp.summary_path).exists())
            if exp.status != "done" and summary_exists:
                exp.status = "done"
                exp.finished_at = exp.finished_at or now
                exp.error = None
                exp.metrics = _load_mapper_summary(exp)
                changed = True
                continue

            if exp.status in {"done", "failed", "cancelled"}:
                continue

            if exp.status == "queued":
                exp.status = "cancelled"
                exp.finished_at = exp.finished_at or now
                exp.error = "Queued mapper job was not resumed after viewer restart"
                changed = True
                continue

            if not _matching_mapper_experiment_pids(exp):
                exp.status = "failed"
                exp.finished_at = exp.finished_at or now
                exp.error = "No running process or summary file found after viewer restart"
                changed = True

        if changed:
            _save_mapper_experiments_db()

def _ensure_mapper_worker():
    # Lazily start the mapper daemon once per server process.
    global _mapper_worker_started
    if _mapper_worker_started:
        return
    _reconcile_mapper_experiments_on_startup()
    threading.Thread(target=_run_mapper_experiment_worker, daemon=True).start()
    _mapper_worker_started = True
