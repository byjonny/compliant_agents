from __future__ import annotations

import json
import queue
import subprocess
import threading
from dataclasses import asdict

from .config import EXPERIMENTS_DB, MAPPER_EXPERIMENTS_DB
from .models import Experiment, MapperExperiment

_experiments: dict[str, Experiment] = {}
_experiment_queue: queue.Queue[str] = queue.Queue()
_experiment_lock = threading.Lock()
_running_processes: dict[str, subprocess.Popen] = {}
_mapper_experiments: dict[str, MapperExperiment] = {}
_mapper_experiment_queue: queue.Queue[str] = queue.Queue()
_mapper_experiment_lock = threading.Lock()
_mapper_running_processes: dict[str, subprocess.Popen] = {}

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

_experiments.update(_load_experiments_db())
_mapper_experiments.update(_load_mapper_experiments_db())
