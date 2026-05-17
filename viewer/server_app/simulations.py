from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .config import DATA, REPO
from .store import _experiment_for_result

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
        "guard_expected_block_count": total_blocks - false_positive_blocks,
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
            "guard_expected_block_count": fp_summary["guard_expected_block_count"],
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
                "guard_expected_block_count": r.get("guard_expected_block_count"),
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
