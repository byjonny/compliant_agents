"""
Guardrail Workbench — benchmark runner.

Loads test cases from guardrail_workbench/test_cases/, instantiates an LLMGuard
per tool, and measures how often the guard correctly blocks the violating call.

Usage (from inside tau2-bench/):
    uv run python guardrail_workbench/run_bench.py
    uv run python guardrail_workbench/run_bench.py --guardrail-config guardrail_configs/airline_llm_guard.json
    uv run python guardrail_workbench/run_bench.py --cases TC-015 TC-016 TC-017 --num-trials 5
    uv run python guardrail_workbench/run_bench.py --cases TC-015 TC-017 --num-trials 3 --verbose

Metrics (all DENY cases — guard has no legitimate ALLOW cases to test FP):
    Recall  = cases correctly blocked (DENY or ESCALATE) / total cases (across all trials)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
CASES_DIR = HERE / "test_cases"
RESULTS_DIR = HERE / "results"
sys.path.insert(0, str(HERE.parent / "src"))

from dotenv import load_dotenv
load_dotenv(HERE.parent / ".env")

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.guardrails.guard import GuardVerdict, PolicyPassage, VerdictType
from tau2.guardrails.guards.llm_policy_guard import LLMGuard
from tau2.guardrails.loader import _build_llm_guard


# ── Message reconstruction ────────────────────────────────────────────────────

def _rebuild_messages(raw: list[dict]) -> list[Message]:
    """Convert raw test-case history dicts back to tau2 Message objects."""
    messages: list[Message] = []
    for m in raw:
        role    = m.get("role", "")
        content = m.get("content") or ""
        raw_tcs = m.get("tool_calls") or []

        if role == "system":
            messages.append(SystemMessage(role="system", content=content))
        elif role == "assistant":
            if raw_tcs:
                tool_calls = [
                    ToolCall(id=tc.get("id", ""), name=tc["name"], arguments=tc.get("arguments", {}))
                    for tc in raw_tcs
                ]
                messages.append(AssistantMessage.text(content=content or None, tool_calls=tool_calls))
            else:
                messages.append(AssistantMessage.text(content=content))
        elif role == "user":
            messages.append(UserMessage.text(content=content))
        elif role == "tool":
            messages.append(ToolMessage(
                id=m.get("tool_call_id") or m.get("id") or "",
                role="tool", content=content,
            ))
    return messages


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    tc_id:      str
    task_id:    str
    tool_name:  str
    key_arg:    str      # reservation_id or user_id
    trial:      int      # 1-based trial index
    verdict:    str      # ALLOW / DENY / ESCALATE / ERROR
    correct:    bool     # True if guard blocked (DENY or ESCALATE)
    latency_s:  float
    reason:     str = ""
    error:      str = ""


@dataclass
class BenchResult:
    config_name:  str
    num_trials:   int
    case_results: list[CaseResult] = field(default_factory=list)

    # ── flat aggregates ───────────────────────────────────────────────────────
    @property
    def total(self)   -> int:  return len(self.case_results)
    @property
    def blocked(self) -> int:  return sum(1 for r in self.case_results if r.correct)
    @property
    def errors(self)  -> int:  return sum(1 for r in self.case_results if r.verdict == "ERROR")
    @property
    def recall(self)  -> float:
        denom = self.total - self.errors
        return self.blocked / denom if denom else 0.0
    @property
    def avg_latency(self) -> float:
        valid = [r.latency_s for r in self.case_results if r.verdict != "ERROR"]
        return sum(valid) / len(valid) if valid else 0.0

    # ── per-case averages (across trials) ─────────────────────────────────────
    def per_case_stats(self) -> dict[str, dict]:
        """Return {tc_id: {blocked, total, block_rate, avg_latency}} across all trials."""
        stats: dict = defaultdict(lambda: {"blocked": 0, "total": 0, "latency_sum": 0.0})
        for r in self.case_results:
            s = stats[r.tc_id]
            s["total"]       += 1
            s["blocked"]     += int(r.correct)
            s["latency_sum"] += r.latency_s
            s.setdefault("tool_name", r.tool_name)
            s.setdefault("key_arg",   r.key_arg)
        for tc_id, s in stats.items():
            s["block_rate"]  = s["blocked"] / s["total"] if s["total"] else 0.0
            s["avg_latency"] = s["latency_sum"] / s["total"] if s["total"] else 0.0
        return dict(stats)


# ── Guard helpers ─────────────────────────────────────────────────────────────

def _load_guard_configs(config_path: Path) -> dict[str, dict]:
    data   = json.loads(config_path.read_text())
    guards = data.get("guards", []) if data.get("type") == "sequential" else []
    return {g["tool_name"]: g for g in guards if g.get("type") == "llm_guard"}


def run_case(tc: dict, trial: int, guard_cache: dict, guard_configs: dict[str, dict],
             verbose: bool) -> CaseResult:
    gc      = tc["guard_config"]
    vtc_raw = tc["violating_tool_call"]
    tool    = gc["tool_name"]

    tool_guard_cfg = guard_configs.get(tool)
    if tool_guard_cfg is None:
        return CaseResult(
            tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool,
            key_arg="—", trial=trial, verdict="ERROR", correct=False, latency_s=0.0,
            error=f"No llm_guard entry for tool '{tool}' in guardrail config.",
        )

    if tool not in guard_cache:
        guard_cache[tool] = _build_llm_guard(tool_guard_cfg)
    guard = guard_cache[tool]

    history   = _rebuild_messages(tc["history"])
    tool_call = ToolCall(id="bench-vtc", name=vtc_raw["name"], arguments=vtc_raw["arguments"])
    args      = vtc_raw["arguments"]
    key_arg   = args.get("reservation_id") or args.get("user_id") or str(list(args.values())[:1])

    t0 = time.perf_counter()
    try:
        verdict: GuardVerdict = guard.check(tool_call, env=None, history=history)
        latency = time.perf_counter() - t0
        blocked = not verdict.allowed
        v_str   = "DENY" if blocked else "ALLOW"
        if verbose:
            icon = "✓" if blocked else "✗"
            print(f"    {icon} {tc['id']} trial={trial} | {v_str:6s} | {latency:.1f}s | {(verdict.reason or '')[:60]}")
        return CaseResult(tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool, key_arg=key_arg,
                          trial=trial, verdict=v_str, correct=blocked, latency_s=latency,
                          reason=verdict.reason or "")
    except Exception as exc:
        latency = time.perf_counter() - t0
        if verbose:
            print(f"    ! {tc['id']} trial={trial} | ERROR: {exc}")
        return CaseResult(tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool, key_arg=key_arg,
                          trial=trial, verdict="ERROR", correct=False, latency_s=latency, error=str(exc))


# ── Display ───────────────────────────────────────────────────────────────────

def _print_summary(bench: BenchResult) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.table import Table
        from rich.text import Text
        rich_ok = True
    except ImportError:
        rich_ok = False

    multi = bench.num_trials > 1

    if not rich_ok:
        print(f"\nRecall: {bench.recall:.1%}  ({bench.blocked}/{bench.total - bench.errors} blocked)")
        for r in bench.case_results:
            s = "✓" if r.correct else ("!" if r.verdict == "ERROR" else "✗")
            t = f" trial={r.trial}/{bench.num_trials}" if multi else ""
            print(f"  {s} {r.tc_id}{t} → {r.verdict}")
        return

    console = Console()

    # ── Individual trial results table ────────────────────────────────────────
    tbl = Table(
        title=f"Guardrail Bench — {bench.config_name}"
              + (f"  ·  {bench.num_trials} trials" if multi else ""),
        box=box.ROUNDED, header_style="bold cyan",
        border_style="bright_black", show_lines=True,
    )
    tbl.add_column("ID",       no_wrap=True)
    tbl.add_column("Task",     justify="center")
    if multi:
        tbl.add_column("Trial", justify="center", min_width=6)
    tbl.add_column("Tool",     style="cyan", no_wrap=True)
    tbl.add_column("Key Arg",  style="dim",  no_wrap=True)
    tbl.add_column("Verdict",  justify="center", min_width=8)
    tbl.add_column("Latency",  justify="right")
    tbl.add_column("Reason",   max_width=50)

    prev_tc = None
    for r in bench.case_results:
        if r.verdict == "ERROR":
            vstyle, icon = "bold red",   "⚠ ERROR"
        elif r.correct:
            vstyle, icon = "bold green", "✓ DENY"
        else:
            vstyle, icon = "bold red",   "✗ ALLOW"

        # Visual separator between different test cases
        end_of_block = (prev_tc is not None and r.tc_id != prev_tc)
        if end_of_block and multi:
            tbl.add_section()
        prev_tc = r.tc_id

        row = [r.tc_id, r.task_id]
        if multi:
            row.append(f"{r.trial}/{bench.num_trials}")
        row += [r.tool_name, r.key_arg, Text(icon, style=vstyle),
                f"{r.latency_s:.1f}s", (r.reason or r.error)[:50]]
        tbl.add_row(*row)

    console.print()
    console.print(tbl)

    # ── Per-case average table (only when num_trials > 1) ─────────────────────
    if multi:
        pc = bench.per_case_stats()
        avg_tbl = Table(
            title="Per-Case Block Rate (averaged across trials)",
            box=box.SIMPLE_HEAVY, header_style="bold magenta",
            border_style="bright_black",
        )
        avg_tbl.add_column("ID",          no_wrap=True)
        avg_tbl.add_column("Tool",        style="cyan", no_wrap=True)
        avg_tbl.add_column("Key Arg",     style="dim")
        avg_tbl.add_column("Block Rate",  justify="right", min_width=14)
        avg_tbl.add_column("Avg Latency", justify="right")

        for tc_id, s in sorted(pc.items()):
            rate = s["block_rate"]
            col  = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
            avg_tbl.add_row(
                tc_id,
                s["tool_name"],
                s["key_arg"],
                Text(f"{s['blocked']}/{s['total']}  ({rate:.0%})", style=f"bold {col}"),
                f"{s['avg_latency']:.1f}s",
            )
        console.print()
        console.print(avg_tbl)

    # ── Overall summary panel ─────────────────────────────────────────────────
    by_tool: dict = defaultdict(lambda: {"total": 0, "blocked": 0})
    for r in bench.case_results:
        by_tool[r.tool_name]["total"]   += 1
        by_tool[r.tool_name]["blocked"] += int(r.correct)

    tool_lines = []
    for tool, stats in sorted(by_tool.items()):
        rec = stats["blocked"] / stats["total"] if stats["total"] else 0
        col = "green" if rec >= 0.8 else "yellow" if rec >= 0.5 else "red"
        tool_lines.append(
            f"[dim]{tool}[/dim]  [{col}]{rec:.0%}[/{col}] ({stats['blocked']}/{stats['total']})"
        )

    rec_col = "green" if bench.recall >= 0.8 else "yellow" if bench.recall >= 0.5 else "red"
    n_unique = len(set(r.tc_id for r in bench.case_results))
    scope = (f"{bench.num_trials} trials × {n_unique} cases" if multi
             else f"{n_unique} cases")
    summary = (
        f"[bold]Recall[/bold]  [{rec_col}]{bench.recall:.1%}[/{rec_col}]"
        f"  ({bench.blocked} blocked / {bench.total - bench.errors} evaluated  ·  {scope})\n"
        f"[bold]Avg latency[/bold]  {bench.avg_latency:.1f}s per call\n\n"
        + "\n".join(tool_lines)
    )
    console.print()
    console.print(Panel(summary, title="Overall Summary", border_style="bold blue", padding=(0, 2)))
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _DEFAULT_CFG = HERE.parent / "guardrail_configs" / "airline_llm_guard.json"

    parser = argparse.ArgumentParser(
        description="Guardrail workbench — benchmark LLMGuard recall",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--guardrail-config", default=str(_DEFAULT_CFG),
        help=(
            f"Path to a guardrail config JSON with llm_guard entries.\n"
            f"Default: {_DEFAULT_CFG.relative_to(HERE.parent)}"
        ),
    )
    parser.add_argument("--cases",      nargs="*",
                        help="TC IDs to run, e.g. TC-015 TC-017 (default: all)")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="How many times to run each test case (default: 1)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print each trial result as it completes")
    args = parser.parse_args()

    # ── Load guardrail config ─────────────────────────────────────────────────
    cfg_path = Path(args.guardrail_config)
    if not cfg_path.is_absolute():
        cfg_path = HERE.parent / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: guardrail config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    guard_configs = _load_guard_configs(cfg_path)
    if not guard_configs:
        print(f"ERROR: no llm_guard entries in {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg_summary = "  ".join(
        f"{t}: llm={g.get('llm','?')} mode={g.get('history_mode','full')}"
        for t, g in guard_configs.items()
    )

    # ── Load test cases ───────────────────────────────────────────────────────
    all_files = sorted(CASES_DIR.glob("TC-*.json"))
    if not all_files:
        print(f"No test cases found in {CASES_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.cases:
        wanted = {c.upper() for c in args.cases}
        files  = [f for f in all_files if f.name.split("_")[0] in wanted]
    else:
        files = all_files

    test_cases = [json.loads(f.read_text()) for f in files]
    n_runs     = len(test_cases) * args.num_trials

    print(f"Guardrail Workbench  |  config: {cfg_path.name}  "
          f"|  {len(test_cases)} cases  ×  {args.num_trials} trial(s)  =  {n_runs} runs")
    print(f"  Guards: {cfg_summary}")
    print()

    guard_cache: dict[str, LLMGuard] = {}
    bench = BenchResult(config_name=cfg_path.stem, num_trials=args.num_trials)
    run_no = 0

    for trial in range(1, args.num_trials + 1):
        if args.num_trials > 1:
            print(f"── Trial {trial}/{args.num_trials} " + "─" * 40)
        for tc in test_cases:
            run_no += 1
            label = (f"trial {trial}/{args.num_trials}  " if args.num_trials > 1 else "")
            print(f"  [{run_no:3d}/{n_runs}] {tc['id']}  {label}"
                  f"task={tc['task_id']}  {tc['violating_tool_call']['name']}",
                  end="", flush=True)
            if not args.verbose:
                print(" ...", end="", flush=True)
            else:
                print()

            result = run_case(tc, trial, guard_cache, guard_configs, args.verbose)
            bench.case_results.append(result)

            if not args.verbose:
                icon = "✓" if result.correct else ("⚠" if result.verdict == "ERROR" else "✗")
                print(f" {icon} {result.verdict} ({result.latency_s:.1f}s)")

    _print_summary(bench)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    suffix   = f"_t{args.num_trials}" if args.num_trials > 1 else ""
    out_path = RESULTS_DIR / f"bench_{ts}_{cfg_path.stem}{suffix}.json"

    per_case = bench.per_case_stats()
    out_data = {
        "config":      bench.config_name,
        "num_trials":  bench.num_trials,
        "recall":      bench.recall,
        "total_runs":  bench.total,
        "blocked":     bench.blocked,
        "errors":      bench.errors,
        "avg_latency_s": bench.avg_latency,
        # Each individual trial result
        "trials": [
            {"tc_id": r.tc_id, "task_id": r.task_id, "tool": r.tool_name, "key_arg": r.key_arg,
             "trial": r.trial, "verdict": r.verdict, "correct": r.correct,
             "latency_s": r.latency_s, "reason": r.reason}
            for r in bench.case_results
        ],
        # Per-case averages across all trials
        "per_case_avg": {
            tc_id: {
                "tool": s["tool_name"], "key_arg": s["key_arg"],
                "blocked": s["blocked"], "total": s["total"],
                "block_rate": round(s["block_rate"], 4),
                "avg_latency_s": round(s["avg_latency"], 3),
            }
            for tc_id, s in per_case.items()
        },
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Results saved → {out_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
