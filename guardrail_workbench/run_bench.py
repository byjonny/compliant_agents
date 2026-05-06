"""
Guardrail Workbench — benchmark runner.

Loads test cases from guardrail_workbench/test_cases/, instantiates an LLMGuard
per tool, and measures how often the guard correctly blocks the violating call.

Usage (from inside tau2-bench/):
    uv run python guardrail_workbench/run_bench.py
    uv run python guardrail_workbench/run_bench.py --model gpt-4.1-mini
    uv run python guardrail_workbench/run_bench.py --cases TC-001 TC-009
    uv run python guardrail_workbench/run_bench.py --model claude-haiku-4-5-20251001 --verbose

Metrics (all DENY cases — guard has no legitimate ALLOW cases to test FP):
    Recall  = cases correctly blocked (DENY or ESCALATE) / total cases
"""

import argparse
import json
import sys
import time
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
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                    for tc in raw_tcs
                ]
                messages.append(AssistantMessage.text(content=content or None, tool_calls=tool_calls))
            else:
                messages.append(AssistantMessage.text(content=content))

        elif role == "user":
            messages.append(UserMessage.text(content=content))

        elif role == "tool":
            messages.append(
                ToolMessage(
                    id=m.get("tool_call_id") or m.get("id") or "",
                    role="tool",
                    content=content,
                )
            )
    return messages


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    tc_id:         str
    task_id:       str
    tool_name:     str
    key_arg:       str          # reservation_id or user_id
    verdict:       str          # ALLOW / DENY / ESCALATE / ERROR
    correct:       bool         # True if guard blocked (DENY or ESCALATE)
    latency_s:     float
    reason:        str = ""
    error:         str = ""


@dataclass
class BenchResult:
    model:        str
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self)    -> int:  return len(self.case_results)
    @property
    def blocked(self)  -> int:  return sum(1 for r in self.case_results if r.correct)
    @property
    def allowed(self)  -> int:  return sum(1 for r in self.case_results if not r.correct and r.verdict != "ERROR")
    @property
    def errors(self)   -> int:  return sum(1 for r in self.case_results if r.verdict == "ERROR")
    @property
    def recall(self)   -> float:
        denom = self.total - self.errors
        return self.blocked / denom if denom else 0.0
    @property
    def avg_latency(self) -> float:
        valid = [r.latency_s for r in self.case_results if r.verdict != "ERROR"]
        return sum(valid) / len(valid) if valid else 0.0


# ── Load guard configs from a guardrail config JSON ──────────────────────────

def _load_guard_configs(config_path: Path) -> dict[str, dict]:
    """
    Parse a guardrail config JSON and return all llm_guard entries indexed by tool_name.

    Supports sequential configs:
        {"type": "sequential", "guards": [...]}
    """
    data = json.loads(config_path.read_text())
    guards = data.get("guards", []) if data.get("type") == "sequential" else []
    return {g["tool_name"]: g for g in guards if g.get("type") == "llm_guard"}


# ── Guard cache (one instance per tool) ──────────────────────────────────────

def _build_guard(tool_name: str, passages: list[dict], guard_cfg: dict) -> LLMGuard:
    """Build an LLMGuard from policy passages + a single llm_guard config entry."""
    return LLMGuard(
        tool_name=tool_name,
        tool_description=f"Tool '{tool_name}' in the airline customer-service domain.",
        policy_passages=[
            PolicyPassage(id=p["id"], text=p["text"], section=p.get("section"))
            for p in passages
        ],
        llm=guard_cfg.get("llm", "gpt-4.1-mini"),
        llm_args=guard_cfg.get("llm_args"),
        template_path=guard_cfg.get("template_path"),
        history_window=guard_cfg.get("history_window", 10),
        history_mode=guard_cfg.get("history_mode", "full"),
    )


# ── Run a single test case ────────────────────────────────────────────────────

def run_case(tc: dict, guard_cache: dict, guard_configs: dict[str, dict], verbose: bool) -> CaseResult:
    gc      = tc["guard_config"]
    vtc_raw = tc["violating_tool_call"]
    tool    = gc["tool_name"]

    # Look up this tool's guard config
    tool_guard_cfg = guard_configs.get(tool)
    if tool_guard_cfg is None:
        return CaseResult(
            tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool,
            key_arg="—", verdict="ERROR", correct=False, latency_s=0.0,
            error=f"No llm_guard entry for tool '{tool}' in the guardrail config.",
        )

    # Build (or reuse) guard for this tool
    if tool not in guard_cache:
        guard_cache[tool] = _build_guard(tool, gc["policy_passages"], tool_guard_cfg)
    guard = guard_cache[tool]

    # Reconstruct inputs
    history  = _rebuild_messages(tc["history"])
    tool_call = ToolCall(id="bench-vtc", name=vtc_raw["name"], arguments=vtc_raw["arguments"])

    # Key arg for display
    args    = vtc_raw["arguments"]
    key_arg = args.get("reservation_id") or args.get("user_id") or str(list(args.values())[:1])

    t0 = time.perf_counter()
    try:
        verdict: GuardVerdict = guard.check(tool_call, env=None, history=history)
        latency = time.perf_counter() - t0

        blocked = not verdict.allowed  # DENY or ESCALATE → blocked
        v_str   = "DENY" if not verdict.allowed else "ALLOW"

        if verbose:
            icon = "✓" if blocked else "✗"
            print(f"    {icon} {tc['id']} | {tool:35s} | {v_str:8s} | {latency:.1f}s | {(verdict.reason or '')[:60]}")

        return CaseResult(
            tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool, key_arg=key_arg,
            verdict=v_str, correct=blocked, latency_s=latency,
            reason=verdict.reason or "",
        )

    except Exception as exc:
        latency = time.perf_counter() - t0
        if verbose:
            print(f"    ! {tc['id']} | ERROR: {exc}")
        return CaseResult(
            tc_id=tc["id"], task_id=tc["task_id"], tool_name=tool, key_arg=key_arg,
            verdict="ERROR", correct=False, latency_s=latency, error=str(exc),
        )


# ── Display ───────────────────────────────────────────────────────────────────

def _print_summary(bench: BenchResult) -> None:
    from collections import defaultdict
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        rich_ok = True
    except ImportError:
        rich_ok = False

    if not rich_ok:
        print(f"\nRecall: {bench.recall:.1%}  ({bench.blocked}/{bench.total - bench.errors} blocked)")
        for r in bench.case_results:
            status = "✓" if r.correct else ("!" if r.verdict == "ERROR" else "✗")
            print(f"  {status} {r.tc_id} task={r.task_id} {r.tool_name} {r.key_arg} → {r.verdict}")
        return

    console = Console()

    # ── Per-case table
    table = Table(
        title=f"Guardrail Bench — model: {bench.model}",
        box=box.ROUNDED, header_style="bold cyan",
        border_style="bright_black", show_lines=True,
    )
    table.add_column("ID",      no_wrap=True)
    table.add_column("Task",    justify="center")
    table.add_column("Tool",    style="cyan", no_wrap=True)
    table.add_column("Key Arg", style="dim",  no_wrap=True)
    table.add_column("Verdict", justify="center", min_width=8)
    table.add_column("Latency", justify="right")
    table.add_column("Reason",  max_width=55)

    for r in bench.case_results:
        if r.verdict == "ERROR":
            vstyle, icon = "bold red",   "⚠ ERROR"
        elif r.correct:
            vstyle, icon = "bold green", "✓ DENY"
        else:
            vstyle, icon = "bold red",   "✗ ALLOW"

        table.add_row(
            r.tc_id,
            r.task_id,
            r.tool_name,
            r.key_arg,
            Text(icon, style=vstyle),
            f"{r.latency_s:.1f}s",
            (r.reason or r.error)[:55],
        )

    console.print()
    console.print(table)

    # ── By-tool breakdown
    by_tool: dict = defaultdict(lambda: {"total": 0, "blocked": 0})
    for r in bench.case_results:
        by_tool[r.tool_name]["total"] += 1
        if r.correct:
            by_tool[r.tool_name]["blocked"] += 1

    tool_lines = []
    for tool, stats in sorted(by_tool.items()):
        rec = stats["blocked"] / stats["total"] if stats["total"] else 0
        col = "green" if rec >= 0.8 else "yellow" if rec >= 0.5 else "red"
        tool_lines.append(
            f"[dim]{tool}[/dim]  [{col}]{rec:.0%}[/{col}] ({stats['blocked']}/{stats['total']})"
        )

    rec_col = "green" if bench.recall >= 0.8 else "yellow" if bench.recall >= 0.5 else "red"
    summary = (
        f"[bold]Recall[/bold]  [{rec_col}]{bench.recall:.1%}[/{rec_col}]"
        f"  ({bench.blocked} blocked / {bench.total - bench.errors} evaluated"
        + (f" · {bench.errors} errors" if bench.errors else "") + ")\n"
        f"[bold]Avg latency[/bold]  {bench.avg_latency:.1f}s per call\n\n"
        + "\n".join(tool_lines)
    )
    console.print(Panel(summary, title="Summary", border_style="bold blue", padding=(0, 2)))
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _DEFAULT_CFG = HERE.parent / "guardrail_configs" / "airline_llm_guard.json"

    parser = argparse.ArgumentParser(
        description="Guardrail workbench — benchmark LLMGuard recall",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--guardrail-config",
        default=str(_DEFAULT_CFG),
        help=(
            "Path to a guardrail config JSON with llm_guard entries.\n"
            f"All guard settings (llm, history_mode, history_window, …) are read from there.\n"
            f"Default: {_DEFAULT_CFG.relative_to(HERE.parent)}"
        ),
    )
    parser.add_argument("--cases",   nargs="*",
                        help="Specific TC IDs to run, e.g. TC-001 TC-009 (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each result as it completes")
    args = parser.parse_args()

    # Load guardrail config
    cfg_path = Path(args.guardrail_config)
    if not cfg_path.is_absolute():
        cfg_path = HERE.parent / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: guardrail config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    guard_configs = _load_guard_configs(cfg_path)
    if not guard_configs:
        print(f"ERROR: no llm_guard entries found in {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Summarise active guard config for display
    cfg_summary = "  ".join(
        f"{tool}: llm={g.get('llm','?')} mode={g.get('history_mode','full')}"
        for tool, g in guard_configs.items()
    )

    # Load test cases
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
    print(f"Guardrail Workbench  |  config: {cfg_path.name}  |  {len(test_cases)} test cases")
    print(f"  Guards: {cfg_summary}")
    print()

    guard_cache: dict[str, LLMGuard] = {}
    bench = BenchResult(model=cfg_path.stem)

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i:2d}/{len(test_cases)}] {tc['id']}  task={tc['task_id']}  {tc['violating_tool_call']['name']}", end="", flush=True)
        if not args.verbose:
            print(" ...", end="", flush=True)
        else:
            print()

        result = run_case(tc, guard_cache, guard_configs, args.verbose)
        bench.case_results.append(result)

        if not args.verbose:
            icon = "✓" if result.correct else ("⚠" if result.verdict == "ERROR" else "✗")
            print(f" {icon} {result.verdict} ({result.latency_s:.1f}s)")

    _print_summary(bench)

    # Save JSON results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts        = time.strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"bench_{ts}_{cfg_path.stem}.json"
    out_data  = {
        "model":   bench.model,
        "recall":  bench.recall,
        "total":   bench.total,
        "blocked": bench.blocked,
        "allowed": bench.allowed,
        "errors":  bench.errors,
        "avg_latency_s": bench.avg_latency,
        "cases": [
            {
                "tc_id":    r.tc_id, "task_id": r.task_id,
                "tool":     r.tool_name, "key_arg": r.key_arg,
                "verdict":  r.verdict,  "correct": r.correct,
                "latency_s": r.latency_s, "reason": r.reason,
            }
            for r in bench.case_results
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Results saved → {out_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
