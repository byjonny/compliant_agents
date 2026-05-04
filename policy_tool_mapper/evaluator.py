"""
Precision / recall / F1 evaluator for policy-tool mappings.

Matching strategy: two statements are considered the same if their
text overlap ratio (SequenceMatcher) exceeds a configurable threshold.
This tolerates minor wording differences between ground-truth and
predicted statement texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

DEFAULT_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolEvalResult:
    tool_id: str
    tool_name: str
    gt_count: int       # |ground-truth statements|
    pred_count: int     # |predicted statements|
    matched_gt: int     # GT statements that were found in predictions  (TP for recall)
    matched_pred: int   # predicted statements that match a GT entry    (TP for precision)

    @property
    def precision(self) -> Optional[float]:
        return self.matched_pred / self.pred_count if self.pred_count else None

    @property
    def recall(self) -> Optional[float]:
        return self.matched_gt / self.gt_count if self.gt_count else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalSummary:
    tool_results: list[ToolEvalResult]
    threshold: float

    # -- macro (per-tool average, equal weight per tool) --------------------

    @property
    def macro_precision(self) -> float:
        vals = [r.precision for r in self.tool_results if r.precision is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [r.recall for r in self.tool_results if r.recall is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        p, r = self.macro_precision, self.macro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    # -- micro (pooled counts across all tools) -----------------------------

    @property
    def micro_precision(self) -> float:
        total_matched = sum(r.matched_pred for r in self.tool_results)
        total_pred = sum(r.pred_count for r in self.tool_results)
        return total_matched / total_pred if total_pred else 0.0

    @property
    def micro_recall(self) -> float:
        total_matched = sum(r.matched_gt for r in self.tool_results)
        total_gt = sum(r.gt_count for r in self.tool_results)
        return total_matched / total_gt if total_gt else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _overlap(a: str, b: str) -> float:
    """Normalized overlap ratio between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def evaluate(
    ground_truth: dict,
    predicted: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> EvalSummary:
    """
    Compare predicted tool-statement mappings against ground truth.

    Both dicts must follow the output_formatter JSON structure:
      {"mappings": [{"tool_id": ..., "tool_name": ..., "statements": [{"id": ..., "text": ...}]}]}
    """
    gt_by_tool: dict[str, dict] = {m["tool_id"]: m for m in ground_truth.get("mappings", [])}
    pred_by_tool: dict[str, dict] = {m["tool_id"]: m for m in predicted.get("mappings", [])}

    all_tool_ids = sorted(set(gt_by_tool) | set(pred_by_tool))
    results: list[ToolEvalResult] = []

    for tool_id in all_tool_ids:
        gt_entry = gt_by_tool.get(tool_id)
        pred_entry = pred_by_tool.get(tool_id)

        gt_stmts = gt_entry["statements"] if gt_entry else []
        pred_stmts = pred_entry["statements"] if pred_entry else []
        tool_name = (gt_entry or pred_entry)["tool_name"]  # type: ignore[index]

        matched_gt: set[str] = set()
        matched_pred: set[str] = set()

        for gt in gt_stmts:
            for pred in pred_stmts:
                if _overlap(gt["text"], pred["text"]) >= threshold:
                    matched_gt.add(gt["id"])
                    matched_pred.add(pred["id"])

        results.append(ToolEvalResult(
            tool_id=tool_id,
            tool_name=tool_name,
            gt_count=len(gt_stmts),
            pred_count=len(pred_stmts),
            matched_gt=len(matched_gt),
            matched_pred=len(matched_pred),
        ))

    return EvalSummary(tool_results=results, threshold=threshold)


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------


def print_summary(summary: EvalSummary, title: str = "Policy-Tool Mapping Evaluation") -> None:
    """Render a colour-coded table + summary panel using Rich."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    def _pct(val: Optional[float]) -> str:
        return f"{val:.1%}" if val is not None else "—"

    def _style(val: Optional[float]) -> str:
        if val is None:
            return "dim"
        if val >= 0.8:
            return "bold green"
        if val >= 0.5:
            return "yellow"
        return "bold red"

    table = Table(
        title=f"{title}  (overlap ≥ {summary.threshold:.0%})",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=True,
    )
    table.add_column("Tool", style="cyan", no_wrap=True, min_width=28)
    table.add_column("GT",   justify="right", style="dim")
    table.add_column("Pred", justify="right", style="dim")
    table.add_column("Matched", justify="right")
    table.add_column("Precision", justify="right", min_width=10)
    table.add_column("Recall",    justify="right", min_width=10)
    table.add_column("F1",        justify="right", min_width=10)

    for r in summary.tool_results:
        p, rec, f = r.precision, r.recall, r.f1
        table.add_row(
            r.tool_name,
            str(r.gt_count),
            str(r.pred_count),
            f"{r.matched_gt} / {r.matched_pred}",
            Text(_pct(p),   style=_style(p)),
            Text(_pct(rec), style=_style(rec)),
            Text(_pct(f),   style=_style(f)),
        )

    console.print()
    console.print(table)

    # Overall summary panel
    mp  = summary.macro_precision
    mr  = summary.macro_recall
    mf  = summary.macro_f1
    up  = summary.micro_precision
    ur  = summary.micro_recall
    uf  = summary.micro_f1

    lines = [
        f"[bold]Macro[/bold]   Precision [bold]{mp:.1%}[/bold]   Recall [bold]{mr:.1%}[/bold]   F1 [{_style(mf)}]{mf:.1%}[/{_style(mf)}]",
        f"[bold]Micro[/bold]   Precision [bold]{up:.1%}[/bold]   Recall [bold]{ur:.1%}[/bold]   F1 [{_style(uf)}]{uf:.1%}[/{_style(uf)}]",
    ]
    console.print(Panel("\n".join(lines), title="Overall", border_style="bold blue", padding=(0, 2)))
    console.print()
