"""
Standalone evaluator — compare a policy-tool mapper output file against a ground truth.

Usage:
    policy-map-eval \\
        --predicted  output/airline-mappings-gpt-5.4.json \\
        --ground-truth ground_truth/airline-ground-truth.json

    policy-map-eval \\
        --predicted  output/airline-mappings-gpt-5.4.json \\
        --ground-truth ground_truth/airline-ground-truth.json \\
        --confidence-high-only \\
        --output output/airline-eval-gpt-5.4.json

    policy-map-eval \\
        --predicted  output/airline-mappings-gpt-5.4.json \\
        --ground-truth ground_truth/airline-ground-truth.json \\
        --threshold 0.7

Output JSON structure:
    {
      "metadata": { ... },
      "overall": { "precision", "recall", "f1" },
      "per_tool": {
        "<tool_id>": {
          "precision", "recall", "f1",
          "n_predicted", "n_gt", "n_matched",
          "precision_paragraphs": [ ... ],   // predicted but NOT in GT  (false positives)
          "recall_paragraphs":    [ ... ]    // in GT but NOT predicted   (false negatives)
        }
      }
    }
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_THRESHOLD = 0.8


# ── Matching helpers ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """
    Normalise formatting differences before comparing texts:
      - Strip markdown bullets (- / * at line start)
      - Collapse newlines and extra whitespace to single spaces
      - Lowercase
    This prevents format differences (semicolons vs bullet lists, etc.)
    from hiding semantic matches.
    """
    import re
    t = text
    t = re.sub(r'^\s*[-*]\s+', ' ', t, flags=re.MULTILINE)  # strip bullet markers
    t = re.sub(r'\s+', ' ', t)                               # collapse whitespace
    return t.strip().lower()


def _overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _match_statements(
    pred_stmts: list[dict],
    gt_stmts:   list[dict],
    threshold:  float,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Bipartite matching between predicted and GT statements via text overlap.

    Returns:
        matched_pred_ids  — predicted IDs that matched a GT entry
        matched_gt_ids    — GT IDs that were matched by a prediction
        unmatched_pred    — predicted IDs with no GT match  (false positives)
        unmatched_gt      — GT IDs with no prediction match (false negatives)
    """
    matched_pred: set[str] = set()
    matched_gt:   set[str] = set()

    for p in pred_stmts:
        for g in gt_stmts:
            if _overlap(p["text"], g["text"]) >= threshold:
                matched_pred.add(p["id"])
                matched_gt.add(g["id"])

    unmatched_pred = {p["id"] for p in pred_stmts} - matched_pred
    unmatched_gt   = {g["id"] for g in gt_stmts}   - matched_gt
    return matched_pred, matched_gt, unmatched_pred, unmatched_gt


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_mappings(
    predicted:     dict,
    ground_truth:  dict,
    threshold:     float = DEFAULT_THRESHOLD,
    high_conf_only: bool = False,
) -> dict:
    pred_by_tool = {m["tool_id"]: m for m in predicted.get("mappings", [])}
    gt_by_tool   = {m["tool_id"]: m for m in ground_truth.get("mappings", [])}

    all_tool_ids = sorted(set(pred_by_tool) | set(gt_by_tool))
    per_tool: dict = {}

    total_tp_pred = total_pred = total_gt = 0

    for tool_id in all_tool_ids:
        pred_entry = pred_by_tool.get(tool_id)
        gt_entry   = gt_by_tool.get(tool_id)

        pred_stmts_raw = pred_entry["statements"] if pred_entry else []
        gt_stmts       = gt_entry["statements"]   if gt_entry   else []

        # Optionally filter to high-confidence predictions only
        if high_conf_only:
            pred_stmts = [s for s in pred_stmts_raw if s.get("confidence") == "high"]
        else:
            pred_stmts = pred_stmts_raw

        matched_pred, matched_gt, unmatched_pred_ids, unmatched_gt_ids = \
            _match_statements(pred_stmts, gt_stmts, threshold)

        n_pred    = len(pred_stmts)
        n_gt      = len(gt_stmts)
        n_matched = len(matched_pred)

        # Both precision and recall are undefined (None) when there is no ground
        # truth for this tool — they are excluded from all aggregations.
        if n_gt == 0:
            precision = None
            recall: float | None = None
            f1: float | None = None
        else:
            precision = (n_matched / n_pred) if n_pred else 0.0
            recall    = n_matched / n_gt
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

        # Build paragraph lists (include full statement objects for readability)
        pred_by_id = {s["id"]: s for s in pred_stmts}
        gt_by_id   = {s["id"]: s for s in gt_stmts}

        precision_paragraphs = [pred_by_id[i] for i in sorted(unmatched_pred_ids)]
        recall_paragraphs    = [gt_by_id[i]   for i in sorted(unmatched_gt_ids)]

        per_tool[tool_id] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall":    round(recall,    4) if recall    is not None else None,
            "f1":        round(f1,        4) if f1        is not None else None,
            "n_predicted":  n_pred,
            "n_gt":         n_gt,
            "n_matched":    n_matched,
            "precision_paragraphs": precision_paragraphs,  # in pred but not GT
            "recall_paragraphs":    recall_paragraphs,     # in GT but not pred
        }

        total_pred    += n_pred
        total_gt      += n_gt
        total_tp_pred += n_matched

    # Macro: skip tools with no GT (None) from all averages
    p_vals = [v["precision"] for v in per_tool.values() if v["precision"] is not None]
    r_vals = [v["recall"]    for v in per_tool.values() if v["recall"]    is not None]
    f_vals = [v["f1"]        for v in per_tool.values() if v["f1"]        is not None]

    macro_p = sum(p_vals) / len(p_vals) if p_vals else 0.0
    macro_r = sum(r_vals) / len(r_vals) if r_vals else 0.0
    macro_f = sum(f_vals) / len(f_vals) if f_vals else 0.0

    # Micro: computed over tools that have GT statements
    micro_p = total_tp_pred / total_pred if total_pred else 0.0
    micro_r = total_tp_pred / total_gt   if total_gt   else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) else 0.0)

    return {
        "metadata": {
            "threshold":        threshold,
            "high_conf_only":   high_conf_only,
            "total_tools":      len(all_tool_ids),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        },
        "overall": {
            "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f, 4)},
            "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f, 4)},
        },
        "per_tool": per_tool,
    }


# ── Rich display ──────────────────────────────────────────────────────────────

def print_eval(result: dict, predicted_file: str = "", gt_file: str = "") -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        rich_ok = True
    except ImportError:
        rich_ok = False

    meta    = result["metadata"]
    overall = result["overall"]
    per_t   = result["per_tool"]

    if not rich_ok:
        print(f"Macro P={overall['macro']['precision']:.1%}  "
              f"R={overall['macro']['recall']:.1%}  F1={overall['macro']['f1']:.1%}")
        for tid, s in per_t.items():
            print(f"  {tid:35s}  P={s['precision']:.1%}  R={s['recall']:.1%}  "
                  f"F1={s['f1']:.1%}  "
                  f"({s['n_matched']}/{s['n_gt']} GT matched, "
                  f"{len(s['precision_paragraphs'])} FP, "
                  f"{len(s['recall_paragraphs'])} FN)")
        return

    console = Console()

    def _col(v: float) -> str:
        return "bold green" if v >= 0.8 else "yellow" if v >= 0.5 else "bold red"

    # ── Per-tool table
    tbl = Table(
        title=(f"Policy-Tool Mapping Evaluation"
               + (f"  ·  confidence=high only" if meta["high_conf_only"] else "")
               + (f"  ·  overlap≥{meta['threshold']:.0%}")),
        box=box.ROUNDED, header_style="bold cyan",
        border_style="bright_black", show_lines=True,
    )
    tbl.add_column("Tool",       style="cyan", no_wrap=True, min_width=28)
    tbl.add_column("GT",         justify="right", style="dim")
    tbl.add_column("Pred",       justify="right", style="dim")
    tbl.add_column("Matched",    justify="right")
    tbl.add_column("Precision",  justify="right", min_width=10)
    tbl.add_column("Recall",     justify="right", min_width=10)
    tbl.add_column("F1",         justify="right", min_width=10)
    tbl.add_column("FP",         justify="right", style="dim", min_width=4)
    tbl.add_column("FN",         justify="right", style="dim", min_width=4)

    def _fmt(v: float | None) -> Text:
        if v is None:
            return Text("—", style="dim")
        return Text(f"{v:.1%}", style=_col(v))

    for tid, s in sorted(per_t.items()):
        p, r, f = s["precision"], s["recall"], s["f1"]
        tbl.add_row(
            tid,
            str(s["n_gt"]),
            str(s["n_predicted"]),
            str(s["n_matched"]),
            _fmt(p),
            _fmt(r),
            _fmt(f),
            str(len(s["precision_paragraphs"])),
            str(len(s["recall_paragraphs"])),
        )

    console.print()
    console.print(tbl)

    # ── Overall panel
    mo, mi = overall["macro"], overall["micro"]
    summary = (
        f"[bold]Macro[/bold]   "
        f"P [{_col(mo['precision'])}]{mo['precision']:.1%}[/{_col(mo['precision'])}]  "
        f"R [{_col(mo['recall'])}]{mo['recall']:.1%}[/{_col(mo['recall'])}]  "
        f"F1 [{_col(mo['f1'])}]{mo['f1']:.1%}[/{_col(mo['f1'])}]\n"
        f"[bold]Micro[/bold]   "
        f"P [{_col(mi['precision'])}]{mi['precision']:.1%}[/{_col(mi['precision'])}]  "
        f"R [{_col(mi['recall'])}]{mi['recall']:.1%}[/{_col(mi['recall'])}]  "
        f"F1 [{_col(mi['f1'])}]{mi['f1']:.1%}[/{_col(mi['f1'])}]"
    )
    console.print(Panel(summary, title="Overall", border_style="bold blue", padding=(0, 2)))
    console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy-tool mapper output file against a ground truth."
    )
    parser.add_argument("--predicted",      required=True,
                        help="Path to the predicted mappings JSON (mapper output).")
    parser.add_argument("--ground-truth",   required=True,
                        help="Path to the ground-truth mappings JSON.")
    parser.add_argument("--output",         default=None,
                        help="Save evaluation result to this JSON file (optional).")
    parser.add_argument("--confidence-high-only", action="store_true",
                        help="Only consider predicted statements with confidence='high'.")
    parser.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD,
                        help=f"Text overlap threshold for statement matching (default {DEFAULT_THRESHOLD}).")
    args = parser.parse_args()

    pred_path = Path(args.predicted)
    gt_path   = Path(args.ground_truth)

    for p, label in [(pred_path, "--predicted"), (gt_path, "--ground-truth")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    predicted    = json.loads(pred_path.read_text())
    ground_truth = json.loads(gt_path.read_text())

    result = evaluate_mappings(
        predicted,
        ground_truth,
        threshold=args.threshold,
        high_conf_only=args.confidence_high_only,
    )

    # Attach source filenames
    result["metadata"]["predicted_file"]    = str(pred_path)
    result["metadata"]["ground_truth_file"] = str(gt_path)

    print_eval(result, str(pred_path), str(gt_path))

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Eval result saved → {out_path}")


if __name__ == "__main__":
    main()
