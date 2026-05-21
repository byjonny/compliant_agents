"""
Policy-Tool Mapper — full pipeline runner.

For each model: run policy-map, then policy-map-eval twice (high-confidence only
and all). All file paths are derived from the domain name + pipeline mode.

File conventions (all inside policy_tool_mapper/):

  Mode: llm  (default)
    output/{domain}-mappings-{model}.json
    output/{domain}-eval-{model}-high.json
    output/{domain}-eval-{model}-all.json

  Mode: retrieval
    output/{domain}-mappings-{model}-retrieval.json
    output/{domain}-eval-{model}-retrieval-high.json
    output/{domain}-eval-{model}-retrieval-all.json

  Summary (always):
    output/{domain}-pipeline-summary-{mode}.json

Usage (from inside tau2-bench/):

    # LLM mode — all models
    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain airline \\
        --models gpt-4.1-mini gpt-5.1 gpt-5.4

    # Retrieval mode
    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain airline \\
        --models gpt-5.4 \\
        --mode retrieval

    # Re-run eval only on existing mappings (fast)
    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain airline \\
        --models gpt-4.1-mini gpt-5.1 gpt-5.4 \\
        --skip-mapping

    # Different domain
    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain telecom \\
        --models gpt-5.4
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent


# ── Path helpers ──────────────────────────────────────────────────────────────

def _mode_suffix(mode: str) -> str:
    """File suffix inserted for non-default modes."""
    return f"-{mode}" if mode != "llm" else ""


def _paths(domain: str, model: str, mode: str = "llm") -> dict[str, Path]:
    """Derive all file paths from domain + model + mode."""
    d   = domain.lower()
    cap = domain[0].upper() + domain[1:]
    sfx = _mode_suffix(mode)
    return {
        "policy":       HERE / "input"        / f"{cap}Policy.md",
        "tools":        HERE / "input"        / f"{cap}Tools.json",
        "ground_truth": HERE / "ground_truth" / f"{d}-ground-truth.json",
        "mappings":     HERE / "output"       / f"{d}-mappings-{model}{sfx}.json",
        "eval_high":    HERE / "output"       / f"{d}-eval-{model}{sfx}-high.json",
        "eval_all":     HERE / "output"       / f"{d}-eval-{model}{sfx}-all.json",
    }


# ── Sub-process helpers ───────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> int:
    print(f"\n  › {label}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"  ✓ done")
    return result.returncode


def run_mapping(paths: dict[str, Path], model: str, mode: str, args) -> bool:
    cmd = [
        "uv", "run", "policy-map",
        "--policy",  str(paths["policy"]),
        "--openapi", str(paths["tools"]),
        "--output",  str(paths["mappings"]),
        "--model",   model,
        "--mode",    mode,
    ]
    if mode == "retrieval":
        cmd += [
            "--embed-model",  args.embed_model,
            "--ce-model",     args.ce_model,
            "--ce-top-k",     str(args.ce_top_k),
        ]
    return _run(cmd, f"policy-map  [{model}]  mode={mode}") == 0


def run_eval(paths: dict[str, Path], model: str, mode: str, high_only: bool) -> bool:
    conf_label = "high" if high_only else "all"
    out        = paths["eval_high"] if high_only else paths["eval_all"]
    cmd = [
        "uv", "run", "policy-map-eval",
        "--predicted",    str(paths["mappings"]),
        "--ground-truth", str(paths["ground_truth"]),
        "--output",       str(out),
    ]
    if high_only:
        cmd.append("--confidence-high-only")
    return _run(cmd, f"policy-map-eval [{model}] mode={mode} conf={conf_label}") == 0


def _read_overall(eval_path: Path) -> dict | None:
    try:
        return json.loads(eval_path.read_text()).get("overall")
    except Exception:
        return None


def _print_comparison_table(rows: list[dict]) -> None:
    """Print a unified comparison table from a list of result dicts."""
    W = 25 + 1 + 12 + 1 + 6 + 1 + 7 + 1 + 7 + 1 + 8 + 1 + 7 + 1 + 7 + 1 + 8
    print(f"\n{'═'*W}")
    print(f"{'Model':<25} {'Mode':<12} {'Conf':<6} "
          f"{'Macro P':>7} {'Macro R':>7} {'Macro F1':>8} "
          f"{'Micro P':>7} {'Micro R':>7} {'Micro F1':>8}")
    print(f"{'-'*25} {'-'*12} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*8}")
    for r in rows:
        s     = r["scores"]
        macro = s.get("macro", {})
        micro = s.get("micro", {})
        print(
            f"{r['model']:<25} {r['mode']:<12} {r['conf']:<6} "
            f"{macro.get('precision', 0):>6.1%} "
            f"{macro.get('recall',    0):>6.1%} "
            f"{macro.get('f1',        0):>7.1%} "
            f"{micro.get('precision', 0):>6.1%} "
            f"{micro.get('recall',    0):>6.1%} "
            f"{micro.get('f1',        0):>7.1%}"
        )
    print(f"{'═'*W}")


def compare(domain: str) -> None:
    """
    Discover all eval files for a domain and print a unified comparison table.

    Filename patterns recognised:
      {domain}-eval-{model}-high.json          → llm mode, high conf
      {domain}-eval-{model}-all.json           → llm mode, all conf
      {domain}-eval-{model}-retrieval-high.json → retrieval mode, high conf
      {domain}-eval-{model}-retrieval-all.json  → retrieval mode, all conf
    """
    d       = domain.lower()
    out_dir = HERE / "output"
    prefix  = f"{d}-eval-"

    rows: list[dict] = []
    for path in sorted(out_dir.glob(f"{prefix}*.json")):
        stem = path.stem[len(prefix):]   # e.g. "gpt-5.4-retrieval-high"

        # Detect confidence suffix
        if stem.endswith("-high"):
            conf = "high"
            stem = stem[:-5]
        elif stem.endswith("-all"):
            conf = "all"
            stem = stem[:-4]
        else:
            continue   # unknown pattern

        # Detect mode suffix
        if stem.endswith("-retrieval"):
            mode  = "retrieval"
            model = stem[:-10]
        else:
            mode  = "llm"
            model = stem

        scores = _read_overall(path)
        if scores is None:
            continue
        rows.append({"model": model, "mode": mode, "conf": conf, "scores": scores})

    if not rows:
        print(f"No eval files found for domain '{domain}' in {out_dir}")
        return

    # Sort: model → mode → conf
    _order = {"llm": 0, "retrieval": 1}
    rows.sort(key=lambda r: (r["model"], _order.get(r["mode"], 9), r["conf"]))

    print(f"\nComparison — domain: {domain}  ({len(rows)} result(s) found)")
    _print_comparison_table(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full policy-tool mapper pipeline for a domain."
    )
    parser.add_argument(
        "--domain", required=True,
        help="Domain name (e.g. airline, telecom). Drives all file paths.",
    )
    parser.add_argument(
        "--models", nargs="+", default=[],
        help="One or more LLM model IDs (e.g. gpt-4.1-mini gpt-5.1 gpt-5.4). "
             "Not required when --compare is used.",
    )
    parser.add_argument(
        "--mode", choices=["llm", "retrieval"], default="llm",
        help=(
            "Pipeline architecture. "
            "'llm' (default): chunker→profiler→mapper→sweeper. "
            "'retrieval': chunker→profiler→BM25+biencoder+cross-encoder→judge→sweeper. "
            "Reflected in all output filenames."
        ),
    )
    parser.add_argument(
        "--embed-model", default="text-embedding-3-large",
        help="OpenAI embedding model for bi-encoder (retrieval mode only).",
    )
    parser.add_argument(
        "--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help=(
            "Cross-encoder reranker model (retrieval mode only, "
            "default: cross-encoder/ms-marco-MiniLM-L-6-v2). "
            "Use 'none' to skip reranking entirely."
        ),
    )
    parser.add_argument(
        "--ce-top-k", type=int, default=20,
        help="Number of cross-encoder reranked candidates to keep per tool (retrieval mode only, default: 20).",
    )
    parser.add_argument(
        "--skip-mapping", action="store_true",
        help="Skip policy-map step; only (re-)run eval on existing mapping files.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Run mapping only. Useful when a new domain has no ground-truth file yet.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help=(
            "Print a unified comparison table across ALL existing eval files for "
            "the domain (all models, all modes). No mapping or eval is re-run."
        ),
    )
    parser.add_argument(
        "--summary", default=None,
        help="Path for the JSON summary. "
             "Defaults to output/{domain}-pipeline-summary-{mode}.json.",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.domain)
        return

    domain = args.domain
    models = args.models
    mode   = args.mode
    if not models:
        print("ERROR: --models requires at least one model", file=sys.stderr)
        sys.exit(1)

    summary_path = Path(args.summary) if args.summary else (
        HERE / "output" / f"{domain.lower()}-pipeline-summary{_mode_suffix(mode)}.json"
    )

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    sample_paths = _paths(domain, models[0], mode)
    required = ("policy", "tools") if args.skip_eval else ("policy", "tools", "ground_truth")
    missing = [
        str(sample_paths[k])
        for k in required
        if not sample_paths[k].exists()
    ]
    if missing:
        print("ERROR: Missing input files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print(f"Pipeline: domain={domain}  mode={mode}  models={models}")
    print(f"  Policy:       {sample_paths['policy']}")
    print(f"  Tools:        {sample_paths['tools']}")
    print(f"  Ground truth: {sample_paths['ground_truth'] if sample_paths['ground_truth'].exists() else 'not used'}")

    # ── Run ────────────────────────────────────────────────────────────────────
    results = []
    for model in models:
        paths = _paths(domain, model, mode)
        print(f"\n{'─'*60}")
        print(f"Model: {model}  mode={mode}")
        print(f"{'─'*60}")

        mapping_ok = True
        if not args.skip_mapping:
            mapping_ok = run_mapping(paths, model, mode, args)
        else:
            if paths["mappings"].exists():
                print(f"  [skip-mapping] Using {paths['mappings'].name}")
            else:
                print(f"  ERROR: mapping file not found: {paths['mappings']}", file=sys.stderr)
                mapping_ok = False

        if not mapping_ok:
            results.append({"model": model, "mode": mode, "status": "mapping_failed"})
            continue

        if args.skip_eval:
            results.append({
                "model":    model,
                "mode":     mode,
                "status":   "mapping_only",
                "mappings": str(paths["mappings"]),
                "eval_high": None,
                "eval_all":  None,
                "scores_high": None,
                "scores_all":  None,
            })
            continue

        eval_high_ok = run_eval(paths, model, mode, high_only=True)
        eval_all_ok  = run_eval(paths, model, mode, high_only=False)

        results.append({
            "model":       model,
            "mode":        mode,
            "status":      "ok" if (eval_high_ok and eval_all_ok) else "partial",
            "mappings":    str(paths["mappings"]),
            "eval_high":   str(paths["eval_high"]),
            "eval_all":    str(paths["eval_all"]),
            "scores_high": _read_overall(paths["eval_high"]),
            "scores_all":  _read_overall(paths["eval_all"]),
        })

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "domain":    domain,
        "mode":      mode,
        "models":    models,
        "skip_eval": args.skip_eval,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results":   results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 25 + 1 + 12 + 1 + 6 + 1 + 7 + 1 + 7 + 1 + 8 + 1 + 7 + 1 + 7 + 1 + 8
    print(f"\n{'═'*W}")
    print(f"Summary  [domain={domain}  mode={mode}]")
    print(f"{'═'*W}")
    print(
        f"{'Model':<25} {'Mode':<12} {'Conf':<6} "
        f"{'Macro P':>7} {'Macro R':>7} {'Macro F1':>8} "
        f"{'Micro P':>7} {'Micro R':>7} {'Micro F1':>8}"
    )
    print(
        f"{'-'*25} {'-'*12} {'-'*6} "
        f"{'-'*7} {'-'*7} {'-'*8} "
        f"{'-'*7} {'-'*7} {'-'*8}"
    )
    for r in results:
        if r.get("status") == "mapping_failed":
            print(f"{r['model']:<25} {'MAPPING FAILED'}")
            continue
        if r.get("status") == "mapping_only":
            print(f"{r['model']:<25} {r['mode']:<12} {'map':<6} {'mapping written':>50}")
            continue
        for conf_label, key in [("high", "scores_high"), ("all", "scores_all")]:
            s = r.get(key)
            if not s:
                continue
            macro = s.get("macro", {})
            micro = s.get("micro", {})
            print(
                f"{r['model']:<25} {r['mode']:<12} {conf_label:<6} "
                f"{macro.get('precision', 0):>6.1%} "
                f"{macro.get('recall',    0):>6.1%} "
                f"{macro.get('f1',        0):>7.1%} "
                f"{micro.get('precision', 0):>6.1%} "
                f"{micro.get('recall',    0):>6.1%} "
                f"{micro.get('f1',        0):>7.1%}"
            )

    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
