"""
Policy-Tool Mapper — full pipeline runner.

For each model: run policy-map, then policy-map-eval twice (high-confidence only
and all). All file paths are derived from the domain name.

File conventions (all inside policy_tool_mapper/):
  input/{Domain}Policy.md               — policy document
  input/{Domain}Tools.json              — tool schemas
  ground_truth/{domain}-ground-truth.json
  output/{domain}-mappings-{model}.json — mapper output
  output/{domain}-eval-{model}-high.json — eval, confidence=high only
  output/{domain}-eval-{model}-all.json  — eval, all confidence levels

Usage (from inside tau2-bench/):
    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain airline \\
        --models gpt-4.1-mini gpt-5.1 gpt-5.4

    uv run python policy_tool_mapper/run_pipeline.py \\
        --domain telecom \\
        --models claude-sonnet-4-6 \\
        --skip-mapping          # only re-run eval on existing mappings
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def _paths(domain: str, model: str) -> dict[str, Path]:
    """Derive all file paths from domain + model."""
    d = domain.lower()
    # Capitalise first letter for input files (airlinePolicy.md)
    cap = domain[0].upper() + domain[1:]
    return {
        "policy":      HERE / "input"        / f"{cap}Policy.md",
        "tools":       HERE / "input"        / f"{cap}Tools.json",
        "ground_truth":HERE / "ground_truth" / f"{d}-ground-truth.json",
        "mappings":    HERE / "output"       / f"{d}-mappings-{model}.json",
        "eval_high":   HERE / "output"       / f"{d}-eval-{model}-high.json",
        "eval_all":    HERE / "output"       / f"{d}-eval-{model}-all.json",
    }


def _run(cmd: list[str], label: str) -> int:
    print(f"\n  › {label}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"  ✓ done")
    return result.returncode


def run_mapping(paths: dict[str, Path], model: str) -> bool:
    cmd = [
        "uv", "run", "policy-map",
        "--policy",  str(paths["policy"]),
        "--openapi", str(paths["tools"]),
        "--output",  str(paths["mappings"]),
        "--model",   model,
    ]
    return _run(cmd, f"policy-map  [{model}]") == 0


def run_eval(paths: dict[str, Path], model: str, high_only: bool) -> bool:
    suffix = "high" if high_only else "all"
    out    = paths["eval_high"] if high_only else paths["eval_all"]
    cmd = [
        "uv", "run", "policy-map-eval",
        "--predicted",    str(paths["mappings"]),
        "--ground-truth", str(paths["ground_truth"]),
        "--output",       str(out),
    ]
    if high_only:
        cmd.append("--confidence-high-only")
    return _run(cmd, f"policy-map-eval [{model}] confidence={suffix}") == 0


def _read_overall(eval_path: Path) -> dict | None:
    try:
        return json.loads(eval_path.read_text()).get("overall")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full policy-tool mapper pipeline for a domain."
    )
    parser.add_argument(
        "--domain", required=True,
        help="Domain name (e.g. airline, telecom). Drives all file paths.",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="One or more LLM model IDs to run (e.g. gpt-4.1-mini gpt-5.1).",
    )
    parser.add_argument(
        "--skip-mapping", action="store_true",
        help="Skip policy-map step; only (re-)run eval on existing mapping files.",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Path to write a JSON summary of all results. "
             "Defaults to output/{domain}-pipeline-summary.json.",
    )
    args = parser.parse_args()

    domain = args.domain
    models = args.models
    summary_path = Path(args.summary) if args.summary else \
        HERE / "output" / f"{domain.lower()}-pipeline-summary.json"

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    sample_paths = _paths(domain, models[0])
    missing = []
    for key in ("policy", "tools", "ground_truth"):
        if not sample_paths[key].exists():
            missing.append(str(sample_paths[key]))
    if missing:
        print("ERROR: Missing input files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print(f"Pipeline: domain={domain}  models={models}")
    print(f"  Policy:       {sample_paths['policy']}")
    print(f"  Tools:        {sample_paths['tools']}")
    print(f"  Ground truth: {sample_paths['ground_truth']}")

    # ── Run ────────────────────────────────────────────────────────────────────
    results = []
    for model in models:
        paths = _paths(domain, model)
        print(f"\n{'─'*60}")
        print(f"Model: {model}")
        print(f"{'─'*60}")

        mapping_ok = True
        if not args.skip_mapping:
            mapping_ok = run_mapping(paths, model)
        else:
            if paths["mappings"].exists():
                print(f"  [skip-mapping] Using existing {paths['mappings'].name}")
            else:
                print(f"  ERROR: mapping file not found: {paths['mappings']}", file=sys.stderr)
                mapping_ok = False

        if not mapping_ok:
            results.append({"model": model, "status": "mapping_failed"})
            continue

        eval_high_ok = run_eval(paths, model, high_only=True)
        eval_all_ok  = run_eval(paths, model, high_only=False)

        results.append({
            "model":        model,
            "status":       "ok" if (eval_high_ok and eval_all_ok) else "partial",
            "mappings":     str(paths["mappings"]),
            "eval_high":    str(paths["eval_high"]),
            "eval_all":     str(paths["eval_all"]),
            "scores_high":  _read_overall(paths["eval_high"]),
            "scores_all":   _read_overall(paths["eval_all"]),
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {
        "domain":    domain,
        "models":    models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results":   results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    W = 25 + 1 + 6 + 1 + 5 + 1 + 7 + 1 + 7 + 1 + 7 + 1 + 7 + 1 + 7 + 1 + 7
    print(f"\n{'═'*W}")
    print(f"Summary")
    print(f"{'═'*W}")
    print(f"{'Model':<25} {'Conf':<6} "
          f"{'Macro P':>7} {'Macro R':>7} {'Macro F1':>8} "
          f"{'Micro P':>7} {'Micro R':>7} {'Micro F1':>8}")
    print(f"{'-'*25} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*8}")
    for r in results:
        if r.get("status") == "mapping_failed":
            print(f"{r['model']:<25} MAPPING FAILED")
            continue
        for label, key in [("high", "scores_high"), ("all", "scores_all")]:
            s = r.get(key)
            if not s:
                continue
            macro = s.get("macro", {})
            micro = s.get("micro", {})
            print(
                f"{r['model']:<25} {label:<6} "
                f"{macro.get('precision',0):>6.1%} "
                f"{macro.get('recall',   0):>6.1%} "
                f"{macro.get('f1',       0):>7.1%} "
                f"{micro.get('precision',0):>6.1%} "
                f"{micro.get('recall',   0):>6.1%} "
                f"{micro.get('f1',       0):>7.1%}"
            )
    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
