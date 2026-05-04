import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from the policy_tool_mapper directory, then from the tau2-bench root
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

from policy_tool_mapper.evaluator import DEFAULT_THRESHOLD, evaluate, print_summary
from policy_tool_mapper.graph import app
from policy_tool_mapper.utils.model import create_llm
from policy_tool_mapper.utils.output_formatter import format_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map policy statements to API tools using a LangGraph pipeline."
    )
    parser.add_argument(
        "--policy", required=True,
        help="Path to the policy document (.md or .txt)",
    )
    parser.add_argument(
        "--openapi", required=True,
        help="Path to the OpenAPI spec or tool list (.yaml, .yml, or .json)",
    )
    parser.add_argument(
        "--output", default="mappings.json",
        help="Output file path (default: mappings.json)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="LLM model identifier (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate output against a ground-truth file after mapping",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground-truth mappings JSON (required when --evaluate is set)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Text overlap threshold for matching statements (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    if args.evaluate and not args.ground_truth:
        print("ERROR: --ground-truth is required when --evaluate is set", file=sys.stderr)
        sys.exit(1)

    # Load policy document
    try:
        policy_text = open(args.policy).read()
    except FileNotFoundError:
        print(f"ERROR: Policy file not found: {args.policy}", file=sys.stderr)
        sys.exit(1)

    # Load OpenAPI / tool spec
    try:
        with open(args.openapi) as f:
            if args.openapi.endswith((".yaml", ".yml")):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: OpenAPI spec not found: {args.openapi}", file=sys.stderr)
        sys.exit(1)

    # Create LLM
    try:
        llm = create_llm(args.model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    initial_state = {
        "raw_policy_text": policy_text,
        "raw_openapi_spec": spec,
        "policy_statements": [],
        "tool_profiles": [],
        "mappings": [],
        "final_mappings": [],
        "sweep_iterations": 0,
    }

    print(f"Policy-Tool Mapper")
    print(f"  Policy:   {args.policy}")
    print(f"  OpenAPI:  {args.openapi}")
    print(f"  Model:    {args.model}")
    print()

    result = asyncio.run(
        app.ainvoke(
            initial_state,
            config={"configurable": {"llm": llm}},
        )
    )

    output = format_output(result, policy_file=args.policy, openapi_file=args.openapi)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    meta = output["metadata"]
    print()
    print(f"Done.")
    print(f"  Policy statements : {meta['total_statements']}")
    print(f"  Tools profiled    : {meta['total_tools']}")
    print(f"  Total mappings    : {meta['total_mappings']}")
    print(f"  Sweep iterations  : {meta['sweep_iterations']}")
    print(f"  Output            : {args.output}")

    # ------------------------------------------------------------------
    # Optional evaluation
    # ------------------------------------------------------------------
    if args.evaluate:
        try:
            ground_truth = json.loads(Path(args.ground_truth).read_text())
        except FileNotFoundError:
            print(f"\nERROR: Ground-truth file not found: {args.ground_truth}", file=sys.stderr)
            sys.exit(1)

        summary = evaluate(ground_truth, output, threshold=args.threshold)
        print_summary(summary)

        # Save eval results alongside the mappings output
        eval_path = Path(args.output).with_suffix(".eval.json")
        eval_json = {
            "threshold": summary.threshold,
            "macro": {
                "precision": summary.macro_precision,
                "recall": summary.macro_recall,
                "f1": summary.macro_f1,
            },
            "micro": {
                "precision": summary.micro_precision,
                "recall": summary.micro_recall,
                "f1": summary.micro_f1,
            },
            "per_tool": [
                {
                    "tool_id": r.tool_id,
                    "tool_name": r.tool_name,
                    "gt_count": r.gt_count,
                    "pred_count": r.pred_count,
                    "matched_gt": r.matched_gt,
                    "matched_pred": r.matched_pred,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                }
                for r in summary.tool_results
            ],
        }
        eval_path.write_text(json.dumps(eval_json, indent=2))
        print(f"  Eval results      : {eval_path}")


if __name__ == "__main__":
    main()
