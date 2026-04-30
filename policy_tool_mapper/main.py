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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
