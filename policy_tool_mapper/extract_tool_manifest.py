"""
Extract tool schemas from a tau2 tools.py file.

Writes the full OpenAI function schema to the output file so the result can be
passed directly to `policy-map --openapi` (the profiler node reads it via
parse_openapi, which requires name / description / parameters).

The terminal shows a compact name + description summary.

Usage (from inside tau2-bench/):
    python policy_tool_mapper/extract_tool_manifest.py --tools-file src/tau2/domains/telecom/tools.py
    python policy_tool_mapper/extract_tool_manifest.py --tools-file src/tau2/domains/telecom/tools.py --output policy_tool_mapper/input/telecomTools.json
"""

import argparse
import json
import sys
from pathlib import Path

# Make tau2 importable when running directly from inside tau2-bench/
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DEFAULT = Path(__file__).parent / "input" / "tools.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tool schemas from a tau2 tools.py (profiler-compatible output)"
    )
    parser.add_argument(
        "--tools-file",
        required=True,
        help="Path to the tau2 tools.py (e.g. src/tau2/domains/telecom/tools.py)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Output path for tools JSON (default: {OUTPUT_DEFAULT})",
    )
    args = parser.parse_args()

    tools_file = Path(args.tools_file).resolve()
    if not tools_file.exists():
        print(f"ERROR: File not found: {tools_file}", file=sys.stderr)
        sys.exit(1)

    # build_tools_json.py lives in the same directory; import relative to it
    sys.path.insert(0, str(Path(__file__).parent))
    from build_tools_json import extract_tools

    print(f"Extracting tools from: {tools_file}")
    # Full OpenAI function schemas — compatible with parse_openapi() / profiler
    schemas = extract_tools(tools_file)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schemas, indent=2))

    # Terminal: compact name + first description line
    print(f"\n{'Tool':<45} Description")
    print("-" * 100)
    for s in schemas:
        fn   = s["function"]
        name = fn["name"]
        desc = fn.get("description", "").split("\n")[0].strip()[:54]
        print(f"  {name:<43} {desc}")
    print(f"\nWrote {len(schemas)} tools → {output_path}")
    print(f"(Pass this file to: policy-map --openapi {output_path})")


if __name__ == "__main__":
    main()
