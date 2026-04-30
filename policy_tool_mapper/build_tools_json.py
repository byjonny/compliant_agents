"""
Extract tools from a tau2 tools.py file and write tools.json to policy_tool_mapper/input/.

Usage (from inside tau2-bench/):
    python policy_tool_mapper/build_tools_json.py --tools-file src/tau2/domains/airline/tools.py
    python policy_tool_mapper/build_tools_json.py --tools-file src/tau2/domains/airline/tools.py --output policy_tool_mapper/input/tools.json
"""

import argparse
import functools
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Make tau2 importable when running directly from inside tau2-bench/
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DEFAULT = Path(__file__).parent / "input" / "tools.json"


def _make_free_function(unbound_method: Callable) -> Callable:
    """
    Return a wrapper of an unbound method with 'self' removed from its signature.
    The wrapper is never called — it exists only so as_tool() can read the
    signature and docstring without needing a class instance.
    """
    sig = inspect.signature(unbound_method)
    params_without_self = [
        p for name, p in sig.parameters.items() if name != "self"
    ]

    @functools.wraps(unbound_method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        pass

    wrapper.__signature__ = sig.replace(parameters=params_without_self)  # type: ignore
    return wrapper


def extract_tools(tools_file: Path) -> list[dict]:
    """
    Dynamically import a tau2 tools.py, find all @is_tool-decorated methods,
    and return a list of OpenAI-style function schemas.
    """
    from tau2.environment.toolkit import TOOL_ATTR, ToolKitBase

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("_tools_module", tools_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {tools_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    # Find all ToolKitBase subclasses defined in this module
    toolkit_classes = [
        obj
        for obj in module.__dict__.values()
        if isinstance(obj, type)
        and issubclass(obj, ToolKitBase)
        and obj is not ToolKitBase
        and obj.__module__ == module.__name__
    ]

    if not toolkit_classes:
        raise ValueError(
            f"No ToolKitBase subclasses found in {tools_file}.\n"
            "Make sure the class inherits from tau2.environment.toolkit.ToolKitBase."
        )

    from tau2.environment.tool import as_tool

    seen: set[str] = set()
    schemas: list[dict] = []

    for cls in toolkit_classes:
        # Collect all @is_tool methods from this class and its bases
        for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not getattr(obj, TOOL_ATTR, False):
                continue
            if name in seen:
                continue
            seen.add(name)

            free_func = _make_free_function(obj)
            tool = as_tool(free_func)
            schemas.append(tool.openai_schema)
            print(f"  + {name}")

    return schemas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tool schemas from a tau2 tools.py and write tools.json"
    )
    parser.add_argument(
        "--tools-file",
        required=True,
        help="Path to the tau2 tools.py file (e.g. src/tau2/domains/airline/tools.py)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Output path for tools.json (default: {OUTPUT_DEFAULT})",
    )
    args = parser.parse_args()

    tools_file = Path(args.tools_file).resolve()
    if not tools_file.exists():
        print(f"ERROR: File not found: {tools_file}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting tools from: {tools_file}")
    schemas = extract_tools(tools_file)

    output_path.write_text(json.dumps(schemas, indent=2))
    print(f"\nWrote {len(schemas)} tools -> {output_path}")


if __name__ == "__main__":
    main()
