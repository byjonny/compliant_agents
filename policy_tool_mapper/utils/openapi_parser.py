import json

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head"})
_MAX_SPEC_CHARS = 2000  # per-tool spec truncation to avoid excessive token use


def parse_openapi(spec: dict | list) -> list[dict]:
    """
    Parse an OpenAPI spec into a flat list of tool dicts.

    Supports:
      - Standard OpenAPI 3.x / Swagger 2.x        (has "paths")
      - Tool list: {"tools": [...]}                (LangChain / agent framework export)
      - OpenAI function array: [{"type":"function","function":{...}}, ...]
      - Flat function array:   [{"name": ...}, ...]
      - Single function definition                  (has "name" + "description")
    """
    if isinstance(spec, list):
        return [_parse_function(_unwrap_openai(f)) for f in spec if isinstance(f, dict)]

    if "tools" in spec and isinstance(spec["tools"], list):
        return [_parse_function(_unwrap_openai(t)) for t in spec["tools"] if isinstance(t, dict)]

    if "paths" in spec:
        return _parse_openapi_paths(spec)

    if "name" in spec and "description" in spec:
        return [_parse_function(spec)]

    return []


def _unwrap_openai(tool: dict) -> dict:
    """Unwrap OpenAI-style {"type": "function", "function": {...}} into the inner dict."""
    if tool.get("type") == "function" and "function" in tool:
        return tool["function"]
    return tool


def _parse_openapi_paths(spec: dict) -> list[dict]:
    tools = []
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue

            tool_id = (
                operation.get("operationId")
                or f"{method.upper()}_{path.replace('/', '_').strip('_')}"
            )
            name = operation.get("summary") or tool_id
            description = operation.get("description") or operation.get("summary") or ""
            params = _collect_path_params(operation)

            tools.append({
                "tool_id": tool_id,
                "name": name,
                "description": description,
                "parameters": params,
                "raw_spec": _truncate_spec({
                    "method": method.upper(),
                    "path": path,
                    "operation": operation,
                }),
            })
    return tools


def _parse_function(func: dict) -> dict:
    """Parse a function-style tool definition (LangChain, OpenAI function calling, MCP, etc.)."""
    name = func.get("name") or func.get("operationId") or "unknown"
    description = func.get("description") or ""

    # Support various parameter schema formats
    params_schema = func.get("parameters") or func.get("inputSchema") or func.get("input_schema") or {}
    props = {}
    if isinstance(params_schema, dict):
        props = params_schema.get("properties", {})

    return {
        "tool_id": name,
        "name": func.get("summary") or name,
        "description": description,
        "parameters": list(props.keys()),
        "raw_spec": _truncate_spec(func),
    }


def _collect_path_params(operation: dict) -> list[str]:
    params: list[str] = []
    for param in operation.get("parameters", []):
        if isinstance(param, dict) and param.get("name"):
            params.append(param["name"])
    req_body = operation.get("requestBody", {})
    if isinstance(req_body, dict):
        for content_def in req_body.get("content", {}).values():
            if isinstance(content_def, dict):
                schema = content_def.get("schema", {})
                if isinstance(schema, dict):
                    params.extend(schema.get("properties", {}).keys())
    return params


def _truncate_spec(spec: dict) -> dict:
    serialized = json.dumps(spec)
    if len(serialized) <= _MAX_SPEC_CHARS:
        return spec
    # Keep only the most informative fields
    return {
        k: spec[k]
        for k in ("name", "operationId", "summary", "description", "parameters", "inputSchema", "method", "path")
        if k in spec
    }
