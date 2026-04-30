from datetime import datetime, timezone

from policy_tool_mapper.state import PipelineState, ToolMapping, PolicyStatement


def format_output(
    state: PipelineState,
    policy_file: str = "",
    openapi_file: str = "",
) -> dict:
    """Format the pipeline result as the final n:m mapping JSON."""
    final_mappings: list[ToolMapping] = state["final_mappings"] or state["mappings"]
    policy_statements: list[PolicyStatement] = state["policy_statements"]
    tool_profiles = {p.tool_id: p for p in state["tool_profiles"]}

    stmt_index: dict[str, PolicyStatement] = {s.id: s for s in policy_statements}

    # Build reverse index: statement_id -> list[tool_id]
    statement_to_tools: dict[str, list[str]] = {}
    for mapping in final_mappings:
        for ms in mapping.statements:
            statement_to_tools.setdefault(ms.id, []).append(mapping.tool_id)

    total_mappings = sum(len(m.statements) for m in final_mappings)

    mappings_out = []
    for mapping in sorted(final_mappings, key=lambda m: m.tool_id):
        profile = tool_profiles.get(mapping.tool_id)
        statements_out = [
            {
                "id": ms.id,
                "text": stmt_index[ms.id].text if ms.id in stmt_index else ms.id,
                "section": stmt_index[ms.id].section if ms.id in stmt_index else None,
                "confidence": ms.confidence,
            }
            for ms in sorted(mapping.statements, key=lambda s: s.id)
            if ms.id in stmt_index
        ]
        mappings_out.append({
            "tool_id": mapping.tool_id,
            "tool_name": profile.name if profile else mapping.tool_id,
            "statements": statements_out,
        })

    statement_index_out = {
        sid: {
            "text": stmt.text,
            "section": stmt.section,
            "mapped_to_tools": sorted(statement_to_tools.get(sid, [])),
        }
        for sid, stmt in sorted(stmt_index.items())
    }

    return {
        "metadata": {
            "policy_file": policy_file,
            "openapi_file": openapi_file,
            "total_tools": len(final_mappings),
            "total_statements": len(policy_statements),
            "total_mappings": total_mappings,
            "sweep_iterations": state.get("sweep_iterations", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "mappings": mappings_out,
        "statement_index": statement_index_out,
    }
