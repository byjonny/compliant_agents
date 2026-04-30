import asyncio
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from policy_tool_mapper.schemas import SweeperOutput
from policy_tool_mapper.state import MappedStatement, PipelineState, PolicyStatement, ToolMapping, ToolProfile

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "sweeper.md"
_UNDER_MAPPED_THRESHOLD = 2
_CONCURRENCY = 10


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def _format_statements(statements: list[PolicyStatement]) -> str:
    lines = []
    for stmt in statements:
        section = f" [{stmt.section}]" if stmt.section else ""
        lines.append(f"{stmt.id}{section}: {stmt.text}")
    return "\n".join(lines)


async def _sweep_tool(
    llm: BaseChatModel,
    system_prompt: str,
    profile: ToolProfile,
    current_mapping: ToolMapping,
    statements_text: str,
) -> SweeperOutput:
    structured = llm.with_structured_output(SweeperOutput)
    current_ids = ", ".join(current_mapping.statement_ids) or "none"
    result: SweeperOutput = await structured.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"TOOL UNDER REVIEW:\n"
                    f"ID: {profile.tool_id}\n"
                    f"Name: {profile.name}\n"
                    f"Description: {profile.description}\n"
                    f"Parameters: {', '.join(profile.parameters) or 'none'}\n"
                    f"Semantic Profile:\n{profile.semantic_profile}\n\n"
                    f"Currently mapped statement IDs: {current_ids}\n\n"
                    f"ALL POLICY STATEMENTS:\n{statements_text}"
                )
            ),
        ]
    )
    return result


async def sweeper_node(state: PipelineState, config: RunnableConfig) -> dict:
    llm: BaseChatModel = config["configurable"]["llm"]
    system_prompt = _load_prompt()

    # Use final_mappings from a previous sweep if available; otherwise start from mapper output
    base_mappings: list[ToolMapping] = state["final_mappings"] if state["final_mappings"] else state["mappings"]
    base_by_tool: dict[str, ToolMapping] = {m.tool_id: m for m in base_mappings}

    # Ensure every profiled tool has an entry (even if mapper returned nothing for it)
    for profile in state["tool_profiles"]:
        if profile.tool_id not in base_by_tool:
            base_by_tool[profile.tool_id] = ToolMapping(tool_id=profile.tool_id, statements=[])

    under_mapped = [
        profile
        for profile in state["tool_profiles"]
        if len(base_by_tool[profile.tool_id].statements) < _UNDER_MAPPED_THRESHOLD
    ]

    if not under_mapped:
        print(f"[sweeper] No under-mapped tools found — passing through")
        return {
            "final_mappings": list(base_by_tool.values()),
            "sweep_iterations": state["sweep_iterations"] + 1,
        }

    print(f"[sweeper] Running adversarial sweep for {len(under_mapped)} under-mapped tools")
    statements_text = _format_statements(state["policy_statements"])
    valid_ids = {s.id for s in state["policy_statements"]}

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(profile: ToolProfile) -> tuple[ToolProfile, SweeperOutput]:
        async with semaphore:
            result = await _sweep_tool(
                llm, system_prompt, profile, base_by_tool[profile.tool_id], statements_text
            )
            return profile, result

    sweep_results = await asyncio.gather(*[_bounded(p) for p in under_mapped])

    # Merge: add new valid findings to existing mappings (deduplicate by statement ID)
    merged = dict(base_by_tool)
    new_total = 0
    for profile, sweep_out in sweep_results:
        existing = merged[profile.tool_id]
        existing_ids = {s.id for s in existing.statements}
        new_stmts = [
            MappedStatement(id=s.id, confidence=s.confidence)
            for s in sweep_out.additional_statements
            if s.id not in existing_ids and s.id in valid_ids
        ]
        new_total += len(new_stmts)
        merged[profile.tool_id] = ToolMapping(
            tool_id=profile.tool_id,
            statements=existing.statements + new_stmts,
        )

    print(f"[sweeper] Added {new_total} new mappings from sweep (iteration {state['sweep_iterations'] + 1})")
    return {
        "final_mappings": list(merged.values()),
        "sweep_iterations": state["sweep_iterations"] + 1,
    }
