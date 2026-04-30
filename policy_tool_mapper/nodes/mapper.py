import asyncio
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from policy_tool_mapper.schemas import MapperOutput
from policy_tool_mapper.state import MappedStatement, PipelineState, PolicyStatement, ToolMapping, ToolProfile

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "mapper.md"
_CONCURRENCY = 10


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def _format_statements(statements: list[PolicyStatement]) -> str:
    lines = []
    for stmt in statements:
        section = f" [{stmt.section}]" if stmt.section else ""
        lines.append(f"{stmt.id}{section}: {stmt.text}")
    return "\n".join(lines)


async def _map_tool(
    llm: BaseChatModel,
    system_prompt: str,
    profile: ToolProfile,
    statements_text: str,
) -> ToolMapping:
    structured = llm.with_structured_output(MapperOutput)
    result: MapperOutput = await structured.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"TOOL:\n"
                    f"ID: {profile.tool_id}\n"
                    f"Name: {profile.name}\n"
                    f"Description: {profile.description}\n"
                    f"Parameters: {', '.join(profile.parameters) or 'none'}\n"
                    f"Semantic Profile:\n{profile.semantic_profile}\n\n"
                    f"POLICY STATEMENTS:\n{statements_text}"
                )
            ),
        ]
    )
    return ToolMapping(
        tool_id=profile.tool_id,
        statements=[
            MappedStatement(id=s.id, confidence=s.confidence)
            for s in result.statements
        ],
    )


async def mapper_node(state: PipelineState, config: RunnableConfig) -> dict:
    llm: BaseChatModel = config["configurable"]["llm"]
    system_prompt = _load_prompt()
    statements_text = _format_statements(state["policy_statements"])

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(profile: ToolProfile) -> ToolMapping:
        async with semaphore:
            return await _map_tool(llm, system_prompt, profile, statements_text)

    mappings = await asyncio.gather(*[_bounded(p) for p in state["tool_profiles"]])

    total = sum(len(m.statements) for m in mappings)
    under = sum(1 for m in mappings if len(m.statements) < 2)
    print(f"[mapper] {total} total mappings across {len(mappings)} tools ({under} under-mapped)")
    return {"mappings": list(mappings)}
