import asyncio
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from policy_tool_mapper.schemas import ToolProfileOutput
from policy_tool_mapper.state import PipelineState, ToolProfile
from policy_tool_mapper.utils.openapi_parser import parse_openapi

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "profiler.md"
_CONCURRENCY = 10


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


async def _profile_tool(
    llm: BaseChatModel,
    system_prompt: str,
    tool_dict: dict,
) -> ToolProfile:
    structured = llm.with_structured_output(ToolProfileOutput)
    spec_str = json.dumps(tool_dict["raw_spec"], indent=2)
    result: ToolProfileOutput = await structured.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Tool ID: {tool_dict['tool_id']}\n"
                    f"Name: {tool_dict['name']}\n"
                    f"Description: {tool_dict['description']}\n"
                    f"Parameters: {', '.join(tool_dict['parameters']) or 'none'}\n\n"
                    f"Full spec:\n{spec_str}"
                )
            ),
        ]
    )
    return ToolProfile(
        tool_id=tool_dict["tool_id"],
        name=tool_dict["name"],
        description=tool_dict["description"],
        semantic_profile=result.semantic_profile,
        parameters=tool_dict["parameters"],
    )


async def profiler_node(state: PipelineState, config: RunnableConfig) -> dict:
    llm: BaseChatModel = config["configurable"]["llm"]
    system_prompt = _load_prompt()

    tools = parse_openapi(state["raw_openapi_spec"])
    if not tools:
        print("[profiler] WARNING: No tools found in OpenAPI spec")
        return {"tool_profiles": []}

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(tool_dict: dict) -> ToolProfile:
        async with semaphore:
            return await _profile_tool(llm, system_prompt, tool_dict)

    profiles = await asyncio.gather(*[_bounded(t) for t in tools])
    print(f"[profiler] Profiled {len(profiles)} tools")
    return {"tool_profiles": list(profiles)}
