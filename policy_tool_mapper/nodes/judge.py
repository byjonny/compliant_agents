"""
Judge node — mode: retrieval.

For each (tool, candidate_sentence) pair from the retriever, asks the LLM:
  "Is this sentence directly relevant to governing/constraining this tool?"

This is the precision gate: the retriever maximises recall (generous threshold),
the judge kills false positives with an LLM relevance decision.

Output is a list[ToolMapping] identical in structure to the mapper node output,
so the sweeper node works unchanged in both pipeline modes.
"""

import asyncio
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from policy_tool_mapper.state import (
    MappedStatement,
    PipelineState,
    PolicyStatement,
    RetrievalCandidate,
    ToolMapping,
    ToolProfile,
)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge.md"
_CONCURRENCY = 20   # each call is small (one sentence), so high concurrency is fine


class JudgeOutput(BaseModel):
    relevant:      bool
    confidence:    Literal["high", "medium"]
    justification: str


async def _judge_pair(
    llm:           BaseChatModel,
    system_prompt: str,
    profile:       ToolProfile,
    statement:     PolicyStatement,
) -> JudgeOutput:
    structured = llm.with_structured_output(JudgeOutput)
    result: JudgeOutput = await structured.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"TOOL:\n"
                    f"ID: {profile.tool_id}\n"
                    f"Name: {profile.name}\n"
                    f"Description: {profile.description}\n"
                    f"Parameters: {', '.join(profile.parameters) or 'none'}\n"
                    f"Semantic profile: {profile.semantic_profile}\n\n"
                    f"POLICY SENTENCE:\n"
                    f"[{statement.id}] {statement.text}"
                )
            ),
        ]
    )
    return result


async def judge_node(state: PipelineState, config: RunnableConfig) -> dict:
    llm:           BaseChatModel          = config["configurable"]["llm"]
    system_prompt: str                    = _PROMPT_PATH.read_text()
    candidates:    list[RetrievalCandidate] = state.get("retrieval_candidates", [])
    profiles_map:  dict[str, ToolProfile] = {p.tool_id: p for p in state["tool_profiles"]}
    stmts_map:     dict[str, PolicyStatement] = {s.id: s for s in state["policy_statements"]}

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(profile: ToolProfile, stmt: PolicyStatement):
        async with semaphore:
            out = await _judge_pair(llm, system_prompt, profile, stmt)
            return profile.tool_id, stmt.id, out.relevant, out.confidence

    # Build all (tool, statement) tasks
    tasks = [
        _bounded(profiles_map[cand.tool_id], stmts_map[sid])
        for cand in candidates
        for sid in cand.statement_ids
        if cand.tool_id in profiles_map and sid in stmts_map
    ]

    print(f"[judge] Verifying {len(tasks)} (tool, sentence) pairs ...")
    results = await asyncio.gather(*tasks)

    # Aggregate into ToolMapping (same schema as mapper output)
    by_tool: dict[str, list[MappedStatement]] = {c.tool_id: [] for c in candidates}
    for tool_id, stmt_id, relevant, confidence in results:
        if relevant:
            by_tool[tool_id].append(MappedStatement(id=stmt_id, confidence=confidence))

    mappings = [
        ToolMapping(tool_id=tid, statements=stmts)
        for tid, stmts in by_tool.items()
    ]

    total = sum(len(m.statements) for m in mappings)
    under = sum(1 for m in mappings if len(m.statements) < 2)
    print(f"[judge] {total} relevant mappings across {len(mappings)} tools ({under} under-mapped)")
    return {"mappings": mappings}
