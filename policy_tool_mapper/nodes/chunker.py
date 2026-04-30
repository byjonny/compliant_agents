import asyncio
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from policy_tool_mapper.schemas import ChunkerOutput, RawStatement
from policy_tool_mapper.state import PipelineState, PolicyStatement

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "chunker.md"
_HEADING_RE = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split the policy text into (heading_label, section_content) pairs.
    The heading line itself is not included in section_content but is
    passed as a label so the LLM can set the 'section' field correctly.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        label = match.group().lstrip("#").strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        if content:
            sections.append((label, content))

    return sections


def _validate_verbatim(
    statements: list[RawStatement], original_text: str
) -> list[RawStatement]:
    """
    Drop statements whose text cannot be found verbatim (modulo whitespace)
    in the original policy. Warns on failures so the caller is aware.
    """
    normalized_orig = _normalize(original_text)
    valid: list[RawStatement] = []
    dropped = 0
    for stmt in statements:
        normalized_stmt = _normalize(stmt.text)
        if normalized_stmt and normalized_stmt in normalized_orig:
            valid.append(stmt)
        else:
            dropped += 1
            preview = stmt.text[:70].replace("\n", " ")
            print(f"[chunker] WARNING non-verbatim statement dropped: «{preview}…»")
    if dropped:
        print(f"[chunker] {dropped}/{len(statements)} statements dropped (non-verbatim)")
    return valid


async def _chunk_section(
    llm: BaseChatModel,
    system_prompt: str,
    heading: str,
    content: str,
) -> list[RawStatement]:
    structured = llm.with_structured_output(ChunkerOutput)
    label = f"Section: {heading}\n\n" if heading else ""
    result: ChunkerOutput = await structured.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Split the following policy text into self-contained statements.\n\n"
                    f"{label}{content}"
                )
            ),
        ]
    )
    # Attach the section heading to each returned statement if not already set
    for stmt in result.statements:
        if stmt.section is None and heading:
            stmt.section = heading
    return result.statements


async def chunker_node(state: PipelineState, config: RunnableConfig) -> dict:
    llm: BaseChatModel = config["configurable"]["llm"]
    system_prompt = _load_prompt()
    raw_text = state["raw_policy_text"]

    sections = _split_by_headings(raw_text)
    tasks = [
        _chunk_section(llm, system_prompt, heading, content)
        for heading, content in sections
        if content.strip()
    ]
    section_results = await asyncio.gather(*tasks)

    all_raw: list[RawStatement] = [stmt for batch in section_results for stmt in batch]
    valid = _validate_verbatim(all_raw, raw_text)

    statements = [
        PolicyStatement(
            id=f"PS-{i + 1:03d}",
            text=stmt.text,
            section=stmt.section,
        )
        for i, stmt in enumerate(valid)
    ]

    print(f"[chunker] Extracted {len(statements)} policy statements")
    return {"policy_statements": statements}
