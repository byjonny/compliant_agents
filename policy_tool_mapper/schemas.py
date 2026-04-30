from typing import Literal
from pydantic import BaseModel, Field


class RawStatement(BaseModel):
    text: str = Field(description="Verbatim text from the policy document")
    section: str | None = Field(None, description="Nearest section heading, or null")


class ChunkerOutput(BaseModel):
    statements: list[RawStatement] = Field(
        description="Self-contained policy statements extracted from the document"
    )


class ToolProfileOutput(BaseModel):
    semantic_profile: str = Field(
        description=(
            "Rich semantic description covering: what the tool does, what data it "
            "reads/writes/deletes, side effects, compliance risks, and policy categories likely to apply."
        )
    )


class LLMMappedStatement(BaseModel):
    id: str = Field(description="Policy statement ID, e.g. PS-001")
    confidence: Literal["high", "medium"] = Field(
        description="'high' for direct, unambiguous relevance; 'medium' for indirect or borderline relevance"
    )


class MapperOutput(BaseModel):
    tool_id: str = Field(description="Tool identifier")
    statements: list[LLMMappedStatement] = Field(
        description="All policy statement IDs that apply to this tool, with confidence"
    )


class SweeperOutput(BaseModel):
    additional_statements: list[LLMMappedStatement] = Field(
        description="Additional policy statement IDs not already in the current mapping"
    )
