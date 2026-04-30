from typing import TypedDict
from pydantic import BaseModel


class PolicyStatement(BaseModel):
    id: str
    text: str
    section: str | None = None


class ToolProfile(BaseModel):
    tool_id: str
    name: str
    description: str
    semantic_profile: str
    parameters: list[str]


class MappedStatement(BaseModel):
    id: str
    confidence: str  # "high" | "medium"


class ToolMapping(BaseModel):
    tool_id: str
    statements: list[MappedStatement] = []

    @property
    def statement_ids(self) -> list[str]:
        return [s.id for s in self.statements]


class PipelineState(TypedDict):
    # Inputs
    raw_policy_text: str
    raw_openapi_spec: dict

    # After chunker
    policy_statements: list[PolicyStatement]

    # After profiler
    tool_profiles: list[ToolProfile]

    # After mapper
    mappings: list[ToolMapping]

    # After sweeper
    final_mappings: list[ToolMapping]
    sweep_iterations: int
