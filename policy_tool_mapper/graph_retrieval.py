"""
LangGraph pipeline — mode: retrieval.

chunker → profiler → retriever → judge → sweeper → END

The retriever runs BM25 + bi-encoder + cross-encoder reranking to surface
candidate (tool, sentence) pairs. The judge uses an LLM to verify each pair.
The sweeper catches under-mapped tools exactly as in the default pipeline.
"""

from langgraph.graph import END, START, StateGraph

from policy_tool_mapper.nodes.chunker import chunker_node
from policy_tool_mapper.nodes.judge import judge_node
from policy_tool_mapper.nodes.profiler import profiler_node
from policy_tool_mapper.nodes.retriever import retriever_node
from policy_tool_mapper.nodes.sweeper import sweeper_node
from policy_tool_mapper.state import PipelineState


def _should_resweep(state: PipelineState) -> str:
    """One resweep pass is enough after the thorough retrieval stage."""
    if state["sweep_iterations"] >= 1:
        return "end"
    mapper_ids = {
        (m.tool_id, sid)
        for m in state["mappings"]
        for sid in m.statement_ids
    }
    final_ids = {
        (m.tool_id, sid)
        for m in state["final_mappings"]
        for sid in m.statement_ids
    }
    return "resweep" if (final_ids - mapper_ids) else "end"


workflow = StateGraph(PipelineState)

workflow.add_node("chunker",   chunker_node)
workflow.add_node("profiler",  profiler_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("judge",     judge_node)
workflow.add_node("sweeper",   sweeper_node)

workflow.add_edge(START,       "chunker")
workflow.add_edge("chunker",   "profiler")
workflow.add_edge("profiler",  "retriever")
workflow.add_edge("retriever", "judge")
workflow.add_edge("judge",     "sweeper")

workflow.add_conditional_edges(
    "sweeper",
    _should_resweep,
    {"resweep": "judge", "end": END},
)

app_retrieval = workflow.compile()
