from langgraph.graph import END, START, StateGraph

from policy_tool_mapper.nodes.chunker import chunker_node
from policy_tool_mapper.nodes.mapper import mapper_node
from policy_tool_mapper.nodes.profiler import profiler_node
from policy_tool_mapper.nodes.sweeper import sweeper_node
from policy_tool_mapper.state import PipelineState


def should_resweep(state: PipelineState) -> str:
    """
    Decide whether to loop back to the mapper for a verification pass.

    Exits immediately if we have already done 2 sweeps.
    Loops back if the sweeper found new mappings that the mapper originally missed.
    """
    if state["sweep_iterations"] >= 2:
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
    if final_ids - mapper_ids:
        return "resweep"
    return "end"


workflow = StateGraph(PipelineState)

workflow.add_node("chunker", chunker_node)
workflow.add_node("profiler", profiler_node)
workflow.add_node("mapper", mapper_node)
workflow.add_node("sweeper", sweeper_node)

workflow.add_edge(START, "chunker")
workflow.add_edge("chunker", "profiler")
workflow.add_edge("profiler", "mapper")
workflow.add_edge("mapper", "sweeper")

workflow.add_conditional_edges(
    "sweeper",
    should_resweep,
    {"resweep": "mapper", "end": END},
)

app = workflow.compile()
