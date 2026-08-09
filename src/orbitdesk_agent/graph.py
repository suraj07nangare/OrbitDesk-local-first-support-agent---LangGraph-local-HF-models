from functools import partial

from langgraph.graph import END, StateGraph

from . import nodes, routing
from .models import GenerationModel
from .retrieval import EmbeddingModel, RetrievalIndex
from .state import AgentState


def build_graph(embedding_model: EmbeddingModel, retrieval_index: RetrievalIndex, generation_model: GenerationModel):
    graph = StateGraph(AgentState)

    graph.add_node("triage", nodes.triage_node)
    graph.add_node("retrieve", partial(nodes.retrieval_node, retrieval_index=retrieval_index))
    graph.add_node("generate", partial(nodes.generation_node, generation_model=generation_model))
    graph.add_node("verify", partial(nodes.verification_node, embedding_model=embedding_model))
    graph.add_node("clarify", nodes.clarify_node)
    graph.add_node("safe_response", nodes.safe_response_node)
    graph.add_node("safe_failure", nodes.safe_failure_node)
    graph.add_node("finalize", nodes.finalize_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        routing.route_after_triage,
        {"retrieve": "retrieve", "safe_response": "safe_response"},
    )
    graph.add_conditional_edges(
        "retrieve",
        routing.route_after_retrieval,
        {"generate": "generate", "clarify": "clarify", "safe_response": "safe_response"},
    )
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        routing.route_after_verification,
        {"regenerate": "generate", "finalize": "finalize", "safe_failure": "safe_failure"},
    )
    graph.add_edge("clarify", "finalize")
    graph.add_edge("safe_response", "finalize")
    graph.add_edge("safe_failure", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
