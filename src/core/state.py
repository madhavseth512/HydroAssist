"""State management for HydroAssist LangGraph workflow."""

from typing import TypedDict, Annotated, Literal, List, Dict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state across all graph nodes.

    Fields:
        messages: Full conversation history (managed by LangGraph)
        user_intent: Classified intent from Manager
        aquifer_context: Aquifer type extracted from conversation
        selected_method: Hydrogeological method identified by Consultant
        calculator_ready: Flag indicating Phase 2 readiness
        retrieved_docs: Document chunks from RAG retrieval
        metadata_filters: Active filters for retrieval
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_intent: Literal["consultation", "calculation", "clarification", "unknown"]
    aquifer_context: Literal["confined", "unconfined", "leaky", "fractured", "unknown"]
    selected_method: str  # e.g., "Theis (1935)", "Cooper-Jacob (1946)"
    calculator_ready: bool
    retrieved_docs: List[Dict]  # {content, metadata, score}
    metadata_filters: Dict  # For retrieval: {"aquifer": "confined", "type": "theory"}


def create_initial_state() -> AgentState:
    """Create initial empty state for new conversation."""
    return AgentState(
        messages=[],
        user_intent="unknown",
        aquifer_context="unknown",
        selected_method="",
        calculator_ready=False,
        retrieved_docs=[],
        metadata_filters={}
    )
