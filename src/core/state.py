"""State management for HydroAssist LangGraph workflow."""

from typing import TypedDict, Annotated, Literal, List, Dict, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state across all graph nodes.

    Fields:
        messages: Full conversation history (managed by LangGraph)
        user_intent: Classified intent from Manager
        aquifer_context: Aquifer type extracted from conversation
        selected_method: Hydrogeological method extracted from user message by Manager
        retrieved_docs: Document chunks from RAG retrieval
        metadata_filters: Active filters for retrieval (only used during calculation)
        method_suggestion: Structured suggestion card {"method", "reason", "confidence"}
        suggestion_confirmed: True once user accepts or overrides the method suggestion
        calculation_input: Merged CSV data + well parameters from the calculator workspace
        calculation_result: Computed T, S, goodness-of-fit and plot data
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_intent: Literal["consultation", "calculation", "clarification", "unknown"]
    aquifer_context: Literal["confined", "unconfined", "leaky", "fractured", "unknown"]
    selected_method: str
    retrieved_docs: List[Dict]
    metadata_filters: Dict
    method_suggestion: Dict
    suggestion_confirmed: bool
    calculation_input: Dict
    calculation_result: Dict


def create_initial_state() -> AgentState:
    """Create initial empty state for new conversation."""
    return AgentState(
        messages=[],
        user_intent="unknown",
        aquifer_context="unknown",
        selected_method="",
        retrieved_docs=[],
        metadata_filters={},
        method_suggestion={},
        suggestion_confirmed=False,
        calculation_input={},
        calculation_result={}
    )
