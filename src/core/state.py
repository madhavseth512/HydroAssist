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
        selected_method: The method the workspace will run — moved only by the
            user accepting a suggestion or running a fit
        discussed_method: The method the current message is about. Informational:
            a question about a method is not an instruction to switch to it
        retrieved_docs: Document chunks from RAG retrieval — this turn's only
        sources_by_message: Each answer's own documents, keyed by its position in
            `messages`. One shared list belonged to the latest answer and was
            wrong under every earlier one
        metadata_filters: Active filters for retrieval (only used during calculation)
        method_suggestion: Structured suggestion card {"method", "reason", "confidence"}
        suggestion_confirmed: True once user accepts or overrides the method suggestion
        calculation_input: Merged CSV data + well parameters from the calculator workspace
        calculation_result: Computed T, S, goodness-of-fit and plot data
        staged_series: Uploaded time/drawdown arrays, so the Manager can read the
            flow regime off the data rather than guessing from the message
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_intent: Literal["consultation", "calculation", "how_to",
                         "clarification", "unknown"]
    aquifer_context: Literal["confined", "unconfined", "leaky", "fractured", "unknown"]
    selected_method: str
    discussed_method: str
    retrieved_docs: List[Dict]
    sources_by_message: Dict
    metadata_filters: Dict
    method_suggestion: Dict
    suggestion_confirmed: bool
    calculation_input: Dict
    calculation_result: Dict
    staged_series: Dict
    # What the guide offered to do next, which the browser renders as a button.
    # Never acted on here: nothing moves until the user clicks.
    proposed_actions: List[Dict]


def create_initial_state() -> AgentState:
    """Create initial empty state for new conversation."""
    return AgentState(
        messages=[],
        user_intent="unknown",
        aquifer_context="unknown",
        selected_method="",
        discussed_method="",
        sources_by_message={},
        retrieved_docs=[],
        metadata_filters={},
        method_suggestion={},
        suggestion_confirmed=False,
        calculation_input={},
        calculation_result={},
        staged_series={},
        proposed_actions=[]
    )
