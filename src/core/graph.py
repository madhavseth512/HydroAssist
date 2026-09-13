"""LangGraph workflow orchestration for HydroAssist."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from src.core.state import AgentState, create_initial_state
from src.core.config import AppConfig
from src.agents.manager import ManagerAgent
from src.agents.consultant import ConsultantAgent
from src.agents.calculator import CalculatorAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_hydroassist_graph(config: AppConfig):
    """
    Build the LangGraph state machine for HydroAssist.

    Flow:
    START → manager → {consultant, calculator, END (clarification/unknown)}
    consultant → END (with loop option via manager for new queries)
    calculator → END

    The manager routes based on classified intent:
    - "consultation" → consultant (RAG-based answers)
    - "calculation" → calculator (Phase 1: stub, Phase 2: actual calculations)
    - "clarification" → END (manager already asked clarifying question)
    - "unknown" → END (polite decline)

    Args:
        config: Application configuration

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building LangGraph workflow...")

    # Initialize agents
    manager = ManagerAgent(config)
    consultant = ConsultantAgent(config)
    calculator = CalculatorAgent(config)

    # Create workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("manager", manager.run)
    workflow.add_node("consultant", consultant.run)
    workflow.add_node("calculator", calculator.run)

    # Define routing logic
    def route_intent(state: AgentState) -> Literal["consultant", "calculator", "__end__"]:
        """
        Route based on classified user intent.

        Args:
            state: Current agent state

        Returns:
            Next node name or END
        """
        intent = state.get("user_intent", "unknown")

        logger.info(f"Routing decision: intent={intent}")

        if intent == "consultation":
            return "consultant"
        elif intent == "calculation":
            return "calculator"
        else:
            # For clarification/unknown, manager already responded
            return END

    # Define edges
    workflow.set_entry_point("manager")

    # Conditional edges from manager
    workflow.add_conditional_edges(
        "manager",
        route_intent,
        {
            "consultant": "consultant",
            "calculator": "calculator",
            END: END
        }
    )

    # Both consultant and calculator end the turn
    workflow.add_edge("consultant", END)
    workflow.add_edge("calculator", END)

    # Compile workflow
    graph = workflow.compile()

    # Plain ASCII: this runs for every new session, and a Windows console
    # cannot encode "✓", which raised a logging error each time.
    logger.info("LangGraph workflow compiled successfully")

    return graph


def run_conversation_turn(
    user_message: str,
    graph,
    current_state: AgentState
) -> AgentState:
    """
    Run one turn of conversation.

    Args:
        user_message: User's message
        graph: Compiled LangGraph workflow
        current_state: Current conversation state

    Returns:
        Updated state after processing
    """
    logger.info(f"Processing user message: '{user_message[:100]}...'")

    # Add user message to state
    current_state["messages"].append(HumanMessage(content=user_message))

    # Run graph
    try:
        result = graph.invoke(current_state)
        logger.info("Conversation turn completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error processing conversation turn: {e}")
        raise


class HydroAssistChat:
    """
    Conversational interface for HydroAssist.

    Manages state across multiple conversation turns.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize chat interface.

        Args:
            config: Application configuration
        """
        self.config = config
        self.graph = create_hydroassist_graph(config)
        self.state = create_initial_state()
        self.logger = setup_logger(self.__class__.__name__)

        self.logger.info("HydroAssist chat initialized")

    def send_message(self, message: str) -> str:
        """
        Send a message and get response.

        Args:
            message: User message

        Returns:
            Assistant response
        """
        self.logger.info(f"User: {message}")

        # Run conversation turn
        self.state = run_conversation_turn(message, self.graph, self.state)

        # Get last assistant message
        assistant_messages = [msg for msg in self.state["messages"]
                             if msg.__class__.__name__ == "AIMessage"]

        if assistant_messages:
            response = assistant_messages[-1].content
            self.logger.info(f"Assistant: {response[:100]}...")
            return response
        else:
            return "No response generated."

    def reset(self):
        """Reset conversation state."""
        self.state = create_initial_state()
        self.logger.info("Conversation reset")

    def get_context(self) -> dict:
        """
        Get current conversation context.

        Returns:
            Dictionary with context information
        """
        return {
            "intent": self.state.get("user_intent", "unknown"),
            "aquifer": self.state.get("aquifer_context", "unknown"),
            "method": self.state.get("selected_method", ""),
            "suggestion_confirmed": self.state.get("suggestion_confirmed", False),
            "method_suggestion": self.state.get("method_suggestion", {}),
            "has_result": bool(self.state.get("calculation_result")),
            "message_count": len(self.state.get("messages", [])),
            "retrieved_docs_count": len(self.state.get("retrieved_docs", []))
        }

    # ── Calculator workspace bridge ───────────────────────────────────────────
    # The workspace runs outside the graph (it needs a file upload and form
    # inputs), so these are the two seams through which it exchanges state with
    # the conversation. Keeping them here means the UI never reaches into the
    # AgentState dict directly.

    def confirm_method(self, method: str) -> None:
        """Record that the user accepted (or overrode) the suggested method.

        Setting `suggestion_confirmed` is what unlocks aquifer-filtered
        retrieval in the Consultant — until a method is settled, filtering by
        aquifer wrongly excludes relevant cross-aquifer theory.
        """
        self.state["selected_method"] = method
        self.state["suggestion_confirmed"] = True
        self.logger.info(f"Method confirmed by user: {method}")

    def record_calculation(self, result: dict, calc_input: dict) -> None:
        """Publish a completed workspace analysis into the conversation state.

        `result` must be the plain-dict form of CalculationResult — numpy
        arrays are dropped by the caller so the state stays serialisable.
        """
        self.state["calculation_result"] = result or {}
        self.state["calculation_input"] = calc_input or {}
        self.logger.info(
            f"Recorded calculation: {result.get('method', 'unknown')} "
            f"success={result.get('success')}"
        )

    def stage_series(self, time_s, drawdown_m, Q=None) -> None:
        """Publish the uploaded series so method selection can read the data.

        This is what lets the Manager recommend from the Bourdet derivative — the
        actual shape of the user's curve — rather than from keywords in their
        message. Arrays are kept as-is; they are never sent to the LLM, only to
        the deterministic diagnosis.
        """
        self.state["staged_series"] = {
            "time_s": time_s, "drawdown_m": drawdown_m, "Q": Q,
        }
        self.logger.info(f"Staged {len(time_s)} data points for diagnosis")

    def clear_staged_series(self) -> None:
        self.state["staged_series"] = {}

    def clear_calculation(self) -> None:
        """Drop any recorded result (e.g. after a new CSV is uploaded)."""
        self.state["calculation_result"] = {}
        self.state["calculation_input"] = {}
