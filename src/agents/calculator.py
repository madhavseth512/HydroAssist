"""Calculator Agent - Phase 2 stub for pumping test calculations."""

from langchain_core.messages import AIMessage

from src.agents.base import BaseAgent
from src.core.state import AgentState
from src.prompts.calculator_prompts import get_calculator_stub_response


class CalculatorAgent(BaseAgent):
    """
    Calculator agent for pumping test analysis.

    Phase 1: Stub that acknowledges calculation requests and explains Phase 2 roadmap
    Phase 2: Will implement actual curve-fitting and parameter estimation
    """

    def run(self, state: AgentState) -> AgentState:
        """
        Phase 1 stub - acknowledge request and explain Phase 2.

        Args:
            state: Current conversation state

        Returns:
            Updated state with stub response
        """
        self._log_execution(state, "calculator_stub")

        # Get context from state
        aquifer_context = state.get("aquifer_context", "unknown")
        selected_method = state.get("selected_method", "")

        # Generate stub response
        response = get_calculator_stub_response(aquifer_context, selected_method)

        # Update state
        state["calculator_ready"] = True  # Architecture ready for Phase 2
        state["messages"].append(AIMessage(content=response))

        self.logger.info("Returned Phase 2 stub response")

        return state
