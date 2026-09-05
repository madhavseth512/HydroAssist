"""Calculator Agent — orchestrates the conversation side of an analysis.

The numerical work happens in src/calculator/, driven by the Calculator
Workspace tab. This agent is the chat-side counterpart: it reports on a
completed analysis, or tells the user exactly what the workspace still needs.

It deliberately does not run the fit itself. Fitting needs a CSV, a pumping
rate and a well distance, which are collected by the workspace UI; duplicating
that here would mean two code paths that could disagree about the same result.
"""

from langchain_core.messages import AIMessage

from src.agents.base import BaseAgent
from src.agents.method_selection import (
    format_suggestion, summarise_result, suggest_method,
)
from src.core.state import AgentState


class CalculatorAgent(BaseAgent):
    """
    Calculator agent for pumping test analysis.

    Three cases, in priority order:
      1. A result already exists in state → summarise it and invite follow-ups.
      2. Data is staged but not yet fitted → say what is still missing.
      3. Nothing staged → give the method suggestion and the required inputs.
    """

    def run(self, state: AgentState) -> AgentState:
        self._log_execution(state, "calculator")

        result = state.get("calculation_result") or {}
        calc_input = state.get("calculation_input") or {}

        if result:
            response = self._report_result(result, calc_input)
            self.logger.info("Reported existing calculation result")
        else:
            response = self._request_inputs(state, calc_input)
            self.logger.info("Requested calculator inputs")

        state["messages"].append(AIMessage(content=response))
        return state

    # ── Case 1: a completed analysis ──────────────────────────────────────────

    def _report_result(self, result: dict, calc_input: dict) -> str:
        summary = summarise_result(result, calc_input)

        if not result.get("success", False):
            return (
                f"{summary}\n\n"
                "Things worth checking: that Q is the constant pumping rate in "
                "the unit you selected, that r is the distance to the "
                "observation well, and that recovery-phase rows are not mixed "
                "into the pumping data. Adjust the inputs in the Calculator "
                "Workspace and run again."
            )

        return (
            f"{summary}\n\n"
            "Ask me anything about this result — whether the fit is "
            "trustworthy, what the validity notes mean, how it compares with "
            "another method, or what the storativity implies about the aquifer."
        )

    # ── Cases 2 and 3: not yet fitted ─────────────────────────────────────────

    def _request_inputs(self, state: AgentState, calc_input: dict) -> str:
        suggestion = state.get("method_suggestion") or suggest_method(
            aquifer_context=state.get("aquifer_context", "unknown"),
            selected_method=state.get("selected_method", ""),
        )

        if not suggestion.get("supported", True):
            return (
                f"{format_suggestion(suggestion)}\n\n"
                "If your test is in a confined aquifer after all, tell me and "
                "I will set the workspace up for it."
            )

        missing = self._missing_inputs(calc_input)
        header = format_suggestion(suggestion)

        if not missing:
            return (
                f"{header}\n\n"
                "Everything needed is staged in the **Calculator Workspace** "
                "tab — press **Run Analysis** there and I will interpret the "
                "result here."
            )

        return (
            f"{header}\n\n"
            "To run this, the **Calculator Workspace** tab still needs:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nOnce the analysis has run, come back here and I will "
              "interpret the numbers."
        )

    @staticmethod
    def _missing_inputs(calc_input: dict) -> list:
        """Which workspace inputs are still outstanding."""
        missing = []
        if not calc_input.get("n_points"):
            missing.append(
                "a **CSV** of time and drawdown (Step 1) — time in s/min/hr, "
                "drawdown in m/ft; column units are detected automatically"
            )
        if not calc_input.get("Q"):
            missing.append("the **pumping rate Q**, constant through the test (Step 2)")
        if not calc_input.get("r"):
            missing.append(
                "the **distance r** from the pumping well to the observation "
                "well (Step 2)"
            )
        return missing
