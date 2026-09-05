"""Integration tests for the Consultation ↔ Calculator Workspace bridge.

Before this work the two halves of the app were disconnected: the workspace ran
the calculators directly and never wrote back to AgentState, while the Manager
never populated `method_suggestion` and `suggestion_confirmed` stayed False
forever — which in turn meant the Consultant's aquifer metadata filter could
never activate.

These tests drive the real agents with a stubbed LLM and a stubbed retriever, so
they exercise the actual wiring without network access. The numerical results
come from the real calculators run on the real bundled sample CSV.
"""

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.calculator import CalculatorAgent
from src.agents.manager import ManagerAgent
from src.agents.method_selection import (
    THEIS, COOPER_JACOB, PAPADOPULOS_COOPER,
)
from src.calculator.base import CalculationInput
from src.calculator.theis import TheisCalculator
from src.core.config import load_config
from src.core.state import create_initial_state


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubLLM:
    """Returns a scripted classification payload; records every prompt."""

    def __init__(self, payload=None, text=None):
        self.payload = payload
        self.text = text
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        body = self.text if self.text is not None else json.dumps(self.payload)
        return type("Resp", (), {"content": body})()


class _StubRetriever:
    """Stands in for the ChromaDB-backed Retriever."""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            {"content": "Theis (1935) assumes a confined aquifer of infinite extent.",
             "metadata": {"source": "Theis-1935", "aquifer": "confined",
                          "method": "Theis_1935", "page": 520},
             "score": 0.81},
        ]
        self.calls = []

    def retrieve(self, query, top_k=None, metadata_filters=None):
        self.calls.append({"query": query, "top_k": top_k,
                           "metadata_filters": metadata_filters})
        return self.docs


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    return load_config()


def _manager(config, monkeypatch, payload):
    """A ManagerAgent whose Gemini client is a stub."""
    llm = _StubLLM(payload)
    monkeypatch.setattr("src.agents.manager.ChatGoogleGenerativeAI",
                        lambda **kw: llm)
    agent = ManagerAgent(config)
    return agent, llm


def _consultant(config, monkeypatch, retriever=None, answer="An answer."):
    """A ConsultantAgent with both the LLM and the retriever stubbed."""
    from src.agents import consultant as consultant_mod

    llm = _StubLLM(text=answer)
    retriever = retriever or _StubRetriever()
    monkeypatch.setattr(consultant_mod, "ChatGoogleGenerativeAI", lambda **kw: llm)
    monkeypatch.setattr(consultant_mod, "Retriever", lambda cfg: retriever)
    agent = consultant_mod.ConsultantAgent(config)
    return agent, llm, retriever


def _state_with(message, **over):
    state = create_initial_state()
    state["messages"].append(HumanMessage(content=message))
    state.update(over)
    return state


def _classification(intent="calculation", aquifer="confined", method=None):
    return {"intent": intent, "confidence": 0.9, "reasoning": "test",
            "extracted_aquifer": aquifer, "extracted_method": method}


# ── Chat → suggestion ─────────────────────────────────────────────────────────

class TestManagerProducesSuggestion:

    def test_calculation_intent_populates_method_suggestion(self, config, monkeypatch):
        """Regression: method_suggestion was declared in AgentState but never written."""
        agent, _ = _manager(config, monkeypatch, _classification())
        out = agent.run(_state_with("analyse my confined aquifer pumping test"))

        assert out["method_suggestion"], "Manager must populate the suggestion card"
        assert out["method_suggestion"]["method"] == THEIS
        assert out["method_suggestion"]["reason"]

    def test_large_diameter_well_suggests_papadopulos_cooper(self, config, monkeypatch):
        agent, _ = _manager(config, monkeypatch, _classification())
        out = agent.run(_state_with(
            "analyse the test on my large-diameter dug well, radius 0.6 m"))

        assert out["method_suggestion"]["method"] == PAPADOPULOS_COOPER
        assert out["selected_method"] == PAPADOPULOS_COOPER

    def test_late_time_phrasing_suggests_cooper_jacob(self, config, monkeypatch):
        agent, _ = _manager(config, monkeypatch, _classification())
        out = agent.run(_state_with(
            "fit the late-time straight line on my confined aquifer data"))

        assert out["method_suggestion"]["method"] == COOPER_JACOB

    def test_extracted_method_is_normalised_to_canonical_label(self, config, monkeypatch):
        """The dropdown matches on exact labels, so 'theis' must become 'Theis (1935)'."""
        agent, _ = _manager(config, monkeypatch,
                            _classification(method="theis"))
        out = agent.run(_state_with("use theis please"))

        assert out["selected_method"] == THEIS

    def test_unconfined_now_routes_to_its_own_method(self, config, monkeypatch):
        """This asserted the opposite until Boulton (1963) was implemented. The
        Manager reads registry.supported_aquifers(), so it corrected itself
        without an edit."""
        agent, _ = _manager(config, monkeypatch,
                            _classification(aquifer="unconfined"))
        out = agent.run(_state_with("analyse my unconfined aquifer test"))

        suggestion = out["method_suggestion"]
        assert suggestion["supported"] is True
        assert suggestion["method"] == "Boulton (1963)"

    def test_consultation_about_a_method_still_primes_the_workspace(self, config,
                                                                     monkeypatch):
        """Users often settle on a method while still asking theory questions."""
        agent, _ = _manager(config, monkeypatch,
                            _classification(intent="consultation",
                                            method="Cooper-Jacob"))
        out = agent.run(_state_with("when is Cooper-Jacob valid?"))

        assert out["selected_method"] == COOPER_JACOB
        assert out["method_suggestion"]["method"] == COOPER_JACOB

    def test_offtopic_turn_does_not_invent_a_suggestion(self, config, monkeypatch):
        agent, _ = _manager(config, monkeypatch,
                            _classification(intent="unknown", aquifer="unknown"))
        out = agent.run(_state_with("what is the weather"))

        assert not out["method_suggestion"]

    def test_confirmed_suggestion_is_not_overwritten(self, config, monkeypatch):
        """Changing the method under the user on a later turn would be surprising."""
        agent, _ = _manager(config, monkeypatch, _classification())
        state = _state_with("now tell me about wellbore storage in a dug well",
                            selected_method=THEIS,
                            suggestion_confirmed=True,
                            method_suggestion={"method": THEIS, "reason": "chosen",
                                               "confidence": 0.9, "supported": True})
        out = agent.run(state)

        assert out["method_suggestion"]["method"] == THEIS
        assert out["selected_method"] == THEIS


# ── Calculator agent ──────────────────────────────────────────────────────────

class TestCalculatorAgentRespondsInChat:

    def test_no_data_lists_what_the_workspace_needs(self, config):
        out = CalculatorAgent(config).run(
            _state_with("calculate T and S", user_intent="calculation",
                        aquifer_context="confined"))
        reply = out["messages"][-1].content

        assert "CSV" in reply
        assert "pumping rate" in reply.lower()
        assert "distance" in reply.lower()

    def test_is_no_longer_the_phase_2_stub(self, config):
        """Regression: this agent used to answer 'Coming in Phase 2'."""
        out = CalculatorAgent(config).run(
            _state_with("analyse my data", user_intent="calculation"))
        reply = out["messages"][-1].content

        assert "Phase 2" not in reply
        assert "Coming in" not in reply

    def test_partial_inputs_only_lists_the_missing_ones(self, config):
        out = CalculatorAgent(config).run(_state_with(
            "run it", user_intent="calculation",
            calculation_input={"n_points": 60, "Q": 0.0058}))
        reply = out["messages"][-1].content

        assert "CSV" not in reply
        assert "distance" in reply.lower()

    def test_complete_inputs_point_at_the_run_button(self, config):
        out = CalculatorAgent(config).run(_state_with(
            "run it", user_intent="calculation",
            calculation_input={"n_points": 60, "Q": 0.0058, "r": 50.0}))
        reply = out["messages"][-1].content

        assert "Run Analysis" in reply

    def test_existing_result_is_summarised_with_real_numbers(self, config):
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        result = TheisCalculator().calculate(
            CalculationInput(df=df, Q=500 / 86400, r=50.0))

        out = CalculatorAgent(config).run(_state_with(
            "what did we get?", user_intent="calculation",
            calculation_result=result.to_state_dict(),
            calculation_input={"Q": 500 / 86400, "r": 50.0,
                               "n_points": len(df)}))
        reply = out["messages"][-1].content

        assert f"{result.T_day:.2f}" in reply
        assert "Theis (1935)" in reply
        assert "R²" in reply

    def test_failed_result_offers_diagnosis(self, config):
        out = CalculatorAgent(config).run(_state_with(
            "why did it fail?", user_intent="calculation",
            calculation_result={"method": THEIS, "success": False,
                                "error_message": "Curve fitting failed"}))
        reply = out["messages"][-1].content

        assert "Curve fitting failed" in reply
        assert "recovery" in reply.lower()

    def test_unsupported_aquifer_explains_the_gap(self, config):
        out = CalculatorAgent(config).run(_state_with(
            "analyse it", user_intent="calculation",
            method_suggestion={"method": "", "reason": "No calculator for leaky",
                               "confidence": 0.9, "supported": False}))
        reply = out["messages"][-1].content

        assert "No calculator for leaky" in reply


# ── Workspace → chat ──────────────────────────────────────────────────────────

class TestResultSerialisation:

    def test_to_state_dict_is_json_serialisable(self):
        """AgentState must stay serialisable — no numpy arrays."""
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        result = TheisCalculator().calculate(
            CalculationInput(df=df, Q=500 / 86400, r=50.0))

        payload = result.to_state_dict()
        json.dumps(payload)  # must not raise

        assert "time_s" not in payload
        assert "drawdown_obs" not in payload

    def test_to_state_dict_preserves_the_reported_values(self):
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        result = TheisCalculator().calculate(
            CalculationInput(df=df, Q=500 / 86400, r=50.0))
        payload = result.to_state_dict()

        assert payload["T"] == pytest.approx(result.T)
        assert payload["S"] == pytest.approx(result.S)
        assert payload["T_day"] == pytest.approx(result.T_day)
        assert payload["r_squared"] == pytest.approx(result.r_squared)
        assert payload["validity_notes"] == result.validity_notes
        assert payload["success"] is True

    def test_failed_result_serialises(self):
        from src.calculator.base import CalculationResult
        payload = CalculationResult(method="Theis (1935)", T=0.0, S=0.0,
                                    success=False,
                                    error_message="nope").to_state_dict()
        json.dumps(payload)

        assert payload["success"] is False
        assert payload["error_message"] == "nope"


# ── Consultant sees the results ───────────────────────────────────────────────

class TestConsultantUsesCalculationContext:

    def _result_state(self, question="Is this fit trustworthy?"):
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        result = TheisCalculator().calculate(
            CalculationInput(df=df, Q=500 / 86400, r=50.0))
        return _state_with(
            question,
            calculation_result=result.to_state_dict(),
            calculation_input={"Q": 500 / 86400, "r": 50.0, "n_points": len(df)},
        ), result

    def test_prompt_carries_the_computed_parameters(self, config, monkeypatch):
        agent, llm, _ = _consultant(config, monkeypatch)
        state, result = self._result_state()
        agent.run(state)

        prompt = llm.prompts[-1]
        assert "ALREADY RUN" in prompt
        assert f"{result.T_day:.2f}" in prompt
        assert "Validity notes" in prompt

    def test_prompt_has_no_analysis_block_before_any_run(self, config, monkeypatch):
        agent, llm, _ = _consultant(config, monkeypatch)
        agent.run(_state_with("What is the Theis method?"))

        assert "ALREADY RUN" not in llm.prompts[-1]

    def test_failed_run_is_surfaced_to_the_consultant(self, config, monkeypatch):
        agent, llm, _ = _consultant(config, monkeypatch)
        agent.run(_state_with(
            "why did my fit fail?",
            calculation_result={"method": THEIS, "success": False,
                                "error_message": "singular matrix"}))

        prompt = llm.prompts[-1]
        assert "FAILED" in prompt
        assert "singular matrix" in prompt

    def test_result_alone_is_enough_context_when_retrieval_is_empty(self, config,
                                                                    monkeypatch):
        """A user asking about their own numbers must get an answer, not a
        'no documents found' apology."""
        agent, llm, _ = _consultant(config, monkeypatch,
                                     retriever=_StubRetriever(docs=[]))
        state, _ = self._result_state()
        out = agent.run(state)

        reply = out["messages"][-1].content
        assert "don't have enough information" not in reply
        assert llm.prompts, "the LLM should still have been consulted"

    def test_no_docs_and_no_result_still_declines_gracefully(self, config,
                                                              monkeypatch):
        agent, llm, _ = _consultant(config, monkeypatch,
                                     retriever=_StubRetriever(docs=[]))
        out = agent.run(_state_with("What is the Boussinesq equation?"))

        assert "don't have enough information" in out["messages"][-1].content
        assert llm.prompts == []


class TestAquiferFilterActivation:
    """Regression: `suggestion_confirmed` was never set, so the aquifer
    metadata filter in the Consultant could never activate."""

    def test_no_filter_before_the_method_is_confirmed(self, config, monkeypatch):
        agent, _, retriever = _consultant(config, monkeypatch)
        agent.run(_state_with("Explain storativity", aquifer_context="confined"))

        assert retriever.calls[-1]["metadata_filters"] is None

    def test_filter_applied_once_confirmed(self, config, monkeypatch):
        agent, _, retriever = _consultant(config, monkeypatch)
        agent.run(_state_with("Explain storativity",
                              aquifer_context="confined",
                              suggestion_confirmed=True))

        assert retriever.calls[-1]["metadata_filters"] == {"aquifer": "confined"}

    def test_unknown_aquifer_never_filters(self, config, monkeypatch):
        agent, _, retriever = _consultant(config, monkeypatch)
        agent.run(_state_with("Explain storativity",
                              aquifer_context="unknown",
                              suggestion_confirmed=True))

        assert retriever.calls[-1]["metadata_filters"] is None
