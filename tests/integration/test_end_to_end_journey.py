"""End-to-end journey through the compiled LangGraph.

This is the test the whole bridge exists for: a user arrives with a question,
is steered to a method, runs a fit in the workspace, and then interrogates the
result in chat — with state carried across every hop.

The LLM and the vector store are stubbed; the calculators, the graph, the
routing and the state reducers are all real.
"""

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.method_selection import THEIS, PAPADOPULOS_COOPER
from src.calculator.base import CalculationInput
from src.calculator.theis import TheisCalculator
from src.core.config import load_config
from src.core.graph import HydroAssistChat


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """Replays a queue of responses; falls back to the last one when exhausted."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self._i = 0
        # Set to an exception to make every call fail, which is how a spent
        # quota or an unreachable model behaves.
        self.fail_with = None

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail_with is not None:
            raise self.fail_with
        if self._i < len(self.responses):
            body = self.responses[self._i]
            self._i += 1
        else:
            body = self.responses[-1]
        if isinstance(body, dict):
            body = json.dumps(body)
        return type("Resp", (), {"content": body})()


class _StubRetriever:
    def __init__(self):
        self.calls = []
        # Settable, so a test can have a second answer read a different paper —
        # which is how "each answer keeps its own sources" is checked at all.
        self.docs = [{
            "content": "Cooper & Jacob (1946): the approximation is tolerable "
                       "where u is less than about 0.02.",
            "metadata": {"source": "Cooper-Jacob-1946", "aquifer": "confined",
                         "method": "Cooper_Jacob_1946", "page": 3},
            "score": 0.78,
        }]

    def retrieve(self, query, top_k=None, metadata_filters=None):
        self.calls.append(metadata_filters)
        return list(self.docs)


def _classification(intent, aquifer="confined", method=None):
    return {"intent": intent, "confidence": 0.9, "reasoning": "t",
            "extracted_aquifer": aquifer, "extracted_method": method}


class _StubHelpRetriever:
    """The help index, without needing one built."""

    def __init__(self, *args, **kwargs):
        self.available = True
        self.calls = []

    def retrieve(self, query, top_k=4):
        self.calls.append(query)
        return [{
            "content": "Setting up the test — The fields. The pumping rate Q "
                       "goes in Step 2, with its unit.",
            "metadata": {"source": "HydroAssist help", "type": "help",
                         "document": "Setting up the test",
                         "section": "The fields",
                         "file": "setting-up-the-test.md"},
            "score": 0.51,
        }]


@pytest.fixture
def wired(monkeypatch):
    """A HydroAssistChat with stubbed LLM + retriever, plus handles to both."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")

    manager_llm = _ScriptedLLM([_classification("calculation")])
    consultant_llm = _ScriptedLLM(["Here is my interpretation of your fit."])
    guide_llm = _ScriptedLLM(["The pumping rate Q goes in Step 2."])
    retriever = _StubRetriever()
    help_retriever = _StubHelpRetriever()

    monkeypatch.setattr("src.agents.guide.ChatGoogleGenerativeAI",
                        lambda **kw: guide_llm)
    monkeypatch.setattr("src.agents.guide.HelpRetriever",
                        lambda cfg: help_retriever)

    monkeypatch.setattr("src.agents.manager.ChatGoogleGenerativeAI",
                        lambda **kw: manager_llm)
    monkeypatch.setattr("src.agents.consultant.ChatGoogleGenerativeAI",
                        lambda **kw: consultant_llm)
    monkeypatch.setattr("src.agents.consultant.Retriever", lambda cfg: retriever)

    chat = HydroAssistChat(load_config())
    chat.guide_llm = guide_llm                 # for the how-to tests
    chat.help_retriever = help_retriever
    return chat, manager_llm, consultant_llm, retriever


def _run_workspace_fit(chat, method=THEIS):
    """Simulate exactly what the Calculator Workspace does after Run Analysis."""
    df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
    Q_si, r = 500 / 86400, 50.0
    result = TheisCalculator().calculate(CalculationInput(df=df, Q=Q_si, r=r))

    chat.record_calculation(
        result.to_state_dict(),
        {"Q": Q_si, "r": r, "method": method, "n_points": len(df)},
    )
    chat.confirm_method(method)
    return result


# ── Graph routing ─────────────────────────────────────────────────────────────

class TestGraphRouting:

    def test_calculation_intent_reaches_the_calculator_agent(self, wired):
        chat, _, _, _ = wired
        reply = chat.send_message("analyse my confined aquifer pumping test")

        assert "Calculator Workspace" in reply
        assert chat.state["user_intent"] == "calculation"

    def test_consultation_intent_reaches_the_consultant(self, wired, monkeypatch):
        chat, manager_llm, consultant_llm, _ = wired
        manager_llm.responses = [_classification("consultation")]
        manager_llm._i = 0

        reply = chat.send_message("what does storativity mean?")

        assert reply == "Here is my interpretation of your fit."
        assert consultant_llm.prompts

    def test_state_survives_across_turns(self, wired):
        chat, manager_llm, _, _ = wired
        chat.send_message("analyse my confined aquifer test")
        manager_llm.responses = [_classification("consultation")]
        manager_llm._i = 0
        chat.send_message("and what about storativity?")

        assert chat.state["aquifer_context"] == "confined"
        assert len(chat.state["messages"]) == 4  # 2 user + 2 assistant


# ── The full journey ──────────────────────────────────────────────────────────

class TestFullJourney:

    def test_chat_to_calculator_to_chat(self, wired):
        chat, manager_llm, consultant_llm, retriever = wired

        # 1. User asks for an analysis — the calculator agent asks for inputs
        first = chat.send_message(
            "I need to analyse a pumping test in a confined aquifer")

        assert "CSV" in first
        assert chat.state["method_suggestion"]["method"] == THEIS
        assert chat.state["suggestion_confirmed"] is False

        # 2. The workspace runs the fit and publishes the result
        result = _run_workspace_fit(chat)

        assert chat.state["suggestion_confirmed"] is True
        assert chat.state["calculation_result"]["success"] is True

        # 3. The user asks about the result — routed to the consultant, which
        #    now sees the actual numbers
        manager_llm.responses = [_classification("consultation")]
        manager_llm._i = 0
        chat.send_message("is this fit trustworthy?")

        prompt = consultant_llm.prompts[-1]
        assert "ALREADY RUN" in prompt
        assert f"{result.T_day:.2f}" in prompt

        # 4. And the aquifer filter is finally active, because a method is settled
        assert retriever.calls[-1] == {"aquifer": "confined"}

    def test_calculator_agent_reports_the_result_on_a_later_turn(self, wired):
        chat, manager_llm, _, _ = wired
        result = _run_workspace_fit(chat)

        manager_llm.responses = [_classification("calculation")]
        manager_llm._i = 0
        reply = chat.send_message("what were the numbers again?")

        assert f"{result.T_day:.2f}" in reply
        assert "Theis (1935)" in reply

    def test_geometry_in_the_message_steers_the_method(self, wired):
        chat, _, _, _ = wired
        chat.send_message(
            "I pumped a dug well of 1.2 m diameter and logged drawdown in the "
            "well itself — what should I use?")

        assert chat.state["method_suggestion"]["method"] == PAPADOPULOS_COOPER
        assert chat.state["selected_method"] == PAPADOPULOS_COOPER

    def test_confirm_method_is_what_unlocks_aquifer_filtering(self, wired):
        """Filtering by aquifer type excludes papers that cross types, so it
        waits until the user has committed to a method. A general question is
        therefore never narrowed by aquifer."""
        chat, manager_llm, _, retriever = wired
        manager_llm.responses = [_classification("consultation")]

        chat.send_message("what is transmissivity?")
        assert retriever.calls[-1] is None

        chat.confirm_method(THEIS)
        chat.send_message("and how is it measured?")
        assert retriever.calls[-1] == {"aquifer": "confined"}

    def test_naming_a_method_narrows_the_search_to_its_papers(self, wired):
        """Unlike the aquifer filter, this one applies from the first question:
        the user said which method they mean, and the corpus is tagged, so the
        search goes to those papers. A filtered search that finds nothing is
        retried unfiltered by the retriever."""
        chat, manager_llm, _, retriever = wired
        manager_llm.responses = [_classification("consultation")]

        chat.send_message("explain the Theis assumptions")

        clause = retriever.calls[-1]
        assert "aquifer" not in json.dumps(clause)
        assert "method_theis_1935" in json.dumps(clause)

    def test_both_filters_apply_once_a_method_is_confirmed(self, wired):
        chat, manager_llm, _, retriever = wired
        manager_llm.responses = [_classification("consultation")]

        chat.confirm_method(THEIS)
        chat.send_message("and the Cooper-Jacob ones?")

        clause = retriever.calls[-1]
        assert clause["$and"][0] == {"aquifer": "confined"}
        assert "method_cooper_jacob_1946" in json.dumps(clause["$and"][1])

    def test_new_upload_clears_the_stale_result(self, wired):
        chat, _, _, _ = wired
        _run_workspace_fit(chat)
        assert chat.get_context()["has_result"] is True

        chat.clear_calculation()

        assert chat.get_context()["has_result"] is False
        assert chat.state["calculation_result"] == {}

    def test_reset_clears_everything(self, wired):
        chat, _, _, _ = wired
        _run_workspace_fit(chat)
        chat.reset()

        ctx = chat.get_context()
        assert ctx["has_result"] is False
        assert ctx["suggestion_confirmed"] is False
        assert ctx["method"] == ""
        assert ctx["message_count"] == 0


# ── Context surface used by the sidebar ───────────────────────────────────────

class TestContextSurface:

    def test_context_exposes_everything_the_sidebar_renders(self, wired):
        chat, _, _, _ = wired
        ctx = chat.get_context()

        for key in ["intent", "aquifer", "method", "suggestion_confirmed",
                    "method_suggestion", "has_result", "message_count",
                    "retrieved_docs_count"]:
            assert key in ctx

    def test_has_result_tracks_the_recorded_calculation(self, wired):
        chat, _, _, _ = wired
        assert chat.get_context()["has_result"] is False

        _run_workspace_fit(chat)
        assert chat.get_context()["has_result"] is True

    def test_failed_run_still_counts_as_a_result_to_discuss(self, wired):
        chat, _, _, _ = wired
        chat.record_calculation(
            {"method": THEIS, "success": False, "error_message": "no fit"},
            {"Q": 0.0058, "r": 50.0, "n_points": 12})

        assert chat.get_context()["has_result"] is True


# ── Data-driven method suggestion through the graph ───────────────────────────

class TestDerivativeDrivenSuggestion:
    """The Manager reading the flow regime off the staged series.

    Before this, the suggestion came only from keywords in the user's sentence.
    Staging the uploaded curve lets the Bourdet derivative override that, which
    matters most when the two disagree — a user who says "confined aquifer test"
    but whose data plainly shows leakage.
    """

    def _leaky_series(self):
        import numpy as np
        from src.calculator.methods.hantush_jacob import hantush_jacob_drawdown

        t = np.logspace(2, 7, 60)
        s = hantush_jacob_drawdown(t, 2.5e-3, 5e-5, 300.0, 500 / 86400, 50.0)
        return t, s

    def _radial_series(self):
        import numpy as np
        from scipy.special import expn

        t = np.logspace(2, 6, 60)
        u = (50.0 ** 2 * 5e-5) / (4.0 * 2.5e-3 * t)
        s = (500 / 86400 / (4.0 * np.pi * 2.5e-3)) * expn(1, np.maximum(u, 1e-12))
        return t, s

    def test_staging_a_series_records_it_in_state(self, wired):
        chat, _, _, _ = wired
        t, s = self._radial_series()
        chat.stage_series(t, s, Q=500 / 86400)

        assert len(chat.state["staged_series"]["time_s"]) == 60

    def test_leaky_data_steers_the_suggestion_away_from_theis(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("calculation", aquifer="confined")]

        # No data staged: keyword rules give the confined default.
        chat.send_message("analyse my confined aquifer pumping test")
        assert chat.state["method_suggestion"]["method"] == THEIS

        # Same sentence, now with the curve staged.
        chat.reset()
        manager_llm._i = 0
        t, s = self._leaky_series()
        chat.stage_series(t, s, Q=500 / 86400)
        chat.send_message("analyse my confined aquifer pumping test")

        suggestion = chat.state["method_suggestion"]
        assert suggestion["method"] == "Hantush-Jacob (1955)"
        assert "evidence" in suggestion

    def test_radial_data_confirms_theis_with_evidence(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("calculation", aquifer="confined")]

        t, s = self._radial_series()
        chat.stage_series(t, s, Q=500 / 86400)
        chat.send_message("analyse my pumping test")

        suggestion = chat.state["method_suggestion"]
        assert suggestion["method"] == THEIS
        assert "radial flow" in suggestion["evidence"]

    def test_the_suggestion_quotes_a_transmissivity_from_the_plateau(self, wired):
        """An estimate that owes nothing to curve fitting, available before the
        user has run anything."""
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("calculation", aquifer="confined")]

        import numpy as np
        from scipy.special import expn
        t = np.logspace(3, 7, 60)
        u = (50.0 ** 2 * 5e-5) / (4.0 * 2.5e-3 * t)
        s = (500 / 86400 / (4.0 * np.pi * 2.5e-3)) * expn(1, np.maximum(u, 1e-12))

        chat.stage_series(t, s, Q=500 / 86400)
        chat.send_message("analyse my pumping test")

        assert "216.0 m²/day" in chat.state["method_suggestion"]["reason"]

    def test_clearing_the_series_restores_keyword_behaviour(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("calculation", aquifer="confined")]

        t, s = self._leaky_series()
        chat.stage_series(t, s, Q=500 / 86400)
        chat.clear_staged_series()
        chat.send_message("analyse my confined aquifer pumping test")

        assert chat.state["method_suggestion"]["method"] == THEIS

    def test_reset_clears_the_staged_series(self, wired):
        chat, _, _, _ = wired
        t, s = self._radial_series()
        chat.stage_series(t, s, Q=500 / 86400)
        chat.reset()

        assert chat.state["staged_series"] == {}


class TestDescribingATestWithoutNamingIt:
    """A user describes what they did; the workspace should be primed for it.

    Found by the consultant evaluation. Asked "the pump stopped after 24 hours
    and I kept reading water levels as they came back up", the consultant
    explained Theis recovery correctly and at length — but no suggestion card
    appeared, so the workspace stayed on whatever method was already selected.

    The rules were never consulted: the manager only attempted a suggestion when
    the intent was calculation, or a method was named, or an aquifer type had
    been detected. That sentence has none of the three, and asking the rules
    directly returns Theis Recovery at 0.85 confidence.
    """

    def test_a_recovery_test_in_plain_words_raises_the_card(self, wired):
        chat, manager_llm, _, _ = wired
        # Consultation, and the classifier finds no aquifer — the case that
        # used to fall through.
        manager_llm.responses = [_classification("consultation", aquifer="unknown")]

        chat.send_message(
            "The pump stopped after 24 hours and I kept reading water levels "
            "as they came back up. How do I get transmissivity from that?")

        suggestion = chat.state["method_suggestion"]
        assert suggestion["method"] == "Theis Recovery (1935)"
        assert chat.state["selected_method"] == "Theis Recovery (1935)"

    @pytest.mark.parametrize("message", [
        "I stopped the pump and measured recovery",
        "the pumping was shut off and the water level rose again",
        "I have residual drawdown readings",
    ])
    def test_the_other_ways_people_say_it(self, wired, message):
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("consultation", aquifer="unknown")]

        chat.send_message(message)

        assert chat.state["method_suggestion"]["method"] == "Theis Recovery (1935)"

    def test_a_theory_question_still_raises_nothing(self, wired):
        """The gate must stay shut for questions that describe no test: a card
        appearing on 'what is transmissivity?' would change the workspace's
        method for no reason."""
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("consultation", aquifer="unknown")]

        chat.send_message("What is transmissivity?")

        assert not chat.state.get("method_suggestion")

    def test_a_confirmed_method_is_never_overwritten(self, wired):
        """Widening the gate must not let a later sentence change a method the
        user already committed to."""
        chat, manager_llm, _, _ = wired
        chat.confirm_method(THEIS)
        manager_llm.responses = [_classification("consultation", aquifer="unknown")]

        chat.send_message("Later the pump stopped and levels recovered.")

        assert chat.state["selected_method"] == THEIS


class TestWhenTheModelIsUnavailable:
    """A spent quota must not cost the user the method suggestion too.

    Nothing in the suggestion uses the language model: it is keyword rules on
    the message, and the Bourdet derivative on any staged readings. But a failed
    classification used to return before the suggestion was built, so when the
    daily allowance ran out the user lost both the answer and the guidance —
    while the calculators, which the failure message promises are unaffected,
    sat waiting on the wrong method.

    On a free Gemini key this is the ordinary case, not an edge one: the
    allowance is 20 requests a day and each question costs two.
    """

    def _quota_error(self):
        return Exception(
            "Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429 "
            "Quota exceeded... limit: 20")

    def test_a_recovery_test_still_raises_its_card(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.fail_with = self._quota_error()

        chat.send_message(
            "The pump stopped after 24 hours and I kept reading water levels "
            "as they came back up.")

        assert chat.state["method_suggestion"]["method"] == "Theis Recovery (1935)"

    def test_the_intent_falls_back_to_consultation(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.fail_with = self._quota_error()

        chat.send_message("What are the assumptions of Theis?")

        assert chat.state["user_intent"] == "consultation"

    def test_a_named_method_is_still_picked_up(self, wired):
        """"Use Cooper-Jacob on this" must still preselect Cooper-Jacob."""
        chat, manager_llm, _, _ = wired
        manager_llm.fail_with = self._quota_error()

        chat.send_message("I want to use the Cooper-Jacob method on my data.")

        assert chat.state["method_suggestion"]["method"] == "Cooper-Jacob (1946)"

    def test_a_theory_question_still_raises_nothing(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.fail_with = self._quota_error()

        chat.send_message("What is transmissivity?")

        assert not chat.state.get("method_suggestion")


class TestSourcesMatchTheAnswer:
    """What the browser lists as "Sources consulted" must be what the answer used.

    Reported from the deployed site: with the daily Gemini allowance spent, the
    quota message appeared under five citations — Walton's bibliography, an EPA
    acknowledgements list, a general preamble about appraising groundwater
    resources. Retrieval had succeeded and the model call had not, so the
    documents were still in state and the browser dutifully printed them.

    Citing papers for an answer that never read them is a small dishonesty, and
    it is worse than useless here: the sources look like evidence for a message
    that has none.
    """

    def _quota_error(self):
        return Exception("429 RESOURCE_EXHAUSTED: quota exceeded, limit: 20")

    def test_a_failed_answer_cites_nothing(self, wired):
        chat, manager_llm, consultant_llm, retriever = wired
        manager_llm.responses = [_classification("consultation")]
        consultant_llm.fail_with = self._quota_error()

        chat.send_message("What are the assumptions of the Theis method?")

        assert retriever.calls, "the retrieval should still have been attempted"
        assert chat.state["retrieved_docs"] == []

    def test_the_failure_message_is_still_shown(self, wired):
        """Clearing the sources must not cost the explanation."""
        chat, manager_llm, consultant_llm, _ = wired
        manager_llm.responses = [_classification("consultation")]
        consultant_llm.fail_with = self._quota_error()

        reply = chat.send_message("What are the assumptions of the Theis method?")

        assert "daily allowance" in reply
        assert "calculators are unaffected" in reply

    def test_a_successful_answer_still_cites_its_sources(self, wired):
        """The guard against over-correcting: a real answer keeps its evidence."""
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("consultation")]

        chat.send_message("When should I use Cooper-Jacob?")

        assert chat.state["retrieved_docs"], "a real answer must keep its sources"


class TestAskingAboutTheApplication:
    """"Where do I enter the pumping rate?" used to search Theis (1935).

    The consultant knows a library of papers and nothing about HydroAssist, so
    a question about the interface was answered from whatever hydrogeology came
    closest — which is how a question about a text field ends up citing 1955.
    """

    def test_a_how_to_question_reaches_the_guide(self, wired):
        chat, manager_llm, _, retriever = wired
        manager_llm.responses = [_classification("how_to", aquifer="unknown")]

        reply = chat.send_message("Where do I enter the pumping rate?")

        assert "Step 2" in reply
        assert chat.help_retriever.calls, "the help index should have been read"
        assert not retriever.calls, "the papers must not be searched for this"

    def test_it_cites_help_and_never_a_paper(self, wired):
        """The rule the sources bug established: what is listed as consulted
        must be what the answer read."""
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("how_to", aquifer="unknown")]

        chat.send_message("How do I upload my data?")

        sources = chat.state["retrieved_docs"]
        assert sources
        assert all(doc["metadata"]["type"] == "help" for doc in sources)

    def test_it_offers_one_place_to_go(self, wired):
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("how_to", aquifer="unknown")]

        chat.send_message("Where do I enter the pumping rate?")

        assert chat.state["proposed_actions"] == [
            {"kind": "open_panel", "target": "setup",
             "label": "Go to Step 2 — Test setup"}]

    def test_the_offer_does_not_outlive_its_turn(self, wired):
        """A "Go to Step 2" button left under a later answer about leakage is
        the same mistake as citing the previous question's sources."""
        chat, manager_llm, _, _ = wired
        manager_llm.responses = [_classification("how_to", aquifer="unknown")]
        chat.send_message("Where do I enter the pumping rate?")
        assert chat.state["proposed_actions"]

        manager_llm.responses = [_classification("consultation")]
        manager_llm._i = 0
        chat.send_message("What is the leakage factor?")

        assert chat.state["proposed_actions"] == []

    def test_a_theory_question_still_goes_to_the_papers(self, wired):
        """The guard against over-routing: the guide must not swallow
        hydrogeology."""
        chat, manager_llm, _, retriever = wired
        manager_llm.responses = [_classification("consultation")]

        chat.send_message("What are the assumptions of the Theis method?")

        assert retriever.calls, "the papers should have been searched"
        assert not chat.help_retriever.calls

    def test_the_guide_answers_from_the_workspace_too(self, wired):
        """"Why were my points excluded?" is answered by this fit's own notes,
        not by a general description of Cooper-Jacob."""
        chat, manager_llm, _, _ = wired
        _run_workspace_fit(chat)
        manager_llm.responses = [_classification("how_to", aquifer="unknown")]
        manager_llm._i = 0

        chat.send_message("Why were some of my points excluded?")

        prompt = chat.guide_llm.prompts[-1]
        assert "Validity notes on that fit" in prompt or "validity" in prompt.lower()
        assert "T = " in prompt


class TestEachAnswerKeepsItsOwnSources:
    """Reported from the deployed site: one "Sources consulted" list sat under the
    whole conversation, so it described the latest answer and was wrong beneath
    every earlier one — and it changed under them as each new question was asked.

    The documents are recorded against the message they supported now.
    """

    def _answer(self, chat, manager_llm, question):
        manager_llm.responses = [_classification("consultation")]
        return chat.send_message(question)

    @staticmethod
    def _reading(document):
        return [{"content": f"A passage from {document}.",
                 "metadata": {"doc_id": f"{document}_page_12", "page": 12,
                              "source": "Custom_Leaky Confined Aquifer",
                              "type": "research_paper",
                              "method": "hantush_jacob_1955", "aquifer": "leaky"},
                 "score": 0.77}]

    def test_an_answer_records_the_documents_it_read(self, wired):
        chat, manager_llm, _, _ = wired

        self._answer(chat, manager_llm, "When should I use Cooper-Jacob?")

        # Keyed by position in the conversation: user at 0, answer at 1.
        assert chat.state["sources_by_message"]["1"]

    def test_a_second_answer_does_not_take_over_the_first_one_s_sources(self, wired):
        chat, manager_llm, _, retriever = wired

        self._answer(chat, manager_llm, "When should I use Cooper-Jacob?")
        first = list(chat.state["sources_by_message"]["1"])

        retriever.docs = self._reading("ISWSB-49")
        self._answer(chat, manager_llm, "and what does leakage look like?")

        assert chat.state["sources_by_message"]["1"] == first
        assert chat.state["sources_by_message"]["3"] != first

    def test_a_failed_answer_records_none(self, wired):
        """The rule this extends: an answer that read nothing cites nothing, and
        that stays true once the citations are kept per message."""
        chat, manager_llm, consultant_llm, _ = wired
        manager_llm.responses = [_classification("consultation")]
        consultant_llm.fail_with = Exception(
            "429 RESOURCE_EXHAUSTED: quota exceeded, limit: 20")

        chat.send_message("What are the assumptions of the Theis method?")

        assert chat.state["sources_by_message"]["1"] == []

    def test_the_browser_gets_them_on_the_message(self, wired):
        """Through the presenter, which is what the chat panel renders."""
        from src.api import presenters

        chat, manager_llm, _, _ = wired
        self._answer(chat, manager_llm, "When should I use Cooper-Jacob?")

        class _Session:
            def __init__(self, chat):
                self.chat = chat

        state = presenters.chat_state(_Session(chat))

        assert state.messages[0].role == "user"
        assert state.messages[0].sources == []
        assert state.messages[1].sources, "the answer should carry its sources"
        # Named, not filed: the folder name must not reach the reader.
        assert not state.messages[1].sources[0].document.startswith("Custom_")
