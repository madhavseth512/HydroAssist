"""Who owns the method: the workspace, or the conversation.

Reported from the deployed site. Theis was chosen in the calculator and run; a
message about Papadopulos-Cooper then changed what the chat said the method was —
to Cooper-Jacob, a third method nobody had mentioned.

Two faults, and this file pins both:

- The classifier's extraction overwrote the method on every turn, including after
  a fit had been run and the method settled.
- Where that extraction was wrong, nothing checked it against the user's own
  words, which the registry can read directly — including a misspelt author name.

The rule taken from it is the one already agreed for the guide's offers: the
conversation proposes, and only the user moves the analysis.

The model is stubbed, as in tests/integration/test_chat_calculator_bridge.py, so
none of this spends a request from the daily allowance.
"""

import json

import pytest
from langchain_core.messages import HumanMessage

from src.agents.manager import ManagerAgent
from src.core.config import load_config
from src.core.state import create_initial_state


class _StubLLM:
    """Answers every call with one scripted classification."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Resp", (), {"content": json.dumps(self.payload)})()


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    return load_config()


CONSULTATION = {"intent": "consultation", "confidence": 0.9,
                "extracted_aquifer": "confined", "extracted_method": None}


@pytest.fixture
def run(config, monkeypatch):
    """Run one turn through the real manager with a scripted classification."""
    def _run(message, extracted_method=None, **state_overrides):
        payload = dict(CONSULTATION, extracted_method=extracted_method)
        monkeypatch.setattr("src.agents.manager.ChatGoogleGenerativeAI",
                            lambda **kw: _StubLLM(payload))
        agent = ManagerAgent(config)
        state = create_initial_state()
        state["messages"].append(HumanMessage(content=message))
        state.update(state_overrides)
        return agent.run(state)
    return _run


AFTER_A_THEIS_FIT = dict(
    selected_method="Theis (1935)",
    suggestion_confirmed=True,          # running a fit confirms the method
    calculation_result={"method": "Theis (1935)", "success": True},
)


class TestWhatTheMessageIsAbout:

    def test_a_named_method_is_recorded_as_discussed(self, run):
        state = run("tell me about papadopulos-cooper",
                    extracted_method="Papadopulos-Cooper")

        assert state["discussed_method"] == "Papadopulos-Cooper (1967)"

    def test_the_users_own_words_outrank_the_extraction(self, run):
        """The reported failure: the extraction said Cooper-Jacob about a message
        that says Papadopulos-Cooper."""
        state = run("what does papadopulos-cooper assume?",
                    extracted_method="Cooper-Jacob")

        assert state["discussed_method"] == "Papadopulos-Cooper (1967)"

    def test_a_misspelt_name_is_still_read_from_the_message(self, run):
        state = run("what does papadulous-cooper assume?",
                    extracted_method="Cooper-Jacob")

        assert state["discussed_method"] == "Papadopulos-Cooper (1967)"

    def test_a_message_naming_nothing_discusses_nothing(self, run):
        state = run("what is transmissivity?")

        assert state["discussed_method"] == ""


class TestWhatTheWorkspaceWillRun:

    def test_before_anything_is_settled_the_conversation_leads(self, run):
        state = run("I think theis is right for this", extracted_method="Theis")

        assert state["selected_method"] == "Theis (1935)"

    def test_a_settled_method_is_not_moved_by_a_later_question(self, run):
        """The case as reported: a fit has been run, which settles the method,
        and then a question about a different method arrives."""
        state = run("and what does papadopulos-cooper assume?",
                    extracted_method="Papadopulos-Cooper", **AFTER_A_THEIS_FIT)

        assert state["selected_method"] == "Theis (1935)"
        # Answered all the same, and visibly about the other method.
        assert state["discussed_method"] == "Papadopulos-Cooper (1967)"

    def test_no_suggestion_card_reappears_over_a_finished_fit(self, run):
        state = run("why was my storativity so small?", **AFTER_A_THEIS_FIT)

        assert not state.get("method_suggestion")

    def test_naming_a_method_over_a_finished_fit_still_offers_it(self, run):
        """Because that is the user asking, not the model guessing."""
        state = run("run cooper-jacob on this instead",
                    extracted_method="Cooper-Jacob",
                    selected_method="Theis (1935)",
                    calculation_result={"method": "Theis (1935)", "success": True})

        assert state["method_suggestion"]["method"] == "Cooper-Jacob (1946)"

    def test_a_wrong_extraction_cannot_choose_the_method_either(self, run):
        """The whole failure in one assertion: the words say one method, the
        model says another, and the workspace must not be handed the model's."""
        state = run("is papadopulos-cooper right for a large-diameter well?",
                    extracted_method="Cooper-Jacob")

        assert state["selected_method"] == "Papadopulos-Cooper (1967)"
