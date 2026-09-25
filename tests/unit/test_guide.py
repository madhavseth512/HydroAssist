"""The guide: answering about HydroAssist rather than about hydrogeology.

The consultant knows a library of papers and nothing about the application.
Asked "where do I enter the pumping rate?", it searched Theis (1935) for an
answer that was never in it. These cover the parts that do not need a language
model: what the guide reads, what it offers, and what it refuses to claim.
"""

import pytest

from src.agents.guide import ACTION_CONFIDENCE, PANEL_FOR_HELP, GuideAgent
from src.prompts.guide_prompts import (
    NO_HELP_RESPONSE, format_help_context, format_workspace_context,
)


def _help(file: str, section: str = "The fields", score: float = 0.5) -> dict:
    return {
        "content": f"Some help about {section}.",
        "metadata": {"source": "HydroAssist help", "type": "help",
                     "document": "Setting up the test", "section": section,
                     "file": file},
        "score": score,
    }


class TestWhatTheGuideOffers:
    """One place to go, taken from the passage the answer came from — never
    from the model, which could invent a button pointing anywhere."""

    def _offer(self, documents):
        return GuideAgent._offer(None, documents)

    @pytest.mark.parametrize("file,target", [
        ("setting-up-the-test.md", "setup"),
        ("getting-data-in.md", "data"),
        ("reading-the-results.md", "results"),
        ("comparing-methods.md", "compare"),
        ("projects-and-saving.md", "project"),
    ])
    def test_each_help_document_leads_somewhere(self, file, target):
        offer = self._offer([_help(file)])

        assert offer == [{"kind": "open_panel", "target": target,
                          "label": PANEL_FOR_HELP[file][1]}]

    def test_at_most_one_offer(self):
        """A list of buttons under every answer is furniture, not help."""
        offers = self._offer([_help("setting-up-the-test.md"),
                              _help("getting-data-in.md")])

        assert len(offers) == 1

    def test_a_loose_match_offers_nothing(self):
        """An offer built on a weak passage sends the user somewhere
        arbitrary, which is worse than not offering."""
        assert self._offer([_help("setting-up-the-test.md",
                                  score=ACTION_CONFIDENCE - 0.01)]) == []

    @pytest.mark.parametrize("file", ["the-consultant.md", "troubleshooting.md",
                                      "unknown.md", ""])
    def test_documents_with_nowhere_to_go_offer_nothing(self, file):
        assert self._offer([_help(file)]) == []

    def test_an_offer_is_never_built_without_a_passage(self):
        """`_offer` reads the best passage, so the caller must not reach it
        empty-handed — `run` returns the no-help answer first."""
        with pytest.raises(IndexError):
            self._offer([])


class TestWhatTheGuideReads:

    def test_help_passages_are_labelled_with_their_section(self):
        text = format_help_context([_help("setting-up-the-test.md")])

        assert "Setting up the test" in text
        assert "The fields" in text

    def test_no_passages_says_so_rather_than_pretending(self):
        assert "No help passages" in format_help_context([])

    def test_the_refusal_sends_theory_to_the_consultant(self):
        """The guide must not answer hydrogeology, and must say who does."""
        assert "consultant" in NO_HELP_RESPONSE
        assert "literature" in NO_HELP_RESPONSE


class TestWhatTheGuideKnowsAboutTheWorkspace:
    """Half these questions cannot be answered from documentation at all.
    "What do I do next?" depends on which fields are empty; "why were my points
    excluded?" is answered by the fit's own notes."""

    def test_an_empty_workspace_says_step_one_is_empty(self):
        text = format_workspace_context({})

        assert "none yet" in text
        assert "No analysis has been run" in text

    def test_it_reports_what_step_two_still_needs(self):
        text = format_workspace_context({
            "calculation_input": {"n_points": 50, "Q": 0.0058},
        })

        assert "50 points" in text
        assert "distance r" in text and "still needs" in text

    def test_a_finished_fit_brings_its_validity_notes(self):
        """These are the answer to "why were 18 of my points excluded?", and
        they are specific to this fit rather than to the method in general."""
        text = format_workspace_context({
            "calculation_input": {"n_points": 50, "Q": 0.0058, "r": 50.0},
            "calculation_result": {
                "success": True, "method": "Cooper-Jacob (1946)",
                "T_day": 215.29, "S": 1.02e-4, "r_squared": 0.9991,
                "validity_notes": ["18 early-time point(s) excluded (u >= 0.02)"],
            },
        })

        assert "Cooper-Jacob (1946)" in text
        assert "215.3" in text
        assert "18 early-time point(s) excluded" in text

    def test_a_failed_fit_says_why(self):
        text = format_workspace_context({
            "calculation_result": {"success": False,
                                   "error_message": "Too few valid points"},
        })

        assert "failed" in text and "Too few valid points" in text

    def test_a_method_the_user_confirmed_is_reported_as_confirmed(self):
        text = format_workspace_context({"selected_method": "Theis (1935)",
                                         "suggestion_confirmed": True})

        assert "Theis (1935)" in text and "confirmed" in text
