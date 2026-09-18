"""Turning a question into a search filter over the tagged corpus.

The tags are only worth having if something uses them, and a filter that fires
too eagerly is worse than none: it hides the paper that would have answered.
So the rule is narrow — a question must name an author before anything is
filtered — and the retriever retries without filters when a filtered search
comes back empty.
"""

import pytest

from src.ingestion.corpus import METHOD_FLAG_PREFIX
from src.retrieval import method_filter

VOCABULARY = (
    "theis_1935", "theis_recovery_1935", "cooper_jacob_1946",
    "papadopulos_cooper_1967", "hantush_jacob_1955", "hantush_1960",
    "boulton_1963", "warren_root_1963", "neuman_1974", "neuman_1975",
    "neuman_witherspoon_1969", "moench_1997", "general_reference",
    "horizontal_well",
)


def _found(question):
    return method_filter.methods_in(question, VOCABULARY)


class TestWhatAQuestionNames:

    def test_a_named_method_is_found(self):
        """Hantush wrote both leaky papers, so both are in scope until the
        question says which."""
        found = _found("What are the assumptions of the Hantush method?")

        assert sorted(found) == ["hantush_1960", "hantush_jacob_1955"]

    def test_an_author_matches_all_their_papers(self):
        """"What did Neuman say about delayed yield?" should reach every Neuman
        paper, not one chosen arbitrarily."""
        found = _found("What did Neuman say about delayed yield?")

        assert set(found) >= {"neuman_1974", "neuman_1975"}

    def test_a_year_narrows_to_that_paper(self):
        assert _found("Explain Neuman 1974") == ["neuman_1974"]

    def test_two_authors_must_both_appear(self):
        """Neuman-Witherspoon needs Witherspoon named; Neuman alone is the
        unconfined work."""
        assert "neuman_witherspoon_1969" not in _found("Tell me about Neuman's method")
        assert "neuman_witherspoon_1969" in _found("Neuman and Witherspoon leaky aquifers")

    def test_case_and_punctuation_do_not_matter(self):
        assert _found("THEIS, 1935?") == _found("theis 1935")

    @pytest.mark.parametrize("question", [
        "What is transmissivity?",
        "How do I read a derivative plot?",
        "My drawdown data looks odd near the end",
        "",
    ])
    def test_a_general_question_names_nothing(self, question):
        assert _found(question) == []

    def test_ids_that_name_no_author_never_match(self):
        """'general_reference' and 'horizontal_well' describe the paper, not a
        person; matching them on the word 'well' would filter every question."""
        assert _found("How does a horizontal well behave?") == []
        assert _found("Is there a general reference for this?") == []


class TestTheFilter:

    def test_one_method_is_a_plain_flag(self):
        assert method_filter.filter_for(["theis_1935"]) == {
            f"{METHOD_FLAG_PREFIX}theis_1935": True}

    def test_several_are_an_or(self):
        clause = method_filter.filter_for(["neuman_1974", "neuman_1975"])

        assert set(clause) == {"$or"}
        assert len(clause["$or"]) == 2

    def test_nothing_named_means_no_filter(self):
        assert method_filter.filter_for([]) is None

    def test_a_question_becomes_a_usable_clause(self, monkeypatch):
        monkeypatch.setattr(method_filter, "_vocabulary", lambda: VOCABULARY)

        assert method_filter.filter_for_question("Explain Neuman 1974") == {
            f"{METHOD_FLAG_PREFIX}neuman_1974": True}
        assert method_filter.filter_for_question("What is storativity?") is None
