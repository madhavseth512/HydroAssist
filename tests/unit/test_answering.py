"""Streaming an answer, and behaving identically when nobody is listening.

An answer takes several seconds to write and its first sentence usually carries
the verdict, so it is sent as it is produced. The property that matters is that
streaming changes only *when* the reader sees the text: what is stored, and what
the caller gets back, are the same either way.
"""

import pytest

from src.agents.answering import answer_text, sink_from


class _Chunk:
    def __init__(self, content):
        self.content = content


class _LLM:
    """Answers either way, and records which way it was asked."""

    def __init__(self, pieces):
        self.pieces = pieces
        self.streamed = False
        self.invoked = False

    def invoke(self, prompt):
        self.invoked = True
        return _Chunk("".join(self.pieces))

    def stream(self, prompt):
        self.streamed = True
        for piece in self.pieces:
            yield _Chunk(piece)


PIECES = ["The fit is ", "trustworthy: ", "R2 = 0.96."]
WHOLE = "The fit is trustworthy: R2 = 0.96."


class TestWithNobodyListening:

    def test_it_invokes_and_returns_the_whole_answer(self):
        llm = _LLM(PIECES)

        assert answer_text(llm, "prompt") == WHOLE
        assert llm.invoked and not llm.streamed


class TestWithSomeoneWaiting:

    def test_the_pieces_arrive_in_order(self):
        llm, seen = _LLM(PIECES), []

        answer_text(llm, "prompt", seen.append)

        assert seen == PIECES
        assert llm.streamed and not llm.invoked

    def test_the_whole_answer_is_still_returned(self):
        """So the caller that stores it needs no special case."""
        llm, seen = _LLM(PIECES), []

        assert answer_text(llm, "prompt", seen.append) == WHOLE

    def test_empty_chunks_are_not_forwarded(self):
        """A model emits them; a reader does not need to know."""
        llm, seen = _LLM(["a", "", None, "b"]), []

        assert answer_text(llm, "prompt", seen.append) == "ab"
        assert seen == ["a", "b"]

    def test_a_sink_that_fails_does_not_lose_the_answer(self):
        """A full queue, or a reader who closed the tab, is not a reason to throw
        away text the model has already produced."""
        def angry(_text):
            raise RuntimeError("nobody is reading")

        assert answer_text(_LLM(PIECES), "prompt", angry) == WHOLE


class TestFindingTheSink:
    """It travels in the graph's config, not in the conversation state: it belongs
    to one call, and the state is serialised into project files."""

    def test_it_is_read_from_the_configurable_section(self):
        def sink(_text): pass

        assert sink_from({"configurable": {"token_sink": sink}}) is sink

    @pytest.mark.parametrize("config", [
        None, {}, {"configurable": {}}, {"configurable": {"token_sink": None}},
        {"configurable": {"token_sink": "not callable"}}, "nonsense",
    ])
    def test_anything_else_means_no_streaming(self, config):
        assert sink_from(config) is None
