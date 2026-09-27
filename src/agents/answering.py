"""Producing an answer, a token at a time when someone is waiting for it.

An answer takes several seconds to generate, and until it arrives the panel shows
a spinner. That is the wrong shape for reading: the first sentence of an
interpretation usually says whether the fit is trustworthy, and there is no
reason to withhold it until the last sentence is written.

So an answering agent asks for its text through `answer_text`, which streams when
a sink is present and behaves exactly as before when one is not. Two properties
are worth keeping in mind:

- **The state is the same either way.** Streaming does not change what is stored,
  only when the reader sees it. The finished text is appended to the message list
  as one message, so a reload shows the same conversation as a non-streamed one.
- **A failure mid-stream is still a failure.** Tokens already sent cannot be
  recalled, so the caller is told, and the browser replaces the partial text with
  the explanation rather than leaving half an answer on screen looking finished.

The sink travels in the graph's own config — `configurable.token_sink` — rather
than in the conversation state, because it is a property of this one call and not
a fact about the conversation, and because the state is serialised into project
files where a callable has no business being.
"""

from typing import Callable, Optional

TokenSink = Callable[[str], None]

CONFIG_KEY = "token_sink"


def sink_from(config) -> Optional[TokenSink]:
    """The token sink a caller passed, or None for an ordinary call."""
    if not config:
        return None
    try:
        sink = (config.get("configurable") or {}).get(CONFIG_KEY)
    except AttributeError:
        return None
    return sink if callable(sink) else None


def answer_text(llm, prompt: str, sink: Optional[TokenSink] = None) -> str:
    """The model's answer, streamed to `sink` as it arrives if one is given.

    Returns the whole text either way, so every caller keeps working unchanged.
    A sink that raises is not allowed to lose an answer that was generated: the
    stream continues and the text is returned, because the caller's queue being
    full is not the conversation's problem.
    """
    if sink is None:
        return llm.invoke(prompt).content

    pieces = []
    for chunk in llm.stream(prompt):
        text = getattr(chunk, "content", "") or ""
        if not text:
            continue
        pieces.append(text)
        try:
            sink(text)
        except Exception:                                   # noqa: BLE001
            pass
    return "".join(pieces)
