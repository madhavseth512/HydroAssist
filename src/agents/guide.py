"""The guide: how to use HydroAssist, as opposed to how pumping tests work.

Asked "where do I enter the pumping rate?", the consultant searched Theis (1935)
for an answer that was never in it, because the only knowledge it had was a
library of papers. This agent answers from `docs/help/` and from the state of
the user's own workspace.

It may also offer to take the user somewhere. It never goes there itself: an
agent moving the workspace under a scientist mid-analysis is a worse failure
than one that answers a question badly.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.agents.errors import explain_model_failure, retry_delay_seconds
from src.core.config import AppConfig
from src.core.state import AgentState
from src.prompts.guide_prompts import (
    GUIDE_SYSTEM_PROMPT, NO_HELP_RESPONSE, format_help_context,
    format_workspace_context,
)
from src.retrieval.help_retriever import HelpRetriever

# Which panel a help document belongs to, and what to call the offer.
#
# Derived from the help passage that was actually retrieved, not asked of the
# model: a button the model invented could point anywhere, and the one thing a
# navigation offer must be is correct.
PANEL_FOR_HELP = {
    "getting-data-in.md": ("data", "Go to Step 1 — Field data"),
    "setting-up-the-test.md": ("setup", "Go to Step 2 — Test setup"),
    "reading-the-results.md": ("results", "Go to the results"),
    "comparing-methods.md": ("compare", "Go to the comparison"),
    "projects-and-saving.md": ("project", "Go to the project panel"),
}

# Below this the retrieved passage is a loose match, and an offer built on it
# would send the user somewhere arbitrary.
ACTION_CONFIDENCE = 0.30


class GuideAgent(BaseAgent):
    """Answers questions about the application itself."""

    def __init__(self, config: AppConfig):
        super().__init__(config)

        self.llm = ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=0.1,
            max_tokens=config.llm.max_tokens,
            google_api_key=config.llm.api_key,
        )
        self.retriever = HelpRetriever(config)
        self.logger.info("Guide agent initialized")

    def run(self, state: AgentState) -> AgentState:
        self._log_execution(state, "guiding")

        if not state["messages"]:
            return state
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            return state

        question = last_message.content

        documents = self.retriever.retrieve(question)
        self.logger.info(f"Retrieved {len(documents)} help passages")

        # The guide cites help or nothing. Leaving the consultant's papers in
        # place would put Theis (1935) under an answer about a text field.
        state["retrieved_docs"] = documents
        state["proposed_actions"] = []

        if not documents:
            state["messages"].append(AIMessage(content=NO_HELP_RESPONSE))
            return state

        prompt = GUIDE_SYSTEM_PROMPT.format(
            help_context=format_help_context(documents),
            workspace_context=format_workspace_context(state),
            question=question,
        )

        try:
            answer = self.llm.invoke(prompt).content
        except Exception as error:
            self.logger.error(f"Guide answer failed: {error}")
            delay = retry_delay_seconds(error)
            if delay:
                self.logger.error(f"The model asked to be retried in {delay:.0f}s")
            state["retrieved_docs"] = []
            state["messages"].append(AIMessage(content=explain_model_failure(error)))
            return state

        state["messages"].append(AIMessage(content=answer))
        state["proposed_actions"] = self._offer(documents)
        return state

    def _offer(self, documents) -> list:
        """At most one place to go, from the passage the answer came from.

        One, because a list of buttons under every answer is furniture rather
        than help.
        """
        best = documents[0]
        if best.get("score", 0.0) < ACTION_CONFIDENCE:
            return []

        target = PANEL_FOR_HELP.get(best.get("metadata", {}).get("file", ""))
        if not target:
            return []

        panel, label = target
        return [{"kind": "open_panel", "target": panel, "label": label}]
