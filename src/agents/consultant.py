"""Consultant Agent - RAG-based Q&A for hydrogeological theory."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.agents.errors import explain_model_failure, retry_delay_seconds
from src.core.state import AgentState
from src.core.config import AppConfig
from src.retrieval.method_filter import filter_for_question as method_filter_for_question
from src.retrieval.retriever import Retriever
from src.prompts.consultant_prompts import (
    CONSULTANT_SYSTEM_PROMPT,
    format_retrieved_context,
    format_analysis_context,
    NO_CONTEXT_RESPONSE
)


class ConsultantAgent(BaseAgent):
    """
    Consultant agent for RAG-based question answering.

    Responsibilities:
    1. Retrieve relevant documents using Manager's metadata filters
    2. Synthesize answer using Gemini with retrieved context
    3. Format citations properly
    4. Update state with identified method
    """

    def __init__(self, config: AppConfig):
        """
        Initialize Consultant agent.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=0.1,  # Low temperature for factual accuracy
            max_tokens=config.llm.max_tokens,
            google_api_key=config.llm.api_key
        )

        # Initialize retriever
        self.retriever = Retriever(config)

        self.logger.info("Consultant agent initialized")

    def run(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant docs and generate answer.

        Args:
            state: Current conversation state

        Returns:
            Updated state with response
        """
        self._log_execution(state, "consulting")

        # Get user question
        if not state["messages"]:
            return state

        last_message = state["messages"][-1]

        if not isinstance(last_message, HumanMessage):
            return state

        question = last_message.content

        # Build metadata filters.
        # Aquifer filtering is only applied when the user has confirmed a method
        # (i.e. they are in calculation mode). During general consultation it is
        # too aggressive and excludes cross-aquifer documents that are often relevant.
        metadata_filters = {}

        if state.get("suggestion_confirmed", False):
            aquifer_context = state.get("aquifer_context", "unknown")
            if aquifer_context != "unknown":
                metadata_filters["aquifer"] = aquifer_context
                self.logger.info(f"Filtering by aquifer: {aquifer_context}")

        # When the question names a method, prefer the papers about it. Until
        # the corpus was tagged this was impossible: every chunk said "multiple",
        # so a question about Hantush competed with 483 chunks of Neuman. A
        # filtered search that finds nothing is retried unfiltered by the
        # retriever, so this narrows without being able to silence an answer.
        method_filter = method_filter_for_question(question)
        if method_filter:
            if metadata_filters:
                metadata_filters = {"$and": [metadata_filters, method_filter]}
            else:
                metadata_filters = method_filter

        # Retrieve relevant documents
        try:
            retrieved_docs = self.retriever.retrieve(
                query=question,
                top_k=self.config.retrieval.top_k,
                metadata_filters=metadata_filters if metadata_filters else None
            )

            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Store retrieved docs in state
            state["retrieved_docs"] = retrieved_docs

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            retrieved_docs = []

        # Check if we have enough context.
        # A completed analysis is itself usable context: when the user asks
        # about their own T and S we can answer from the result even if the
        # retriever came back empty.
        if not retrieved_docs and not state.get("calculation_result"):
            self.logger.warning("No documents retrieved")
            state["messages"].append(AIMessage(content=NO_CONTEXT_RESPONSE))
            return state

        # Format context for prompt
        context = format_retrieved_context(retrieved_docs)

        # Generate response
        try:
            # Any completed workspace analysis travels with the prompt, so the
            # consultant can answer questions about the user's own T and S
            # rather than only about theory.
            analysis_context = format_analysis_context(
                state.get("calculation_result") or {},
                state.get("calculation_input") or {},
            )

            prompt = CONSULTANT_SYSTEM_PROMPT.format(
                retrieved_context=context,
                analysis_context=analysis_context,
                question=question
            )

            response = self.llm.invoke(prompt)
            answer = response.content

            self.logger.info(f"Generated answer ({len(answer)} chars)")

            # Add response to state
            state["messages"].append(AIMessage(content=answer))

            return state

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            delay = retry_delay_seconds(e)
            if delay:
                self.logger.error(f"The model asked to be retried in {delay:.0f}s")
            # Say what actually went wrong. "Try rephrasing your question" is
            # false when the daily quota is gone: the question was fine, and
            # rephrasing it will fail identically.
            state["messages"].append(AIMessage(content=explain_model_failure(e)))
            return state

