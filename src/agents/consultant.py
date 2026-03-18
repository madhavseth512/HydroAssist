"""Consultant Agent - RAG-based Q&A for hydrogeological theory."""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.core.state import AgentState
from src.core.config import AppConfig
from src.retrieval.retriever import Retriever
from src.prompts.consultant_prompts import (
    CONSULTANT_SYSTEM_PROMPT,
    format_retrieved_context,
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

        # Build metadata filters
        metadata_filters = {}

        aquifer_context = state.get("aquifer_context", "unknown")
        if aquifer_context != "unknown":
            metadata_filters["aquifer"] = aquifer_context
            self.logger.info(f"Filtering by aquifer: {aquifer_context}")

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

        # Check if we have enough context
        if not retrieved_docs:
            self.logger.warning("No documents retrieved")
            state["messages"].append(AIMessage(content=NO_CONTEXT_RESPONSE))
            return state

        # Format context for prompt
        context = format_retrieved_context(retrieved_docs)

        # Generate response
        try:
            prompt = CONSULTANT_SYSTEM_PROMPT.format(
                retrieved_context=context,
                question=question
            )

            response = self.llm.invoke(prompt)
            answer = response.content

            self.logger.info(f"Generated answer ({len(answer)} chars)")

            # Extract method if mentioned
            extracted_method = self._extract_method_from_response(answer)
            if extracted_method and state.get("selected_method", "") == "":
                state["selected_method"] = extracted_method
                self.logger.info(f"Extracted method: {extracted_method}")

            # Add response to state
            state["messages"].append(AIMessage(content=answer))

            return state

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            error_response = "I encountered an error generating a response. Please try rephrasing your question."
            state["messages"].append(AIMessage(content=error_response))
            return state

    def _extract_method_from_response(self, response: str) -> str:
        """
        Extract pumping test method name from response.

        Args:
            response: Generated response text

        Returns:
            Method name or empty string
        """
        # Common method patterns
        patterns = [
            r"(Theis)\s*\(?\s*(\d{4})\)?",
            r"(Cooper[-\s]Jacob)\s*\(?\s*(\d{4})\)?",
            r"(Hantush[-\s]Jacob)\s*\(?\s*(\d{4})\)?",
            r"(Neuman)\s*\(?\s*(\d{4})\)?",
            r"(Boulton)\s*\(?\s*(\d{4})\)?",
            r"(Moench)\s*\(?\s*(\d{4})\)?",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                method_name = match.group(1).replace('-', '_').replace(' ', '_')
                year = match.group(2)
                return f"{method_name}_{year}"

        return ""
