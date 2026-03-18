"""Manager Agent - Intent classification and routing."""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.core.state import AgentState
from src.core.config import AppConfig
from src.prompts.manager_prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    CLARIFICATION_PROMPT,
    UNKNOWN_INTENT_RESPONSE
)


class ManagerAgent(BaseAgent):
    """
    Manager agent for intent classification and routing.

    Responsibilities:
    1. Classify user intent using Gemini
    2. Extract aquifer context from conversation
    3. Determine routing decision
    4. Ask clarification questions for ambiguous queries
    """

    def __init__(self, config: AppConfig):
        """
        Initialize Manager agent.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            google_api_key=config.llm.api_key
        )

        self.logger.info(f"Manager agent initialized with model: {config.llm.model}")

    def run(self, state: AgentState) -> AgentState:
        """
        Classify intent and route conversation.

        Args:
            state: Current conversation state

        Returns:
            Updated state with classified intent
        """
        self._log_execution(state, "classify_intent")

        # Get last user message
        if not state["messages"]:
            self.logger.warning("No messages in state")
            return state

        last_message = state["messages"][-1]

        if not isinstance(last_message, HumanMessage):
            self.logger.warning("Last message is not from user")
            return state

        user_message = last_message.content

        # Get conversation history (last 5 messages before current)
        history = self._format_history(state["messages"][-6:-1] if len(state["messages"]) > 1 else [])

        # Call Gemini for classification
        try:
            classification = self._classify_intent(user_message, history)

            self.logger.info(f"Classification: {classification}")

            # Update state
            state["user_intent"] = classification["intent"]
            state["aquifer_context"] = classification.get("extracted_aquifer", "unknown")

            if classification.get("extracted_method"):
                state["selected_method"] = classification["extracted_method"]

            # Handle clarification and unknown intents
            if classification["intent"] == "clarification":
                response = self._ask_clarification(user_message)
                state["messages"].append(AIMessage(content=response))

            elif classification["intent"] == "unknown":
                state["messages"].append(AIMessage(content=UNKNOWN_INTENT_RESPONSE))

            return state

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            # Default to consultation on error
            state["user_intent"] = "consultation"
            return state

    def _classify_intent(self, user_message: str, history: str) -> dict:
        """
        Classify user intent using Gemini.

        Args:
            user_message: Current user message
            history: Formatted conversation history

        Returns:
            Classification dictionary
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_message=user_message,
            history=history if history else "No previous conversation"
        )

        # Call Gemini
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse JSON (handle markdown code blocks)
        clean_response = self._clean_json_response(response_text)

        try:
            classification = json.loads(clean_response)

            # Validate confidence threshold
            confidence = classification.get("confidence", 0.0)
            if confidence < 0.7 and classification["intent"] not in ["clarification", "unknown"]:
                self.logger.warning(f"Low confidence: {confidence}")
                classification["intent"] = "clarification"

            return classification

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Response: {response_text}")
            # Return safe default
            return {
                "intent": "consultation",
                "confidence": 0.5,
                "reasoning": "JSON parse error",
                "extracted_aquifer": "unknown",
                "extracted_method": None
            }

    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response from Gemini (remove markdown formatting).

        Args:
            response: Raw response string

        Returns:
            Cleaned JSON string
        """
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith('```'):
            # Find the JSON content between ```
            match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Try to extract anything between first { and last }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    response = response[start:end+1]

        return response.strip()

    def _ask_clarification(self, user_message: str) -> str:
        """
        Generate clarification question.

        Args:
            user_message: User's unclear message

        Returns:
            Clarification question
        """
        prompt = CLARIFICATION_PROMPT.format(user_message=user_message)

        response = self.llm.invoke(prompt)

        return response.content

    def _format_history(self, messages: list) -> str:
        """
        Format conversation history for prompt.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted history string
        """
        if not messages:
            return ""

        formatted = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")

        return "\n".join(formatted[-5:])  # Last 5 messages
