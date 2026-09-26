"""Manager Agent - Intent classification and routing."""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.calculator import registry
from src.core.state import AgentState
from src.core.config import AppConfig
from src.agents.method_selection import (
    canonical_method, describes_recovery, suggest_method,
)
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

        # Classification is a separate, smaller call. Two reasons, both of them
        # things a user of the deployed site feels: a reasoning model pauses to
        # think before emitting three JSON fields, which is latency spent on
        # nothing, and the two models have separate free-tier allowances, so a
        # day of testing affords twice as many questions. `thinking_budget=0`
        # is the part that removes the pause.
        self.classifier = ChatGoogleGenerativeAI(
            model=config.llm.classifier_model,
            temperature=0.0,
            max_tokens=config.llm.classifier_max_tokens,
            thinking_budget=0,
            google_api_key=config.llm.api_key,
        )

        self.logger.info(
            f"Manager agent initialized with model: {config.llm.model} "
            f"(classifier: {config.llm.classifier_model})")

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

            extracted = classification.get("extracted_method")
            if extracted:
                # Normalise to the canonical label the calculator dropdown uses,
                # so "theis method" and "Theis (1935)" select the same entry.
                state["selected_method"] = canonical_method(extracted) or extracted

            # Build the method suggestion card that bridges chat → calculator.
            self._update_method_suggestion(state, user_message)

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

            # Still offer a method. Nothing in the suggestion needs the model —
            # it is keyword rules on the message and the Bourdet derivative on
            # any staged readings — so a failed classification is no reason to
            # withhold it. This is the ordinary case on a free key, where the
            # daily allowance runs out while the calculators carry on working;
            # losing the suggestion too would strand the user twice over.
            try:
                self._update_method_suggestion(state, user_message)
            except Exception as suggestion_error:          # never mask the first
                self.logger.error(f"Suggestion also failed: {suggestion_error}")

            return state

    def _update_method_suggestion(self, state: AgentState, user_message: str):
        """Attach a method suggestion card to state.

        Runs for calculation intent, and for consultations that name a method or
        an aquifer — the user often decides which method to run while still
        asking theory questions, and the workspace should be primed for it.

        A suggestion the user has already confirmed is left alone; overwriting it
        on a later unrelated turn would silently change the method under them.
        """
        if state.get("suggestion_confirmed", False):
            return

        intent = state.get("user_intent", "unknown")
        aquifer = state.get("aquifer_context", "unknown")
        method = state.get("selected_method", "")

        staged = state.get("staged_series") or {}

        relevant = (
            intent == "calculation"
            or bool(method)
            or aquifer in {"confined", "unconfined", "leaky", "fractured"}
            # A message can describe the test without naming the aquifer.
            # "The pump stopped and I kept reading water levels as they came
            # back up" is a recovery test in plain words, and the rules resolve
            # it confidently — but the gate used to return first, so no card
            # appeared and the workspace stayed on the wrong method.
            or describes_recovery(user_message)
            # The method named in the sentence, read by the registry's own
            # aliases rather than by the model. `method` above is the
            # classifier's extraction, so without this a message saying
            # "use Cooper-Jacob" raised nothing whenever the model was
            # unavailable — which on a free key is whenever the day's
            # allowance is gone.
            or bool(registry.canonical_id(user_message))
            # Uploaded readings can drive the suggestion on their own, through
            # the Bourdet derivative, without a word being typed.
            or bool(staged.get("drawdown_m"))
        )
        if not relevant:
            return

        # With data the suggestion comes from the Bourdet derivative reading the
        # actual flow regime; without it, from keyword rules on the message.
        suggestion = suggest_method(
            aquifer_context=aquifer,
            selected_method=method,
            user_message=user_message,
            time_s=staged.get("time_s"),
            drawdown_m=staged.get("drawdown_m"),
            Q=staged.get("Q"),
        )
        state["method_suggestion"] = suggestion

        # Keep selected_method in step so the workspace preselects correctly.
        if suggestion.get("method"):
            state["selected_method"] = suggestion["method"]

        self.logger.info(
            f"Method suggestion: {suggestion.get('method') or 'none'} "
            f"(confidence {suggestion.get('confidence', 0):.2f})"
        )

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

        response_text = self._classify_call(prompt)

        # Parse JSON (handle markdown code blocks)
        clean_response = self._clean_json_response(response_text)

        try:
            classification = json.loads(clean_response)

            # Log confidence for observability but never override intent based on it.
            # Hydrogeological queries are often highly technical and Gemini may return
            # low confidence on queries that are obviously consultations to a domain expert.
            confidence = classification.get("confidence", 0.0)
            if confidence < 0.5:
                self.logger.warning(f"Low confidence ({confidence}) — keeping intent as-is: {classification['intent']}")

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

    def _classify_call(self, prompt: str) -> str:
        """The classification call, falling back to the answering model.

        A key that has no access to the smaller model, or has spent its
        allowance, would otherwise fail every classification — and the failure
        path defaults to "consultation", so the router would quietly stop
        routing. One retry on the model we know works costs a request only when
        something is already wrong.
        """
        try:
            return self.classifier.invoke(prompt).content
        except Exception as error:
            self.logger.warning(
                f"Classifier model {self.config.llm.classifier_model} failed "
                f"({error}); retrying on {self.config.llm.model}")
            return self.llm.invoke(prompt).content

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
