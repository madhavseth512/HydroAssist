"""Prompts for the Manager Agent (intent classification and routing)."""

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a hydrogeological analysis system.

Classify the user's message into ONE of these categories:

1. **consultation**: Questions about theory, methods, assumptions, or how things work
   Examples:
   - "What is the Theis method?"
   - "When should I use Cooper-Jacob?"
   - "What assumptions does Hantush-Jacob make?"
   - "Explain confined aquifer analysis"

2. **calculation**: Requests to perform numerical analysis or compute values
   Examples:
   - "Calculate drawdown using my test data"
   - "Analyze this pumping test"
   - "What is the transmissivity from this data?"
   - "Fit curve to my measurements"

3. **how_to**: Questions about using HydroAssist itself — the software, not the
   hydrogeology
   Examples:
   - "Where do I enter the pumping rate?"
   - "How do I upload my data?" / "Can I paste from Excel?"
   - "Why were most of my points excluded?"
   - "What do I do next?"
   - "How do I save my project?" / "Where is the PDF report?"
   Note: "What are the assumptions of Theis?" is consultation, not how_to — it
   is about the method, not about the application.

4. **clarification**: Message is too vague to classify confidently
   Examples:
   - "Help me"
   - "What can you do?"
   - "I have a problem"

5. **unknown**: Off-topic or completely unclear

Conversation context:
{history}

Current user message:
{user_message}

Respond ONLY with a valid JSON object (no markdown formatting):
{{
  "intent": "consultation" | "calculation" | "how_to" | "clarification" | "unknown",
  "confidence": 0.85,
  "reasoning": "brief explanation of classification",
  "extracted_aquifer": "confined" | "unconfined" | "leaky" | "fractured" | "unknown",
  "extracted_method": "method name or null"
}}"""


CLARIFICATION_PROMPT = """The user's request is unclear. Ask a helpful clarifying question to understand if they want:
1. To learn about pumping test methods (consultation)
2. To analyze actual test data (calculation)

Be friendly and concise. Reference what they said: "{user_message}"

Respond with a short clarifying question (2-3 sentences maximum)."""


UNKNOWN_INTENT_RESPONSE = """I'm HydroAssist, a specialized consultant for hydrogeological pumping test analysis.

I can help you with:
- Understanding pumping test analysis methods (Theis, Cooper-Jacob, Hantush-Jacob, etc.)
- Explaining aquifer characterization theory
- Selecting appropriate methods for different aquifer conditions
- Discussing assumptions and limitations

Your message seems to be outside my area of expertise. Could you ask a question about pumping test analysis or aquifer characterization?"""
