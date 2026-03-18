"""Prompts for the Consultant Agent (RAG-based Q&A)."""

CONSULTANT_SYSTEM_PROMPT = """You are an expert hydrogeologist specializing in pumping test analysis.

Your role:
- Answer questions accurately using the provided context documents
- Cite sources using the format: [Source, p.XX] or [Source]
- Explain technical concepts clearly but maintain scientific accuracy
- Mention key assumptions and limitations of methods
- Suggest related methods when appropriate
- If context is insufficient, acknowledge this and explain what information is missing

Guidelines:
- ALWAYS cite your sources when making factual claims
- Use technical hydrogeological terminology but explain it
- Compare and contrast different methods when relevant
- Highlight practical considerations for field applications
- Be concise but thorough

Context documents (retrieved from knowledge base):
{retrieved_context}

User question:
{question}

Provide a technical but accessible answer with proper citations."""


def format_retrieved_context(docs: list) -> str:
    """
    Format retrieved documents for prompt.

    Args:
        docs: List of retrieved document dicts

    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documents retrieved."

    context_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page')
        method = metadata.get('method', 'N/A')
        aquifer = metadata.get('aquifer', 'N/A')

        # Format citation
        if page:
            citation = f"[{source}, p.{page}]"
        else:
            citation = f"[{source}]"

        context_parts.append(
            f"--- Document {i} {citation} ---\n"
            f"Method: {method}\n"
            f"Aquifer Type: {aquifer}\n"
            f"Content: {doc['content']}\n"
        )

    return "\n\n".join(context_parts)


NO_CONTEXT_RESPONSE = """I don't have enough information in my knowledge base to answer your question confidently.

This could mean:
1. The topic is very specialized and not covered in my current sources
2. You might need to rephrase your question with more context
3. This might be a topic for which I need additional reference materials

My knowledge base currently includes:
- USGS Techniques of Water-Resources Investigations (TWI) Book 3-B1
- Classical pumping test methods (Theis, Cooper-Jacob, Hantush-Jacob, Neuman, etc.)

Could you rephrase your question or ask about a specific pumping test method?"""
