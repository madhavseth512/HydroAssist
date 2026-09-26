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
{analysis_context}
User question:
{question}

Provide a technical but accessible answer with proper citations."""


def format_analysis_context(result: dict, calc_input: dict = None) -> str:
    """Format a completed calculation for the consultant prompt.

    Returns an empty string when no analysis has been run, so the prompt reads
    naturally in the pure-consultation case. When a result exists the consultant
    can answer questions about the user's own numbers ("is this S plausible?")
    instead of only about theory.
    """
    if not result:
        return ""

    if not result.get("success", False):
        return (
            "\nThe user's most recent analysis FAILED to converge: "
            f"{result.get('error_message', 'unknown error')}\n"
            "If they ask about it, help them diagnose the cause.\n"
        )

    lines = [
        "",
        "The user has ALREADY RUN an analysis on their own field data.",
        "Refer to these numbers directly when the question is about their results:",
        f"  Method:        {result.get('method', 'unknown')}",
        f"  Transmissivity T = {result.get('T', 0):.4e} m2/s "
        f"({result.get('T_day', 0):.2f} m2/day)",
        (f"  Storativity   S = {result.get('S', 0):.4e}"
         if not result.get("parameters")
         or any(p.get("name") == "S" for p in result["parameters"])
         else "  Storativity   S = not determined by this method"),
        f"  Goodness of fit: R2 = {result.get('r_squared', 0):.4f}, "
        f"RMSE = {result.get('rmse', 0) * 1000:.2f} mm",
    ]

    if calc_input:
        lines.append(
            f"  Test setup:    Q = {calc_input.get('Q', 0) * 86400:.1f} m3/day, "
            f"r = {calc_input.get('r', 0):.1f} m, "
            f"n = {calc_input.get('n_points', 0)} points"
        )

    notes = result.get("validity_notes", [])
    if notes:
        lines.append("  Validity notes produced by the calculator:")
        lines += [f"    - {n}" for n in notes]

    lines.append("")
    return "\n".join(lines)


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

    from src.retrieval import sources as source_names

    for i, doc in enumerate(docs, 1):
        metadata = doc.get('metadata', {})
        method = metadata.get('method', 'N/A')
        aquifer = metadata.get('aquifer', 'N/A')

        # The citation the model will copy into its answer. It used to be built
        # from the folder the paper was filed in, so answers cited "[Custom_
        # Fractured Aquifers, p.5]" — a reader could not tell what had been read,
        # let alone go and check it.
        citation = source_names.resolve(metadata).citation

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
