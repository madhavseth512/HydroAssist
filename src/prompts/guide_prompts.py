"""What the guide is told, and what it is told not to do.

The consultant explains hydrogeology. The guide explains HydroAssist: where a
field is, what a note means, what to do next. Two different jobs, two different
sources, and the important rule is that neither borrows the other's authority —
an answer about where the pumping rate goes must never cite Theis (1935).
"""

GUIDE_SYSTEM_PROMPT = """You are the guide to HydroAssist, a pumping test analysis application.

Your job is to help the user work the application: where to find something, what
a field means, why the software did what it did, and what to do next.

Rules:
- Answer ONLY from the help passages and the workspace state given below. They
  describe this application as it actually is.
- If they do not cover the question, say so plainly and suggest what might. Never
  invent a button, a menu or a file format.
- If the question is about hydrogeology rather than about using the software —
  what an assumption means, which method suits an aquifer, how a solution is
  derived — say that the consultant answers those, and answer only the part that
  concerns the interface.
- Refer to what is on screen by the words the interface uses: "Step 1 — Field
  data", "Step 2", "Run analysis", "Save to project", "Download report".
- Be short. Two or three sentences answers most of these questions. Use a list
  only when the answer really is a sequence of steps.
- Never cite a paper. You are not reading papers.

Help passages:
{help_context}

The user's workspace right now:
{workspace_context}

Question:
{question}

Answer as the guide."""


NO_HELP_RESPONSE = (
    "I could not find anything in the HydroAssist help that answers that.\n\n"
    "I can explain how to get data in, fill in Step 2, read a result, compare "
    "methods, or save a project. For hydrogeology itself — what a method "
    "assumes, which one suits your aquifer — ask the consultant and it will "
    "answer from the literature."
)


def format_help_context(documents) -> str:
    """The retrieved help, labelled so the model knows what it is reading."""
    if not documents:
        return "No help passages were found for this question."

    parts = []
    for index, document in enumerate(documents, 1):
        metadata = document.get("metadata", {})
        parts.append(
            f"--- Passage {index}: {metadata.get('document', 'Help')} — "
            f"{metadata.get('section', '')} ---\n{document.get('content', '')}")
    return "\n\n".join(parts)


def format_workspace_context(state) -> str:
    """What the user actually has in front of them.

    This is what lets "what do I do next?" be answered rather than guessed, and
    "why were my points excluded?" be answered from the fit's own notes instead
    of from a general description of Cooper-Jacob.
    """
    calc_input = state.get("calculation_input") or {}
    result = state.get("calculation_result") or {}
    staged = state.get("staged_series") or {}

    lines = []

    n_points = calc_input.get("n_points") or len(staged.get("drawdown_m") or [])
    lines.append(f"Readings loaded: {n_points} points" if n_points
                 else "Readings loaded: none yet (Step 1 is empty)")

    known = [name for name, value in (("pumping rate Q", calc_input.get("Q")),
                                      ("distance r", calc_input.get("r")))
             if value]
    missing = [name for name in ("pumping rate Q", "distance r")
               if name not in known]
    if known:
        lines.append(f"Step 2 has: {', '.join(known)}")
    if missing:
        lines.append(f"Step 2 still needs: {', '.join(missing)}")

    if state.get("selected_method"):
        lines.append(f"Method selected: {state['selected_method']}"
                     + (" (confirmed)" if state.get("suggestion_confirmed") else ""))

    if result:
        if result.get("success"):
            lines.append(
                f"A fit has been run with {result.get('method', 'a method')}: "
                f"T = {result.get('T_day', 0):.4g} m2/day, "
                f"S = {result.get('S') if result.get('S') else 'not estimated'}, "
                f"R2 = {result.get('r_squared', 0):.4f}")
            notes = result.get("validity_notes") or []
            if notes:
                lines.append("Validity notes on that fit, which are the answer "
                             "to any question about excluded points or "
                             "assumptions:")
                lines.extend(f"  - {note}" for note in notes)
        else:
            lines.append(f"The last fit failed: "
                         f"{result.get('error_message', 'no reason recorded')}")
    else:
        lines.append("No analysis has been run yet.")

    # Which methods have been run, so "what have we tried?" is answerable. The
    # guide is asked this as often as the consultant is.
    history = state.get("analysis_history") or []
    if len(history) > 1:
        tried = ", ".join(row.get("method", "unknown") for row in history)
        lines.append(f"{len(history)} analyses have been run this session, in "
                     f"order: {tried}. They are all on the same readings.")

    return "\n".join(lines)
