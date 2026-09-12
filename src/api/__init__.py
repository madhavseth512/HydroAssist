"""The HTTP API that replaces the Streamlit script.

`src/services` holds what the application does; this package exposes it over
HTTP and owns the one thing a web server has to decide for itself — what
belongs to a visitor and what is shared.

Per visitor (`sessions.py`): the conversation, the open project, the last
result. Shared and stateless: the calculators, the registry, the vector store,
the embedding model.
"""
