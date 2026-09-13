"""Application actions with no user interface attached.

Everything here is what the workspace *does* — prepare an uploaded file, run a
fit, record it against a project, build the report — expressed over the existing
calculator, project and data modules and nothing else.

It exists because `app.py`, the Streamlit script this project started with, had
grown its own copies of that logic, reachable only from Streamlit. These modules
are what the HTTP API sits on, and no module in this package may import a UI
framework; a test enforces that.
"""
