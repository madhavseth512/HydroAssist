"""Application actions with no user interface attached.

Everything here is what the workspace *does* — prepare an uploaded file, run a
fit, record it against a project, build the report — expressed over the existing
calculator, project and data modules and nothing else.

It exists because `app.py` had grown its own copies of that logic, which made it
reachable only from Streamlit. These modules are the seam the HTTP API and the
Streamlit script both sit on (plan Phase 5, M1), and no module in this package
may import a UI framework; a test enforces that.
"""
