# Streamlit → React parity

Every feature of the Streamlit app (`app.py`), and where it lives in the React workspace. This was checked before removing Streamlit (plan Phase 5, M4): once `app.py` is deleted, anything missing here would be lost silently.

**Status legend**
- **ported** — the React app does it.
- **changed** — it does it differently, and the reason is given.
- **dropped** — deliberately not ported, and the reason is given.

## Sidebar → `ProjectSidebar`, header, `ChatDrawer`

| Streamlit | React | Status |
|---|---|---|
| Context metrics: aquifer, message count | Badges in `ChatDrawer`: aquifer, the settled method, "result loaded" | changed — shown where the conversation is |
| Intent, detected method, "✅ confirmed" | Method badge with ✓ once confirmed (`ChatDrawer`) | changed — intent dropped; it is an internal routing label |
| "Analysis loaded — the consultant can discuss it" | "result loaded" badge (`ChatDrawer`) | ported |
| Project title and summary | Title plus summary; the summary badge also appears in the header | ported |
| Open a project file | **Open** (`ProjectSidebar`) | ported |
| Active-test selector | **Active test** select | ported |
| Project details: title, client, number, location, date, analyst, notes | **Project and test details** → Save details | ported |
| Test name, pumping well, observation well | Same section; saved when a field loses focus | ported |
| Add test | ➕ button | ported |
| Save (download the project) | **Save** | ported |
| Start a new project (with a warning) | **New project**, which asks before discarding | ported |
| Reset conversation | **Reset conversation** (`ChatDrawer`) | ported |
| Export chat | **Export** (`ChatDrawer`) | ported |

## Consultation tab → `ChatDrawer`

| Streamlit | React | Status |
|---|---|---|
| Message history | Message list | ported |
| Welcome text with an example question | Empty-state text | ported |
| Suggestion card: method, reason, confidence | Suggestion card | ported |
| "Use this method" / choose another and "Use selected instead" | "Use this method" / other method + "Use selected" | ported |
| Warning when the aquifer type has no calculator | Yellow alert, with nothing to accept | ported |
| "Method set to X — preselected in the workspace" | The method badge turns to ✓; the setup form preselects it | changed — no separate banner |
| Retrieved sources with metadata and preview | **Sources consulted** accordion | ported |
| Chat input | Textarea: Enter sends, Shift+Enter adds a new line | ported |
| — | Notice that the free tier may use messages to improve Google's products | new (required by the $0 plan) |

## Step 1 → `DataPanel`

| Streamlit | React | Status |
|---|---|---|
| CSV upload | **Choose CSV** / **Replace CSV** | ported |
| Detected columns, units and confidence | Badges | ported |
| Column-detection issues | Yellow alerts | ported |
| Conversions applied | Accordion | ported |
| Rows removed | Accordion | ported |
| "Using N readings saved in this project from …" | "From <file>" under the summary | ported |
| Summary: point count, time and drawdown range | Summary line | ported |
| Preview of the first 10 rows | Accordion table | ported |
| Diagnostic derivative plot, regime, evidence, notes, consistent methods | Diagnosis card with the figure | ported |
| Validation errors, warnings, data notes; "cannot be analysed as-is" | Red and yellow alerts, a notes accordion, and the unusable message | ported |

## Steps 2–3 → `SetupPanel`

| Streamlit | React | Status |
|---|---|---|
| Test type | Segmented control | ported |
| Pumping rate Q and its unit | Number input and unit select | ported |
| Observation well distance r | Number input | ported |
| Method select with registry help | Select, with the chosen method's help below it | ported |
| "Not offered for this dataset: …" | Same caption | ported |
| Pumped-well notice and r_w / r_c inputs | Alert and two inputs, shown only for pumped-well methods | ported |
| α caption | Caption under the radii | ported |
| Recovery: pumping duration t_p and time-column clock | Number input and segmented control | ported |
| "Q = … m³/s" | Caption, from factors checked against the server by `tests/unit/test_frontend_constants.py` | ported |
| "Upload a CSV in Step 1 to enable analysis" | "Upload readings in Step 1 to run an analysis" | ported |
| Stehfest numerical-inversion note | — | **dropped** — Streamlit showed it only for Papadopulos-Cooper, although Hantush-Jacob, Boulton and Warren-Root use the same inversion. Each method's registry help describes the method, so a note tied to one of them would mislead. |
| Run analysis, with success and failure messages | **Run analysis**; the result or the error appears in the results panel | ported |

## Results and Step 5 → `ResultsPanel`

| Streamlit | React | Status |
|---|---|---|
| Metrics: T (m²/day), T (m²/s), S or "not determined", R² | Metric cards; the intervals and RMSE are shown on the cards | ported |
| Parameter table: derived, at bound, low reliability | Table with badges; the reliability note is shown on hover | ported |
| "Interpret these results in chat" | **Interpret in chat**, which asks the same question | ported |
| T and S 95% CI, RMSE | On the metric cards | ported |
| Validity notes | Accordion | ported |
| Save this analysis to the test (once) | **Save to project**, which becomes "Saved to project" | ported |
| Generate and download the PDF report | **Download report** — one click | changed — no separate generate step |
| Fit diagnostic plot | **Fit diagnostic** tab | ported |
| Manual matching: T and S sliders, manual vs automatic RMSE | **Manual matching** tab; S is hidden for methods without storativity | ported — the figure is redrawn when a slider is released |
| "Accept manual values as final estimate" | — | **dropped** — in Streamlit it showed a success message and saved nothing, so a React copy would be a button that does nothing. Recording a manual estimate against the project is a feature still to design. |

## Comparison → `ComparePanel`

| Streamlit | React | Status |
|---|---|---|
| Saved analyses ranked by AICc: ΔAICc, weight, evidence | Table with evidence badges | ported |
| Explanation of the ranking | Same caption | ported |
| Count of stale analyses | Same note | ported |
| — | Remove an analysis | new |

## Verified by

- **Unit tests:** 48+ Vitest tests, one or more per row above.
- **Browser tests:** the Playwright tests in `frontend/e2e/`, run against the real API.
- **API tests:** `tests/api/`, which re-express the Streamlit AppTest behaviours.
