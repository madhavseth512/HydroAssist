# HydroAssist — Intelligent Hydrogeological Analysis Platform

**MTP Part 1, IIT Kharagpur.** A browser workspace that analyses aquifer pumping
tests and tracer tests, with a consultant grounded in the hydrogeology literature
that explains, and argues for, every result.

**Live:** <https://hydroassist.onrender.com> (password protected — see DEPLOY.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/MTP%20part%201-complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-1835%20python%20%2B%20187%20frontend%20%2B%2011%20browser-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)]()
[![Methods](https://img.shields.io/badge/methods-9%20across%20pumping%20and%20tracer%20tests-blue.svg)]()

---

## Overview

HydroAssist analyses two kinds of aquifer test:

- **Pumping tests** — drawdown against time, giving transmissivity (T) and
  storativity (S), with the flow regime read off the data first.
- **Tracer tests** — concentration against time, giving velocity or travel time,
  longitudinal dispersivity and effective porosity, which is what contaminant
  transport and wellhead protection depend on.

What it adds to a commercial curve-fitting package is the second half of the job:
every result comes with the assumptions behind the method, the ones the data
violate, the papers those conditions come from, and a comparison against the
other methods that could have been used.

### What was built, phase by phase

**Phase 1 — Consultant**
- RAG chatbot on pumping-test theory, method selection and aquifer characterisation
- Grounded in 50 hydrogeology papers (4,239 passages), cited by document and page
- Intent classification and routing through a LangGraph state machine
- Retrieval filtered by aquifer type and by the method a question names

**Phase 2 — Calculator**
- Data intake with column detection, unit conversion and validation — CSV,
  delimited text, Excel, pasted columns, Solinst / Van Essen Diver / In-Situ logger
  exports, and OCR of a photographed field sheet
- Theis (1935), Cooper-Jacob (1946), Papadopulos-Cooper (1967)
- T and S with 95% confidence intervals, interactive type curves, residuals,
  manual curve matching and a PDF report

**Phase 3 — Integrated workflow**
- The consultant proposes a method from the conversation; the user decides
- Completed analyses flow back into the chat, so the consultant interprets the
  user's own result
- `DataValidator` decides which methods the workspace offers for a dataset

**Phase 4 — Extensible calculator**
- A new method is a model plus a parameter specification, not a new module
- All four aquifer types: confined, leaky (Hantush-Jacob), unconfined (Boulton),
  fractured (Warren-Root), plus Theis recovery
- Vectorised Stehfest Laplace engine — Papadopulos-Cooper fitting from 10–30 s to 0.04 s
- **Bourdet log-derivative**, which reads the flow regime off the data and drives
  method selection
- Methods compared by AICc, not R²

**Phase 5 — Web application and deployment**
- FastAPI back end and a React + TypeScript workspace, replacing Streamlit
- Projects with several tests each, saved as a file, restored after a reload
- Consultant answers streamed as they are written, with the passages they used
- Rate limits and a fallback, so the calculators work without the language model
- Deployed at no cost on Render, behind a password

**Phase 6 — Tracer tests (closes MTP Part 1)**
- A concentration column switches a test to *Tracer* by itself; concentrations
  are never unit-converted
- **Ogata-Banks (1961)**, natural gradient: velocity, longitudinal dispersivity,
  dispersion coefficient, Péclet number, and n_e = K·i/v
- **Thiéry (2022)**, forced gradient, radially convergent: travel time,
  dispersivity, n_e from Q, b and r, and mass recovery. Thiéry's closed form
  reproduces the exact radial solution for Péclet 1–1000; Sauty's (1980) older
  approximation overstates dispersivity by a median 69% on field tests, so it is
  not used
- Pulse or continuous injection and a background concentration; validity rules for
  a missed peak, a truncated tail, an unreached plateau, a low Péclet number, and
  implausible porosity or recovery
- Results, manual matching, comparison, PDF report, saved projects and the
  consultant all speak in tracer quantities
- Checked against Thiéry's published equations and against an independent
  numerical solution of the radial advection-dispersion equation: α_L within about
  5% and t_c within 4% for Péclet 3–50

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│   React workspace (frontend/)  ──HTTP──▶  FastAPI (src/api)      │
│   project · data · setup · results · compare · consultant drawer │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                        │
│   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐ │
│   │ Manager Agent│──▶│Consultant Agent│   │ Calculator Agent │ │
│   │ (routing +   │   │ (RAG Q&A +     │   │ (reports results,│ │
│   │  suggestion) │   │  result interp)│   │  requests inputs)│ │
│   └──────────────┘   └────────────────┘   └──────────────────┘ │
│           ▲                                        │             │
│           └──── method_suggestion / calculation_result ─────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────────────┐
          ▼                ▼                        ▼
   ┌────────────┐   ┌────────────────┐   ┌─────────────────────┐
   │   Gemini   │   │  RAG Pipeline  │   │  Calculator Engine  │
   │  2.5 Flash │   │  ChromaDB +    │   │  scipy / Stehfest   │
   │    LLM     │   │  ONNX MiniLM   │   │  + Plotly + PDF     │
   └────────────┘   └────────────────┘   └─────────────────────┘
```

### Agent State

All agents share a typed `AgentState` with these key fields:

| Field | Type | Purpose |
|---|---|---|
| `messages` | `List[AnyMessage]` | Full conversation history |
| `user_intent` | `str` | consultation / calculation / clarification |
| `aquifer_context` | `str` | confined / unconfined / leaky / fractured |
| `selected_method` | `str` | Method extracted from user message |
| `method_suggestion` | `Dict` | Structured suggestion card for UI |
| `suggestion_confirmed` | `bool` | Gates the calculator workspace |
| `calculation_input` | `Dict` | Q, r, method, n_points — written by the workspace |
| `calculation_result` | `Dict` | T, S, CIs, R², validity notes — written by the workspace, read by the consultant |

### The chat ↔ calculator bridge

The workspace runs outside the LangGraph (it needs a file upload and form
inputs), so the two halves exchange state through three seams on
`HydroAssistChat`:

| Seam | Direction | Effect |
|---|---|---|
| `method_suggestion` | chat → workspace | Manager recommends a method; the card renders in the chat tab |
| `confirm_method()` | chat → workspace | Sets `selected_method` + `suggestion_confirmed`; the workspace dropdown preselects it, and aquifer-filtered retrieval activates |
| `record_calculation()` | workspace → chat | Publishes the fitted T, S and validity notes so the consultant can interpret them |

---

## Project Structure

```
hydroassist/
├── frontend/                       # React + TypeScript workspace (Vite, Mantine)
├── Dockerfile, docker-compose.yml  # Container build + Caddy (see DEPLOY.md)
├── requirements.txt
├── configs/                        # YAML configs
├── src/
│   ├── api/                        # FastAPI app: sessions, routes, rate limit
│   ├── services/                   # Workspace actions with no UI (analysis, intake)
│   ├── project/                    # Project model and file format
│   ├── core/
│   │   ├── config.py               # AppConfig dataclass (Pydantic)
│   │   ├── state.py                # AgentState TypedDict
│   │   └── graph.py                # LangGraph workflow compilation
│   ├── agents/
│   │   ├── base.py                 # Abstract BaseAgent
│   │   ├── manager.py              # Intent classification + routing + suggestion
│   │   ├── consultant.py           # RAG Q&A agent (also interprets results)
│   │   ├── calculator.py           # Reports results / requests missing inputs
│   │   └── method_selection.py     # Deterministic method-suggestion rules
│   ├── retrieval/
│   │   ├── embeddings.py           # ONNX all-MiniLM-L6-v2, via chromadb
│   │   ├── vectorstore.py          # ChromaDB manager
│   │   └── retriever.py            # Metadata-filtered semantic search
│   ├── ingestion/
│   │   ├── pipeline.py             # Ingestion orchestration
│   │   ├── processors/             # PDF + HTML processors, chunker
│   │   └── scrapers/               # USGS + AQTESOLV scrapers
│   ├── data/
│   │   ├── csv_inspector.py        # Column detection + unit inference (3-layer)
│   │   ├── formatter.py            # Unit conversion → standard schema
│   │   └── validator.py            # Physical plausibility + method suitability
│   ├── calculator/
│   │   ├── registry.py             # THE method table — one entry per method
│   │   ├── base.py                 # CalculationInput / CalculationResult
│   │   ├── geometry.py             # Aquifer, well, aquitard, fracture, schedule
│   │   ├── parameters.py           # FittedParameter — value, CI, reliability
│   │   ├── fitting.py              # curve_fit, CIs, bound detection, AIC/AICc
│   │   ├── validation.py           # Shared input checks
│   │   ├── rules.py                # Validity rules as data
│   │   ├── templates.py            # CurveFitCalculator — the fixed eight steps
│   │   ├── laplace.py              # Vectorised Stehfest inversion
│   │   ├── laplace_solutions.py    # Shared kernel + one q(p) per method family
│   │   ├── derivative.py           # Bourdet derivative + flow-regime diagnosis
│   │   ├── theis.py                # Theis (1935)
│   │   ├── cooper_jacob.py         # Cooper-Jacob (1946)
│   │   ├── papadopulos_cooper.py   # Papadopulos-Cooper (1967)
│   │   ├── methods/
│   │   │   ├── hantush_jacob.py    # Hantush-Jacob (1955) — leaky
│   │   │   ├── boulton.py          # Boulton (1963) — unconfined
│   │   │   ├── warren_root.py      # Warren-Root (1963) — fractured
│   │   │   └── tracer.py           # Ogata-Banks (1961), Thiéry (2022) — tracer tests
│   │   ├── plots.py                # Plotly diagnostic plots
│   │   └── report.py               # ReportLab PDF report generation
│   └── prompts/                    # LLM prompts for each agent
├── tests/
│   ├── unit/                       # calculators, data pipeline, retrieval, project model
│   ├── validation/                 # published well-function tables and field datasets
│   ├── integration/                # agents, graph, retrieval regression
│   ├── api/                        # HTTP API tests (FastAPI TestClient)
│   └── *_sample.csv                # sample datasets, one per method family
├── frontend/e2e/                   # Playwright browser tests
├── knowledge source/               # the PDF corpus, one folder per aquifer type
└── data/
    ├── raw/                        # USGS PDFs, AQTESOLV metadata
    ├── processed/                  # Chunked documents
    └── vectordb/                   # ChromaDB persistence
```

---

## Phase 2 — Calculator

### Supported Methods

| Method | Aquifer | Estimates | Valid when |
|---|---|---|---|
| **Theis** (1935) | confined | T, S | All times; the general full-curve method |
| **Cooper-Jacob** (1946) | confined | T, S | u < 0.02 — late-time straight line (paper p. 3) |
| **Papadopulos-Cooper** (1967) | confined | T, S | Large-diameter well; data from the pumped well |
| **Hantush-Jacob** (1955) | leaky | T, S, B | Vertical leakage through an aquitard with no storage of its own |
| **Boulton** (1963) | unconfined | T, S, S_y, α | Delayed gravity drainage from the water table |
| **Warren-Root** (1963) | fractured | T, S, ω, λ | Double porosity — fractures transmit, matrix stores |
| **Theis recovery** (1935) | confined | T | Residual drawdown after pumping stops (Eq. 7) |
| **Ogata-Banks** (1961) | porous, tracer | v, α_L, D_L, n_e | Natural-gradient tracer test, uniform flow |
| **Thiéry** (2022) | porous, tracer | t_c, α_L, n_e, recovery | Forced-gradient tracer test, radially convergent; Péclet 1–1000 |

Every method must pass three acceptance gates before it is merged:

1. **An exact analytical limit** it collapses to when its extra physics is
   switched off — Hantush-Jacob → Theis as leakage vanishes, Boulton → Theis with
   S + S_y as drainage becomes instantaneous, Warren-Root → Theis as all storage
   moves into the fractures, Papadopulos-Cooper → Theis as the casing radius
   shrinks. All computed in closed form, so none depends on a transcribed table.
2. **Round-trip recovery** of known parameters from synthetic noisy data.
3. **Physics invariants** — positivity, monotonicity, linearity in Q, the correct
   direction of response to each parameter.

### Diagnostic-first workflow

The **Bourdet log-derivative** (`ds/d ln t`) is computed as soon as a CSV is
uploaded, before any method is chosen — which is the order modern well-test
practice actually works in. Drawdown curves for different aquifer types all look
like smooth rising lines; their derivatives do not:

| Derivative shape | Flow regime | Method |
|---|---|---|
| Flat | Infinite-acting radial | Theis, Cooper-Jacob |
| Unit slope, early, lying on the drawdown | Wellbore storage | Papadopulos-Cooper |
| Falls and stays low | Leakage or recharge boundary | Hantush-Jacob |
| Dips then recovers | Double porosity | Warren-Root |
| Two plateaus with a dip between | Delayed yield | Boulton |
| Doubles at late time | No-flow boundary | *not implemented* |
| Half slope | Linear flow to a vertical fracture | *not implemented* |

On a radial plateau the derivative height is exactly `Q/(4πT)`, so **T can be read
off the plot without fitting anything** — the implementation recovers it to 0.001%
on synthetic data. That reading also drives `suggest_method`: a user who describes
a "confined aquifer test" over a curve that plainly shows leakage is steered to
Hantush-Jacob rather than Theis.

### Tracer tests

```
natural gradient (Ogata-Banks 1961), at distance x downstream:
    pulse       C = A / sqrt(4 pi D t) * exp(-(x - vt)^2 / 4Dt),           D = alpha_L v
    continuous  C = C0/2 [erfc((x - vt)/2sqrt(Dt)) + exp(vx/D) erfc((x + vt)/2sqrt(Dt))]

forced gradient, radially convergent (Thiéry 2022, unbounded domain):
    t_c = pi r^2 b n_e / Q,   t_D = t / t_c,   x = f_T t_D,   P' = (r/alpha_L) / f_P
    pulse (eqs 21-22)       C = (M/Q)/t_c * f_T/2 sqrt(P'/4 pi x) (x+1)/x exp(-P'(1-x)^2/4x)
    continuous (eqs 15-16)  C = C0/2 erfc(sqrt(P'/4x) (1-x))
```

f_T and f_P are Thiéry's correction factors, functions of the Péclet number alone.
The radial method needs Q; thickness b gives effective porosity, and the injected
mass gives the fraction recovered. The natural-gradient method needs no pumping
rate; hydraulic conductivity and gradient give effective porosity.

### Four-Step Workflow

```
STEP 1 — Upload CSV
  ↓  CSVInspector → 3-layer detection (regex → statistics → LLM fallback)
  ↓  DataFormatter → unit conversion, t=0 removal, deduplication
  ↓  Preview: column map, conversions applied, data summary

STEP 2 — Well Parameters
  ↓  Q (value + unit), r, method selection
  ↓  r_w and r_c inputs shown conditionally (Papadopulos-Cooper only)

STEP 3 — Run Analysis
  ↓  scipy.optimize.curve_fit (Theis, CJ) or Stehfest inversion (PC)
  ↓  Returns T, S, 95% CIs, R², RMSE, validity notes

STEP 4 — Results
     ├── Metric cards: T (m²/day), T (m²/s), S, R²
     ├── Confidence intervals + RMSE
     ├── Plotly interactive type curve + residuals panel
     ├── Validity & Interpretation (expandable notes)
     ├── 💬 Interpret these results in chat
     └── 📄 Download PDF Report
```

### CSV Pipeline

```
CSV upload
  ↓
CSVInspector   — 3-layer detection: regex → statistics → LLM fallback
  ↓
DataFormatter  — unit conversion (min→s, ft→m), t=0 removal, deduplication
  ↓
DataValidator  — min points, monotonic time, log-cycle span, method suitability
  ↓
Standard schema: time_s | drawdown_m | well_id (optional)
```

### Theis (1935) Calculator

Key implementation decisions grounded in the original paper:

- **W(u) computation**: `scipy.special.expn(1, u)` — exact computation of the Theis well function
- **u guard**: `np.maximum(u, 1e-10)` — prevents divergence as u → 0 (paper Eq. 3 series diverges at −ln(u) → ∞)
- **Fitting**: `scipy.optimize.curve_fit` with Trust Region Reflective; `ftol=xtol=gtol=1e-12` (tightened — T and S span 4–5 orders of magnitude)
- **Bounds**: T ∈ [10⁻⁹, 1.0] m²/s (covers karst at 86 400 m²/day); S ∈ [10⁻⁹, 10⁻²] (confined only)

**Smoke test (T=2.5×10⁻³ m²/s, S=5×10⁻⁵, σ=5 mm noise, 60 points):**

| Metric | Result |
|--------|--------|
| T recovery error | < 0.1% |
| S recovery error | < 1% |
| R² | > 0.999 |

### Cooper-Jacob (1946) Calculator

Key implementation decisions grounded in the original paper:

- **Validity threshold**: `u < 0.02` — paper p.3: *"tolerable where u is less than about 0.02"* (stricter than textbook 0.05)
- **Equations 8 & 9**: `T = 2.303Q / (4π × Δs)`, `S = 2.25 × T × t₀ / r²`
- **Iterative data selection**: Converges to a self-consistent (T, S, u-filter) triple — not a fixed 2-pass; typically converges in 2–3 iterations
- **Post-fit u check**: Authoritative recheck using final T, S on all fitted points

**Smoke test (same synthetic data as Theis):**

| Metric | Result |
|--------|--------|
| T recovery error | < 5% |
| S recovery error | < 30% |
| R² on filtered points | > 0.97 |

### Papadopulos-Cooper (1967) Calculator

Key implementation decisions grounded in the original paper:

- **Core equation (Eq. 12)**: `s_w = Q/(4πT) × F(u_w, α)` where `α = r_w²S/r_c²`
- **Numerical method**: Stehfest (1970) N=12 Laplace inversion of the analytical Laplace-domain formula (paper Eq. 10), using K₀, K₁ modified Bessel functions (`scipy.special.kv`) — replaces direct integration of the oscillatory Eq. 13 which gave 23–115% errors in testing
- **Fitting**: Same `curve_fit` TRF as Theis with `ftol=xtol=gtol=1e-10`
- **Critical limitation (paper p. 244)**: S has order-of-magnitude uncertainty; T is the reliable output — always surfaced in validity notes

**Smoke test (T=1.0×10⁻³ m²/s, S=1.0×10⁻⁴, σ=3 mm noise, 50 points):**

| Metric | Result |
|--------|--------|
| T recovery error | < 0.1% |
| S recovery error | < 1% |
| R² | > 0.999 |
| Late-time Theis convergence error | 0.02% |

---

## Tests

1,836 Python tests (1,835 pass, 1 skipped), 187 frontend unit tests and 11
Playwright browser tests, all passing on the commit that is deployed.

```bash
pytest tests/ -q                 # backend, API and integration
pytest tests/ -q -m "not slow"   # skip the slow end-to-end tests

cd frontend
npm test                         # component tests (Vitest + Testing Library)
npm run build && npm run e2e     # browser tests (Playwright, real API, consultant off)
```

Run through `python -m pytest` if the venv's `.exe` shims carry a stale path.

### Is the mathematics right?

| Check | Tests | What it compares against |
|------|------|------|
| `validation/test_published_values.py` | 46 | W(u) and W(u, r/B) as printed in the literature (AQTESOLV guide; Hantush 1956 via Walton's Bulletin 49) |
| `validation/test_field_datasets.py` | 22 | Real published field tests against the analyst's own answer — e.g. Walton (1962), Gridley, within 1.9% |
| `unit/test_tracer.py` | 96 | Thiéry's equations written out from the paper; an independent Crank-Nicolson solve of the radial equation |
| `unit/test_golden_regression.py` | 20 | The shipped fits, pinned to 1e-9 |
| `unit/test_laplace.py` | 47 | The inverter against known transform pairs and every analytical limit |

### Calculators

| File | Tests |
|------|-------|
| `test_tracer.py` | 96 |
| `test_method_conformance.py` | 75 — every registered method obeys the same contract |
| `test_theis.py` | 46 |
| `test_papadopulos_cooper.py` | 41 |
| `test_hantush_jacob.py` | 38 |
| `test_boulton_warren_root.py` | 38 |
| `test_template_ports.py` | 31 |
| `test_theis_recovery.py` | 23 |
| `test_cooper_jacob.py` | 17 |

### Shared machinery

| File | Tests | Covers |
|------|-------|--------|
| `test_registry.py` | 97 | Method table integrity, alias resolution, import weight |
| `test_geometry.py` | 65 | Geometry invariants and backward compatibility |
| `test_validation_rules.py` | 44 | Shared input checks and the validity-rule library |
| `test_fitting.py` | 42 | `fit_curve`, CIs, bound detection, AIC/AICc, log-scale |
| `test_parameters.py` | 32 | Bidirectional `CalculationResult` reconciliation |
| `test_derivative.py` | 29 | Bourdet derivative and flow-regime diagnosis |

### Data, projects, retrieval

| File | Tests |
|------|-------|
| `test_method_selection.py` | 84 |
| `test_ingestion.py` | 63 |
| `test_services.py` | 61 |
| `test_plots_report.py` | 41 |
| `test_csv_inspector.py` | 40 |
| `test_table_ocr.py` / `test_table_reader.py` | 39 / 36 |
| `test_project.py` / `test_project_workspace.py` / `test_project_runs.py` | 36 / 23 / 14 |
| `test_logger_import.py` | 29 |
| `test_formatter.py` / `test_validator.py` | 26 / 24 |
| `test_corpus_methods.py` | 24 |

### Integration and API

| File | Tests | What it covers |
|------|-------|----------------|
| `tests/api/` | 202 | The HTTP API — sessions, projects, fits, tracer runs, chat, auth, rate limit, resilience |
| `test_end_to_end_journey.py` | 50 | Full journey through the compiled LangGraph |
| `test_chat_calculator_bridge.py` | 26 | Agents with stubbed LLM + retriever |
| `test_method_filtered_retrieval.py` | 21 | Method tags narrow the search without silencing it |

No test needs a network connection or an API key. The LLM and vector store are
stubbed; the calculators, graph, routing and state reducers are real.

### Golden-fit regression

`tests/fixtures/golden_fits.json` pins the exact T, S, RMSE and confidence
intervals the original three calculators produced before the architecture
refactor, at 1e-9 relative tolerance. Steps of the migration that are supposed to
be pure code moves must reproduce them; a change means the optimisation moved, not
just the code. The one intentional re-baseline — Papadopulos-Cooper, when
`scipy.special.kv` was replaced by the exponentially scaled `kve` — is recorded in
the fixture's `_provenance` block with the evidence for why the new values are the
more accurate ones.

### Sample Data Files

| File | Method | Parameters |
|------|--------|-----------|
| `tests/theis_cooper_jacob_sample.csv` | Theis / Cooper-Jacob | T=2.5×10⁻³ m²/s, S=5×10⁻⁵, Q=500 m³/day, r=50 m |
| `tests/papadopulos_cooper_sample.csv` | Papadopulos-Cooper | T=1.0×10⁻³ m²/s, S=1.0×10⁻⁴, Q=300 m³/day, r_w=0.30 m, r_c=0.20 m |
| `tests/theis_recovery_sample.csv` | Theis recovery | residual drawdown after pumping stops |
| `tests/tracer_natural_sample.csv` | Ogata-Banks | natural gradient, x=25 m |
| `tests/tracer_convergent_sample.csv` | Thiéry | pulse, r=15 m, Q=500 m³/day, b=10 m; t_c=67.86 h, α_L=1 m, n_e=0.2, 80% of 5 kg recovered |

The two tracer samples are synthetic, generated from the models themselves; they
prove the fitting, not the physics. The physics is checked against the radial
equation solved numerically.

---

## Quick Start

### 1. Clone and Setup

> **On Windows, turn on long paths first.** Several papers in `knowledge source/`
> have names long enough that, inside a deep directory, the full path exceeds the
> 260-character limit and `git clone` fails with `Filename too long` — leaving a
> repository whose working tree is only half checked out.
>
> ```bash
> git config --global core.longpaths true
> ```
>
> Cloning into a short path such as `C:\dev\` also avoids it. Linux and macOS
> are unaffected, which is why deployment hosts never see this.


```bash
git clone https://github.com/madhavseth512/HydroAssist.git
cd HydroAssist

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Only to rebuild the knowledge base, or to run tests/unit/test_ingestion.py:
pip install -r requirements-ingest.txt
```

### 2. Configure Environment

```bash
echo GEMINI_API_KEY=your_api_key_here > .env
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 3. Ingest Knowledge Base

The shipped vector store is built from the PDF corpus under `knowledge source/`,
organised by aquifer type:

```bash
python scripts/ingest_knowledge_source.py
# Chunks every PDF under "knowledge source/", tags each chunk with the aquifer
# type taken from its folder, and populates ChromaDB.
# Takes ~5–10 minutes on first run.
```

Current contents — 50 papers, 4,239 chunks:

| Source folder | Aquifer tag | Chunks |
|---|---|---|
| Unconfined Aquifer | `unconfined` | 1,358 |
| General | `general` (237 pinned to `unconfined` by the manifest) | 919 |
| Confined Aquifer | `confined` | 617 |
| Leaky Confined Aquifer | `leaky` | 572 |
| Fractured Aquifers | `fractured` | 539 |
| Tracer Tests | `porous` | 222 |
| Step drawdown test | `general` | 12 |

`configs/corpus_methods.yaml` tags every paper with the methods it covers, and every
assignment has been reviewed. To add one paper without rebuilding everything:

```bash
python scripts/ingest_knowledge_source.py --only "part of its filename"
```

`scripts/ingest.py --all` is a separate path that fetches USGS TWI Book 3-B1;
it is **not** what produced the bundled store.

> **Note on the PDF corpus.** `knowledge source/` contains publisher PDFs
> (SPE, Elsevier, AGU/Wiley). These are for local research use and should not
> be redistributed — see *Known Issues* below.

### 4. Launch

```bash
# Build the browser workspace once
cd frontend && npm ci && npm run build && cd ..

# Serve the API and the built workspace together
python -m uvicorn src.api.app:app --port 8000
# Opens at http://localhost:8000
```

For frontend work with hot reload, keep the API running as above and, in a second
terminal, run `cd frontend && npm run dev` — it serves http://localhost:5173 and
proxies `/api` to the API.

---

## Configuration

### Environment Variables

```bash
GEMINI_API_KEY=your_key_here
HYDROASSIST_ENV=default          # default | development | production

# Optional overrides
HYDROASSIST_LLM_MODEL=gemini-2.5-flash
HYDROASSIST_LLM_TEMPERATURE=0.1
HYDROASSIST_CHUNK_SIZE=1000
HYDROASSIST_TOP_K=5
HYDROASSIST_LOG_LEVEL=INFO
```

### configs/default.yaml

```yaml
retrieval:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  similarity_threshold: 0.2

llm:
  provider: "gemini"
  model: "gemini-2.5-flash"
  temperature: 0.1
  max_tokens: 8192
```

---

## Roadmap

### Done in MTP Part 1
- [x] RAG consultant, LangGraph routing, filtered retrieval, cited answers
- [x] Data intake in the forms field data arrives in, including OCR of a photograph
- [x] Nine methods: Theis, Theis recovery, Cooper-Jacob, Papadopulos-Cooper,
      Hantush-Jacob, Boulton, Warren-Root, Ogata-Banks, Thiéry
- [x] Bourdet derivative driving method selection; AICc comparison
- [x] Projects, runs, saved analyses, PDF reports
- [x] React workspace, FastAPI back end, browser tests
- [x] Deployed on Render (see DEPLOY.md)
- [x] Tracer tests, natural and forced gradient

### Next
- [ ] Validate the tracer methods against field breakthrough curves — Thiéry's
      (2022) supplementary data for tests 1–8, with his fitted α_L and n_e as the answer
- [ ] Add the remaining tracer literature to the corpus: Ogata & Banks (1961),
      Sauty (1980), Moench (1989), USGS WRI 98-4126A, EPA QTRACER2
- [ ] Finite-duration injections, sorption (retardation) and decay for tracers
- [ ] Step-drawdown (Birsoy-Summers 1980)
- [ ] Neuman (1974) unconfined — needs a transcendental root series
- [ ] Slug tests — a different input shape, and Stehfest is the wrong inverter
- [ ] Forward simulation / test design
- [ ] Named wells with coordinates

---

## Known Issues

| Issue | Impact | Status |
|---|---|---|
| `knowledge source/` contains paywalled publisher PDFs tracked in git | Redistribution of copyrighted material if the repository or site were made public | **Mitigated** — the repository is private, the PDFs are kept out of the Docker image, and the site is password protected. A licensing decision before going public |
| ~~`method` metadata is `multiple` for 3,837 of 3,843 chunks~~ | Method-level retrieval filtering was ineffective | **Fixed** — `configs/corpus_methods.yaml` tags each paper; chunks carry a flag per method and the consultant filters on it when a question names one. Every assignment has been reviewed |
| Tracer methods have no field validation yet | Checked against published equations and a numerical solution, not measured breakthrough curves | Open — see Roadmap |
| ~~The radial tracer method was labelled Sauty (1980) but used a moment approximation~~ | α_L out by up to +48% at low Péclet, +9% for continuous injection | **Fixed** — replaced by Thiéry (2022); projects saved under the old name still open |
| ~~`canonical_method("Hantush-Jacob")` resolves to Cooper-Jacob~~ | Shared surname | **Fixed** — implementing the method resolved it; the longer alias now wins |
| ~~Agarwal (recovery) and one other PDF are image-only scans~~ | They contributed **zero** chunks; the recovery paper was absent | **Fixed** — OCR (pypdfium2 + Tesseract) recovered 20/20 and 14/14 pages. Ingestion now reports any document that yields too little instead of skipping it |
| Boulton, Warren-Root, Papadopulos-Cooper and recovery have no published table in the corpus | They are checked against analytical limits and their own forward models, not against values published elsewhere | Open — `tests/validation/` covers Theis, Cooper-Jacob and Hantush-Jacob; the rest needs AQTESOLV or new references |
| Stehfest loses relative accuracy on decaying functions | Fine for drawdown, which grows; wrong for slug-test head recovery, which decays | Documented in `laplace.py`; slug tests will need de Hoog |
| `src/utils/citations.py`, `src/utils/metadata.py` untested | No coverage on those helpers | Open — `src/ingestion/` itself is now covered by `tests/unit/test_ingestion.py` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | HTTP API |
| React, Mantine, Plotly.js (`frontend/`) | Browser workspace |
| `langchain`, `langgraph` | Multi-agent orchestration |
| `langchain-google-genai` | Gemini 2.5 Flash LLM |
| `chromadb` | Vector store (RAG) |
| `onnxruntime` (via `chromadb`) | Local embeddings (ONNX all-MiniLM-L6-v2) |
| `scipy` | `curve_fit` (TRF fitting), `expn` (Theis W(u)), `kv` (Bessel K₀/K₁ for Stehfest) |
| `numpy` | Array operations, polyfit (CJ slope) |
| `pandas` | CSV processing and tabular data |
| `plotly` | Interactive type curves, semi-log plots, residuals |
| `matplotlib` | Static figures embedded in PDF reports |
| `reportlab` | PDF report generation |
| `pypdf` | PDF text extraction (ingestion) |
| `langchain-text-splitters` (`requirements-ingest.txt`) | Chunking, when rebuilding the knowledge base. Not installed by the app or the image. |
| `pydantic` | AppConfig validation |

---

## References

1. **Theis, C.V. (1935)**. The relation between the lowering of the piezometric surface and the rate and duration of discharge of a well using ground-water storage. *Transactions of the American Geophysical Union*, 16, 519–524.
2. **Cooper, H.H. & Jacob, C.E. (1946)**. A generalized graphical method for evaluating formation constants and summarizing well-field history. *Transactions of the American Geophysical Union*, 27(4), 526–534.
3. **Hantush, M.S. & Jacob, C.E. (1955)**. Non-steady radial flow in an infinite leaky aquifer. *Transactions of the American Geophysical Union*, 36(1), 95–100.
4. **Boulton, N.S. (1963)**. Analysis of data from non-equilibrium pumping tests allowing for delayed yield from storage. *Proceedings of the Institution of Civil Engineers*, 26, 469–482.
5. **Warren, J.E. & Root, P.J. (1963)**. The behavior of naturally fractured reservoirs. *Society of Petroleum Engineers Journal*, 3(3), 245–255.
6. **Papadopulos, I.S. & Cooper, H.H. (1967)**. Drawdown in a well of large diameter. *Water Resources Research*, 3(1), 241–244.
7. **Stehfest, H. (1970)**. Algorithm 368: Numerical inversion of Laplace transforms. *Communications of the ACM*, 13(1), 47–49.
8. **Ogata, A. & Banks, R.B. (1961)**. A solution of the differential equation of longitudinal dispersion in porous media. *USGS Professional Paper 411-A*.
9. **Sauty, J.-P. (1980)**. An analysis of hydrodispersive transfer in aquifers. *Water Resources Research*, 16(1), 145–158.
10. **Gelhar, L.W., Welty, C. & Rehfeldt, K.R. (1992)**. A critical review of data on field-scale dispersion in aquifers. *Water Resources Research*, 28(7), 1955–1974.
11. **Thiéry, D. (2022)**. Simple approximate closed-form expressions for tracer injection in aquifer with a radially converging or diverging flow field. *Journal of Hydrology*, 608, 127661.
12. **Walton, W.C. (1962)**. Selected analytical methods for well and aquifer evaluation. *Illinois State Water Survey Bulletin 49*.

---

## Acknowledgments

- USGS for public domain hydrogeological publications
- Google for Gemini API
- LangChain & LangGraph teams
- IIT Kharagpur — project supervision by Prof. Anirban Dhar

---

**Developed at IIT Kharagpur as a BTP and MTP Part 1 project**
