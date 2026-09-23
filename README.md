# HydroAssist — Intelligent Hydrogeological Analysis Platform

**BTP-1 → BTP-2**: RAG-based consultant chatbot + confined aquifer pumping test calculator, powered by LangGraph and Gemini AI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase](https://img.shields.io/badge/phase-5%20in%20progress-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-1182%20python%20%2B%2054%20frontend-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-75%25-green.svg)]()
[![Methods](https://img.shields.io/badge/methods-6%20across%204%20aquifer%20types-blue.svg)]()

---

## Overview

HydroAssist is a two-phase hydrogeology assistant targeting confined aquifer pumping test analysis.

**Phase 1 — Consultant (complete)**
- RAG-based chatbot answering questions about pumping test theory, method selection, and aquifer characterisation
- Grounded in a curated corpus of hydrogeology papers (see Knowledge Base below) plus AQTESOLV method metadata
- Metadata-filtered retrieval (aquifer type, method, section)
- Intent classification and conversation routing via LangGraph

**Phase 2 — Calculator (complete)**
- CSV upload with automatic column detection, unit conversion, and data validation
- Confined aquifer curve-fitting: Theis (1935), Cooper-Jacob (1946), Papadopulos-Cooper (1967)
- Outputs T (transmissivity) and S (storativity) with 95% confidence intervals
- Interactive Plotly diagnostic plots (type curves, residuals, manual matching)
- PDF report generation (ReportLab)

**Phase 3 — Integrated workflow (complete)**
- The consultant suggests a method from the conversation, and the workspace follows it
- Completed analyses flow back into the chat, so the consultant can interpret the user's own T and S
- `DataValidator` gates which methods the workspace offers for a given dataset

**Phase 4 — Extensible calculator architecture (in progress)**
- Refactored so a new method is a transfer function plus a parameter spec, not a new module
- **All four aquifer types covered**: confined, leaky, unconfined, fractured
- Shared Stehfest Laplace engine, vectorised — Papadopulos-Cooper fitting went from 10–30 s to 0.04 s
- **Bourdet log-derivative**, which reads the flow regime off the data and drives method selection
- 795-test suite (75% coverage)

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
│   │   │   └── warren_root.py      # Warren-Root (1963) — fractured
│   │   ├── plots.py                # Plotly diagnostic plots
│   │   └── report.py               # ReportLab PDF report generation
│   └── prompts/                    # LLM prompts for each agent
├── tests/
│   ├── unit/                       # 298 tests
│   │   ├── test_theis.py                 # 46
│   │   ├── test_cooper_jacob.py          # 17
│   │   ├── test_papadopulos_cooper.py    # 41
│   │   ├── test_csv_inspector.py         # 40
│   │   ├── test_formatter.py             # 26
│   │   ├── test_validator.py             # 24
│   │   ├── test_plots_report.py          # 34
│   │   └── test_method_selection.py      # 70
│   ├── integration/                # agents, graph, retrieval regression
│   └── api/                        # HTTP API tests (FastAPI TestClient)
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

1,182 Python tests and 54 frontend unit tests, plus 3 Playwright browser tests.

```bash
pytest tests/ -q                 # backend, API and integration
pytest tests/ -q -m "not slow"   # skip the slow end-to-end tests

cd frontend
npm test                         # component tests (Vitest + Testing Library)
npm run build && npm run e2e     # browser tests (Playwright, real API, consultant off)
```

Run through `python -m pytest` if the venv's `.exe` shims carry a stale path.

### Calculators

| File | Tests | Module coverage |
|------|-------|-----------------|
| `test_theis.py` | 46 | `theis.py` 91% |
| `test_papadopulos_cooper.py` | 41 | `papadopulos_cooper.py` 91% |
| `test_hantush_jacob.py` | 38 | `hantush_jacob.py` 97% |
| `test_boulton_warren_root.py` | 38 | `boulton.py` 98%, `warren_root.py` 98% |
| `test_cooper_jacob.py` | 17 | `cooper_jacob.py` 81% |
| `test_golden_regression.py` | 20 | pins the shipped fits to 1e-9 |

### Shared machinery

| File | Tests | Covers |
|------|-------|--------|
| `test_registry.py` | 70 | Method table integrity, alias resolution, import weight |
| `test_laplace.py` | 47 | Inverter against known transform pairs; every analytical limit |
| `test_fitting.py` | 42 | `fit_curve`, CIs, bound detection, AIC/AICc, log-scale |
| `test_validation_rules.py` | 44 | Shared input checks and the validity-rule library |
| `test_parameters.py` | 32 | Bidirectional `CalculationResult` reconciliation |
| `test_geometry.py` | 65 | Geometry invariants and backward compatibility |
| `test_derivative.py` | 29 | Bourdet derivative and flow-regime diagnosis |

### Data pipeline, plotting, method selection

| File | Tests |
|------|-------|
| `test_method_selection.py` | 84 |
| `test_csv_inspector.py` | 40 |
| `test_plots_report.py` | 34 |
| `test_formatter.py` | 26 |
| `test_validator.py` | 24 |

### Integration

| File | Tests | What it covers |
|------|-------|----------------|
| `test_chat_calculator_bridge.py` | 26 | Agents with stubbed LLM + retriever |
| `test_end_to_end_journey.py` | 18 | Full journey through the compiled LangGraph |
| `tests/api/` | 112 | The HTTP API through FastAPI's TestClient — sessions, projects, fits, chat, rate limit |

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

---

## Quick Start

### 1. Clone and Setup

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

Current contents — 3,843 chunks:

| Source folder | Aquifer tag | Chunks |
|---|---|---|
| Unconfined Aquifer | `unconfined` | 1,574 |
| Confined Aquifer | `confined` | 613 |
| General | `general` | 597 |
| Leaky Confined Aquifer | `leaky` | 572 |
| Fractured Aquifers | `fractured` | 472 |
| AQTESOLV method metadata | per method | 6 |

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

### Phase 1 — Consultant (Complete)
- [x] RAG-based consultant chatbot
- [x] LangGraph multi-agent state machine
- [x] Intent classification and conversation routing
- [x] Metadata-filtered retrieval (aquifer type, method, source)
- [x] Streamlit web interface

### Phase 2 — Calculator (Complete)
- [x] CSV inspector — 3-layer column/unit detection with LLM fallback
- [x] Data formatter — unit conversion, t=0 removal, wide-to-long melt
- [x] Data validator — physical plausibility checks, method suitability pre-filtering
- [x] Theis (1935) calculator — verified against original paper
- [x] Cooper-Jacob (1946) calculator — iterative u-filter (u < 0.02), paper-verified
- [x] Papadopulos-Cooper (1967) calculator — Stehfest (1970) Laplace inversion
- [x] Plotly interactive diagnostic plots (type curves + residuals for all 3 methods)
- [x] PDF report generation (ReportLab)
- [x] Calculator workspace UI (4-step workflow)

### Phase 3 — Integration (Complete)
- [x] Method suggestion card: the Manager recommends a method from the conversation
- [x] Workspace preselects the confirmed method
- [x] Completed analyses published back into `AgentState`
- [x] Consultant interprets the user's own results, with citations
- [x] `DataValidator` wired in — gates the method dropdown per dataset
- [x] Test coverage for the data pipeline, plotting and PDF layers
- [x] Streamlit UI smoke tests

### Phase 4 — Extensible architecture and domain increment (In progress)
- [x] Method registry — one declarative table replacing ten hardcoded call sites
- [x] Golden-fit regression baseline, enforced at 1e-9
- [x] Shared fitting machinery: `ParameterSpec`, CIs, bound detection, AIC/AICc
- [x] Log-scale fitting, without which multi-parameter methods do not converge
- [x] Parameter-correlation diagnostic for non-unique fits
- [x] `CalculationResult` carries a variable-length parameter set
- [x] Geometry dataclasses — aquifer, well, aquitard, fracture, pumping schedule
- [x] Stehfest inverter extracted and **vectorised** (PC fitting 10–30 s → 0.04 s)
- [x] `CurveFitCalculator` template — a new method is a model plus a spec
- [x] **Hantush-Jacob (1955)** — leaky
- [x] **Boulton (1963)** — unconfined
- [x] **Warren-Root (1963)** — fractured
- [x] **Bourdet log-derivative** driving data-based method selection
- [x] Theis, Cooper-Jacob and Papadopulos-Cooper ported onto the template
- [x] Recovery analysis — Theis (1935) recovery, Eq. 7
- [ ] Step-drawdown (Birsoy-Summers 1980)
- [ ] Neuman (1974) unconfined — needs a transcendental root series
- [ ] Forward simulation / test design
- [ ] Slug tests — a different input shape, and Stehfest is the wrong inverter
- [ ] Deployment — one image, two free hosts: Render (no card) or Oracle Cloud Always Free (see DEPLOY.md)
- [x] Projects — tests, readings and analyses saved as a file, compared by AICc
- [ ] Named wells with coordinates, logger import
- [ ] IIT KGP email authentication (deferred)

### Phase 5 — Web application (In progress)
- [x] Workspace logic extracted into UI-free services
- [x] FastAPI HTTP API with per-visitor sessions
- [x] React + TypeScript workspace replacing Streamlit (see PARITY.md)
- [x] Consultant rate limit, and a fallback when the model is unavailable
- [x] Browser end-to-end tests (Playwright)
- [ ] Deploy — Render or Oracle Cloud Always Free

---

## Known Issues

| Issue | Impact | Status |
|---|---|---|
| `knowledge source/` contains paywalled publisher PDFs tracked in git | Redistribution of copyrighted material from a public repo | **Open** — needs `git rm --cached` plus a DOI manifest |
| ~~`method` metadata is `multiple` for 3,837 of 3,843 chunks~~ | Method-level retrieval filtering was ineffective | **Fixed** — `configs/corpus_methods.yaml` tags each paper; chunks carry a flag per method and the consultant filters on it when a question names one. Assignments are `reviewed: false` until a hydrogeologist confirms them |
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

3. **Papadopulos, I.S. & Cooper, H.H. (1967)**. Drawdown in a well of large diameter. *Water Resources Research*, 3(1), 241–244.

4. **Stehfest, H. (1970)**. Algorithm 368: Numerical inversion of Laplace transforms. *Communications of the ACM*, 13(1), 47–49.

5. **USGS TWI Book 3-B1** — Techniques of Water-Resources Investigations, Book 3, Chapter B1.

---

## Acknowledgments

- USGS for public domain hydrogeological publications
- Google for Gemini API
- LangChain & LangGraph teams
- IIT Kharagpur — BTP supervision

---

**Developed as part of BTP (Bachelor Thesis Project) at IIT Kharagpur**
