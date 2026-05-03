# HydroAssist — Intelligent Hydrogeological Analysis Platform

**BTP-1 → BTP-2**: RAG-based consultant chatbot + confined aquifer pumping test calculator, powered by LangGraph and Gemini AI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase](https://img.shields.io/badge/phase-2%20complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-107%20passing-brightgreen.svg)]()

---

## Overview

HydroAssist is a two-phase hydrogeology assistant targeting confined aquifer pumping test analysis.

**Phase 1 — Consultant (complete)**
- RAG-based chatbot answering questions about pumping test theory, method selection, and aquifer characterisation
- Grounded in USGS TWI Book 3-B1 and curated AQTESOLV method metadata
- Metadata-filtered retrieval (aquifer type, method, section)
- Intent classification and conversation routing via LangGraph

**Phase 2 — Calculator (complete)**
- CSV upload with automatic column detection, unit conversion, and data validation
- Confined aquifer curve-fitting: Theis (1935), Cooper-Jacob (1946), Papadopulos-Cooper (1967)
- Outputs T (transmissivity) and S (storativity) with 95% confidence intervals
- Interactive Plotly diagnostic plots (type curves, residuals, manual matching)
- PDF report generation (ReportLab)
- 107-test unit test suite covering all three calculators

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                            │
│         Tab 1: Consultation Chat  |  Tab 2: Calculator           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                        │
│   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐ │
│   │ Manager Agent│──▶│Consultant Agent│   │ Calculator Agent │ │
│   │ (routing +   │   │ (RAG Q&A)      │   │ (orchestrator)   │ │
│   │  suggestion) │   └────────────────┘   └──────────────────┘ │
│   └──────────────┘                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────────────┐
          ▼                ▼                        ▼
   ┌────────────┐   ┌────────────────┐   ┌─────────────────────┐
   │   Gemini   │   │  RAG Pipeline  │   │  Calculator Engine  │
   │  2.5 Flash │   │  ChromaDB +    │   │  scipy / Stehfest   │
   │    LLM     │   │  Sentence-Trns │   │  + Plotly + PDF     │
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
| `calculation_input` | `Dict` | CSV data + well parameters |
| `calculation_result` | `Dict` | T, S, confidence intervals, plots |

---

## Project Structure

```
hydroassist/
├── app.py                          # Streamlit entry point (2-tab layout)
├── requirements.txt
├── configs/                        # YAML configs
├── src/
│   ├── core/
│   │   ├── config.py               # AppConfig dataclass (Pydantic)
│   │   ├── state.py                # AgentState TypedDict
│   │   └── graph.py                # LangGraph workflow compilation
│   ├── agents/
│   │   ├── base.py                 # Abstract BaseAgent
│   │   ├── manager.py              # Intent classification + routing
│   │   ├── consultant.py           # RAG Q&A agent
│   │   └── calculator.py           # Calculator orchestrator
│   ├── retrieval/
│   │   ├── embeddings.py           # Sentence-transformers (all-MiniLM-L6-v2)
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
│   │   ├── base.py                 # CalculationInput / CalculationResult / BaseCalculator
│   │   ├── theis.py                # Theis (1935)
│   │   ├── cooper_jacob.py         # Cooper-Jacob (1946)
│   │   ├── papadopulos_cooper.py   # Papadopulos-Cooper (1967) — Stehfest inversion
│   │   ├── plots.py                # Plotly interactive diagnostic plots
│   │   └── report.py               # ReportLab PDF report generation
│   └── prompts/                    # LLM prompts for each agent
├── tests/
│   └── unit/
│       ├── test_theis.py           # 46 tests
│       ├── test_cooper_jacob.py    # 20 tests
│       └── test_papadopulos_cooper.py  # 41 tests
└── data/
    ├── raw/                        # USGS PDFs, AQTESOLV metadata
    ├── processed/                  # Chunked documents
    └── vectordb/                   # ChromaDB persistence
```

---

## Phase 2 — Calculator

### Supported Methods (Confined Aquifer)

| Method | Reference | Valid When | Key Output |
|---|---|---|---|
| **Theis** | Theis (1935) | All times, homogeneous confined aquifer | T reliable; S reliable |
| **Cooper-Jacob** | Cooper & Jacob (1946) | u < 0.02 (paper threshold) | T reliable; S via t₀ extrapolation |
| **Papadopulos-Cooper** | Papadopulos & Cooper (1967) | Large-diameter wells with wellbore storage | T reliable; S order-of-magnitude only |

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
     ├── 📄 Download PDF Report
     └── 📊 Download Results CSV
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

## Unit Tests

107 tests across 14 test classes, all passing.

```bash
pytest tests/unit/ -v
# 107 passed in ~7.7s
```

| File | Classes | Tests | Coverage |
|------|---------|-------|----------|
| `test_theis.py` | 6 | 46 | 91% |
| `test_cooper_jacob.py` | 3 | 20 | 88% |
| `test_papadopulos_cooper.py` | 5 | 41 | 91% |

Test categories per method: well function validation, physical limits (monotonicity, linearity in Q), end-to-end smoke test, input validation (all error paths), validity notes (paper-mandated warnings).

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
```

### 2. Configure Environment

```bash
echo GEMINI_API_KEY=your_api_key_here > .env
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 3. Ingest Knowledge Base

```bash
python scripts/ingest.py --all
# Downloads USGS TWI Book 3-B1, chunks text, populates ChromaDB
# Takes ~5–10 minutes on first run
```

### 4. Launch

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

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
- [x] 107-test unit test suite
- [ ] IIT KGP email authentication (deferred)

### Phase 3 — Future
- [ ] Recovery analysis (Theis 1935, Eq. 7)
- [ ] Derivative diagnostic plots (ds/d ln(t))
- [ ] Unconfined and leaky aquifer methods
- [ ] Multi-user project management
- [ ] REST API

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web interface |
| `langchain`, `langgraph` | Multi-agent orchestration |
| `langchain-google-genai` | Gemini 2.5 Flash LLM |
| `chromadb` | Vector store (RAG) |
| `sentence-transformers` | Local embeddings (all-MiniLM-L6-v2) |
| `scipy` | `curve_fit` (TRF fitting), `expn` (Theis W(u)), `kv` (Bessel K₀/K₁ for Stehfest) |
| `numpy` | Array operations, polyfit (CJ slope) |
| `pandas` | CSV processing and tabular data |
| `plotly` | Interactive type curves, semi-log plots, residuals |
| `matplotlib` | Static figures embedded in PDF reports |
| `reportlab` | PDF report generation |
| `pypdf` | PDF text extraction (ingestion) |
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
