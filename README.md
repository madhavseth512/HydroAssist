# HydroAssist — Intelligent Hydrogeological Analysis Platform

**BTP-1 → BTP-2**: RAG-based consultant chatbot + confined aquifer pumping test calculator, powered by LangGraph and Gemini AI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase](https://img.shields.io/badge/phase-2%20in%20progress-orange.svg)]()

---

## Overview

HydroAssist is a two-phase hydrogeology assistant targeting confined aquifer pumping test analysis.

**Phase 1 — Consultant (complete)**
- RAG-based chatbot answering questions about pumping test theory, method selection, and aquifer characterisation
- Grounded in USGS TWI Book 3-B1 and curated AQTESOLV method metadata
- Metadata-filtered retrieval (aquifer type, method, section)
- Intent classification and conversation routing via LangGraph

**Phase 2 — Calculator (in progress)**
- CSV upload with automatic column detection, unit conversion, and data validation
- Confined aquifer curve-fitting analysis: Theis (1935), Cooper-Jacob (1946), Papadopulos-Cooper (1967)
- Outputs T (transmissivity) and S (storativity) with 95% confidence intervals
- Diagnostic plots and PDF report generation
- IIT KGP email-based authentication

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                            │
│   consultation mode  →  transition/suggestion  →  workspace     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                        │
│   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐ │
│   │ Manager Agent│──▶│Consultant Agent│   │ Calculator Agent │ │
│   │ (routing +   │   │ (RAG Q&A)      │   │ (Phase 2)        │ │
│   │  suggestion) │   └────────────────┘   └──────────────────┘ │
│   └──────────────┘                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────────────┐
          ▼                ▼                        ▼
   ┌────────────┐   ┌────────────────┐   ┌─────────────────────┐
   │   Gemini   │   │  RAG Pipeline  │   │  Calculator Engine  │
   │  2.5 Flash │   │  ChromaDB +    │   │  src/calculator/    │
   │    LLM     │   │  Sentence-Trns │   │  scipy curve_fit    │
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
├── app.py                     # Streamlit entry point (3 UI layout modes)
├── requirements.txt
├── configs/                   # YAML configs (default, development, production)
├── src/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── state.py           # AgentState TypedDict
│   │   └── graph.py           # LangGraph workflow
│   ├── agents/
│   │   ├── base.py            # Abstract BaseAgent
│   │   ├── manager.py         # Intent classification + routing
│   │   ├── consultant.py      # RAG Q&A agent
│   │   └── calculator.py      # Phase 2 calculator orchestrator
│   ├── retrieval/
│   │   ├── embeddings.py      # Sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── vectorstore.py     # ChromaDB manager
│   │   └── retriever.py       # Metadata-filtered semantic search
│   ├── ingestion/
│   │   ├── pipeline.py        # Ingestion orchestration
│   │   ├── processors/        # PDF + HTML processors, chunker
│   │   └── scrapers/          # USGS + AQTESOLV scrapers
│   ├── data/                  # Phase 2 — CSV pipeline
│   │   ├── csv_inspector.py   # Column detection + unit inference (3-layer)
│   │   ├── formatter.py       # Unit conversion → standard schema
│   │   └── validator.py       # Physical plausibility + method suitability
│   ├── calculator/            # Phase 2 — numerical methods
│   │   ├── base.py            # CalculationInput / CalculationResult / BaseCalculator
│   │   └── theis.py           # Theis (1935) curve fitting
│   └── prompts/               # LLM prompts for each agent
├── data/
│   ├── raw/                   # USGS PDFs, AQTESOLV metadata, custom uploads
│   ├── processed/             # Chunked documents
│   └── vectordb/              # ChromaDB persistence
└── tests/
```

---

## Phase 2 — Calculator

### Supported Methods (Confined Aquifer)

| Method | Reference | Valid When | Outputs |
|---|---|---|---|
| **Theis** | Theis (1935) | All times, homogeneous confined aquifer | T, S |
| **Cooper-Jacob** | Cooper & Jacob (1946) | u < 0.05 (large time / small r) | T, S |
| **Papadopulos-Cooper** | Papadopulos & Cooper (1967) | Large-diameter wells with wellbore storage | T, S |

### CSV Pipeline

The data pipeline handles messy real-world field data in 3 stages:

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

### Theis Calculator — Verified Against Theis (1935)

Key implementation decisions grounded in the original paper:

- **W(u) computation**: `scipy.special.expn(1, u)` — the exponential integral E₁(u), exact computation of the Theis well function
- **u guard**: `np.maximum(u, 1e-10)` — prevents divergence as u → 0 (paper eq. 3 series diverges at -ln(u) → ∞)
- **Factory pattern**: `make_theis_model(Q, r)` — Q and r are invariants of the test setup per Theis
- **S bounds**: S_max = 1e-2 — enforces compaction-based storage (confined), not gravity drainage (unconfined)
- **T bounds**: T_max = 1.0 m²/s — covers karst and fractured systems (86,400 m²/day)
- **Tolerances**: `ftol=xtol=gtol=1e-12` — tightened for parameters spanning 4-5 orders of magnitude
- **CI computation**: Guards against `inf` and negative diagonal in covariance matrix

**Smoke test results on synthetic data (T=2.5e-3 m²/s, S=5e-5, σ=5mm noise):**
- T recovery error: **0.09%**
- S recovery error: **0.98%**
- R²: **0.9999**, RMSE: **4.55 mm**

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
# Create .env file
echo GEMINI_API_KEY=your_api_key_here > .env
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 3. Ingest Knowledge Base

```bash
python scripts/ingest.py --all
# Downloads USGS TWI Book 3-B1, chunks text, populates ChromaDB
# Takes ~5-10 minutes on first run
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

### Phase 2 — Calculator (In Progress)
- [x] AgentState extended with `method_suggestion`, `suggestion_confirmed`, `calculation_input`, `calculation_result`
- [x] Manager agent fixes: confidence threshold, method extraction location, aquifer filter scope
- [x] CSV inspector — 3-layer column/unit detection with LLM fallback
- [x] Data formatter — unit conversion, t=0 removal, wide-to-long melt
- [x] Data validator — physical plausibility checks, method suitability pre-filtering
- [x] Theis (1935) calculator — verified against original paper
- [ ] Cooper-Jacob (1946) calculator
- [ ] Papadopulos-Cooper (1967) calculator
- [ ] Diagnostic plots (log-log type curve, semi-log straight line)
- [ ] PDF report generation
- [ ] IIT KGP email authentication
- [ ] Calculator workspace UI (3-mode layout)

### Phase 3 — Future
- [ ] Recovery analysis (Theis 1935, eq. 7)
- [ ] Derivative diagnostic plots (ds/dln(t))
- [ ] Interactive type curve adjustment
- [ ] Unconfined and leaky aquifer methods
- [ ] Multi-user project management
- [ ] REST API

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langgraph` | Multi-agent orchestration |
| `langchain-google-genai` | Gemini 2.5 Flash LLM |
| `chromadb` | Vector store |
| `sentence-transformers` | Local embeddings (all-MiniLM-L6-v2) |
| `scipy` | curve_fit, expn (Theis W(u) well function) |
| `numpy` | Array operations, polyfit |
| `matplotlib` | Diagnostic plots |
| `pandas` | CSV processing |
| `reportlab` | PDF report generation |
| `streamlit` | Web interface |
| `pypdf` | PDF text extraction |

---

## References

1. **Theis, C.V. (1935)**. The relation between the lowering of the piezometric surface and the rate and duration of discharge of a well using ground-water storage. *Transactions of the American Geophysical Union*, 16, 519–524.

2. **Cooper, H.H. & Jacob, C.E. (1946)**. A generalized graphical method for evaluating formation constants and summarizing well-field history. *Transactions of the American Geophysical Union*, 27(4), 526–534.

3. **Papadopulos, I.S. & Cooper, H.H. (1967)**. Drawdown in a well of large diameter. *Water Resources Research*, 3(1), 241–244.

4. **Hantush, M.S. & Jacob, C.E. (1955)**. Non-steady radial flow in an infinite leaky aquifer. *Transactions of the American Geophysical Union*, 36(1), 95–100.

5. **USGS TWI Book 3-B1** — Techniques of Water-Resources Investigations, Book 3, Chapter B1.

---

## Acknowledgments

- USGS for public domain hydrogeological publications
- Google for Gemini API
- LangChain & LangGraph teams
- IIT Kharagpur — BTP supervision

---

**Developed as part of BTP (Bachelor Thesis Project) at IIT Kharagpur**
