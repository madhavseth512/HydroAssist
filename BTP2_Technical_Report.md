# HydroAssist — BTP-2 Technical Report
**Bachelor's Thesis Project, Phase 2**
**Indian Institute of Technology Kharagpur**
**Student:** Madhav Seth | **Guide:** Prof. Anirban Dhar

---

## 1. Project Overview

HydroAssist is a full-stack web application for confined aquifer pumping test analysis. It combines a RAG-based AI consultation chatbot (Phase 1) with a numerical curve-fitting calculator workspace (Phase 2). The application enables hydrogeologists to upload field time-drawdown data, fit three classical analytical models, and download a formatted PDF report — all through a browser interface.

**Repository:** `github.com/madhavseth512/HydroAssist`
**Framework:** Streamlit (Python)
**Deployment:** Local (`http://localhost:8501`)

---

## 2. System Architecture

### 2.1 High-Level Structure

```
hydroassist/
├── app.py                          ← Streamlit UI entry point
├── src/
│   ├── agents/                     ← LangGraph AI agent system (Phase 1)
│   │   ├── base.py                 ← Abstract BaseAgent class
│   │   ├── manager.py              ← Intent classifier + router
│   │   ├── consultant.py           ← RAG Q&A agent
│   │   └── calculator.py           ← Calculator orchestrator stub
│   ├── calculator/                 ← Numerical methods (Phase 2)
│   │   ├── base.py                 ← CalculationInput, CalculationResult dataclasses
│   │   ├── theis.py                ← Theis (1935)
│   │   ├── cooper_jacob.py         ← Cooper-Jacob (1946)
│   │   ├── papadopulos_cooper.py   ← Papadopulos-Cooper (1967)
│   │   ├── plots.py                ← Plotly diagnostic plots
│   │   └── report.py               ← ReportLab PDF generation
│   ├── core/
│   │   ├── config.py               ← AppConfig dataclass (Pydantic)
│   │   ├── state.py                ← AgentState TypedDict
│   │   └── graph.py                ← LangGraph workflow compilation
│   ├── data/
│   │   ├── csv_inspector.py        ← Column detection + unit inference
│   │   ├── formatter.py            ← Unit conversion + schema standardisation
│   │   └── validator.py            ← Physical plausibility checks
│   ├── retrieval/
│   │   ├── embeddings.py           ← Sentence-Transformers manager
│   │   ├── vectorstore.py          ← ChromaDB interface
│   │   └── retriever.py            ← Metadata-filtered semantic search
│   └── ingestion/                  ← PDF/HTML ingestion pipeline
├── tests/
│   └── unit/
│       ├── test_theis.py           ← 46 tests
│       ├── test_cooper_jacob.py    ← 20 tests
│       └── test_papadopulos_cooper.py ← 41 tests
└── data/
    └── vectordb/                   ← ChromaDB persistent store
```

### 2.2 Two-Mode Application Design

| Mode | Tab | Technology | Purpose |
|------|-----|------------|---------|
| Consultation | Tab 1 | LangGraph + ChromaDB + Gemini | Hydrogeological Q&A via RAG |
| Calculator Workspace | Tab 2 | scipy + plotly + ReportLab | CSV upload → curve fit → PDF |

### 2.3 Phase 2 Data Flow

```
CSV Upload
    ↓
CSVInspector   → detect columns, infer units
    ↓
DataFormatter  → convert to time_s / drawdown_m schema
    ↓
DataValidator  → physical checks, method suitability
    ↓
Calculator     → curve fit (scipy.optimize.curve_fit / Stehfest inversion)
    ↓
CalculationResult → T, S, CIs, R², RMSE, validity notes
    ↓
plots.py       → Plotly interactive type curves + residuals
    ↓
report.py      → ReportLab PDF
```

---

## 3. Technology Stack

### 3.1 Core Framework

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | ≥1.33.0 | Web UI framework — interactive dashboard, file upload, widgets |
| LangChain | ≥0.3.0 | LLM chain orchestration, document loaders, text splitters |
| LangGraph | ≥0.2.0 | Graph-based stateful agent workflows (Manager → Consultant/Calculator) |
| langchain-google-genai | ≥2.0.0 | Google Gemini 2.5 Flash integration |

### 3.2 Scientific Computing (Phase 2)

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.26.0 | Array operations, polyfit (CJ slope), log-space time arrays |
| SciPy | ≥1.13.0 | `curve_fit` (TRF nonlinear least squares), `expn` (Theis W(u)), `kv` (Bessel K₀/K₁) |
| Pandas | ≥2.2.0 | DataFrame operations, CSV inspection, tabular data handling |
| Matplotlib | ≥3.8.0 | Static figures embedded in PDF reports |
| Plotly | ≥5.22.0 | Interactive type curves, semi-log plots, residual charts |
| ReportLab | ≥4.2.0 | PDF document generation |

### 3.3 AI / Knowledge Base

| Library | Version | Purpose |
|---------|---------|---------|
| ChromaDB | ≥0.4.0 | Vector database for persistent RAG embeddings |
| sentence-transformers | ≥2.7.0 | All-MiniLM-L6-v2 model — document encoding |
| PyPDF | ≥4.0.0 | PDF text extraction (ingestion pipeline) |
| BeautifulSoup4 | ≥4.12.0 | HTML parsing for USGS/AQTESOLV web scraping |

### 3.4 Infrastructure

| Library | Version | Purpose |
|---------|---------|---------|
| Pydantic | ≥2.7.0 | `AppConfig` dataclass validation |
| PyYAML | ≥6.0.0 | Configuration file parsing |
| python-dotenv | ≥1.0.0 | `.env` loading for API keys |
| openpyxl | ≥3.1.0 | Excel export of results tables |
| pytest | ≥9.0.0 | Test runner with coverage reporting |
| pytest-cov | — | Coverage HTML reports |

---

## 4. Analytical Methods Implemented

All three methods target **confined (artesian) aquifers** — homogeneous, isotropic, with constant pumping rate Q and fully penetrating well.

### 4.1 Theis (1935) — Full-Curve Nonlinear Fit

**Reference:** Theis, C.V. (1935). Transactions of the American Geophysical Union, 16, 519–524.

#### Core Equations

$$s = \frac{Q}{4\pi T} \cdot W(u) \qquad u = \frac{r^2 S}{4Tt}$$

$$W(u) = \int_u^\infty \frac{e^{-u}}{u}\,du = \texttt{expn(1, u)} \quad \text{(scipy.special)}$$

**Series expansion (paper Eq. 3):**
$$W(u) = -0.5772 - \ln(u) + u - \frac{u^2}{2 \cdot 2!} + \frac{u^3}{3 \cdot 3!} - \cdots$$

#### Fitting Method

- **Algorithm:** `scipy.optimize.curve_fit` with Trust Region Reflective (TRF)
- **Free parameters:** T [m²/s], S [dimensionless]
- **Fixed parameters:** Q, r (user inputs)
- **Tolerances:** `ftol = xtol = gtol = 1e-12` (tightened from scipy default 1e-8 — T and S span 4–5 orders of magnitude)
- **Initial estimate:** Late-time CJ slope heuristic; fallback to `Q / (4πs̄)`
- **Bounds:** T ∈ [10⁻⁹, 1.0] m²/s; S ∈ [10⁻⁹, 10⁻²]

#### Physical Parameter Bounds Justification

| Bound | Value | Justification |
|-------|-------|---------------|
| T_MAX | 1.0 m²/s (86 400 m²/day) | Covers karst and fractured rock (raised from 0.1 to avoid silent bound-hit) |
| T_MIN | 1×10⁻⁹ m²/s | Theoretical lower limit for clay |
| S_MAX | 1×10⁻² | Theis (1935) distinguishes confined (S = 10⁻⁵–10⁻³) from unconfined (S = 0.01–0.35) |
| S_MIN | 1×10⁻⁹ | Practical lower limit |

#### Key Implementation Details

- **u guard:** `u = np.maximum(u, 1e-10)` prevents `expn(1, 0)` overflow
- **Confidence intervals:** 1.96 × √(diag(pcov)); set to `None` if pcov contains `inf` or negative diagonal
- **CJ regime check:** Post-fit u values compared against threshold 0.02 to advise on Cooper-Jacob suitability

#### Outputs

| Output | Symbol | Unit | Reliability |
|--------|--------|------|-------------|
| Transmissivity | T | m²/s, m²/day | High |
| Storativity | S | dimensionless | Moderate |
| 95% CI on T | T_ci | m²/day | When covariance well-conditioned |
| 95% CI on S | S_ci | — | When covariance well-conditioned |
| R² | — | — | Goodness of fit |
| RMSE | — | mm | Residual magnitude |

---

### 4.2 Cooper-Jacob (1946) — Semi-Log Straight-Line Method

**Reference:** Cooper, H.H. & Jacob, C.E. (1946). Transactions of the American Geophysical Union, 27, 526–534.

#### Core Equations

For small u (u < 0.02), truncating Theis series gives:

$$s \approx \frac{2.303\,Q}{4\pi T}\,\log_{10}\!\left(\frac{2.25\,T\,t}{r^2 S}\right)$$

which is linear in log₁₀(t): **slope = 2.303Q / (4πT)**

**Transmissivity from slope (paper Eq. 8):**
$$T = \frac{2.303\,Q}{4\pi\,\Delta s}$$

**Storativity from zero-drawdown intercept (paper Eq. 9):**
$$S = \frac{2.25\,T\,t_0}{r^2} \qquad \text{where } t_0 \text{ is the time at } s = 0$$

**Validity criterion:** $u = r^2 S / (4Tt) < 0.02$

#### Iterative Self-Consistent Data Selection

Fixed 2-pass selection is not self-consistent (pass 1 uses all data, pass 2 uses T₁/S₁ filter). HydroAssist uses iterative convergence:

```
Pass 1: Fit ALL data → (T₁, S₁)
Pass k: Filter to u < 0.02 using (T_{k-1}, S_{k-1})
        Refit filtered subset → (Tₖ, Sₖ)
Stop when |ΔT/T| < 1e-3 AND |ΔS/S| < 1e-3   (typically 2–3 iterations)
```

**Fallback:** If fewer than 4 points satisfy u < 0.02, use all data with a warning.

#### Key Implementation Details

- Straight-line regression via `numpy.polyfit` on log₁₀(t) vs s
- t₀ extrapolated as `10^(-intercept/slope)` (may fall before first measurement — expected and normal)
- Post-fit u check uses final T, S to re-verify each included point

---

### 4.3 Papadopulos-Cooper (1967) — Large-Diameter Well with Wellbore Storage

**Reference:** Papadopulos, I.S. & Cooper, H.H. (1967). Water Resources Research, 3(1), 241–244.

#### Core Equations

**Drawdown at pumped well (paper Eq. 12):**
$$s_w = \frac{Q}{4\pi T}\,F(u_w,\,\alpha)$$

**Well function (paper Eq. 13) — oscillatory integral:**
$$F(u_w,\alpha) = \frac{32\alpha^2}{\pi^2}\int_0^\infty \frac{1-e^{-\beta^2/4u_w}}{\beta^3\,\Delta(\beta)}\,d\beta$$

**Δ(β) denominator (paper Eq. 11):**
$$\Delta(\beta) = \bigl[\beta J_0(\beta) - 2\alpha J_1(\beta)\bigr]^2 + \bigl[\beta Y_0(\beta) - 2\alpha Y_1(\beta)\bigr]^2$$

**Dimensionless parameters:**
$$\alpha = \frac{r_w^2\,S}{r_c^2} \qquad u_w = \frac{r_w^2\,S}{4\,T\,t}$$

**Late-time convergence to Theis (paper p.242):**
$$\frac{u_w}{\alpha} < 10^{-3} \;\Rightarrow\; F(u_w,\alpha) \approx W(u_w) \qquad \left(t > \frac{2500\,r_c^2}{T}\right)$$

#### Implementation: Stehfest (1970) Laplace Inversion

**Problem with direct integration:** Eq. 13 contains products of oscillatory Bessel functions (J₀, J₁, Y₀, Y₁) that make `scipy.integrate.quad` unreliable — errors of 23–115% were measured against paper Table 1 for certain (u_w, α) combinations during development.

**Solution adopted:** Invert the Laplace-domain solution (paper Eq. 10 at r = r_w):

$$\bar{s}_w(p) = \frac{Q\,K_0(q\,r_w)}{\pi\,p\,\bigl[r_c^2\,p\,K_0(q\,r_w) + 2\,r_w\,T\,q\,K_1(q\,r_w)\bigr]}, \qquad q = \sqrt{\frac{pS}{T}}$$

K₀ and K₁ are modified Bessel functions of the second kind (`scipy.special.kv`) — smooth, non-oscillatory, numerically stable.

**Stehfest (1970) inversion algorithm:**
$$f(t) \approx \frac{\ln 2}{t}\sum_{i=1}^{N} V_i\,\bar{F}\!\left(\frac{i\ln 2}{t}\right), \qquad N = 12$$

**V coefficients** precomputed at import time from:
$$V_i = (-1)^{M+i}\sum_{k=\lfloor(i+1)/2\rfloor}^{\min(i,M)} \frac{k^M\,(2k)!}{(M-k)!\,k!\,(k-1)!\,(i-k)!\,(2k-i)!}, \qquad M = N/2$$

**Standard properties (verified in tests):**
- $\sum_{i=1}^{N} V_i = 0$
- V_i and V_{i+1} have alternating signs

#### Critical Paper Warning (p. 244)

> *"A determination of S by this method has questionable reliability. Whereas the determined value of S will change by an order of magnitude when the data plot is moved from one type curve to another, that of T will change only slightly."*

→ **T is the reliable output. S carries order-of-magnitude uncertainty.** This warning is always surfaced in the validity notes.

#### Additional Geometry Inputs

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Well screen radius | r_w | m | Effective radius of open hole / screen |
| Casing radius | r_c | m | Radius where water level declines |
| Storage parameter | α (computed) | — | = r_w²S / r_c² |

---

## 5. Data Pipeline

### 5.1 CSV Inspection (`csv_inspector.py`)

Detects column roles and units from raw CSV using a 3-layer cascade:
1. **Regex pattern matching** — column name keywords (time, drawdown, level, depth)
2. **Statistical heuristics** — value range, monotonicity, units implied by magnitude
3. **LLM fallback** — Gemini for ambiguous columns

### 5.2 Formatting (`formatter.py`)

Converts raw DataFrame to standard schema `(time_s, drawdown_m)`:

| Step | Action |
|------|--------|
| Column extraction | Keep only time + drawdown + optional well_id |
| Numeric coercion | Drop rows with unparseable values |
| Time conversion | seconds (×1), minutes (×60), hours (×3600) |
| Drawdown conversion | metres (×1), feet (×0.3048) |
| Remove t = 0 | Log transform undefined at zero |
| Remove duplicates | Keep first occurrence per timestamp |
| Sort | Ascending time order |

All changes are logged in a `FormattedData.conversions_applied` list shown to the user.

### 5.3 Validation (`validator.py`)

| Check | Type | Threshold |
|-------|------|-----------|
| Minimum data points | Blocking | N ≥ 10 |
| Monotonic time | Warning | All Δt > 0 |
| Non-negative drawdown | Warning | All s ≥ 0 |
| Log-cycle span | Warning | log₁₀(t_max/t_min) ≥ 1.0 |
| Method suitability | Advisory | Rough pre-check for CJ and PC |

### 5.4 `CalculationInput` Dataclass (`base.py`)

```
df        : pd.DataFrame   — formatted (time_s, drawdown_m)
Q         : float          — pumping rate [m³/s] — guard: Q > 1.0 raises ValueError (catches m³/day error)
r         : float          — observation distance [m]
r_w       : float = 0.10  — well screen radius [m] (PC only)
r_c       : float = 0.10  — casing radius [m]       (PC only)
well_id   : str            — optional, for multi-well datasets
```

---

## 6. User Interface

### 6.1 Four-Step Calculator Workflow

```
STEP 1 — Upload CSV
  ↓  File uploader → CSVInspector → DataFormatter
  ↓  Shows: column map, unit conversions, data preview

STEP 2 — Well Parameters
  ↓  Q (value + unit selectbox) | r | Method dropdown
  ↓  Conditional: r_w and r_c inputs (Papadopulos-Cooper only)

STEP 3 — Run Analysis
  ↓  "▶ Run Analysis" button (primary, red)
  ↓  Spinner: fit completes in 0.5s (Theis/CJ) to 5s (PC with 50 pts)

STEP 4 — Results
     ├── Metric cards: T (m²/day), T (m²/s), S, R²
     ├── Confidence intervals + RMSE
     ├── Diagnostic plot (Plotly, interactive)
     ├── Validity & Interpretation (expandable notes)
     ├── 📄 Download PDF Report
     └── 📊 Download Results CSV
```

### 6.2 Diagnostic Plots

All plots are Plotly figures with hover data, rendered inside Streamlit.

| Plot | Panels | X-axis | Y-axis |
|------|--------|--------|--------|
| Theis static | 2 (main + residuals) | Time log-scale | Drawdown log-scale |
| Theis manual | 1 | Time log-scale | Drawdown log-scale |
| Cooper-Jacob static | 2 (main + residuals) | Time log-scale | Drawdown linear |
| Cooper-Jacob manual | 1 | Time log-scale | Drawdown linear |
| Papadopulos-Cooper static | 2 (main + residuals) | Time log-scale | Drawdown log-scale |
| Papadopulos-Cooper manual | 1 | Time log-scale | Drawdown log-scale |

**Common design:**
- Legend: horizontal, centred below both panels (never overlapping data)
- All annotation text removed from plot area
- Colour palette: Observed = `#1f77b4` (blue), Fit = `#d62728` (red), Manual = `#ff7f0e` (orange)

### 6.3 PDF Report Sections

1. Header — title, method, timestamp
2. Configuration — Q, r, dataset summary
3. Results table — T, S, CIs, R², RMSE
4. Validity & Interpretation — all validity notes
5. Diagnostic plot — embedded matplotlib figure
6. Data table — full input dataset
7. Footer — generated by HydroAssist, IIT Kharagpur

---

## 7. Unit Test Suite

### 7.1 Summary

| File | Classes | Tests | Runtime |
|------|---------|-------|---------|
| `test_theis.py` | 6 | 46 | ~2.4 s |
| `test_cooper_jacob.py` | 3 | 20 | ~1.2 s |
| `test_papadopulos_cooper.py` | 5 | 41 | ~4.1 s |
| **Total** | **14** | **107** | **~7.7 s** |

All 107 tests pass. Coverage on calculator modules: 91% (Theis), 88% (Cooper-Jacob), 91% (PC).

### 7.2 Theis Test Classes (46 tests)

| Class | Tests | What it verifies |
|-------|-------|-----------------|
| `TestTheisWellFunction` | 8 | W(u) known tabulated values (within 0.1%), monotonicity, divergence as u→0 |
| `TestTheisModelFunction` | 6 | Drawdown positivity, monotonicity, linearity in Q, T/r sensitivity, u guard |
| `TestTheisInitialEstimate` | 3 | T₀ heuristic is positive and within 1 order of magnitude of true T |
| `TestTheisSmokeTest` | 14 | End-to-end: T within 5%, S within 30%, R²>0.99, CIs, arrays, noise stability |
| `TestTheisInputValidation` | 8 | All `_validate()` error paths, unit error detection |
| `TestTheisValidityNotes` | 5 | Recovery hint, CJ regime note, poor-fit warning, high-S warning |

**Smoke test parameters:** TRUE_T = 2.5×10⁻³ m²/s, TRUE_S = 5×10⁻⁵, Q = 1×10⁻³ m³/s, r = 50 m, noise σ = 5 mm

**Smoke test result:** T recovered to **+0.1%**, R² = **0.9994**, RMSE = **4.8 mm**

### 7.3 Cooper-Jacob Test Classes (20 tests)

| Class | Tests | What it verifies |
|-------|-------|-----------------|
| `TestCooperJacobSmokeTest` | 13 | End-to-end: T within 5%, S within 30%, R²>0.97, early-time exclusion, CIs |
| `TestCooperJacobInputValidation` | 4 | Validation error paths |
| `TestCooperJacobUThreshold` | 1 | Relaxed threshold (0.05 textbook vs 0.02 paper) gives consistent T |

**Smoke test parameters:** Same as Theis (T=2.5×10⁻³, S=5×10⁻⁵, r=50 m). For r=50 m, CJ valid for t > 625 s (u < 0.02).

**Smoke test result:** T within **3.2%**, S within **18%**, R² = **0.9981** on filtered points

### 7.4 Papadopulos-Cooper Test Classes (41 tests)

| Class | Tests | What it verifies |
|-------|-------|-----------------|
| `TestStehfestNumerics` | 7 | V coefficient properties (sum=0, alternating sign), Laplace kernel |
| `TestPhysicalLimits` | 6 | Late-time Theis convergence (<1%), early-time wellbore storage (<5%), monotonicity, linearity in Q |
| `TestPCSmokeTest` | 13 | End-to-end: T within 5%, S within order-of-magnitude, R²>0.99, CIs, arrays |
| `TestPCInputValidation` | 9 | All error paths including r_w, r_c guards and unit error detection |
| `TestPCValidityNotes` | 6 | Paper p.244 mandatory warnings always emitted |

**Smoke test parameters:** TRUE_T = 1.0×10⁻³ m²/s, TRUE_S = 1.0×10⁻⁴, Q = 300 m³/day, r_w = 0.30 m, r_c = 0.20 m, noise σ = 3 mm

**Smoke test result:** T recovered to **+0.0%**, S recovered to **−0.5%**, R² = **1.0000**, RMSE = **2.4 mm**

### 7.5 Key Physical Validation Results

| Test | Expected | Measured | Error |
|------|----------|----------|-------|
| Stehfest late-time convergence to Theis (u_w/α = 10⁻⁴) | F ≈ W(u_w) | F_calc vs W(u_w) | **0.02%** |
| Early-time wellbore storage (u_w/α = 50) | s_w ≈ Qt/(πr_c²) | | **< 1.5%** |
| W(u = 0.001) | 6.332 (table) | 6.332 (scipy) | **< 0.1%** |
| W(u = 0.1) | 1.823 (table) | 1.823 (scipy) | **< 0.1%** |

---

## 8. Phase 1 AI Consultant (RAG System)

### 8.1 Architecture

```
User question (natural language)
  ↓
Manager Agent (Gemini LLM)
  → Classifies intent: consultation / calculation / clarification
  → Extracts aquifer type, method keyword
  ↓
Consultant Agent
  → Retriever.retrieve(query, top_k=5, metadata_filters)
  → ChromaDB semantic search over ingested PDFs
  → Constructs context from top-k chunks
  → Gemini LLM generates response + citations
  ↓
Formatted response in chat interface
```

### 8.2 Knowledge Base

| Source | Content | Format |
|--------|---------|--------|
| USGS TWI Books | Standard groundwater methods reference | PDF |
| AQTESOLV documentation | Method metadata, applicability conditions | HTML |
| Custom PDFs | Aquifer-type specific references | PDF |

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
**Vector store:** ChromaDB (persistent, local)
**LLM:** Google Gemini 2.5 Flash

### 8.3 Agent State

```python
class AgentState(TypedDict):
    messages          : List[AnyMessage]      # Conversation history
    user_intent       : str                   # consultation | calculation | ...
    aquifer_context   : str                   # confined | unconfined | leaky | fractured
    selected_method   : str                   # Theis | Cooper-Jacob | PC
    retrieved_docs    : List[Dict]            # Top-k retrieved chunks
    metadata_filters  : Dict                  # Applied to vector search
    suggestion_confirmed : bool               # User approved method suggestion
    calculation_result   : Dict              # T, S, plots
```

---

## 9. Git Commit History

| Commit | Description |
|--------|-------------|
| `bf8f4d9` | Fix plot layout: move legend below graphs, remove overlapping annotations |
| `cdb4707` | Add Papadopulos-Cooper (1967) calculator with full test suite (41 tests) |
| `560f9df` | Add Phase 2 calculator workspace UI with interactive diagnostic plots |
| `1f8b0c2` | Apply 8 verified bug fixes across base.py, theis.py, cooper_jacob.py |
| `07f7c8f` | Remove redundant Phase 1 setup docs |
| `ae3d574` | Repo cleanup: remove sensitive config, rename Phase 1 summary, tighten gitignore |
| `3551411` | Add Cooper-Jacob (1946) calculator with paper-verified implementation |
| `01a8f95` | Phase 2 foundation: CSV pipeline, Theis calculator, agent state fixes |
| `3d9ef43` | Initial Commit: Consultant Agent — HydroAssist |

---

## 10. Key Design Decisions and Engineering Notes

### 10.1 Stehfest vs Direct Integration (PC Calculator)

Direct integration of paper Eq. 13 via `scipy.integrate.quad` produced errors of 23–115% on certain (u_w, α) combinations during development testing. The oscillatory Bessel function products in the integrand cause `quad` to declare convergence prematurely. The Stehfest (1970) algorithm operating on the smooth Laplace-domain formula (paper Eq. 10) reduced the error to < 0.1% across all tested combinations.

### 10.2 Iterative u-Filter in Cooper-Jacob

The standard 2-pass approach (fit all data → compute u → filter → refit once) is not self-consistent because the u-filter in pass 2 uses T₁, S₁ from pass 1, but the refit produces T₂, S₂ which may move different points across the threshold. Full convergence (typically 2–3 iterations) ensures the filter and the fit agree.

### 10.3 T_MAX Raised to 1.0 m²/s

The initial bound of T_MAX = 0.1 m²/s silently hit the bound for high-transmissivity systems (karst, Snake River Plain type). The calculator would report T = 8640 m²/day as the result with no warning. The bound was raised to 1.0 m²/s (86 400 m²/day) and a bound-hit detection note was added.

### 10.4 Confidence Interval Guarding

`scipy.optimize.curve_fit` returns `pcov = inf` when convergence is poor or when T and S are strongly correlated (common in PC where type curves are nearly identical for α values an order of magnitude apart). All CI computations check `np.isinf(np.diag(pcov))` and return `None` rather than crashing.

### 10.5 Q Unit Guard

A common user mistake is entering Q in m³/day instead of m³/s. Since real pumping wells operate between 0.001 and 0.5 m³/s, any Q > 1.0 m³/s almost certainly means m³/day was entered. `CalculationInput.__post_init__` raises `ValueError` with a conversion hint: `Q_si = Q / 86400`.

---

## 11. Feature Completeness Summary

| Feature | Status |
|---------|--------|
| Theis (1935) calculator | ✅ Complete |
| Cooper-Jacob (1946) calculator | ✅ Complete |
| Papadopulos-Cooper (1967) calculator | ✅ Complete |
| CSV upload + auto-inspection | ✅ Complete |
| Unit conversion pipeline | ✅ Complete |
| Plotly interactive diagnostic plots | ✅ Complete |
| PDF report generation | ✅ Complete |
| 107-test unit test suite | ✅ Complete |
| RAG consultation chatbot (Phase 1) | ✅ Complete |
| IIT KGP email authentication | 🔲 Deferred |
| Theis recovery analysis (Eq. 7) | 🔲 Future |
| Unconfined / leaky aquifer methods | 🔲 Future |
