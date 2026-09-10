# Deploying HydroAssist on Streamlit Community Cloud

The free-tier target. Streamlit Community Cloud is free, builds straight from a
GitHub repository, and supports the persistent WebSocket connection Streamlit
needs. (Hugging Face Spaces offers more RAM but now charges for Docker Spaces, and
AWS App Runner cannot hold Streamlit's WebSocket at all.)

## What had to change to fit the free tier

| Constraint | What was done |
|---|---|
| Roughly 0.7–2.7 GB RAM from a shared pool | Removed `sentence-transformers` / `torch`. Queries now use ChromaDB's ONNX build of the same `all-MiniLM-L6-v2` model — the one the index was always built with. Retrieval is unchanged: 12/12 benchmark queries return the identical top-5 (`tests/integration/test_retrieval_regression.py`). **Measured in a clean install from `requirements.txt`: 977 MB on disk (was 1.7 GB), and 415 MB peak RAM** for a full workload — booting the chat graph, four retrievals, all six fits, the derivative plot and a PDF. |
| Builds from the repository, no persistent disk | `data/vectordb/` (38 MB) is committed. Without it the app boots with an empty knowledge base and the consultant answers nothing. |
| No `.env` file on the host | The Gemini key goes in the app's Secrets; Streamlit exposes it as the `GEMINI_API_KEY` environment variable the config already reads. |

## Steps

1. **Push the branch to GitHub.** The repository can be private — Community Cloud
   can deploy from a private repo, which is also the safer choice while the
   publisher PDFs under `knowledge source/` remain in the history (see below).
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose
   **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `madhavseth512/HydroAssist`
   - **Branch:** the branch you pushed
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version: 3.11.** Verified on a clean 3.11 install from
     `requirements.txt`, which resolves to the versions the host will get
     (Streamlit 1.63, chromadb 1.5.9): the full suite passes — 805 unit and
     integration tests plus the 14 Streamlit UI tests that boot the real
     `app.py` — with no deprecation warnings raised. 3.14 is only what the local
     venv happens to use.
   - **Secrets:** paste the contents of `.streamlit/secrets.toml.example` with
     your real key.
5. **Deploy.** The first build installs dependencies (a few minutes).

## What to expect

- **First visitor after a cold start waits longer.** ChromaDB downloads its ONNX
  model (~80 MB) on first use and caches it for the life of the container.
- **The app sleeps after 12 hours without traffic.** The next visitor wakes it,
  with another cold start.
- **Uploaded CSVs and results do not persist.** Nothing is saved between sessions
  on the free tier — which is also why saved projects are planned as a file the
  user downloads and re-opens rather than server-side storage.

## Verifying the deployment

1. The page loads and the sidebar shows the model and message count.
2. Ask the consultant *"What are the assumptions of the Theis method?"* — the
   answer should cite sources in the **Retrieved Sources** expander. An answer
   saying it lacks information means the vector store did not deploy.
3. Upload `tests/theis_cooper_jacob_sample.csv`, set Q = 500 m³/day and r = 50 m,
   and run **Theis (1935)**. Expect T ≈ 215.3 m²/day and S ≈ 5.05×10⁻⁵.

## Before making the app public

`knowledge source/` contains roughly thirty publisher PDFs (SPE, Elsevier,
Wiley/AGU) tracked in git. The app does not need them at runtime — only the built
index — but they are in the repository and its history. Deploying from a private
repository avoids redistributing them; making the repository public does not. This
is a licensing decision, not a technical one, and should be settled first.
