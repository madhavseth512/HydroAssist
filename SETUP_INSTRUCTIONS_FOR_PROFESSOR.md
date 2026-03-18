# HydroAssist - Setup Instructions

**HydroAssist Phase 1 (BTP-1)**: An open-source intelligent hydrogeological consultant chatbot powered by RAG (Retrieval Augmented Generation) with LangGraph.

---

## Quick Start Guide

### Prerequisites
- **Python 3.10+** (recommended: Python 3.10 or 3.11)
- **Conda** or **venv** for virtual environment
- **Gemini API Key** (free tier available at https://ai.google.dev/)

### Step 1: Extract the Project
```bash
unzip hydroassist_demo.zip
cd hydroassist
```

### Step 2: Create Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n hydroassist python=3.10 -y
conda activate hydroassist
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Important:** If you encounter dependency conflicts, use these specific versions:
```bash
pip install langchain-google-genai==2.0.10 google-generativeai==0.8.5
pip install 'langchain<1.0' 'langchain-core<1.0'
```

### Step 4: Get Gemini API Key

1. Visit: https://ai.google.dev/
2. Click "Get API Key in Google AI Studio"
3. Create a new API key (free tier available)
4. Copy the API key

### Step 5: Configure API Key

Create a `.env` file in the project root:
```bash
# Create .env file
cat > .env << 'EOF'
GEMINI_API_KEY=your_api_key_here
EOF
```

**Replace** `your_api_key_here` with your actual Gemini API key.

### Step 6: Launch the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## What's Included

### Knowledge Base (3,843 chunks)
- **47 Custom PDFs** organized by aquifer type:
  - 10 Confined Aquifer papers
  - 14 Unconfined Aquifer papers
  - 12 Leaky Confined Aquifer papers
  - 7 Fractured Aquifer papers
  - 5 General hydrogeology papers
- **6 Classical Methods** (AQTESOLV metadata):
  - Theis (1935)
  - Cooper-Jacob (1946)
  - Hantush-Jacob (1955)
  - Neuman (1972)
  - Boulton (1963)
  - Moench (1985)

### Technology Stack
- **LangGraph**: State machine orchestration
- **Gemini 2.5 Flash**: Google's latest LLM
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Local embedding generation
- **Streamlit**: Web interface
- **RAG Architecture**: Retrieval Augmented Generation

### Project Structure
```
hydroassist/
├── app.py                          # Streamlit web interface
├── configs/
│   └── default.yaml                # Configuration file
├── src/
│   ├── agents/                     # Agent implementations
│   │   ├── manager.py              # Intent classification agent
│   │   ├── consultant.py           # RAG-based Q&A agent
│   │   └── calculator.py           # Phase 2 stub
│   ├── core/
│   │   ├── config.py               # Configuration loader
│   │   ├── state.py                # LangGraph state management
│   │   └── graph.py                # Workflow orchestration
│   ├── retrieval/
│   │   ├── embeddings.py           # Embedding manager
│   │   ├── vectorstore.py          # ChromaDB interface
│   │   └── retriever.py            # Metadata-filtered retrieval
│   ├── ingestion/                  # Data ingestion pipeline
│   └── utils/                      # Utilities
├── scripts/
│   ├── ingest.py                   # Ingest standard sources
│   ├── ingest_knowledge_source.py  # Ingest custom PDFs
│   └── test_retrieval.py           # Test retrieval system
├── data/
│   ├── vectordb/                   # ChromaDB persistent storage
│   └── knowledge source/           # PDF collection (organized by type)
└── requirements.txt                # Python dependencies
```

---

## Usage Examples

### Sample Queries to Try

1. **Basic Method Inquiry:**
   ```
   "What is the Theis method?"
   ```

2. **Method Comparison:**
   ```
   "Compare the Theis and Neuman methods"
   ```

3. **Aquifer-Specific Questions:**
   ```
   "How does delayed yield affect unconfined aquifer tests?"
   ```

4. **Application Questions:**
   ```
   "When should I use the Cooper-Jacob approximation instead of Theis?"
   ```

5. **Theoretical Concepts:**
   ```
   "Explain the assumptions of leaky aquifer analysis"
   ```

### Interface Features

- **Chat Interface**: Natural language conversation
- **Retrieved Sources**: View sources with citations and similarity scores
- **Current Context**: See aquifer type, intent, and selected method
- **Export Chat**: Download conversation history
- **Reset Conversation**: Start fresh session

---

## Troubleshooting

### Issue: "API key not found"
**Solution:** Ensure `.env` file exists with `GEMINI_API_KEY=your_key`

### Issue: "404 models/gemini-xxx not found"
**Solution:** The config uses `gemini-2.5-flash`. If unavailable, edit `configs/default.yaml`:
```yaml
llm:
  model: "gemini-pro"  # Fallback model
```

### Issue: Rate limit errors
**Solution:**
- Free tier has limits: 15 requests/minute
- Wait a few minutes between queries
- Or upgrade to paid tier

### Issue: Dependency conflicts
**Solution:**
```bash
pip install langchain-google-genai==2.0.10 google-generativeai==0.8.5 --force-reinstall
pip install 'langchain<1.0' 'langchain-core<1.0'
```

### Issue: Empty vector database
**Solution:** Re-run ingestion:
```bash
python scripts/ingest_knowledge_source.py
```

### Issue: Slow first query
**Note:** First query loads embedding model (~6 seconds). Subsequent queries are fast.

---

## Testing the System

### Test Retrieval
```bash
python scripts/test_retrieval.py "What is the Theis method?"
```

### Check Database Stats
```bash
python -c "
from src.core.config import load_config
from src.retrieval.vectorstore import VectorStoreManager

config = load_config()
vm = VectorStoreManager(
    persist_directory=config.vectorstore.persist_directory,
    collection_name=config.vectorstore.collection_name
)
print(vm.get_collection_stats())
"
```

---

## Performance Metrics

- **Total Documents**: 3,843 chunks
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Average Retrieval Time**: 0.1-0.5 seconds
- **Average LLM Response Time**: 2-4 seconds
- **Similarity Threshold**: 0.2 (configurable)
- **Top-K Results**: 5 (configurable)

---

## Project Details

**Developed By:** BTP-1 Team
**Institution:** [Your Institution]
**Phase:** 1 - Consultation (Active)
**Future Phase:** 2 - Calculation & Curve-Fitting

### Phase 1 Features (Current)
✅ Intent classification (Manager Agent)
✅ RAG-based Q&A (Consultant Agent)
✅ Citation generation
✅ Metadata-filtered retrieval
✅ Multi-agent orchestration with LangGraph
✅ Web interface with Streamlit

### Phase 2 Roadmap
🚧 Calculator Agent implementation
🚧 Curve-fitting analysis
🚧 Parameter estimation (T, S, K)
🚧 Diagnostic plots
🚧 CSV data upload

---

## Documentation

- **README.md**: Comprehensive project overview
- **SETUP_GUIDE.md**: Detailed setup instructions
- **PROJECT_SUMMARY.md**: Technical architecture
- **CUSTOM_PDF_GUIDE.md**: Adding custom PDFs
- **BATCH_INGESTION_GUIDE.md**: Batch processing guide

---

## Support & Contact

For issues or questions:
1. Check troubleshooting section above
2. Review documentation files
3. Contact: [Your Email]
4. Project Repository: [Your Repo URL]

---

## License

Open Source - Built for educational and research purposes.

**Citation:**
If you use this work, please cite:
```
HydroAssist Phase 1 (BTP-1): An Intelligent Hydrogeological Consultant
Powered by RAG and LangGraph
[Your Institution], 2025
```

---

## System Requirements

**Minimum:**
- 4GB RAM
- 2GB free disk space
- Internet connection (for API calls)

**Recommended:**
- 8GB+ RAM
- 5GB free disk space
- Stable internet connection

---

**Last Updated:** November 26, 2025
**Version:** 1.0.0 (Phase 1)
