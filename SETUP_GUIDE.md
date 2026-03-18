# HydroAssist - Quick Setup Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure API Key

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
nano .env  # or use any text editor
```

Add this line to `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 3: Ingest Knowledge Base

```bash
# This downloads USGS documents and loads method metadata
python scripts/ingest.py --all

# Expected output:
# - Downloading USGS TWI Book 3-B1...
# - Processing PDF...
# - Creating chunks...
# - Adding to vector store...
# - Total chunks: ~500-1000
# Takes 5-10 minutes on first run
```

### Step 4: Test the System

```bash
# Test retrieval
python scripts/test_retrieval.py "What is the Theis method?"

# Should return 5 relevant documents with scores
```

### Step 5: Launch Web Interface

```bash
# Start Streamlit
streamlit run app.py

# Opens browser at http://localhost:8501
```

## ✅ Verification Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Gemini API key configured in `.env`
- [ ] Knowledge base ingested (vector DB populated)
- [ ] Retrieval test successful
- [ ] Web interface launches

## 🔧 Common Issues

### Issue: ModuleNotFoundError

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: API Key Not Found

```bash
# Check .env file exists
ls -la .env

# Verify key is set
cat .env | grep GEMINI_API_KEY

# Load environment manually for testing
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GEMINI_API_KEY'))"
```

### Issue: Vector Store Empty

```bash
# Re-run ingestion
python scripts/ingest.py --all --force-download

# Check database stats
python -c "from src.retrieval.vectorstore import VectorStoreManager; vm = VectorStoreManager(); print(vm.get_collection_stats())"
```

### Issue: Embedding Model Download Fails

```bash
# Manually download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## 📝 Example Queries

Once the system is running, try these queries:

### Theoretical Questions
- "What is the Theis method?"
- "Explain the Cooper-Jacob approximation"
- "What assumptions does Hantush-Jacob make?"
- "When should I use Neuman's method?"

### Comparative Questions
- "What's the difference between Theis and Cooper-Jacob?"
- "Compare confined and unconfined aquifer analysis"
- "Which method is best for leaky aquifers?"

### Practical Questions
- "How do I analyze a confined aquifer?"
- "What data do I need for pumping test analysis?"
- "Explain transmissivity and storativity"

### Calculation Requests (Phase 2 stub)
- "Calculate transmissivity from my data"
- "Analyze this pumping test"

## 🎓 Understanding the System

### How It Works

1. **User asks a question** → Enters chat interface
2. **Manager Agent** → Classifies intent (consultation/calculation)
3. **Consultant Agent** → Retrieves relevant documents from vector DB
4. **Gemini LLM** → Synthesizes answer with citations
5. **User receives response** → With source citations

### Data Flow

```
User Query
    ↓
Manager (Intent Classification)
    ↓
Consultant (RAG Retrieval)
    ↓
[Vector DB] → Retrieve Top-K Similar Chunks
    ↓
Format Context + Prompt
    ↓
[Gemini LLM] → Generate Answer
    ↓
Response with Citations
```

### Key Files

- `app.py` - Streamlit web interface
- `src/core/graph.py` - LangGraph orchestration
- `src/agents/` - Agent implementations
- `src/retrieval/retriever.py` - RAG retrieval logic
- `scripts/ingest.py` - Data ingestion pipeline

## 🔄 Development Workflow

### Making Changes

```bash
# 1. Make code changes
nano src/agents/consultant.py

# 2. Test locally
python scripts/test_retrieval.py "test query"

# 3. Run web interface
streamlit run app.py

# 4. If changing data processing
python scripts/reset_db.py --confirm
python scripts/ingest.py --all
```

### Adding New Data Sources

1. Create scraper in `src/ingestion/scrapers/`
2. Add processor logic in `src/ingestion/processors/`
3. Update `pipeline.py` to handle new source
4. Run ingestion: `python scripts/ingest.py --source new_source`

### Customizing Prompts

Edit prompt files in `src/prompts/`:
- `manager_prompts.py` - Intent classification
- `consultant_prompts.py` - RAG synthesis
- `calculator_prompts.py` - Phase 2 responses

## 📊 Monitoring

### Check System Status

```bash
# Vector store stats
python -c "
from src.core.config import load_config
from src.retrieval.vectorstore import VectorStoreManager
config = load_config()
vm = VectorStoreManager(config.vectorstore.persist_directory, config.vectorstore.collection_name)
print(vm.get_collection_stats())
"

# Test retrieval quality
python scripts/test_retrieval.py "confined aquifer" --top-k 10
```

### Logs

Check logs in Streamlit output or configure file logging in `configs/default.yaml`:

```yaml
logging:
  level: "DEBUG"  # More verbose
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 🎯 Next Steps

1. **Explore the codebase** - Read through `src/` modules
2. **Try different queries** - Test edge cases
3. **Customize prompts** - Adjust for your use case
4. **Add more data** - Ingest additional sources
5. **Prepare for Phase 2** - Start designing calculation module

## 📚 Additional Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Gemini API Docs](https://ai.google.dev/docs)

## 💬 Need Help?

1. Check [README.md](README.md) for detailed documentation
2. Review [Troubleshooting](#-common-issues) section above
3. Check logs for error messages
4. Verify all dependencies are installed correctly

---

**Happy analyzing! 🌊**
