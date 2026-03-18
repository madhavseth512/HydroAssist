# 💧 HydroAssist - Intelligent Hydrogeological Consultant

**Phase 1 (BTP-1)**: An open-source RAG-based chatbot for hydrogeological pumping test analysis, powered by LangGraph and Gemini AI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

HydroAssist is an intelligent consultant that helps hydrogeologists understand pumping test analysis methods, select appropriate techniques for different aquifer conditions, and explains the theoretical foundations of aquifer characterization.

**Current Capabilities (Phase 1):**
- 📖 Answer questions about pumping test analysis methods (Theis, Cooper-Jacob, Hantush-Jacob, etc.)
- 🎯 Method selection guidance for different aquifer types
- 📐 Explain assumptions and limitations of various methods
- 🔬 Discuss aquifer characterization theory
- 📚 Cite sources from USGS publications and scientific literature

**Phase 2 Roadmap (BTP-2):**
- Curve-fitting analysis using nonlinear least squares
- Automatic parameter estimation (T, S, K)
- Diagnostic plots generation
- Data validation and quality checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Manager  │───▶│Consultant│    │Calculator│              │
│  │  Agent   │    │  Agent   │    │  Agent   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌─────────────┐              ┌──────────────────┐
    │   Gemini    │              │   RAG Pipeline   │
    │ 2.0 Flash   │              │  ┌────────────┐  │
    │     LLM     │              │  │ ChromaDB   │  │
    └─────────────┘              │  │VectorStore │  │
                                 │  └────────────┘  │
                                 │  ┌────────────┐  │
                                 │  │ Sentence   │  │
                                 │  │Transformers│  │
                                 │  └────────────┘  │
                                 └──────────────────┘
```

### Key Components

- **Manager Agent**: Intent classification and conversation routing using Gemini
- **Consultant Agent**: RAG-based Q&A with retrieval from knowledge base
- **Calculator Agent**: Phase 2 stub for numerical analysis (coming soon)
- **Vector Store**: ChromaDB with metadata-filtered similarity search
- **Embeddings**: Sentence-transformers for local embedding generation
- **LLM**: Google Gemini 2.0 Flash for reasoning and response generation

## 📋 Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM (for embedding model)
- 2GB+ disk space (for vector database)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hydroassist.git
cd hydroassist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Ingest Knowledge Base

```bash
# Download and process USGS documents + AQTESOLV metadata
python scripts/ingest.py --all

# This will:
# - Download USGS TWI Book 3-B1 (public domain)
# - Load curated pumping test method metadata
# - Chunk documents (1000 chars, 200 overlap)
# - Create embeddings and populate vector store
# - Takes ~5-10 minutes on first run
```

### 4. Launch Web Interface

```bash
# Start Streamlit app
streamlit run app.py

# Opens at http://localhost:8501
```

### 5. Test the System

```bash
# Test retrieval quality
python scripts/test_retrieval.py "What is the Theis method?"

# Test with filters
python scripts/test_retrieval.py "confined aquifer analysis" --aquifer confined --top-k 10
```

## 💻 Usage Examples

### Web Interface

1. **Open the app**: `streamlit run app.py`
2. **Ask questions**:
   - "What is the Theis method?"
   - "When should I use Cooper-Jacob instead of Theis?"
   - "What assumptions does Hantush-Jacob make?"
   - "Explain confined aquifer analysis"
3. **View context**: Check sidebar for aquifer type, selected method, and intent
4. **Export chat**: Download conversation history as text file

### CLI Scripts

#### Ingest Data

```bash
# Ingest all sources
python scripts/ingest.py --all

# Ingest specific source
python scripts/ingest.py --source usgs

# Force re-download
python scripts/ingest.py --all --force-download

# Custom chunk size
python scripts/ingest.py --all --chunk-size 800
```

#### Test Retrieval

```bash
# Basic query
python scripts/test_retrieval.py "What is the Theis method?"

# With filters
python scripts/test_retrieval.py "drawdown analysis" \
    --aquifer confined \
    --top-k 10

# Different config
python scripts/test_retrieval.py "Cooper-Jacob" \
    --config production
```

#### Reset Database

```bash
# Reset vector store (requires confirmation)
python scripts/reset_db.py --confirm
```

### Python API

```python
from src.core.graph import HydroAssistChat
from src.core.config import load_config

# Initialize chat
config = load_config()
chat = HydroAssistChat(config)

# Send message
response = chat.send_message("What is the Theis method?")
print(response)

# Get context
context = chat.get_context()
print(f"Aquifer: {context['aquifer']}")
print(f"Method: {context['method']}")

# Reset conversation
chat.reset()
```

## 📁 Project Structure

```
hydroassist/
├── src/
│   ├── core/              # State management, config, LangGraph orchestration
│   ├── agents/            # Manager, Consultant, Calculator agents
│   ├── retrieval/         # Embeddings, vector store, retriever
│   ├── ingestion/         # Data scrapers, processors, chunker
│   ├── prompts/           # LLM prompts for each agent
│   └── utils/             # Logger, metadata, citations
├── scripts/               # CLI tools (ingest, test, reset)
├── configs/               # YAML configurations (default, dev, prod)
├── data/                  # Raw data, processed chunks, vector DB
├── tests/                 # Unit and integration tests
├── app.py                 # Streamlit web interface
└── requirements.txt       # Python dependencies
```

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your_api_key_here
HYDROASSIST_ENV=default  # or development, production

# Optional overrides
HYDROASSIST_LLM_MODEL=gemini-2.0-flash-exp
HYDROASSIST_LLM_TEMPERATURE=0.1
HYDROASSIST_CHUNK_SIZE=1000
HYDROASSIST_TOP_K=5
HYDROASSIST_LOG_LEVEL=INFO
```

### Configuration Files

Edit `configs/default.yaml` (or create custom configs):

```yaml
retrieval:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  similarity_threshold: 0.7

vectorstore:
  persist_directory: "./data/vectordb"
  collection_name: "hydrogeology_kb"

llm:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  temperature: 0.1
  max_tokens: 8192
```

## 🧪 Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_retrieval.py

# Run with verbose output
pytest -v
```

## 📊 Data Sources

### USGS TWI Book 3-B1

- **Source**: USGS Techniques of Water-Resources Investigations
- **Document**: Book 3, Chapter B1 - "Methods of Determining Permeability, Transmissibility, and Drawdown"
- **Status**: Public domain
- **Content**: Comprehensive theory and methods for pumping test analysis

### AQTESOLV Method Metadata

- **Source**: Manually curated from peer-reviewed literature
- **Content**: Method names, aquifer types, basic assumptions
- **Methods Included**:
  - Theis (1935) - Confined aquifers
  - Cooper-Jacob (1946) - Confined aquifers, large time
  - Hantush-Jacob (1955) - Leaky confined aquifers
  - Neuman (1972) - Unconfined aquifers
  - Boulton (1963) - Unconfined aquifers
  - Moench (1985) - Fractured aquifers

**Note**: If you have official collaboration with AQTESOLV, update `src/ingestion/scrapers/aqtesolv_scraper.py` with proper API access.

## 🔧 Troubleshooting

### Vector Store is Empty

```bash
# Check if ingestion ran successfully
python scripts/ingest.py --all --validate

# Verify database
python scripts/test_retrieval.py "test query"
```

### API Key Issues

```bash
# Verify .env file exists and has correct key
cat .env | grep GEMINI_API_KEY

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

### Embedding Model Download

On first run, the sentence-transformers model (~90MB) will download automatically:

```bash
# If download fails, manually install
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Memory Issues

If you encounter memory errors:

```python
# Edit configs/default.yaml
retrieval:
  chunk_size: 800  # Reduce from 1000
  top_k: 3         # Reduce from 5
```

## 🛣️ Roadmap

### Phase 1 (Current) ✅
- [x] RAG-based consultant chatbot
- [x] LangGraph state management
- [x] Intent classification and routing
- [x] Metadata-filtered retrieval
- [x] Streamlit web interface
- [x] CLI tools for ingestion and testing

### Phase 2 (BTP-2) 🚧
- [ ] Pumping test data upload
- [ ] Curve-fitting analysis (nonlinear least squares)
- [ ] Parameter estimation (T, S, K)
- [ ] Diagnostic plots (time-drawdown, residuals)
- [ ] Goodness-of-fit metrics
- [ ] Multi-well analysis

### Phase 3 (Future) 💡
- [ ] Multi-user support with authentication
- [ ] Project management (save/load analyses)
- [ ] Export to PDF reports
- [ ] Batch processing
- [ ] REST API
- [ ] Integration with GIS tools

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests before committing
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **USGS** for public domain hydrogeological publications
- **Google** for Gemini API
- **LangChain & LangGraph** for orchestration framework
- **ChromaDB** for vector storage
- **Sentence-Transformers** for embeddings

## 📧 Contact

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 📚 References

### Key Papers

1. Theis, C.V. (1935). The relation between the lowering of the piezometric surface and the rate and duration of discharge of a well using groundwater storage. *Transactions of the American Geophysical Union*, 16(2), 519-524.

2. Cooper, H.H., & Jacob, C.E. (1946). A generalized graphical method for evaluating formation constants and summarizing well-field history. *Transactions of the American Geophysical Union*, 27(4), 526-534.

3. Hantush, M.S., & Jacob, C.E. (1955). Non-steady radial flow in an infinite leaky aquifer. *Transactions of the American Geophysical Union*, 36(1), 95-100.

4. Neuman, S.P. (1972). Theory of flow in unconfined aquifers considering delayed response of the water table. *Water Resources Research*, 8(4), 1031-1045.

### Additional Resources

- [USGS Groundwater Information](https://www.usgs.gov/mission-areas/water-resources/science/groundwater-information)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Built with ❤️ for hydrogeologists and groundwater professionals**
