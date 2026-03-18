# Adding Your Custom Hydrogeology PDFs

## 📁 Where to Put Your PDFs

You can place your PDF files in:
```
/home/nightfury/myStuff/testing-random/hydroassist/data/raw/custom_pdfs/
```

Or keep them in any directory you prefer and point the script to it.

## 🚀 How to Ingest Your PDFs

### Option 1: Ingest All PDFs from a Directory

```bash
# If your PDFs are in the default location
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs

# If your PDFs are in a different location
python scripts/ingest_custom_pdfs.py --pdf-dir /path/to/your/pdfs
```

### Option 2: Ingest a Single PDF

```bash
python scripts/ingest_custom_pdfs.py --pdf-file /path/to/document.pdf
```

### Option 3: With Custom Metadata

You can specify metadata for better retrieval:

```bash
# For confined aquifer documents
python scripts/ingest_custom_pdfs.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --source "Research Papers" \
    --aquifer confined \
    --method "Theis_1935"

# For general hydrogeology textbooks
python scripts/ingest_custom_pdfs.py \
    --pdf-dir ./my_textbooks \
    --source "Textbooks" \
    --aquifer general \
    --method multiple
```

## 📊 Metadata Options

### Source (`--source`)
Give your documents a recognizable source name:
- `"Research Papers"`
- `"Class Notes"`
- `"Textbooks"`
- `"Lab Reports"`
- `"Project Documents"`

### Aquifer Type (`--aquifer`)
Choose the most relevant type:
- `confined` - For confined aquifer studies
- `unconfined` - For unconfined/water table aquifers
- `leaky` - For leaky confined aquifers
- `fractured` - For fractured rock aquifers
- `general` - Mixed or general hydrogeology (default)

### Method (`--method`)
Specify if documents focus on a specific method:
- `Theis_1935`
- `Cooper_Jacob_1946`
- `Hantush_Jacob_1955`
- `Neuman_1972`
- `multiple` - Multiple methods or general (default)

## 📝 Complete Workflow

### Step 1: Copy Your PDFs

```bash
# Create directory if needed
mkdir -p data/raw/custom_pdfs

# Copy your PDFs there
cp /path/to/your/*.pdf data/raw/custom_pdfs/
```

### Step 2: Ingest Standard Sources (One-time)

```bash
# This gets USGS and AQTESOLV data
python scripts/ingest.py --all
```

### Step 3: Ingest Your Custom PDFs

```bash
# Ingest your custom documents
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs
```

### Step 4: Test Retrieval

```bash
# Test if your documents are searchable
python scripts/test_retrieval.py "query related to your PDFs"
```

### Step 5: Use the Web Interface

```bash
streamlit run app.py
```

## 🎯 Examples

### Example 1: Class Lecture Notes

```bash
# You have lecture PDFs in ~/Documents/Hydro_Lectures/
python scripts/ingest_custom_pdfs.py \
    --pdf-dir ~/Documents/Hydro_Lectures/ \
    --source "Class Lectures" \
    --aquifer general
```

### Example 2: Theis Method Paper

```bash
# You have the original Theis (1935) paper
python scripts/ingest_custom_pdfs.py \
    --pdf-file ./papers/Theis_1935_original.pdf \
    --source "Original Paper" \
    --aquifer confined \
    --method "Theis_1935"
```

### Example 3: Multiple Research Papers

```bash
# You have various research papers
python scripts/ingest_custom_pdfs.py \
    --pdf-dir ./research_papers/ \
    --source "Research" \
    --aquifer general \
    --method multiple
```

## 🔍 What Happens During Ingestion

1. **PDF Processing**: Extracts text page by page
2. **Cleaning**: Removes headers/footers, normalizes text
3. **Chunking**: Splits into ~1000 character chunks with 200 character overlap
4. **Metadata**: Attaches source, page number, aquifer type, etc.
5. **Embedding**: Creates vector embeddings using sentence-transformers
6. **Storage**: Adds to ChromaDB vector store
7. **Indexing**: Indexes by metadata for filtered retrieval

## ⚡ Pro Tips

### 1. **Organize by Topic**

Instead of one big directory, organize by category:

```
data/raw/custom_pdfs/
├── textbooks/
├── papers/
├── reports/
└── lecture_notes/
```

Then ingest each separately with appropriate metadata:

```bash
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs/textbooks --source "Textbooks"
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs/papers --source "Research Papers"
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs/reports --source "Field Reports"
```

### 2. **Verify After Ingestion**

Check the collection stats:

```python
from src.retrieval.vectorstore import VectorStoreManager
from src.core.config import load_config

config = load_config()
vm = VectorStoreManager(config.vectorstore.persist_directory, config.vectorstore.collection_name)
print(vm.get_collection_stats())
```

### 3. **Re-ingest if Needed**

The system handles duplicates by chunk_id, so you can re-run ingestion if needed. Or reset and start fresh:

```bash
# Reset database
python scripts/reset_db.py --confirm

# Re-ingest everything
python scripts/ingest.py --all
python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs
```

## 🚨 Troubleshooting

### PDF Not Processing?

```bash
# Check if PDF is readable
python -c "from pypdf import PdfReader; print(len(PdfReader('your.pdf').pages))"
```

### Low Quality Extraction?

Some PDFs (especially scanned ones) may have poor text extraction. Consider:
- Using better quality PDFs
- Running OCR preprocessing
- Adjusting chunk size: `--chunk-size 800`

### Memory Issues?

If processing many large PDFs:
- Process in batches
- Reduce chunk size
- Close other applications

## 📚 Your Custom PDFs + Standard Sources

After ingestion, your system will have:

1. **USGS TWI Book 3-B1** (standard reference)
2. **AQTESOLV Methods** (classical methods metadata)
3. **Your Custom PDFs** (your specific documents)

The RAG system will search across **ALL** sources and return the most relevant information with proper citations!

---

**Ready to add your PDFs?** Just put them in a folder and run the ingestion script! 🚀