# Batch Ingestion Guide for 50 PDFs

## 🚀 Quick Start for Your 50 PDFs

### **Step 1: Organize Your PDFs**

Put all 50 PDFs in one place:
```bash
data/raw/custom_pdfs/
```

Or keep them wherever they are and point the script to that location.

### **Step 2: Run Batch Ingestion**

```bash
# Process all 50 PDFs in batches of 10
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10 \
    --source "My Research Collection"
```

Expected time: **1.5-2.5 hours** (depending on PDF sizes)

### **Step 3: Monitor Progress**

You'll see output like:
```
========================================
BATCH 1/5
Processing PDFs 1 to 10
========================================

[1/50] Processing: Theis_1935.pdf...
  ✓ 20 pages → 150 chunks (5.2s)

[2/50] Processing: Cooper_Jacob_1946.pdf...
  ✓ 30 pages → 220 chunks (7.1s)

...

Batch 1 complete:
  • Processed: 10 PDFs
  • Chunks added: 1,850
  • Time: 12.5 minutes
  • Avg: 75s per PDF
```

## ⚙️ Batch Processing Options

### **Standard Processing (Recommended)**
```bash
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10
```

### **Faster (Smaller Batches)**
If you have memory constraints:
```bash
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 5
```

### **Resume After Interruption**
If processing stops at PDF #25:
```bash
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10 \
    --start-from 25
```

### **With Custom Metadata**
```bash
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10 \
    --source "Hydrogeology Papers" \
    --aquifer general \
    --method multiple
```

## 📊 What to Expect

### **Timeline for 50 PDFs**

| Batch | PDFs | Estimated Time | Progress |
|-------|------|----------------|----------|
| 1 | 1-10 | 15-20 min | ████░░░░░░ 20% |
| 2 | 11-20 | 15-20 min | ████████░░ 40% |
| 3 | 21-30 | 15-20 min | ████████████░ 60% |
| 4 | 31-40 | 15-20 min | ████████████████░ 80% |
| 5 | 41-50 | 15-20 min | ████████████████████ 100% |

**Total: 1.5-2.5 hours**

### **System Resources During Processing**

```
CPU Usage: 50-70% (single core)
RAM Usage: 2-4 GB
Disk I/O: Moderate
Network: None (all local)
```

### **Final Database Stats**

After 50 PDFs (estimated):
```
Total chunks: 5,000-10,000
Vector DB size: 500 MB - 1 GB
Sources: Custom (50 PDFs)
Retrieval speed: 300-500ms
```

## 🎯 Optimization Tips

### **1. Organize by Category First**

If your 50 PDFs fall into categories:

```bash
# Textbooks (10 PDFs)
python scripts/ingest_batch.py \
    --pdf-dir ./pdfs/textbooks \
    --source "Textbooks" \
    --batch-size 10

# Research Papers (20 PDFs)
python scripts/ingest_batch.py \
    --pdf-dir ./pdfs/papers \
    --source "Research Papers" \
    --batch-size 10

# Reports (20 PDFs)
python scripts/ingest_batch.py \
    --pdf-dir ./pdfs/reports \
    --source "Field Reports" \
    --batch-size 10
```

### **2. Test with Small Batch First**

Process just 5-10 PDFs to test:
```bash
# Create test folder
mkdir -p ./data/raw/custom_pdfs/test
# Copy 5-10 important PDFs there

# Test ingestion
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs/test \
    --batch-size 5

# If successful, proceed with all 50
```

### **3. Run Overnight**

For 50 PDFs, consider running overnight:
```bash
# Run in background with logging
nohup python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10 \
    > ingestion.log 2>&1 &

# Check progress
tail -f ingestion.log
```

## 🔧 Memory Optimization

If you encounter memory issues with 50 PDFs:

### **Option 1: Reduce Batch Size**
```bash
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 5  # Smaller batches
```

### **Option 2: Adjust Chunk Size**

Edit `configs/default.yaml`:
```yaml
retrieval:
  chunk_size: 800  # Reduced from 1000
  chunk_overlap: 150  # Reduced from 200
```

### **Option 3: Process in Multiple Sessions**

```bash
# Session 1: First 25 PDFs
python scripts/ingest_batch.py --pdf-dir ./pdfs --batch-size 10 --start-from 0

# Close everything, free memory

# Session 2: Next 25 PDFs
python scripts/ingest_batch.py --pdf-dir ./pdfs --batch-size 10 --start-from 25
```

## 📝 Complete Workflow for 50 PDFs

### **Day 1: Setup & Standard Sources**

```bash
# 1. Set up environment (DONE)
conda activate hydroassist

# 2. Add Gemini API key to .env (DO THIS)
nano .env

# 3. Ingest standard sources (10-15 min)
python scripts/ingest.py --all
```

### **Day 2: Your 50 Custom PDFs**

```bash
# 4. Copy your 50 PDFs
cp /path/to/your/pdfs/*.pdf ./data/raw/custom_pdfs/

# 5. Run batch ingestion (1.5-2.5 hours)
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --batch-size 10 \
    --source "My Collection"
```

### **Day 3: Test & Deploy**

```bash
# 6. Test retrieval
python scripts/test_retrieval.py "your test query"

# 7. Launch web interface
streamlit run app.py
```

## 🚨 Troubleshooting

### **Processing Stops/Crashes?**

**Resume from where it stopped:**
```bash
# If it stopped at PDF 23
python scripts/ingest_batch.py \
    --pdf-dir ./data/raw/custom_pdfs \
    --start-from 23
```

### **Some PDFs Fail?**

The script will continue and report failed PDFs at the end:
```
⚠ Failed PDFs (3):
  • scanned_document.pdf: Text extraction failed
  • encrypted.pdf: PDF is password protected
  • corrupted.pdf: Invalid PDF structure
```

You can process them separately or skip them.

### **Check What's Already Ingested**

```bash
python -c "
from src.retrieval.vectorstore import VectorStoreManager
from src.core.config import load_config

config = load_config()
vm = VectorStoreManager(config.vectorstore.persist_directory, config.vectorstore.collection_name)
stats = vm.get_collection_stats()

print(f'Total chunks: {stats.get(\"total_chunks\", 0):,}')
print('Sources:', stats.get('sources', {}))
"
```

## 🎯 Final Tips for 50 PDFs

1. **Start with standard sources** (USGS + AQTESOLV) to test the system
2. **Then add your 50 PDFs** in batches
3. **Use meaningful source names** (helps with citations later)
4. **Monitor the first batch** to ensure everything works
5. **Run longer batches overnight** if convenient
6. **Don't worry about interruptions** - you can always resume

## ⏱️ Time Investment

- **Setup**: 10 minutes (already done!)
- **Standard ingestion**: 10-15 minutes
- **Your 50 PDFs**: 1.5-2.5 hours
- **Testing**: 5-10 minutes
- **Total**: ~2-3 hours one-time setup

After this, you'll have a **fully loaded knowledge base** with:
- USGS authoritative references
- AQTESOLV method descriptions
- Your 50 custom PDFs
- **~10,000-15,000 searchable chunks** total!

The chatbot will search across **everything** and give you cited answers! 🚀

---

**Ready when you are!** Just add your API key and let me know if you want to start with standard sources or jump straight to your 50 PDFs.