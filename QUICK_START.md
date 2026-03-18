# HydroAssist - Quick Start (30 seconds)

## 🚀 Fastest Way to Run

```bash
# 1. Extract and enter directory
unzip hydroassist_demo.zip
cd hydroassist

# 2. Create environment and install
conda create -n hydroassist python=3.10 -y
conda activate hydroassist
pip install -r requirements.txt

# 3. Add your Gemini API key to .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Launch the app
streamlit run app.py
```

## 📝 Get Free Gemini API Key
Visit: https://ai.google.dev/ → "Get API Key in Google AI Studio"

## ✅ What Works Out of the Box
- ✅ **3,843 pre-indexed chunks** from 47 hydrogeology PDFs
- ✅ **Vector database** already populated (ChromaDB)
- ✅ **6 classical methods** (Theis, Cooper-Jacob, Neuman, etc.)
- ✅ **Web interface** with chat, sources, and citations

## 🎯 Try These Queries
1. "What is the Theis method?"
2. "Compare Theis and Neuman methods"
3. "How does delayed yield affect unconfined aquifer tests?"

## 📦 What's Included (63 MB)
- Source code (Python)
- 47 hydrogeology research papers (PDF)
- Pre-built vector database (ChromaDB)
- Documentation (5 guides)
- Configuration files

## 🔧 Troubleshooting
**If you get dependency errors:**
```bash
pip install langchain-google-genai==2.0.10 google-generativeai==0.8.5
pip install 'langchain<1.0' 'langchain-core<1.0'
```

## 📖 Full Instructions
See `SETUP_INSTRUCTIONS_FOR_PROFESSOR.md` for complete details.

---
**Ready in 30 seconds!** 🎉
