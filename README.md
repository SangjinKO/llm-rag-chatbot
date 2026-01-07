# Multi RAG + LLM Comparison

**Date:** 2026.01  

## Goal
Build a **transparent, evidence-grounded RAG system** that allows:
- Comparing **multiple LLMs** under identical conditions
- Observing how **prompt strategy (STRICT vs GENEROUS)** changes answers
- Minimizing hallucination by forcing answers to stay within retrieved context
- Clearly visualizing **retrieved evidence → prompt → answer** flow



## What This Project Focuses On

- End-to-end RAG pipeline (PDF → chunks → embeddings → FAISS → retrieval)
- **Prompt strategy comparison**
  - STRICT: hallucination-safe, evidence-only
  - GENEROUS: inference-friendly but still evidence-grounded
- Behavioral differences across LLM providers
- Lightweight, interactive Streamlit demo suitable for interviews and portfolios


## Tech Stack

### Core
- **Language:** Python 3.x
- **UI:** Streamlit
- **PDF Parsing:** pypdf
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (CPU)

### LLM Runtime
- **Local:** Ollama(LLaMA)
- **Cloud:** OpenAI API (ChatGPT), Google Gemini API



## Key Features

- Upload and process **text-based PDF documents**
- Semantic retrieval using FAISS
- Automatic **Top-k evidence extraction**
- Side-by-side comparison:
  - STRICT vs GENEROUS
  - Across multiple LLMs
- Transparent display of:
  - Retrieved chunks
  - Merged context block
  - Prompt text (optional)
- Supports **local-only** or **local + cloud hybrid** execution



## Prompt Strategies

### STRICT
- Answers **only if explicitly supported** by retrieved evidence
- Returns *“Not specified in the document”* when information is missing
- Designed to minimize hallucination

### GENEROUS
- Allows **reasonable inference** when clearly implied by evidence
- Produces more user-friendly summaries
- Still restricted to retrieved context (no external knowledge)


## Application Flow

1. Upload a PDF document  
2. Build FAISS index (cached)  
3. Enter a question  
4. Retrieve Top-k evidence chunks  
5. Generate merged context  
6. Compare answers:
   - STRICT vs GENEROUS  
   - Across Ollama / ChatGPT / Gemini  


## Repository Structure
This repository contains two versions of the project:
- **v1_local**: Initial version (local-only RAG with Ollama)
- **v2_local-to-cloud**: The 1nd version (multi-LLM comparison with local + cloud models)

```text
llm-rag-chatbot/
├── .env
├── v1_local/                   # v1: Local-only RAG chatbot
│   └── app.py
└── v2_local-to-cloud/          # v2: Local → Local+Cloud expansion
    ├── app/
        └── app.py  

```

## How to Run

### 1. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and run Ollama (local LLM server)
- download Ollama from https://ollama.com/download
- launch Ollama.app
- pull the test model: 
```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

### 3. Configure LLM environment (.env)
- create a `.env` file in the project root (`v2_local-to-cloud/`) and set the LLM models and API keys.

```env
# Local LLM (Ollama)
OLLAMA_MODEL=llama3.2:3b

# OpenAI (ChatGPT)
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini

# Google Gemini
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run the Streamlit app
```bash
cd ./v2_local-to-cloud/
streamlit run app/app.py
```

<img width="1136" height="621" alt="image" src="https://github.com/user-attachments/assets/434ed7c6-7cc2-4e3d-a9e9-855a82a3ddf6" />

