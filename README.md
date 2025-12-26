# llm-rag-chatbot

Date: 2025.12

Goal: Build a local, cost-free LLM-based RAG (Retrieval-Augmented Generation) chatbot that answers user questions strictly grounded in PDF documents, while making the retrieval and prompt behavior transparent and explainable.

This project focuses on:
- Understanding and implementing the full RAG pipeline
- Comparing strict vs inference-allowed prompting strategies
- Avoiding hallucination while maintaining practical usability
- Demonstrating the system via a lightweight web UI

Dev Environment:
- Language: Python 3.x
- UI Framework: Streamlit
- Document Processing: pypdf
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Vector Store: FAISS (CPU)
- LLM Runtime: Ollama (local LLM, llama3.2:3b)

Features:
- Upload and process text-based PDF documents
- Semantic search using vector similarity (FAISS)
- Evidence-based question answering (RAG)
- Side-by-side comparison of (1) STRICT prompt (hallucination-safe) and (2) GENEROUS prompt (inference-friendly)
- Transparent display of retrieved context (evidence)
- Fully local execution (no paid APIs, no external data leakage)

Prompt Strategies: 
- STRICT Prompt: Answers only when information is explicitly present. Responds with “Not specified in the document” when insufficient. Designed to minimize hallucination
- GENEROUS Prompt: Allows reasonable inference from clearly implied instructions. Produces more user-friendly summaries. Still constrained to retrieved document context

Application Flow (UI Demo): 
- Upload a PDF document
- Enter a question (e.g. “What are the prerequisites before installation?”)
- Inspect retrieved evidence chunks (Top-k)
- Compare answers: STRICT and GENEROUS
- Validate how prompt strategy affects answer quality

<img width="1040" height="582" alt="image" src="https://github.com/user-attachments/assets/819bda9b-a7fa-4073-9f5d-91b5fd557e21" />

How to Run:
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
```


### 3. Run the Streamlit app
```bash
streamlit run app.py
```

