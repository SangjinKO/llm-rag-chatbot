from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st

import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat

from pypdf import PdfReader
# Reuse your existing PDF extractor
# from src.pdf_loader import extract_pdf_text


# -----------------------------
# Prompts (Options: build_strict_prompt,build_generous_prompt)
# -----------------------------
def build_strict_prompt(question: str, context_block: str) -> str:
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "Not specified in the document" and list what is missing.

Return the answer as:
- A short title
- A bullet list of prerequisites/requirements

Question:
{question}

Context:
{context_block}
"""


def build_generous_prompt(question: str, context_block: str) -> str:
    return f"""You are a helpful assistant.

Based ONLY on the provided context:
- Identify any information that can reasonably be interpreted as prerequisites,
  requirements, or preparations needed before installation or first use.
- You may infer prerequisites if they are clearly implied by the context
  (e.g., required registrations, initial setup steps, or mandatory checks).

If the document does not explicitly list prerequisites, summarize them
as "Inferred prerequisites based on the document".

Return the answer as:
- A short title
- A bullet list of prerequisites/requirements

Question:
{question}

Context:
{context_block}
"""

# -----------------------------
# extract_pdf_text
# -----------------------------
@dataclass
class PageText:
    page_num: int
    text: str

def extract_pdf_text(pdf_path: Path) -> List[PageText]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        
        # Clean up the format (remove spaces)
        cleaned = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
        
        pages.append(PageText(page_num=i, text=cleaned))

    return pages

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# -----------------------------
# Index bundle
# -----------------------------
@dataclass
class IndexBundle:
    doc_id: str
    chunks: List[str]
    embeddings: np.ndarray  # (N, dim), float32
    index: faiss.Index


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def build_index_from_pdf_bytes(
    pdf_bytes: bytes,
    chunk_size: int,
    overlap: int,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> IndexBundle:
    """
    Build FAISS index from uploaded PDF bytes.
    Cached by Streamlit (resource cache) using function args.
    """
    doc_id = _hash_bytes(pdf_bytes)

    # Save PDF temporarily to read with your existing loader
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_pdf = tmp_dir / f"{doc_id}.pdf"
    tmp_pdf.write_bytes(pdf_bytes)

    pages = extract_pdf_text(tmp_pdf)

    full_text = "\n".join(p.text for p in pages)

    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    model = get_embed_model(embed_model_name)
    emb = model.encode(chunks, show_progress_bar=False)
    embeddings = np.asarray(emb, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return IndexBundle(doc_id=doc_id, chunks=chunks, embeddings=embeddings, index=index)


def search_top_k(bundle: IndexBundle, question: str, top_k: int, embed_model_name: str) -> Tuple[List[int], List[float]]:
    model = get_embed_model(embed_model_name)
    q_emb = np.asarray(model.encode([question], show_progress_bar=False), dtype="float32")
    distances, indices = bundle.index.search(q_emb, top_k)
    return indices[0].tolist(), distances[0].tolist()


def call_ollama(model_name: str, prompt: str) -> str:
    resp = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄🔎 PDF RAG Chatbot (FAISS + SentenceTransformers + Ollama)")
st.caption("Option A: Evidence shown | Option C: STRICT vs GENEROUS prompt comparison")

with st.sidebar:
    st.header("Settings")
    embed_model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    ollama_model = st.text_input("Ollama model", value="llama3.2:3b")
    top_k = st.slider("Top-k (retrieval)", min_value=2, max_value=10, value=5, step=1)
    chunk_size = st.slider("Chunk size (chars)", min_value=300, max_value=1200, value=500, step=50)
    overlap = st.slider("Overlap (chars)", min_value=0, max_value=300, value=100, step=25)
    show_context_block = st.checkbox("Show merged context block", value=False)
    show_prompt_text = st.checkbox("Show prompt text", value=False)

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

bundle: IndexBundle | None = None

if uploaded is None:
    st.info("Upload a PDF to build the vector index.")
    st.stop()

pdf_bytes = uploaded.read()

with st.spinner("Building / loading FAISS index..."):
    bundle = build_index_from_pdf_bytes(
        pdf_bytes=pdf_bytes,
        chunk_size=chunk_size,
        overlap=overlap,
        embed_model_name=embed_model_name,
    )

st.success(f"Index ready ✅  doc_id={bundle.doc_id} | chunks={len(bundle.chunks)}")

question_default = "What are the prerequisites or requirements before installation or first use?"
question = st.text_input("Question", value=question_default)

if st.button("🔍 Search + Generate Answers", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching relevant chunks..."):
        idxs, dists = search_top_k(bundle, question, top_k=top_k, embed_model_name=embed_model_name)

    # Evidence (retrieved chunks)
    st.subheader("Evidence (Top-k Retrieved Chunks)")
    for rank, (i, d) in enumerate(zip(idxs, dists), start=1):
        with st.expander(f"#{rank}  idx={i}  dist={d:.4f}", expanded=(rank <= 2)):
            st.write(bundle.chunks[i])

    contexts = [bundle.chunks[i] for i in idxs]
    context_block = "\n\n---\n\n".join(contexts)

    if show_context_block:
        st.subheader("Merged Context Block")
        st.code(context_block)

    strict_prompt = build_strict_prompt(question, context_block)
    generous_prompt = build_generous_prompt(question, context_block)

    if show_prompt_text:
        colp1, colp2 = st.columns(2)
        with colp1:
            st.markdown("### STRICT prompt text")
            st.code(strict_prompt)
        with colp2:
            st.markdown("### GENEROUS prompt text")
            st.code(generous_prompt)

    # Compare answers side by side (two prompt options)
    st.subheader("Answer Comparison (STRICT vs GENEROUS)")

    col1, col2 = st.columns(2)

    with st.spinner("Calling local LLM (Ollama) twice..."):
        with col1:
            st.markdown("### 🧊 STRICT")
            strict_answer = call_ollama(ollama_model, strict_prompt)
            st.write(strict_answer)

        with col2:
            st.markdown("### 🔥 GENEROUS")
            generous_answer = call_ollama(ollama_model, generous_prompt)
            st.write(generous_answer)

    st.divider()
    st.markdown("### Quick Interpretation")
    st.write(
        "STRICT tends to avoid inference and may respond with 'Not specified in the document'. "
        "GENEROUS allows reasonable inference from clearly implied instructions, producing more practical summaries."
    )
