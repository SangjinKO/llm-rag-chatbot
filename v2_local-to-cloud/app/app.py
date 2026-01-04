from __future__ import annotations
import sys
import os
import time
import streamlit as st

# Make project root importable (Streamlit-safe)
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.util import load_env_file
from llm.local_ollama import LocalOllamaLLM
from llm.chatgpt import ChatGPTLLM
from llm.gemini import GeminiLLM
from prompt.templates import build_prompt
from rag.rag_core import build_index_from_pdf_bytes, search_top_k, make_context_block

# Keys loaded from env
load_env_file(".env")
openai_key = os.getenv("OPENAI_API_KEY", "").strip()
gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

QUESTION_DEFAULT = "What are the prerequisites before installation or first use?"


def render_prompt_panel(title: str, prompt_text: str):
    st.markdown(f"#### {title}")
    st.code(prompt_text)

def render_answer(col, llm_label: str, llm_obj, prompt_text: str):
    with col:
        t0 = time.time()
        result = llm_obj.generate(prompt_text)
        latency_ms = int((time.time() - t0) * 1000)

        st.caption(f"Model: `{llm_obj.model_name}`  |  Latency: {latency_ms} ms")

        if result.error:
            st.error(result.error)
        else:
            st.write(result.text)


def render_model_section(llm_label: str, llm_obj, strict_prompt: str, generous_prompt: str):
    st.markdown(f"## 🤖 {llm_label}")
    left, right = st.columns(2)

    with left:
        st.markdown("### 🧊 STRICT")
    with right:
        st.markdown("### 🔥 GENEROUS")

    render_answer(left, llm_label, llm_obj, strict_prompt)
    render_answer(right, llm_label, llm_obj, generous_prompt)

    st.divider()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Multi RAG+LLM", layout="wide")
st.title("Multi RAG+LLM Comparison")
st.caption("PDF → Retrieval(FAISS) → Evidence → Compare STRICT vs GENEROUS across models.")

with st.sidebar:
    st.header("RAG Settings")
    embed_model_name = st.text_input("Embedding model", value=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    top_k = st.slider("Top-k (retrieval)", 2, 10, 5, 1)
    chunk_size = st.slider("Chunk size (chars)", 300, 1200, 500, 50)
    overlap = st.slider("Overlap (chars)", 0, 300, 100, 25)

    st.divider()
    st.header("LLM Models")
    use_ollama = st.checkbox("Ollama (Local)", value=True)
    use_chatgpt = st.checkbox("ChatGPT (OpenAI)", value=True)
    use_gemini = st.checkbox("Gemini (Google)", value=True)

    st.divider()
    st.header("LLM Model Settings")
    ollama_model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", ""))
    openai_model = st.text_input("ChatGPT model", value=os.getenv("OPENAI_MODEL", ""))
    gemini_model = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", ""))

    st.divider()
    st.header("Debug")
    show_evidence = st.checkbox("Show evidence context", value=True)
    show_prompts = st.checkbox("Show prompt text", value=False)

#PDF Upload
uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded is None:
    st.info("Upload a PDF to build the vector index (Day 3).")
    st.stop()

pdf_bytes = uploaded.read()
with st.spinner("Building / loading FAISS index..."):
    bundle = build_index_from_pdf_bytes(
        pdf_bytes=pdf_bytes,
        chunk_size=chunk_size,
        overlap=overlap,
        embed_model_name=embed_model_name,
    )
st.success(f"Index ready ✅ doc_id={bundle.doc_id} | chunks={len(bundle.chunks)}")


# Input: question 
question = st.text_input("Question", value=QUESTION_DEFAULT)

# Run button
if st.button("▶ Run Comparison", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching relevant chunks..."):
        idxs, dists = search_top_k(bundle, question, top_k=top_k, embed_model_name=embed_model_name)

    st.subheader("Evidence (Top-k Retrieved Chunks)")
    for rank, (i, d) in enumerate(zip(idxs, dists), start=1):
        with st.expander(f"#{rank} idx={i} dist={d:.4f}", expanded=(rank <= 2)):
            st.write(bundle.chunks[i])

    # Source of truth: retrieved context
    context_block = make_context_block(bundle, idxs)

    # show merged context
    if show_evidence:
        st.subheader("Merged Context Block")
        st.code(context_block)

    # build prompts from retrieved context
    strict_prompt = build_prompt("strict", question, context_block)
    generous_prompt = build_prompt("generous", question, context_block)

    st.subheader("Results (STRICT vs GENEROUS)")
    # Instantiate LLMs
    ollama = LocalOllamaLLM(model_name=ollama_model)
    chatgpt = ChatGPTLLM(model_name=openai_model, api_key=openai_key)
    gemini = GeminiLLM(model_name=gemini_model, api_key=gemini_key)

    if show_prompts:
        st.subheader("Prompts")
        p1, p2 = st.columns(2)
        with p1:
            render_prompt_panel("🧊 STRICT Prompt", strict_prompt)
        with p2:
            render_prompt_panel("🔥 GENEROUS Prompt", generous_prompt)

    # Render selected models in order
    if use_ollama:
        render_model_section("Ollama (Local)", ollama, strict_prompt, generous_prompt)
    if use_chatgpt:
        render_model_section("ChatGPT (OpenAI)", chatgpt, strict_prompt, generous_prompt)
    if use_gemini:
        render_model_section("Gemini (Google)", gemini, strict_prompt, generous_prompt)

else:
    st.info("Upload a PDF, enter a question, then click **Run Comparison**.")