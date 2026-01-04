from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader



# extract_pdf_text
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
        cleaned = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
        pages.append(PageText(page_num=i, text=cleaned))

    return pages


# Chunking 
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


# Index bundle
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
    Cached by Streamlit using (pdf_bytes, chunk params, embed model).
    """
    doc_id = _hash_bytes(pdf_bytes)

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


def search_top_k(
    bundle: IndexBundle,
    question: str,
    top_k: int,
    embed_model_name: str,
) -> Tuple[List[int], List[float]]:
    model = get_embed_model(embed_model_name)
    q_emb = np.asarray(model.encode([question], show_progress_bar=False), dtype="float32")
    distances, indices = bundle.index.search(q_emb, top_k)
    return indices[0].tolist(), distances[0].tolist()


def make_context_block(bundle: IndexBundle, idxs: List[int]) -> str:
    """
    Merge selected chunks into a single context block.
    """
    contexts = [bundle.chunks[i] for i in idxs if 0 <= i < len(bundle.chunks)]
    return "\n\n---\n\n".join(contexts).strip()
