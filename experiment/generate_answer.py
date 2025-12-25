from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)

    STRICT_PROMPT = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "Not specified in the document" and list what is missing.

Return the answer as:
- A short title
- A bullet list of prerequisites/requirements

Question:
{question}

Context:
{context_block}
"""
    
    # By slightly relaxing the prompt to allow inference from clearly implied instructions,the system can then produce a practical and user-friendly summary while still staying grounded in the document.”
    GENEROUS_PROMPT = f"""You are a helpful assistant.

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
    return GENEROUS_PROMPT


def main():
    root = Path(__file__).resolve().parents[1]
    index_path = root / "output" / "faiss.index"
    chunks_path = root / "output" / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    question = "What are the prerequisites or requirements before installation or first use?"
    q_emb = model.encode([question]).astype("float32")

    top_k = 5
    distances, indices = index.search(q_emb, top_k)

    contexts = [chunks[i] for i in indices[0]]

    prompt = build_prompt(question, contexts)

    # Local LLM via Ollama
    resp = chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n[Q]", question)
    print("\n[Retrieved Chunks]")
    for r, i in enumerate(indices[0], start=1):
        print(f"- #{r} idx={i} dist={distances[0][r-1]:.4f}")

    print("\n[Answer]")
    print(resp["message"]["content"])


if __name__ == "__main__":
    main()
