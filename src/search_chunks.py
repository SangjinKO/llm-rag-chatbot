from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


def main():
    root = Path(__file__).resolve().parents[1]

    index_path = root / "output" / "faiss.index"
    chunks_path = root / "output" / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(index_path))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # QUESTION
    QUESTION = "What are the prerequisites or requirements before installation or first use?"
    q_emb = model.encode([QUESTION])
    q_emb = np.array(q_emb).astype("float32")

    top_k = 5
    distances, indices = index.search(q_emb, top_k)

    print(f"\n[Q] {QUESTION}")
    print(f"[OK] Top-{top_k} results:\n")

    for rank, idx in enumerate(indices[0], start=1):
        chunk = chunks[idx]
        preview = chunk[:450].replace("\n", " ")
        print(f"--- #{rank} | idx={idx} | dist={distances[0][rank-1]:.4f} ---")
        print(preview)
        print()

if __name__ == "__main__":
    main()
