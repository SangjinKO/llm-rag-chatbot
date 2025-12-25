from pathlib import Path
import numpy as np
import json
from pdf_loader import extract_pdf_text
from chunker import chunk_text
from embedder import Embedder
from vector_store import FaissStore


def main():
    root = Path(__file__).resolve().parents[1]
    # PDF_PATH
    PDF_PATH = root / "data" / "sample_en_short.pdf"
    index_path = root / "output" / "faiss.index"

    pages = extract_pdf_text(PDF_PATH)
    full_text = "\n".join(p.text for p in pages)

    # extract and save chunks
    chunks = chunk_text(full_text)
    print(f"[OK] Total chunks: {len(chunks)}")

    #CHUNKS_PATH
    CHUNKS_PATH = root / "output" / "chunks.json"
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[OK] Chunks saved to {CHUNKS_PATH}")

    # save embeddings
    embedder = Embedder()
    embeddings = embedder.embed(chunks)
    embeddings = np.array(embeddings).astype("float32")
    store = FaissStore(dim=embeddings.shape[1], index_path=index_path)
    store.add(embeddings)
    store.save()

    print(f"[OK] FAISS index saved to {index_path}")


if __name__ == "__main__":
    main()
