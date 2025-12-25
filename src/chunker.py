
from __future__ import annotations
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
