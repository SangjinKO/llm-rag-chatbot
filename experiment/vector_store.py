from __future__ import annotations
from pathlib import Path
from typing import List

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int, index_path: Path):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
