from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2 is a lightweight and efficient sentence embedding model
        # - Pretrained for general-purpose semantic similarity tasks
        # - Provides a good balance between embedding quality and inference speed
        # - Suitable for document retrieval in RAG pipelines
        # - Runs fully locally and is free to use
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=True)
