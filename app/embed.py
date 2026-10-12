from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    # Imported and loaded on first use so the API (and tests) start without loading PyTorch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    # Unit-length vectors, so inner product in FAISS equals cosine similarity
    return get_model().encode(texts, normalize_embeddings=True).astype("float32")
