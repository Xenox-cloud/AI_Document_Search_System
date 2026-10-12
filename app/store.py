import json
import threading
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """FAISS inner-product index over normalised embeddings (cosine similarity), saved to disk."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.index = None
        self.chunks = []  # metadata for each vector, in index order
        self.lock = threading.Lock()
        self._load()

    def add(self, doc_id: str, title: str, texts: list[str], vectors: np.ndarray):
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            for i, text in enumerate(texts):
                self.chunks.append({"doc_id": doc_id, "title": title, "chunk_index": i, "text": text})
            self._save()

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[dict]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            k = min(k, self.index.ntotal)
            scores, ids = self.index.search(query_vector.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # FAISS pads with -1 when it has fewer than k results
                continue
            results.append({**self.chunks[idx], "score": round(float(score), 4)})
        return results

    def stats(self) -> dict:
        return {
            "documents": len({c["doc_id"] for c in self.chunks}),
            "chunks": len(self.chunks),
        }

    def _save(self):
        if not self.data_dir:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.data_dir / "index.faiss"))
        (self.data_dir / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    def _load(self):
        if not self.data_dir or not (self.data_dir / "index.faiss").exists():
            return
        self.index = faiss.read_index(str(self.data_dir / "index.faiss"))
        self.chunks = json.loads((self.data_dir / "chunks.json").read_text(encoding="utf-8"))
