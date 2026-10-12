"""Tests that run without downloading the embedding model: a small bag-of-words embedder stands in.

Run: pip install -r requirements.txt -r requirements-dev.txt && pytest
"""
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import embed, main
from app.chunking import chunk_text
from app.store import VectorStore

DIM = 64


def fake_embed(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            out[row, zlib.crc32(word.encode()) % DIM] += 1
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", fake_embed)
    monkeypatch.setattr(main, "store", VectorStore(tmp_path))
    return TestClient(main.app)


def test_search_ranks_relevant_document_first(client):
    client.post("/documents", json={"title": "pets", "text": "cats and dogs are popular pets"})
    client.post("/documents", json={"title": "db", "text": "postgres indexes speed up sql queries"})
    res = client.post("/search", json={"query": "sql indexes", "top_k": 2}).json()
    assert res["results"][0]["title"] == "db"
    assert res["results"][0]["score"] > res["results"][1]["score"]


def test_fewer_documents_than_top_k_has_no_duplicates(client):
    # FAISS pads missing results with -1; these must not turn into repeated documents
    client.post("/documents", json={"text": "only one document here"})
    results = client.post("/search", json={"query": "document", "top_k": 5}).json()["results"]
    assert len(results) == 1


def test_empty_index_and_validation(client):
    assert client.post("/search", json={"query": "anything"}).json()["results"] == []
    assert client.post("/documents", json={"text": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 50}).status_code == 422


def test_index_persists_across_restarts(tmp_path):
    store = VectorStore(tmp_path)
    store.add("d1", "t", ["hello world"], fake_embed(["hello world"]))
    reloaded = VectorStore(tmp_path)
    assert reloaded.stats() == {"documents": 1, "chunks": 1}
    assert reloaded.search(fake_embed(["hello"])[0], 3)[0]["doc_id"] == "d1"


def test_chunking_overlaps_and_covers_all_words():
    words = [f"w{i}" for i in range(450)]
    chunks = chunk_text(" ".join(words), chunk_words=200, overlap=40)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w160"  # 40-word overlap with the first chunk
    assert chunks[-1].split()[-1] == "w449"
