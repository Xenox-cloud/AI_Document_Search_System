# AI Document Search API

Semantic search over your documents using Sentence Transformers embeddings and a FAISS vector index. It is the retrieval half of a RAG pipeline: given a question, it returns the most relevant passages.

## Features
- **Document ingestion with chunking**: long texts are split into overlapping 200-word chunks so each section is searchable on its own
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions), normalised to unit length
- **Cosine similarity search**: FAISS `IndexFlatIP` over normalised vectors, results returned with scores
- **Persistence**: the index and chunk metadata are saved to disk (`DATA_DIR`, default `./data`) and reloaded on startup
- **Thread-safe** index updates
- Input validation (`top_k` between 1 and 20, no empty documents)
- Tests that run without downloading the model

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs. The embedding model downloads on first use.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | Status plus document and chunk counts |
| POST | `/documents` | `{"title": "Transformers", "text": "..."}` | `{"doc_id": "3f9c2a1b7d4e", "chunks": 4}` |
| POST | `/search` | `{"query": "what is attention?", "top_k": 3}` | Ranked chunks with `doc_id`, `title`, `chunk_index`, `text`, `score` |

## Example

```bash
curl -X POST http://localhost:8000/documents -H "Content-Type: application/json" \
  -d '{"title": "Postgres", "text": "Indexes speed up lookups on large tables..."}'

curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "how do I make queries faster?", "top_k": 3}'
```

## How it works

```text
POST /documents → chunk_text() → embed_texts() → FAISS IndexFlatIP (+ chunks.json) → saved to DATA_DIR
POST /search    → embed_texts(query) → inner-product search = cosine similarity → top_k chunks with scores
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests swap in a tiny bag-of-words embedder, so they run in seconds without PyTorch. They cover ranking, persistence across restarts, chunk overlap, validation, and FAISS's `-1` padding when fewer than `top_k` chunks exist.

## Tech Stack
FastAPI, Sentence Transformers, FAISS, NumPy
