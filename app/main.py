import os
import uuid
from fastapi import FastAPI, HTTPException
from . import embed
from .chunking import chunk_text
from .logger import logger
from .schemas import DocumentInput, QueryInput
from .store import VectorStore

app = FastAPI(title="AI Document Search API")
store = VectorStore(os.getenv("DATA_DIR", "./data"))

@app.get("/")
def root():
    return {"message": "AI Document Search API running", **store.stats()}

# Add document: chunk, embed and index it
@app.post("/documents", status_code=201)
def add(doc: DocumentInput):
    chunks = chunk_text(doc.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no text")

    doc_id = uuid.uuid4().hex[:12]
    store.add(doc_id, doc.title, chunks, embed.embed_texts(chunks))
    logger.info(f"Indexed document {doc_id} ({len(chunks)} chunks)")
    return {"doc_id": doc_id, "chunks": len(chunks)}

# Semantic search over all chunks
@app.post("/search")
def search(q: QueryInput):
    vec = embed.embed_texts([q.query])[0]
    return {"query": q.query, "results": store.search(vec, q.top_k)}
