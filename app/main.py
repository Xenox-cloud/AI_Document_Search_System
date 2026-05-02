from fastapi import FastAPI
from .embed import embed_text
from .store import add_document, search_documents
from .schemas import DocumentInput, QueryInput

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Document Search API running"}

# Add document
@app.post("/add")
def add(doc: DocumentInput):
    vec = embed_text(doc.text)
    add_document(doc.text, vec)
    return {"status": "document added"}

# Search
@app.post("/search")
def search(q: QueryInput):
    vec = embed_text(q.query)
    results = search_documents(vec)
    return {"results": results}