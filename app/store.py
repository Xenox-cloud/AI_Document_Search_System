import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []

def add_document(text, vector):
    documents.append(text)
    index.add(np.array([vector]).astype("float32"))

def search_documents(query_vector, k=3):
    if len(documents) == 0:
        return ["No documents found"]

    D, I = index.search(np.array([query_vector]).astype("float32"), k)

    results = []
    for idx in I[0]:
        if idx < len(documents):
            results.append(documents[idx])

    return results