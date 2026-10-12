def chunk_text(text: str, chunk_words: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows so long documents are searchable by section."""
    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks
