def chunk_words(text: str, size: int, overlap: int) -> list:
    """Zerlegt Text in überlappende Wort-Chunks. Gibt immer mindestens einen Chunk zurück."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk-Größe muss > 0 sein und der Overlap kleiner als die Chunk-Größe.")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    chunks, start, step = [], 0, size - overlap
    while True:
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            return chunks
        start += step
