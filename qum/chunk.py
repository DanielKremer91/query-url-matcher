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


_HEADING_MAX_WORDS = 14


def is_heading(line: str) -> bool:
    """Zwischenüberschrift: eigene, kurze Zeile ohne Satzende (Fragezeichen und Doppelpunkt sind erlaubt)."""
    words = line.split()
    return 0 < len(words) <= _HEADING_MAX_WORDS and not line.rstrip().endswith((".", ",", ";", "!"))


def chunk_sections(text: str, size: int, overlap: int) -> list:
    """Je Chunk (wie chunk_words) die Zwischenüberschrift, unter der die meisten seiner Wörter stehen; "" ohne
    Überschrift. Nur zur Anzeige: gerechnet wird weiter mit den festen Wortfenstern."""
    owner = []  # je Wort die Überschrift seines Abschnitts
    current = ""
    for line in str(text).splitlines():
        if is_heading(line):
            current = " ".join(line.split())
        owner.extend([current] * len(line.split()))
    chunks = chunk_words(text, size, overlap)
    if len(owner) <= size:
        return [_dominant(owner)] if chunks else []
    step = size - overlap
    return [_dominant(owner[k * step : k * step + size]) for k in range(len(chunks))]


def _dominant(owners: list) -> str:
    counts = {}
    for heading in owners:
        counts[heading] = counts.get(heading, 0) + 1
    return max(counts, key=counts.get) if counts else ""
