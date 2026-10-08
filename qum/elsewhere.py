import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import best_rankings, format_position

COLUMNS = [
    L.C_QUERY, L.C_CHUNK_URL, L.C_BEST_CHUNK, L.C_SECTION, L.C_S_BEST_CHUNK, L.C_S_FULL_CHUNK_URL,
    L.C_OVERALL_URL, L.C_S_FULL_OVERALL, L.C_POSITION, L.C_RANK_URL, L.C_RANK_IS,
]


def _ranking_page(hit, u_index, chunk_j, overall_j) -> str:
    if hit is None:
        return L.CMP_NOT_RANKING
    j = u_index.get(hit.url_norm)
    if j is None:
        return L.CMP_NOT_IN_EXPORT
    if j == chunk_j:
        return L.RANK_IS_CHUNK
    return L.RANK_IS_OVERALL if j == overall_j else L.RANK_IS_OTHER


def chunk_elsewhere(result, rankings=None) -> pd.DataFrame:
    """Queries, deren bester Chunk nicht auf der besten Seite insgesamt (höchster Gesamt-URL-Score) liegt. Unabhängig
    von Schwelle und Bewertungsgrundlage: Eine Seite dreht sich insgesamt um das Thema, die konkreteste Antwort steht
    aber auf einer anderen. Stärkster Chunk zuerst."""
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    best = best_rankings(rankings)
    rows = []
    for i, query in enumerate(result.queries):
        chunk_j = int(np.argmax(result.chunk_scores[i]))
        overall_j = int(np.argmax(result.full_scores[i]))
        if chunk_j == overall_j:
            continue
        hit = best.get(normalize_query(query))
        rows.append({
            L.C_QUERY: query,
            L.C_CHUNK_URL: result.urls[chunk_j],
            L.C_BEST_CHUNK: result.chunks[chunk_j][int(result.best_chunk_idx[i, chunk_j])],
            L.C_SECTION: result.best_section(i, chunk_j),
            L.C_S_BEST_CHUNK: round(float(result.chunk_scores[i, chunk_j]), 4),
            L.C_S_FULL_CHUNK_URL: round(float(result.full_scores[i, chunk_j]), 4),
            L.C_OVERALL_URL: result.urls[overall_j],
            L.C_S_FULL_OVERALL: round(float(result.full_scores[i, overall_j]), 4),
            L.C_POSITION: format_position(hit.position) if hit is not None else "",
            L.C_RANK_URL: hit.url if hit is not None else "",
            L.C_RANK_IS: _ranking_page(hit, u_index, chunk_j, overall_j) if rankings is not None else "",
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(L.C_S_BEST_CHUNK, ascending=False, kind="stable").reset_index(drop=True)
