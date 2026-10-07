import pandas as pd

from . import labels as L
from .normalize import normalize_query
from .verdict import best_rankings, format_position

COLUMNS = [L.C_QUERY, L.C_BEST_SCORE, L.C_BEST_URL, L.C_POSITION, L.C_TOPIC]


def find_gaps(result, lead, threshold, rankings=None, topics=None, below=0.0, max_position=0) -> pd.DataFrame:
    """Potentielle Content-Lücken: bester Leit-Score unter below (0 = Schwelle aus Schritt 7b).
    max_position > 0: nur Queries ohne eigenes Ranking bis zu dieser Position; ohne Rankings wirkungslos.
    topics: SERP-Cluster je Query (0 = ohne Cluster) oder None ohne SERPs."""
    limit = below or threshold
    best = best_rankings(rankings)
    rows = []
    for i, query in enumerate(result.queries):
        j = int(lead[i].argmax())
        if lead[i, j] >= limit:
            continue
        hit = best.get(normalize_query(query))
        if max_position > 0 and hit is not None and hit.position <= max_position:
            continue
        topic = topics[i] if topics is not None and topics[i] else None
        rows.append((query, round(float(lead[i, j]), 4), result.urls[j], format_position(hit.position) if hit else "", topic))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df[L.C_TOPIC] = df[L.C_TOPIC].astype("Int64")
    return df


def count_new_pages(gaps: pd.DataFrame) -> int:
    """Eine neue Seite je Thema, Lücken ohne Thema zählen einzeln."""
    topics = gaps[L.C_TOPIC]
    return int(topics.dropna().nunique() + topics.isna().sum())
