import numpy as np
import pandas as pd

from . import labels as L
from .verdict import assess, format_position

COLUMNS = [L.C_QUERY, L.C_BEST_SCORE, L.C_TO_THRESHOLD, L.C_BEST_URL, L.C_POSITION, L.C_RANK_URL, L.C_TOPIC]


def find_gaps(result, lead, threshold, rankings=None, topics=None, good_position=10, gap_position=0) -> pd.DataFrame:
    """Potentielle Content-Lücken: genau die Queries mit dem Urteil Content-Lücke (dieselbe Regel wie die Übersicht),
    die sichersten zuerst (größter Abstand unter der Schwelle). topics: SERP-Cluster je Query (0 = ohne Cluster) oder
    None ohne SERPs."""
    assessments = assess(result, lead, threshold, rankings, good_position, gap_position=gap_position)
    rows = []
    for i, (query, a) in enumerate(zip(result.queries, assessments)):
        if a.verdict != L.V_GAP:
            continue
        j = int(lead[i].argmax())
        best = float(lead[i, j])
        position, ranking_url = (format_position(a.hit.position), a.hit.url) if a.hit is not None else ("", "")
        topic = topics[i] if topics is not None and topics[i] else None
        rows.append((query, round(best, 4), float(np.round(best - threshold, 4)) + 0.0, result.urls[j], position,
                     ranking_url, topic))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df[L.C_TOPIC] = df[L.C_TOPIC].astype("Int64")
    return df.sort_values(L.C_TO_THRESHOLD, kind="stable").reset_index(drop=True)


def count_new_pages(gaps: pd.DataFrame) -> int:
    """Eine neue Seite je Thema, Lücken ohne Thema zählen einzeln."""
    topics = gaps[L.C_TOPIC]
    return int(topics.dropna().nunique() + topics.isna().sum())
