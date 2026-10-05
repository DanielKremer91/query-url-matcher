from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url


MIN_PAIRS = 20  # weniger Ranking-Paare ergeben keine Kalibrierung


@dataclass
class ThresholdProposal:
    value: float
    source: str  # "rankings" | "median"
    n_pairs: int = 0


def calibration_scores(result, lead, rankings, max_position=5) -> list:
    """Leit-Scores der Paare aus Query und eigener URL bis max_position, die beide im Lauf vorkommen."""
    if rankings is None:
        return []
    q_index = {normalize_query(q): i for i, q in enumerate(result.queries)}
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    # jedes Paar aus Query und URL zählt einmal, mit seiner besten Position
    pairs = rankings.sort_values("position", kind="stable").drop_duplicates(["query_norm", "url_norm"])
    scores = []
    for row in pairs[pairs["position"] <= max_position].itertuples():
        i, j = q_index.get(row.query_norm), u_index.get(row.url_norm)
        if i is not None and j is not None:
            scores.append(float(lead[i, j]))
    return scores


# einmal gerundet: angezeigte und angewandte Schwelle sind dieselbe Zahl
def calibrated_threshold(result, lead, rankings, max_position=5, min_pairs=MIN_PAIRS) -> ThresholdProposal | None:
    """25. Perzentil der gut rankenden Paare. None ohne Rankings oder mit weniger als min_pairs Paaren."""
    scores = calibration_scores(result, lead, rankings, max_position)
    if len(scores) < min_pairs:
        return None
    return ThresholdProposal(round(float(np.percentile(scores, 25)), 4), "rankings", len(scores))


def median_threshold(result, lead) -> ThresholdProposal:
    """Mittlerer bester Score: nicht kalibriert, etwa die Hälfte der Queries liegt darunter."""
    return ThresholdProposal(round(float(np.median(lead.max(axis=1))), 4), "median")


def examples_around(result, lead, threshold, n=5) -> pd.DataFrame:
    best = lead.argmax(axis=1)
    raw = "_raw"  # ungerundeter Score: entscheidet über Seite und Reihenfolge, gerundet wird nur die Anzeige
    rows = [
        {
            L.C_QUERY: query,
            L.C_URL: result.urls[j],
            L.C_CHUNK: result.best_chunk(i, j),
            L.C_SCORE: round(float(lead[i, j]), 4),
            raw: lead[i, j],  # gleicher Typ wie in build_decisions, damit der Vergleich mit der Schwelle derselbe ist
        }
        for i, (query, j) in enumerate(zip(result.queries, best))
    ]
    df = pd.DataFrame(rows)
    above = df[df[raw] >= threshold].sort_values(raw, kind="stable").head(n).copy()
    below = df[df[raw] < threshold].sort_values(raw, ascending=False, kind="stable").head(n).copy()
    above.insert(0, L.C_SIDE, L.SIDE_ABOVE)
    below.insert(0, L.C_SIDE, L.SIDE_BELOW)
    return pd.concat([above, below], ignore_index=True).drop(columns=raw)
