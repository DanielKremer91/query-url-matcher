from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url


@dataclass
class ThresholdProposal:
    value: float
    source: str  # "rankings" | "median"
    n_pairs: int


def propose_threshold(result, lead, rankings=None, max_position=5, min_pairs=20) -> ThresholdProposal:
    scores = []
    if rankings is not None:
        q_index = {normalize_query(q): i for i, q in enumerate(result.queries)}
        u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
        for row in rankings[rankings["position"] <= max_position].itertuples():
            i, j = q_index.get(row.query_norm), u_index.get(row.url_norm)
            if i is not None and j is not None:
                scores.append(float(lead[i, j]))
    if len(scores) >= min_pairs:
        return ThresholdProposal(float(np.percentile(scores, 25)), "rankings", len(scores))
    return ThresholdProposal(float(np.median(lead.max(axis=1))), "median", len(scores))


def examples_around(result, lead, threshold, n=5) -> pd.DataFrame:
    best = lead.argmax(axis=1)
    rows = [
        {
            L.C_QUERY: query,
            L.C_URL: result.urls[j],
            L.C_CHUNK: result.best_chunk(i, j),
            "Score": round(float(lead[i, j]), 4),
        }
        for i, (query, j) in enumerate(zip(result.queries, best))
    ]
    df = pd.DataFrame(rows)
    above = df[df["Score"] >= threshold].sort_values("Score").head(n).copy()
    below = df[df["Score"] < threshold].sort_values("Score", ascending=False).head(n).copy()
    above.insert(0, L.C_SIDE, "knapp über der Schwelle")
    below.insert(0, L.C_SIDE, "knapp unter der Schwelle")
    return pd.concat([above, below], ignore_index=True)
