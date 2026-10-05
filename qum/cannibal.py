import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import format_position

def _describe(url, score=None, position="") -> str:
    parts = []
    if position:
        parts.append(f"Position {position}")
    if score is not None:
        parts.append(f"Score {round(float(score), 4)}")
    return f"{url} ({', '.join(parts)})"


def _rankings_by_query(rankings) -> dict:
    """Je Query die eigenen URLs, beste Position zuerst, jede URL einmal."""
    if rankings is None:
        return {}
    ordered = rankings.sort_values("position", kind="stable").drop_duplicates(["query_norm", "url_norm"])
    return {query: list(group.itertuples()) for query, group in ordered.groupby("query_norm", sort=False)}


def find_cannibalization(
    result, lead, threshold, decisions, rankings=None, margin=0.02, visible_position=20
) -> pd.DataFrame:
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    by_query = _rankings_by_query(rankings)

    def score_of(i, url_norm):
        j = u_index.get(url_norm)
        return None if j is None else lead[i, j]

    rows = []
    for i, query in enumerate(result.queries):
        decision = decisions.iloc[i]
        own = by_query.get(normalize_query(query), [])
        order = np.argsort(-lead[i], kind="stable")
        best = order[0]
        # verglichen werden die angezeigten Werte (4 Nachkommastellen), sonst kippt der Rand durch Rundungsfehler
        top = round(float(lead[i, best]), 4)
        close = [
            j for j in order if lead[i, j] >= threshold and top - round(float(lead[i, j]), 4) <= margin + 1e-9
        ]
        if decision[L.C_VERDICT] == L.V_RISK:
            rank_url = decision[L.C_RANK_URL]
            best_position = next((r.position for r in own if r.url_norm == normalize_url(result.urls[best])), None)
            competing = [
                _describe(rank_url, score_of(i, normalize_url(rank_url)), decision[L.C_POSITION]),
                _describe(result.urls[best], lead[i, best], format_position(best_position)),
            ]
            rows.append((query, L.STAGE_RISK, L.REASON_BETTER, " | ".join(competing)))
        elif len(close) >= 2:
            competing = [_describe(result.urls[j], lead[i, j]) for j in close]
            rows.append((query, L.STAGE_RISK, L.REASON_CLOSE, " | ".join(competing)))
        visible = [r for r in own if r.position <= visible_position]
        if len(visible) >= 2:
            competing = [_describe(r.url, score_of(i, r.url_norm), format_position(r.position)) for r in visible]
            rows.append((query, L.STAGE_VISIBLE, L.REASON_RANKING, " | ".join(competing)))
    return pd.DataFrame(rows, columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING])
