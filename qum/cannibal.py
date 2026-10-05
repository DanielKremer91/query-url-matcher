import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query
from .verdict import format_position

REASON_BETTER = "Eine andere Seite passt besser als die rankende"
REASON_CLOSE = "Mehrere Seiten passen fast gleich gut"
REASON_RANKING = "Mehrere eigene Seiten ranken für die Query"


def _score(value) -> str:
    return f"Score {round(float(value), 4)}"


def find_cannibalization(
    result, lead, threshold, decisions, rankings=None, margin=0.02, visible_position=20
) -> pd.DataFrame:
    rows = []
    for i, query in enumerate(result.queries):
        decision = decisions.iloc[i]
        order = np.argsort(-lead[i], kind="stable")
        top = lead[i, order[0]]
        close = [j for j in order if lead[i, j] >= threshold and top - lead[i, j] <= margin]
        if decision[L.C_VERDICT] == L.V_RISK:
            competing = [
                f"{decision[L.C_RANK_URL]} (Position {decision[L.C_POSITION]})",
                f"{decision[L.C_BEST_URL]} ({_score(top)})",
            ]
            rows.append((query, L.STAGE_RISK, REASON_BETTER, " | ".join(competing)))
        elif len(close) >= 2:
            competing = [f"{result.urls[j]} ({_score(lead[i, j])})" for j in close]
            rows.append((query, L.STAGE_RISK, REASON_CLOSE, " | ".join(competing)))
        if rankings is not None:
            own = rankings[
                (rankings["query_norm"] == normalize_query(query)) & (rankings["position"] <= visible_position)
            ]
            own = own.sort_values("position", kind="stable").drop_duplicates("url_norm")
            if len(own) >= 2:
                competing = [f"{row.url} (Position {format_position(row.position)})" for row in own.itertuples()]
                rows.append((query, L.STAGE_VISIBLE, REASON_RANKING, " | ".join(competing)))
    return pd.DataFrame(rows, columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING])
