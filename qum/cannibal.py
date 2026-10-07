import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import (
    RISK_CLEAR,
    RISK_NOT_IN_EXPORT,
    RISK_PLAIN,
    check_margin,
    close_to_ranking,
    format_position,
    risk_kind,
    within_margin,
)

# gleiche Unterscheidung wie die Empfehlung im Blatt Entscheidung
_RISK_REASON = {
    RISK_CLEAR: L.REASON_BETTER,
    RISK_PLAIN: L.REASON_BETTER_PLAIN,
    RISK_NOT_IN_EXPORT: L.REASON_NOT_IN_EXPORT,
}

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
    result, lead, threshold, decisions, rankings=None, margin=0.01, visible_position=20
) -> pd.DataFrame:
    check_margin(margin)
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
        close = [j for j in order if lead[i, j] >= threshold and within_margin(lead[i, best], lead[i, j], margin)]
        if decision[L.C_VERDICT] == L.V_RISK:
            rank_url = decision[L.C_RANK_URL]
            best_position = next((r.position for r in own if r.url_norm == normalize_url(result.urls[best])), None)
            competing = [
                _describe(rank_url, score_of(i, normalize_url(rank_url)), decision[L.C_POSITION]),
                _describe(result.urls[best], lead[i, best], format_position(best_position)),
            ]
            reason = _RISK_REASON[risk_kind(lead[i], u_index.get(normalize_url(rank_url)), margin)]
            rows.append((query, L.STAGE_RISK, reason, " | ".join(competing)))
        elif decision[L.C_VERDICT] == L.V_OK:
            # dieselbe Regel wie der Hinweis in der Empfehlung: weitere passende Seiten nah an der rankenden
            ranking_j = u_index[normalize_url(decision[L.C_RANK_URL])]
            others = close_to_ranking(lead[i], ranking_j, threshold, margin)
            if others:
                position_of = {r.url_norm: format_position(r.position) for r in own}
                competing = [_describe(result.urls[ranking_j], lead[i, ranking_j], decision[L.C_POSITION])] + [
                    _describe(result.urls[k], lead[i, k], position_of.get(normalize_url(result.urls[k]), ""))
                    for k in others
                ]
                rows.append((query, L.STAGE_RISK, L.REASON_OK_CLOSE, " | ".join(competing)))
        elif len(close) >= 2:
            competing = [_describe(result.urls[j], lead[i, j]) for j in close]
            rows.append((query, L.STAGE_RISK, L.REASON_CLOSE, " | ".join(competing)))
        visible = [r for r in own if r.position <= visible_position]
        if len(visible) >= 2:
            competing = [_describe(r.url, score_of(i, r.url_norm), format_position(r.position)) for r in visible]
            rows.append((query, L.STAGE_VISIBLE, L.REASON_RANKING, " | ".join(competing)))
    return pd.DataFrame(rows, columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING])


def annotate(decisions: pd.DataFrame, cannibal: pd.DataFrame) -> pd.DataFrame:
    """Kopie der Urteile mit der Spalte Kannibalisierung: Stufe und konkurrierende URLs je Query, sonst leer."""
    hints = {}
    for query, stage, competing in zip(cannibal[L.C_QUERY], cannibal[L.C_STAGE], cannibal[L.C_COMPETING]):
        hints.setdefault(query, []).append(f"{stage}: {competing}")
    out = decisions.copy()
    out[L.C_CANNIBAL] = ["; ".join(hints.get(query, [])) for query in out[L.C_QUERY]]
    return out
