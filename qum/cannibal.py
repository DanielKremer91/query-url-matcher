import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import RISK_CLEAR, RISK_NOT_IN_EXPORT, RISK_PLAIN, assess, format_position, risk_kind

_PLACES = [
    (L.C_URL_1, L.C_SCORE_1, L.C_POS_1),
    (L.C_URL_2, L.C_SCORE_2, L.C_POS_2),
    (L.C_URL_3, L.C_SCORE_3, L.C_POS_3),
]
COLUMNS = (
    [L.C_QUERY] + [column for place in _PLACES for column in place]
    + [L.C_POSITION, L.C_RANK_URL, L.C_STAGE, L.C_REASON, L.C_PRIORITY]
)

_RISK_REASON = {
    RISK_CLEAR: L.REASON_BETTER,
    RISK_PLAIN: L.REASON_BETTER_PLAIN,
    RISK_NOT_IN_EXPORT: L.REASON_NOT_IN_EXPORT,
}


def _danger_reason(a, scores, margin):
    """Grund der Stufe Gefahr oder None.

    Gutes Ranking: auch wenn heute die richtige Seite rankt, wird gemeldet, wenn eine andere eigene Seite passt und
    besser oder fast gleich gut ist, denn Google kann die rankende Seite wechseln. Der Grund betrifft die rankende URL.
    """
    if a.verdict == L.V_CANNIBAL:
        return L.REASON_CLOSE
    if not a.ranks_well or not a.close:
        return None
    if a.ranking_j in a.close:
        return L.REASON_OK_CLOSE if len(a.close) >= 2 else None
    return _RISK_REASON[risk_kind(scores, a.ranking_j, margin)]


def _priority(a, stage, best_j, rankings_loaded) -> str:
    """Wie dringend. Ohne Rankings offen. Stufe Möglich (Konkurrenz deutlich dahinter): mittel, wenn eine andere als die
    beste Seite rankt (egal wo), sonst niedrig. Gefahr und bereits sichtbar: ohne Top-Ranking hoch, Top-Ranking der
    besten Seite niedrig, Top-Ranking einer anderen Seite mittel."""
    if not rankings_loaded:
        return L.PRIO_OPEN
    best_ranks = a.ranking_j is not None and a.ranking_j == best_j
    if stage == L.STAGE_POSSIBLE:
        return L.PRIO_MID if a.hit is not None and not best_ranks else L.PRIO_LOW
    if not a.ranks_well:
        return L.PRIO_HIGH
    return L.PRIO_LOW if best_ranks else L.PRIO_MID


def _row(query, hit, stage, reason, competing, priority_of) -> dict:
    """hit: beste eigene Ranking-Zeile der Query oder None. competing: (URL, Score oder None, Position als Text) je URL.
    Was über drei URLs hinausgeht, zählt der Grund."""
    if len(competing) > len(_PLACES):
        reason = f"{reason} … und {len(competing) - len(_PLACES)} weitere"
    row = {
        L.C_QUERY: query, L.C_STAGE: stage, L.C_REASON: reason,
        L.C_POSITION: format_position(hit.position) if hit is not None else "",
        L.C_RANK_URL: hit.url if hit is not None else "",
        L.C_PRIORITY: priority_of(stage),
    }
    for k, (url_col, score_col, pos_col) in enumerate(_PLACES):
        url, score, position = competing[k] if k < len(competing) else ("", None, "")
        row.update({url_col: url, score_col: None if score is None else round(float(score), 4), pos_col: position})
    return row


def _own_rankings(rankings) -> dict:
    """Je Query die eigenen URLs, beste Position zuerst, jede URL einmal."""
    if rankings is None:
        return {}
    ordered = rankings.sort_values("position", kind="stable").drop_duplicates(["query_norm", "url_norm"])
    return {query: list(group.itertuples()) for query, group in ordered.groupby("query_norm", sort=False)}


def find_cannibalization(
    result, lead, threshold, rankings=None, good_position=10, margin=0.01, visible_position=20, gap_position=0
) -> pd.DataFrame:
    """Stufe Gefahr: jedes Urteil Kannibalisierungsgefahr und bei gutem Ranking jede andere passende Seite, die besser
    oder fast gleich gut ist als die rankende. Stufe Möglich: sonst weitere Seiten, die die Schwelle erreichen.
    Stufe Kannibalisierung bereits sichtbar: mehrere eigene URLs ranken bis visible_position."""
    assessments = assess(result, lead, threshold, rankings, good_position, margin, gap_position)
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    by_query = _own_rankings(rankings)
    rows = []
    for i, (query, a) in enumerate(zip(result.queries, assessments)):
        own = by_query.get(normalize_query(query), [])
        position_of = {r.url_norm: format_position(r.position) for r in own}
        pages = [(result.urls[j], lead[i, j], position_of.get(normalize_url(result.urls[j]), "")) for j in a.close]
        reason = _danger_reason(a, lead[i], margin)
        best_j = int(lead[i].argmax())

        def priority(stage, a=a, best_j=best_j):
            return _priority(a, stage, best_j, rankings is not None)

        fitting = [int(j) for j in np.argsort(-lead[i], kind="stable") if lead[i, j] >= threshold]
        if reason is None and len(fitting) >= 2:
            further = [(result.urls[j], lead[i, j], position_of.get(normalize_url(result.urls[j]), "")) for j in fitting]
            rows.append(_row(query, a.hit, L.STAGE_POSSIBLE, L.REASON_FURTHER, further, priority))
        elif reason == L.REASON_CLOSE:
            rows.append(_row(query, a.hit, L.STAGE_DANGER, reason, pages, priority))
        elif reason is not None:
            score = None if a.ranking_j is None else lead[i, a.ranking_j]
            first = (a.hit.url, score, format_position(a.hit.position))
            others = [page for j, page in zip(a.close, pages) if j != a.ranking_j]
            rows.append(_row(query, a.hit, L.STAGE_DANGER, reason, [first] + others, priority))
        visible = [r for r in own if r.position <= visible_position]
        if len(visible) >= 2:
            competing = [
                (r.url, lead[i, u_index[r.url_norm]] if r.url_norm in u_index else None, format_position(r.position))
                for r in visible
            ]
            rows.append(_row(query, a.hit, L.STAGE_VISIBLE, L.REASON_RANKING, competing, priority))
    return pd.DataFrame(rows, columns=COLUMNS)


def annotate(decisions: pd.DataFrame, cannibal: pd.DataFrame) -> pd.DataFrame:
    """Kopie der Übersicht mit der Spalte Kannibalisierungsgefahr: "ja" bei Stufe Gefahr oder Kannibalisierung bereits sichtbar,
    "möglich" nur bei Stufe Möglich, sonst "nein"."""
    possible = cannibal[L.C_STAGE] == L.STAGE_POSSIBLE
    flagged = set(cannibal.loc[~possible, L.C_QUERY])
    maybe = set(cannibal.loc[possible, L.C_QUERY])
    out = decisions.copy()
    out[L.C_CANNIBAL] = [
        L.YES if query in flagged else L.MAYBE if query in maybe else L.NO for query in out[L.C_QUERY]
    ]
    return out
