import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import RISK_CLEAR, RISK_NOT_IN_EXPORT, RISK_PLAIN, assess, format_position, risk_kind

# Einzelne Fälle (intern, mit Stufe und Grund): eine Zeile je Query, Stufe und konkurrierender URL
CASE_COLUMNS = [
    L.C_QUERY, L.C_NO, L.C_COMP_URL, L.C_COMP_SCORE, L.C_GAP_TO_BEST, L.C_COMP_POS,
    L.C_POSITION, L.C_RANK_URL, L.C_STAGE, L.C_REASON, L.C_PRIORITY,
]
# Blatt Kannibalisierungsgefahr (Langformat): eine Zeile je Query und konkurrierender URL, die Query steht in jeder Zeile
COLUMNS = [
    L.C_QUERY, L.C_NO, L.C_COMP_URL, L.C_COMP_SCORE, L.C_GAP_TO_BEST, L.C_COMP_POS,
    L.C_POSITION, L.C_RANK_URL, L.C_PRIORITY,
]
# dringendste zuerst; je Query gilt die dringendste Einordnung ihrer Fälle
_URGENCY = [L.PRIO_VERY_HIGH, L.PRIO_HIGH, L.PRIO_MID, L.PRIO_LOW, L.PRIO_VERY_LOW, L.PRIO_OPEN]

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


def _priority(a, stage, best_j, rankings_loaded, band_end) -> str:
    """Wie dringend. Ohne Rankings offen. Stufe Möglich (Konkurrenz deutlich dahinter): mittel, wenn eine andere als die
    beste Seite rankt (egal wo), sonst sehr niedrig. Gefahr und bereits sichtbar nach Ranking-Band: Top-Ranking der
    besten Seite niedrig, einer anderen Seite mittel; knapp dahinter bis band_end sehr hoch (fast oben, die Konkurrenz
    bremst vermutlich); schlechter oder kein Ranking hoch."""
    if not rankings_loaded:
        return L.PRIO_OPEN
    best_ranks = a.ranking_j is not None and a.ranking_j == best_j
    if stage == L.STAGE_POSSIBLE:
        return L.PRIO_MID if a.hit is not None and not best_ranks else L.PRIO_VERY_LOW
    if a.ranks_well:
        return L.PRIO_LOW if best_ranks else L.PRIO_MID
    if a.hit is not None and a.hit.position <= band_end:
        return L.PRIO_VERY_HIGH
    return L.PRIO_HIGH


def _rows(query, hit, stage, reason, competing, priority_of, best_score) -> list:
    """Eine Zeile je konkurrierender URL. hit: beste eigene Ranking-Zeile der Query oder None.
    competing: (URL, Score oder None, Position als Text) je URL, in der gewünschten Reihenfolge.
    best_score: Score der semantisch besten URL der Query, für den Abstand je Zeile."""
    shared = {
        L.C_POSITION: format_position(hit.position) if hit is not None else "",
        L.C_RANK_URL: hit.url if hit is not None else "",
        L.C_STAGE: stage, L.C_REASON: reason, L.C_PRIORITY: priority_of(stage),
    }
    return [
        {L.C_QUERY: query, L.C_NO: number, L.C_COMP_URL: url,
         L.C_COMP_SCORE: None if score is None else round(float(score), 4),
         L.C_GAP_TO_BEST: None if score is None else round(float(best_score) - float(score), 4) + 0.0,
         L.C_COMP_POS: position, **shared}
        for number, (url, score, position) in enumerate(competing, start=1)
    ]


def _own_rankings(rankings) -> dict:
    """Je Query die eigenen URLs, beste Position zuerst, jede URL einmal."""
    if rankings is None:
        return {}
    ordered = rankings.sort_values("position", kind="stable").drop_duplicates(["query_norm", "url_norm"])
    return {query: list(group.itertuples()) for query, group in ordered.groupby("query_norm", sort=False)}


def cannibal_cases(
    result, lead, threshold, rankings=None, good_position=10, margin=0.01, visible_position=20, gap_position=0
) -> pd.DataFrame:
    """Einzelne Fälle mit Stufe und Grund (Grundlage für das Blatt und seine Einordnung). Stufe Gefahr: jedes Urteil Kannibalisierungsgefahr und bei gutem Ranking jede andere passende Seite, die besser
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
            return _priority(a, stage, best_j, rankings is not None, visible_position)

        fitting = [int(j) for j in np.argsort(-lead[i], kind="stable") if lead[i, j] >= threshold]
        if reason is None and len(fitting) >= 2:
            further = [(result.urls[j], lead[i, j], position_of.get(normalize_url(result.urls[j]), "")) for j in fitting]
            rows.extend(_rows(query, a.hit, L.STAGE_POSSIBLE, L.REASON_FURTHER, further, priority, lead[i].max()))
        elif reason == L.REASON_CLOSE:
            rows.extend(_rows(query, a.hit, L.STAGE_DANGER, reason, pages, priority, lead[i].max()))
        elif reason is not None:
            score = None if a.ranking_j is None else lead[i, a.ranking_j]
            first = (a.hit.url, score, format_position(a.hit.position))
            others = [page for j, page in zip(a.close, pages) if j != a.ranking_j]
            rows.extend(_rows(query, a.hit, L.STAGE_DANGER, reason, [first] + others, priority, lead[i].max()))
        visible = [r for r in own if r.position <= visible_position]
        if len(visible) >= 2:
            # Seiten aus dem Frog-Export in dessen Schreibweise, damit dieselbe Seite überall gleich aussieht
            competing = [
                (result.urls[u_index[r.url_norm]], lead[i, u_index[r.url_norm]], format_position(r.position))
                if r.url_norm in u_index else (r.url, None, format_position(r.position))
                for r in visible
            ]
            rows.extend(_rows(query, a.hit, L.STAGE_VISIBLE, L.REASON_RANKING, competing, priority, lead[i].max()))
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def find_cannibalization(
    result, lead, threshold, rankings=None, good_position=10, margin=0.01, visible_position=20, gap_position=0
) -> pd.DataFrame:
    """Blatt Kannibalisierungsgefahr: je Query alle konkurrierenden URLs aus allen Fällen (passende Seiten nah an der
    besten, weitere passende Seiten, mehrere rankende Seiten), jede URL einmal, nach Score sortiert (ohne Score nach
    Position). Einordnung: die dringendste der Fälle der Query."""
    cases = cannibal_cases(result, lead, threshold, rankings, good_position, margin, visible_position, gap_position)
    rows = []
    for query, group in cases.groupby(L.C_QUERY, sort=False):
        urgency = min(group[L.C_PRIORITY], key=_URGENCY.index)
        ordered = group.assign(
            _score=group[L.C_COMP_SCORE].astype(float).fillna(-np.inf),
            _position=pd.to_numeric(group[L.C_COMP_POS].replace("", np.nan), errors="coerce").fillna(np.inf),
            _key=group[L.C_COMP_URL].map(normalize_url),
        ).sort_values(["_score", "_position"], ascending=[False, True], kind="stable")
        ordered = ordered.drop_duplicates("_key")  # www- und Frog-Schreibweise derselben Seite nur einmal
        for number, row in enumerate(ordered.to_dict("records"), start=1):
            rows.append({column: row[column] for column in COLUMNS} | {L.C_NO: number, L.C_PRIORITY: urgency})
    return pd.DataFrame(rows, columns=COLUMNS)

