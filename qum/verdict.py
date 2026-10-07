import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url

def format_position(position) -> str:
    if position is None or pd.isna(position):
        return ""
    return str(int(position)) if float(position).is_integer() else f"{position:.1f}"


def within_margin(top, score, margin) -> bool:
    """Verglichen werden die angezeigten Werte (4 Nachkommastellen), sonst kippt der Rand durch Rundungsfehler."""
    return round(float(top), 4) - round(float(score), 4) <= margin + 1e-9


def check_margin(margin) -> None:
    if margin < 0:
        raise ValueError(f"Der Abstand 'fast gleich' darf nicht negativ sein ({margin}).")


def close_to_ranking(scores, ranking_j, threshold, margin) -> list:
    """Weitere passende URLs, deren Score höchstens um margin von dem der rankenden URL abweicht, beste zuerst."""
    order = np.argsort(-scores, kind="stable")
    return [
        int(k)
        for k in order
        if k != ranking_j
        and scores[k] >= threshold
        and within_margin(scores[k], scores[ranking_j], margin)
        and within_margin(scores[ranking_j], scores[k], margin)
    ]


# Art der Gefahr: entscheidet über den Grund im Blatt Kannibalisierungsgefahr
RISK_CLEAR, RISK_PLAIN, RISK_NOT_IN_EXPORT = "deutlich", "knapp", "nicht im Export"


def risk_kind(scores, ranking_j, margin) -> str:
    """Nichts verglichen (rankende URL fehlt im Export), deutlich besser (mehr als ein positiver Abstand) oder knapp."""
    if ranking_j is None:
        return RISK_NOT_IN_EXPORT
    if margin > 0 and not within_margin(scores.max(), scores[ranking_j], margin):
        return RISK_CLEAR
    return RISK_PLAIN


def lead_gap(scores, j):
    """Vorsprung der URL j vor der besten anderen URL (negativ, wenn eine andere vorn liegt). Leer bei nur einer URL."""
    if len(scores) < 2:
        return None
    return round(float(scores[j]) - float(np.delete(scores, j).max()), 4)


def build_decisions(result, lead, threshold, rankings=None, good_position=10, weight=0.7, margin=0.01) -> pd.DataFrame:
    check_margin(margin)
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    best_ranking = {}
    if rankings is not None:
        ordered = rankings.sort_values("position", kind="stable")
        for row in ordered.itertuples():
            best_ranking.setdefault(row.query_norm, row)
    combined = result.combined(weight)

    rows = []
    for i, query in enumerate(result.queries):
        j = int(lead[i].argmax())
        fits = lead[i, j] >= threshold
        hit = best_ranking.get(normalize_query(query))
        rank_url = hit.url if hit is not None else ""
        position = format_position(hit.position) if hit is not None else ""
        note = ""
        if rankings is None:
            verdict = L.V_MATCH if fits else L.V_GAP
        elif hit is not None and hit.position <= good_position:
            ranking_j = u_index.get(hit.url_norm)
            if ranking_j is None:
                note = L.NOTE_NOT_IN_EXPORT
            if not fits:
                verdict = L.V_WATCH
            elif (
                ranking_j is not None
                and lead[i, ranking_j] >= threshold
                and within_margin(lead[i, j], lead[i, ranking_j], margin)
            ):
                # die rankende Seite gehört zu den besten Treffern: die Zeile beschreibt sie
                verdict = L.V_OK
                j = ranking_j
            else:
                verdict = L.V_CANNIBAL
        else:
            verdict = L.V_USE if fits else L.V_GAP
        rows.append(
            {
                L.C_QUERY: query,
                L.C_VERDICT: verdict,
                L.C_BEST_URL: result.urls[j],
                L.C_CHUNK: result.best_chunk(i, j),
                L.C_S_CHUNK: round(float(result.chunk_scores[i, j]), 4),
                L.C_S_FULL: round(float(result.full_scores[i, j]), 4),
                L.C_S_COMBI: round(float(combined[i, j]), 4),
                L.C_LEAD_GAP: lead_gap(lead[i], j),
                L.C_RANK_URL: rank_url,
                L.C_POSITION: position,
                L.C_NOTE: note,
            }
        )
    return pd.DataFrame(rows)
