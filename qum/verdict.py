from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .match import MATCH_COLUMNS, best_matches
from .normalize import normalize_query, normalize_url

OVERVIEW_COLUMNS = MATCH_COLUMNS + [L.C_POSITION, L.C_RANK_URL, L.C_RANK_IS_BEST, L.C_VERDICT, L.C_CANNIBAL]


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


def close_urls(scores, threshold, margin) -> list:
    """Passende URLs, die höchstens um margin unter dem besten Score liegen, beste zuerst."""
    order = np.argsort(-scores, kind="stable")
    top = scores[order[0]]
    return [int(k) for k in order if scores[k] >= threshold and within_margin(top, scores[k], margin)]


# Art der Gefahr bei gutem Ranking: entscheidet über den Grund im Blatt Kannibalisierungsgefahr
RISK_CLEAR, RISK_PLAIN, RISK_NOT_IN_EXPORT = "deutlich", "knapp", "nicht im Export"


def risk_kind(scores, ranking_j, margin) -> str:
    """Nichts verglichen (rankende URL fehlt im Export), deutlich besser (mehr als ein positiver Abstand) oder knapp."""
    if ranking_j is None:
        return RISK_NOT_IN_EXPORT
    if margin > 0 and not within_margin(scores.max(), scores[ranking_j], margin):
        return RISK_CLEAR
    return RISK_PLAIN


def best_rankings(rankings) -> dict:
    """Je Query die Ranking-Zeile mit der besten eigenen Position."""
    best = {}
    if rankings is not None:
        for row in rankings.sort_values("position", kind="stable").itertuples():
            best.setdefault(row.query_norm, row)
    return best


@dataclass
class Assessment:
    verdict: str
    close: list  # passende URLs nah am besten Score, beste zuerst
    hit: object  # beste eigene Ranking-Zeile oder None
    ranking_j: int | None  # Index der rankenden URL; None ohne Ranking oder wenn sie nicht im Frog-Export steht
    ranks_well: bool


def assess(result, lead, threshold, rankings=None, good_position=10, margin=0.01) -> list:
    """Urteil je Query. Prüfreihenfolge: nichts passt, dann gutes Ranking, dann eine oder mehrere passende Seiten.

    Bei gutem Ranking zählt die rankende Seite: erreicht sie die Schwelle oder steht sie nicht im Frog-Export (nicht
    prüfbar), ist es in Ordnung, sonst rankt sie trotz schwachem Match. Konkurrierende Seiten meldet cannibal.py.
    """
    check_margin(margin)
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    best = best_rankings(rankings)
    out = []
    for i, query in enumerate(result.queries):
        close = close_urls(lead[i], threshold, margin)
        hit = best.get(normalize_query(query))
        ranking_j = u_index.get(hit.url_norm) if hit is not None else None
        ranks_well = hit is not None and hit.position <= good_position
        if not close:
            verdict = L.V_WATCH if ranks_well else L.V_GAP
        elif ranks_well:
            fits = ranking_j is None or lead[i, ranking_j] >= threshold
            verdict = L.V_OK if fits else L.V_WATCH
        elif len(close) >= 2:
            verdict = L.V_CANNIBAL
        else:
            verdict = L.V_MATCH
        out.append(Assessment(verdict, close, hit, ranking_j, ranks_well))
    return out


def _compare(a: Assessment, best_j, rankings_loaded) -> str:
    """Wert der Spalte "Rankende URL = beste URL?"."""
    if not rankings_loaded:
        return ""
    if a.hit is None:
        return L.CMP_NOT_RANKING
    if a.ranking_j is None:
        return L.CMP_NOT_IN_EXPORT
    if a.ranking_j == best_j:
        return L.YES
    return L.NO


def build_decisions(result, lead, threshold, rankings=None, good_position=10, weight=0.7, margin=0.01) -> pd.DataFrame:
    """Übersicht ohne die Spalte Kannibalisierungsgefahr, die setzt cannibal.annotate."""
    out = best_matches(result, lead, weight)
    assessments = assess(result, lead, threshold, rankings, good_position, margin)
    best_j = lead.argmax(axis=1)
    out[L.C_POSITION] = [format_position(a.hit.position) if a.hit is not None else "" for a in assessments]
    out[L.C_RANK_URL] = [a.hit.url if a.hit is not None else "" for a in assessments]
    out[L.C_RANK_IS_BEST] = [_compare(a, best_j[i], rankings is not None) for i, a in enumerate(assessments)]
    out[L.C_VERDICT] = [a.verdict for a in assessments]
    return out
