import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url

# Feste Textbausteine. Platzhalter: {best}, {rank_url}, {position}
ADVICE = {
    L.V_MATCH: (
        "Es gibt bereits eine passende Seite: {best}. Prüfen, ob sie die Query abdeckt oder ausgebaut werden kann, "
        "bevor eine neue Seite entsteht."
    ),
    L.V_GAP: "Keine Seite erreicht die Schwelle. Kandidat für eine neue Seite, nach Prüfung der besten Treffer.",
    L.V_OK: "Die rankende Seite gehört semantisch zu den besten Treffern. Kein Hinweis auf Handlungsbedarf.",
    L.V_RISK: (
        "{rank_url} rankt auf Position {position}, semantisch passt {best} deutlich besser. "
        "Prüfen, welche Seite die Query bedienen soll."
    ),
    L.V_WATCH: (
        "{rank_url} rankt auf Position {position}, obwohl keine Seite die Schwelle erreicht. "
        "Beobachten und die Schwelle prüfen."
    ),
    L.V_USE: (
        "{best} passt semantisch zur Query, rankt aber nicht gut. Prüfen, ob diese Seite ausgebaut und intern "
        "gestärkt werden kann, statt eine neue zu bauen."
    ),
    L.V_CHECK: (
        "Vor Neuerstellung prüfen: {best} bedient bereits ein Keyword mit stark überlappender SERP. "
        "Lässt sich die Seite erweitern?"
    ),
}
# "In Ordnung", aber eine weitere passende Seite liegt fast gleich auf: gleiche Aussage wie im Blatt Kannibalisierung
ADVICE_OK_CLOSE = (
    "Die rankende Seite gehört semantisch zu den besten Treffern. {other} passt fast gleich gut, siehe Blatt Kannibalisierung."
)
# Risiko ohne deutlichen Abstand (Abstand 0 oder die rankende Seite erreicht die Schwelle nicht)
ADVICE_RISK_PLAIN = (
    "{rank_url} rankt auf Position {position}, semantisch passt {best} besser. "
    "Prüfen, welche Seite die Query bedienen soll."
)
# Rankende URL fehlt im Frog-Export: verglichen wurde nichts, das Urteil bleibt
ADVICE_NOT_IN_EXPORT = (
    "{rank_url} rankt auf Position {position}, steht aber nicht im Frog-Export und wurde nicht verglichen. "
    "Semantisch bester Treffer im Export: {best}. "
    "Prüfen, ob die rankende Seite im Export fehlt, bevor daraus Schlüsse gezogen werden."
)


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


# Art des Risikos: entscheidet über die Empfehlung und den Grund im Blatt Kannibalisierung
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
        advice = None
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
                others = close_to_ranking(lead[i], j, threshold, margin)
                if others:
                    advice = ADVICE_OK_CLOSE.format(other=result.urls[others[0]])
            else:
                verdict = L.V_RISK
                if risk_kind(lead[i], ranking_j, margin) == RISK_PLAIN:
                    advice = ADVICE_RISK_PLAIN.format(best=result.urls[j], rank_url=rank_url, position=position)
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
                L.C_ADVICE: advice
                or (ADVICE_NOT_IN_EXPORT if note else ADVICE[verdict]).format(
                    best=result.urls[j], rank_url=rank_url, position=position
                ),
            }
        )
    return pd.DataFrame(rows)
