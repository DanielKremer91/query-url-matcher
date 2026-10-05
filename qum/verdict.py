import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url

# Feste Textbausteine. Platzhalter: {best}, {rank_url}, {position}
ADVICE = {
    L.V_MATCH: "Es gibt bereits eine passende Seite: {best}. Vor einer neuen Seite prüfen, ob sie ausgebaut werden kann.",
    L.V_GAP: "Keine passende Seite gefunden. Kandidat für eine neue Seite.",
    L.V_OK: "Die rankende Seite ist auch semantisch der beste Treffer. Kein Handlungsbedarf.",
    L.V_RISK: (
        "{rank_url} rankt auf Position {position}, semantisch passt {best} besser. "
        "Prüfen, welche Seite die Query bedienen soll, und die andere abgrenzen."
    ),
    L.V_WATCH: "{rank_url} rankt auf Position {position}, obwohl keine Seite semantisch gut passt. Beobachten.",
    L.V_USE: "Keine neue Seite bauen: {best} passt bereits. Seite ausbauen und intern stärken.",
    L.V_CHECK: (
        "Vor Neuerstellung prüfen: {best} bedient bereits ein Keyword mit stark überlappender SERP. "
        "Seite erweitern statt neu bauen?"
    ),
}
# Rankende URL fehlt im Frog-Export: verglichen wurde nichts, das Urteil bleibt
ADVICE_NOT_IN_EXPORT = (
    "{rank_url} rankt auf Position {position}, steht aber nicht im Frog-Export und wurde nicht verglichen. "
    "Semantisch bester Treffer im Export: {best}. Prüfen, ob die rankende Seite im Export fehlt."
)


def format_position(position) -> str:
    if position is None or pd.isna(position):
        return ""
    return str(int(position)) if float(position).is_integer() else f"{position:.1f}"


def build_decisions(result, lead, threshold, rankings=None, good_position=10, weight=0.7) -> pd.DataFrame:
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
            elif ranking_j is not None and lead[i, ranking_j] >= lead[i, j]:
                verdict = L.V_OK
            else:
                verdict = L.V_RISK
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
                L.C_RANK_URL: rank_url,
                L.C_POSITION: position,
                L.C_NOTE: note,
                L.C_ADVICE: (ADVICE_NOT_IN_EXPORT if note else ADVICE[verdict]).format(
                    best=result.urls[j], rank_url=rank_url, position=position
                ),
            }
        )
    return pd.DataFrame(rows)
