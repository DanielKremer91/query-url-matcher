import math
from collections import deque

import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import ADVICE, format_position

# Urteile, bei denen "Beste URL" eine passende Seite ist
_HAS_PAGE = {L.V_MATCH, L.V_USE, L.V_OK, L.V_RISK}
_CANDIDATE_COLUMNS = [
    L.C_CAND, L.C_CAND_KW, L.C_CAND_OVERLAP, L.C_CAND_SCORE, L.C_CAND_CHUNK, L.C_CAND_POS, L.C_CAND_MORE,
]


def top_urls_per_keyword(serps: pd.DataFrame) -> dict:
    out = {}
    for keyword, group in serps.groupby("query_norm", sort=False):
        ordered = group.sort_values("position", kind="stable").drop_duplicates("url_norm")
        out[keyword] = ordered["url_norm"].head(10).tolist()
    return out


def overlap_edges(kw_urls: dict, min_overlap: float = 0.5) -> dict:
    keywords = sorted(kw_urls)
    sets = {k: set(kw_urls[k]) for k in keywords}
    edges = {}
    for a_idx, a in enumerate(keywords):
        for b in keywords[a_idx + 1 :]:
            denominator = min(len(sets[a]), len(sets[b]), 10)
            if denominator == 0:
                continue
            overlap = len(sets[a] & sets[b]) / denominator
            if overlap >= min_overlap:
                edges[(a, b)] = overlap
    return edges


def _neighbours(kw_urls, edges):
    neighbours = {k: set() for k in kw_urls}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    return neighbours


def _components(nodes: set, neighbours: dict) -> list:
    seen, out = set(), []
    for start in sorted(nodes):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend((neighbours[node] & nodes) - component)
        seen |= component
        out.append(component)
    return out


def cluster_keywords(kw_urls: dict, min_overlap: float = 0.5, min_density: float = 0.5) -> dict:
    neighbours = _neighbours(kw_urls, overlap_edges(kw_urls, min_overlap))
    assignments = {k: 0 for k in kw_urls}
    next_id = 1
    todo = deque(_components(set(kw_urls), neighbours))
    while todo:
        component = todo.popleft()
        nodes = set(component)
        # Dichte: wer mit zu wenigen Mitgliedern verbunden ist, fliegt raus (gegen den Ketteneffekt)
        while len(nodes) >= 2:
            needed = math.ceil(min_density * (len(nodes) - 1))
            weak = {n for n in nodes if len(neighbours[n] & nodes) < needed}
            if not weak:
                break
            nodes -= weak
        for sub in _components(nodes, neighbours):
            if len(sub) >= 2:
                for keyword in sub:
                    assignments[keyword] = next_id
                next_id += 1
        # Rausgeflogene bekommen unter sich eine zweite Chance (nur wenn die Menge schrumpft, sonst Endlosschleife)
        dropped = component - nodes
        if dropped and nodes:
            todo.extend(_components(dropped, neighbours))
    return assignments


def apply_serp(decisions, result, lead, serps, rankings=None, min_overlap=0.5, min_density=0.5) -> pd.DataFrame:
    kw_urls = top_urls_per_keyword(serps)
    edges = overlap_edges(kw_urls, min_overlap)
    neighbours = _neighbours(kw_urls, edges)
    clusters = cluster_keywords(kw_urls, min_overlap, min_density)

    out = decisions.copy()
    norms = [normalize_query(q) for q in result.queries]
    row_of = {norm: i for i, norm in enumerate(norms)}
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    positions = {}
    if rankings is not None:
        for row in rankings.sort_values("position", kind="stable").itertuples():
            positions.setdefault((row.query_norm, row.url_norm), row.position)

    out[L.C_CLUSTER] = [clusters.get(norm, 0) for norm in norms]
    for column in _CANDIDATE_COLUMNS:
        # object-Spalte: nimmt später Text und Zahlen auf
        out[column] = pd.Series([""] * len(out), index=out.index, dtype=object)

    for i, norm in enumerate(norms):
        if decisions.iloc[i][L.C_VERDICT] != L.V_GAP:
            continue
        candidates = []
        for other in neighbours.get(norm, ()):
            k = row_of.get(other)
            if k is None or decisions.iloc[k][L.C_VERDICT] not in _HAS_PAGE:
                continue
            overlap = edges[tuple(sorted((norm, other)))]
            candidates.append((overlap, result.queries[k], decisions.iloc[k][L.C_BEST_URL]))
        if not candidates:
            continue
        candidates.sort(key=lambda c: (-c[0], c[1]))
        overlap, keyword, url = candidates[0]
        j = u_index[normalize_url(url)]
        label = out.index[i]
        out.loc[label, L.C_VERDICT] = L.V_CHECK
        out.loc[label, L.C_ADVICE] = ADVICE[L.V_CHECK].format(best=url, rank_url="", position="")
        out.loc[label, L.C_CAND] = url
        out.loc[label, L.C_CAND_KW] = keyword
        out.loc[label, L.C_CAND_OVERLAP] = f"{round(overlap * 100)} %"
        out.loc[label, L.C_CAND_SCORE] = round(float(lead[i, j]), 4)
        out.loc[label, L.C_CAND_CHUNK] = result.best_chunk(i, j)
        out.loc[label, L.C_CAND_POS] = format_position(positions.get((norm, normalize_url(url))))
        out.loc[label, L.C_CAND_MORE] = " | ".join(
            f"{u} (Nachbar: {kw}, {round(o * 100)} %)" for o, kw, u in candidates[1:] if u != url
        )
    return out


def gap_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    gaps = decisions[decisions[L.C_VERDICT] == L.V_GAP]
    clusters = gaps[L.C_CLUSTER] if L.C_CLUSTER in gaps.columns else pd.Series(0, index=gaps.index)
    rows = []
    for cluster in sorted(c for c in clusters.unique() if c != 0):
        queries = gaps.loc[clusters == cluster, L.C_QUERY].tolist()
        rows.append(
            {L.C_CLUSTER: str(cluster), L.C_GAP_COUNT: len(queries), L.C_NEW_PAGES: 1, L.C_GAP_QUERIES: " | ".join(queries)}
        )
    loose = gaps.loc[clusters == 0, L.C_QUERY].tolist()
    if loose:
        rows.append(
            {L.C_CLUSTER: L.NO_CLUSTER, L.C_GAP_COUNT: len(loose), L.C_NEW_PAGES: len(loose), L.C_GAP_QUERIES: " | ".join(loose)}
        )
    return pd.DataFrame(rows, columns=[L.C_CLUSTER, L.C_GAP_COUNT, L.C_NEW_PAGES, L.C_GAP_QUERIES])


def count_new_pages(decisions: pd.DataFrame) -> int:
    return int(gap_summary(decisions)[L.C_NEW_PAGES].sum())
