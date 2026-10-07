import math
from collections import Counter, defaultdict, deque

import pandas as pd

from . import labels as L
from .normalize import normalize_query


def top_urls_per_keyword(serps: pd.DataFrame) -> dict:
    out = {}
    for keyword, group in serps.groupby("query_norm", sort=False):
        ordered = group.sort_values("position", kind="stable").drop_duplicates("url_norm")
        out[keyword] = ordered["url_norm"].head(10).tolist()
    return out


def overlap_edges(kw_urls: dict, min_overlap: float = 0.5) -> dict:
    """Kanten (a, b) mit a < b. Gezählt werden nur Paare mit gemeinsamer URL, über einen Index URL -> Keywords."""
    sets = {k: set(urls) for k, urls in kw_urls.items()}
    by_url = defaultdict(list)
    for keyword in sorted(sets):
        for url in sets[keyword]:
            by_url[url].append(keyword)
    shared = Counter()
    for keywords in by_url.values():
        for a_idx, a in enumerate(keywords):
            for b in keywords[a_idx + 1 :]:
                shared[(a, b)] += 1
    edges = {}
    for (a, b), count in sorted(shared.items()):
        overlap = count / min(len(sets[a]), len(sets[b]), 10)
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


def cluster_keywords(kw_urls: dict, min_overlap: float = 0.5, min_density: float = 0.5, edges=None) -> dict:
    if edges is None:
        edges = overlap_edges(kw_urls, min_overlap)
    neighbours = _neighbours(kw_urls, edges)
    assignments = {k: 0 for k in kw_urls}
    next_id = 1
    todo = deque([set(kw_urls)])
    while todo:
        for component in _components(todo.popleft(), neighbours):
            if len(component) < 2:
                continue
            needed = math.ceil(round(min_density * (len(component) - 1), 9))
            degree = {n: len(neighbours[n] & component) for n in component}
            weak = {n for n in component if degree[n] < needed}
            if not weak:
                for keyword in component:
                    assignments[keyword] = next_id
                next_id += 1
                continue
            # Dichte: nur die schwächsten Mitglieder fliegen raus (gegen den Ketteneffekt),
            # die übrigen und die Rausgeflogenen werden getrennt neu geprüft
            lowest = min(degree[n] for n in weak)
            removed = {n for n in weak if degree[n] == lowest}
            if removed == component:
                continue  # alle gleich schwach: kein Cluster, sonst Endlosschleife
            todo.append(component - removed)
            todo.append(removed)
    return assignments


def topics(queries, serps, min_overlap=0.5, min_density=0.5) -> list:
    """Je Query die Nummer ihres SERP-Clusters, 0 ohne Cluster."""
    clusters = cluster_keywords(top_urls_per_keyword(serps), min_overlap, min_density)
    return [clusters.get(normalize_query(query), 0) for query in queries]


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
