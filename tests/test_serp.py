import pandas as pd

from qum.serp import (
    cluster_keywords,
    overlap_edges,
    top_urls_per_keyword,
    topics,
)


def _urls(*ids):
    return [f"https://s.de/{i}" for i in ids]


def _serps(kw_urls):
    rows = []
    for kw, urls in kw_urls.items():
        for pos, url in enumerate(urls, start=1):
            rows.append((kw, kw, url, url, float(pos)))
    return pd.DataFrame(rows, columns=["keyword", "query_norm", "url", "url_norm", "position"])


def test_top_urls_per_keyword_sorts_dedupes_and_limits():
    serps = pd.DataFrame(
        [("k", "k", f"https://s.de/{i}", f"https://s.de/{i}", float(i)) for i in range(12, 0, -1)]
        + [("k", "k", "https://s.de/1", "https://s.de/1", 99.0)],
        columns=["keyword", "query_norm", "url", "url_norm", "position"],
    )
    urls = top_urls_per_keyword(serps)["k"]
    assert urls == _urls(*range(1, 11))


def test_overlap_uses_smaller_list_as_denominator():
    edges = overlap_edges({"a": _urls(1, 2, 3, 4), "b": _urls(1, 2)}, min_overlap=0.5)
    assert edges == {("a", "b"): 1.0}


def test_overlap_below_threshold_is_no_edge():
    assert overlap_edges({"a": _urls(1, 2, 3, 4), "b": _urls(1, 5, 6, 7)}, min_overlap=0.5) == {}


def test_chain_is_broken_by_density():
    # a-b, b-c, c-d überschneiden sich jeweils zur Hälfte, a und d haben nichts gemeinsam
    kw = {
        "a": _urls(1, 2, 3, 4),
        "b": _urls(3, 4, 5, 6),
        "c": _urls(5, 6, 7, 8),
        "d": _urls(7, 8, 9, 10),
    }
    clusters = cluster_keywords(kw, 0.5, 0.5)
    assert clusters["a"] == 0 and clusters["d"] == 0
    assert clusters["b"] == clusters["c"] != 0


def test_single_link_into_clique_is_dropped():
    clique = {k: _urls(1, 2, 3, 4) for k in ["a", "b", "c", "d"]}
    clique["x"] = _urls(1, 2, 20, 21, 22, 23, 24, 25)  # nur 2 von 4 gemeinsam mit jedem: Kante zu allen
    clique["y"] = _urls(20, 21, 22, 23, 30, 31, 32, 33)  # hängt nur an x
    clusters = cluster_keywords(clique, 0.5, 0.5)
    assert len({clusters[k] for k in ["a", "b", "c", "d", "x"]}) == 1
    assert clusters["y"] == 0


def test_bridge_between_dense_groups_keeps_both_groups():
    kw = {f"a{i}": _urls(1, 2, 3, 4) for i in (2, 3, 4)}
    kw.update({f"b{i}": _urls(11, 12, 13, 14) for i in (2, 3, 4)})
    # Brücke: a1 und b1 teilen 4 von 8 URLs (50 %), a1 mit Gruppe a und b1 mit Gruppe b je 4 von 4
    kw["a1"] = _urls(1, 2, 3, 4, 21, 22, 23, 24)
    kw["b1"] = _urls(11, 12, 13, 14, 21, 22, 23, 24)
    assert ("a1", "b1") in overlap_edges(kw, 0.5)
    clusters = cluster_keywords(kw, 0.5, 0.5)
    assert clusters["a2"] == clusters["a3"] == clusters["a4"] != 0
    assert clusters["b2"] == clusters["b3"] == clusters["b4"] != 0
    assert clusters["a2"] != clusters["b2"]
    # die Brücke selbst bleibt ein eigenes Zweier-Cluster
    assert clusters["a1"] == clusters["b1"] != 0
    assert clusters["a1"] not in {clusters["a2"], clusters["b2"]}


def test_path_between_two_cliques_keeps_both_cliques():
    kw = {f"a{i}": _urls(1, 2, 3, 4) for i in (2, 3, 4)}
    kw.update({f"b{i}": _urls(11, 12, 13, 14) for i in (2, 3, 4)})
    kw["a1"] = _urls(1, 2, 21, 22, 23, 24, 25, 26)  # 2 von 4 mit Gruppe a, 2 von 8 mit m
    kw["b1"] = _urls(11, 12, 31, 32, 33, 34, 35, 36)
    kw["m"] = _urls(21, 22, 31, 32)  # 2 von 4 mit a1 und mit b1, sonst nichts
    edges = overlap_edges(kw, 0.5)
    assert ("a1", "m") in edges and ("b1", "m") in edges
    assert {pair for pair in edges if "m" in pair} == {("a1", "m"), ("b1", "m")}
    clusters = cluster_keywords(kw, 0.5, 0.5)
    assert len({clusters[k] for k in ["a1", "a2", "a3", "a4"]}) == 1 and clusters["a1"] != 0
    assert len({clusters[k] for k in ["b1", "b2", "b3", "b4"]}) == 1 and clusters["b1"] != 0
    assert clusters["a1"] != clusters["b1"]
    assert clusters["m"] == 0


def test_all_equally_weak_component_terminates_without_cluster():
    cycle = {"a": _urls(1, 2), "b": _urls(2, 3), "c": _urls(3, 4), "d": _urls(4, 1)}
    assert len(overlap_edges(cycle, 0.5)) == 4
    assert cluster_keywords(cycle, 0.5, 1.0) == {"a": 0, "b": 0, "c": 0, "d": 0}


def test_density_threshold_is_adjustable():
    kw = {"a": _urls(1, 2, 3, 4), "b": _urls(3, 4, 5, 6), "c": _urls(5, 6, 7, 8), "d": _urls(7, 8, 9, 10)}
    strict = cluster_keywords(kw, 0.5, 0.5)
    assert strict["a"] == 0 and strict["d"] == 0
    loose = cluster_keywords(kw, 0.5, 0.3)
    assert len({loose[k] for k in "abcd"}) == 1 and loose["a"] != 0


def test_isolated_keyword_has_no_cluster():
    assert cluster_keywords({"a": _urls(1), "b": _urls(2)}) == {"a": 0, "b": 0}


def test_topics_are_cluster_numbers_per_query_in_query_order():
    serps = _serps({"a": _urls(1, 2), "b": _urls(1, 2), "c": _urls(7, 8)})
    assert topics(["b", "c", "A ", "fehlt"], serps) == [1, 0, 1, 0]


def test_topics_use_the_sliders():
    serps = _serps({"a": _urls(1, 2, 3, 4), "b": _urls(1, 2, 7, 8)})
    assert topics(["a", "b"], serps, min_overlap=0.5) == [1, 1]
    assert topics(["a", "b"], serps, min_overlap=0.6) == [0, 0]


def _brute_force_edges(kw_urls, min_overlap):
    keywords = sorted(kw_urls)
    sets = {k: set(kw_urls[k]) for k in keywords}
    edges = {}
    for a_idx, a in enumerate(keywords):
        for b in keywords[a_idx + 1 :]:
            denominator = min(len(sets[a]), len(sets[b]), 10)
            if denominator and len(sets[a] & sets[b]) / denominator >= min_overlap:
                edges[(a, b)] = len(sets[a] & sets[b]) / denominator
    return edges


def test_indexed_edges_equal_brute_force_on_random_input():
    import random

    rng = random.Random(7)
    for _ in range(20):
        kw_urls = {
            f"k{i}": _urls(*rng.sample(range(25), rng.randint(0, 10))) for i in range(rng.randint(2, 30))
        }
        for min_overlap in (0.1, 0.3, 0.5, 0.8, 1.0):
            assert overlap_edges(kw_urls, min_overlap) == _brute_force_edges(kw_urls, min_overlap)


def test_cluster_keywords_accepts_precomputed_edges():
    kw = {k: _urls(1, 2, 3, 4) for k in ["a", "b", "c"]}
    kw["d"] = _urls(7, 8, 9)
    edges = overlap_edges(kw, 0.5)
    assert cluster_keywords(kw, 0.5, 0.5, edges=edges) == cluster_keywords(kw, 0.5, 0.5)
    assert cluster_keywords(kw, 0.5, 0.5, edges={}) == {k: 0 for k in kw}
