import pandas as pd

from qum import labels as L
from qum.serp import (
    apply_serp,
    cluster_keywords,
    count_new_pages,
    gap_summary,
    overlap_edges,
    top_urls_per_keyword,
)
from qum.verdict import build_decisions
from tests.conftest import make_result


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


def test_isolated_keyword_has_no_cluster():
    assert cluster_keywords({"a": _urls(1), "b": _urls(2)}) == {"a": 0, "b": 0}


OWN = ["https://a.de/getreidefrei", "https://a.de/anderes"]


def _decisions(queries, scores):
    result = make_result(queries, OWN, scores)
    lead = result.lead("chunk")
    return result, lead, build_decisions(result, lead, 0.6)


def test_neighbour_hint_only_for_direct_neighbours():
    queries = ["getreidefrei", "ohne getreide", "allergie"]
    result, lead, decisions = _decisions(queries, [[0.9, 0.1], [0.4, 0.1], [0.3, 0.1]])
    serps = _serps(
        {
            "getreidefrei": _urls(1, 2, 3, 4),
            "ohne getreide": _urls(1, 2, 3, 9),  # direkter Nachbar von "getreidefrei"
            "allergie": _urls(3, 9, 11, 12),  # Nachbar von "ohne getreide", nicht von "getreidefrei"
        }
    )
    out = apply_serp(decisions, result, lead, serps)
    assert out[L.C_VERDICT].tolist() == [L.V_MATCH, L.V_CHECK, L.V_GAP]
    row = out.iloc[1]
    assert row[L.C_CAND] == OWN[0]
    assert row[L.C_CAND_KW] == "getreidefrei"
    assert row[L.C_CAND_OVERLAP] == "75 %"
    assert row[L.C_CAND_SCORE] == 0.4
    assert row[L.C_CAND_CHUNK] == f"Text {OWN[0]}"
    assert OWN[0] in row[L.C_ADVICE]
    assert out.iloc[2][L.C_CAND] == ""


def test_cluster_column_is_added_and_original_untouched():
    result, lead, decisions = _decisions(["a", "b"], [[0.9, 0.1], [0.9, 0.1]])
    out = apply_serp(decisions, result, lead, _serps({"a": _urls(1, 2), "b": _urls(1, 2)}))
    assert out[L.C_CLUSTER].tolist() == [1, 1]
    assert L.C_CLUSTER not in decisions.columns


def test_query_missing_in_serps_gets_cluster_zero():
    result, lead, decisions = _decisions(["a"], [[0.9, 0.1]])
    out = apply_serp(decisions, result, lead, _serps({"z": _urls(1)}))
    assert out[L.C_CLUSTER].tolist() == [0]


def test_several_candidates_are_ordered_by_overlap():
    queries = ["luecke", "nachbar eins", "nachbar zwei"]
    result, lead, decisions = _decisions(queries, [[0.2, 0.3], [0.9, 0.1], [0.1, 0.9]])
    serps = _serps(
        {
            "luecke": _urls(1, 2, 3, 4),
            "nachbar eins": _urls(1, 2, 7, 8),  # 50 %
            "nachbar zwei": _urls(1, 2, 3, 9),  # 75 %
        }
    )
    row = apply_serp(decisions, result, lead, serps).iloc[0]
    assert row[L.C_CAND] == OWN[1]
    assert row[L.C_CAND_KW] == "nachbar zwei"
    assert row[L.C_CAND_MORE] == f"{OWN[0]} (Nachbar: nachbar eins, 50 %)"


def test_candidate_position_comes_from_rankings():
    queries = ["luecke", "nachbar"]
    result, lead, decisions = _decisions(queries, [[0.2, 0.1], [0.9, 0.1]])
    serps = _serps({"luecke": _urls(1, 2), "nachbar": _urls(1, 2)})
    rankings = pd.DataFrame(
        [("luecke", OWN[0], OWN[0], 44.0)], columns=["query_norm", "url", "url_norm", "position"]
    )
    row = apply_serp(decisions, result, lead, serps, rankings).iloc[0]
    assert row[L.C_CAND_POS] == "44"


def test_gap_summary_counts_one_page_per_cluster():
    decisions = pd.DataFrame(
        {
            L.C_QUERY: ["a", "b", "c", "d", "e"],
            L.C_VERDICT: [L.V_GAP, L.V_GAP, L.V_GAP, L.V_GAP, L.V_MATCH],
            L.C_CLUSTER: [1, 1, 0, 0, 1],
        }
    )
    summary = gap_summary(decisions)
    assert summary.to_dict("records") == [
        {L.C_CLUSTER: "1", "Lücken-Queries": 2, "Neue Seiten": 1, "Queries": "a | b"},
        {L.C_CLUSTER: L.NO_CLUSTER, "Lücken-Queries": 2, "Neue Seiten": 2, "Queries": "c | d"},
    ]
    assert count_new_pages(decisions) == 3


def test_count_new_pages_without_cluster_column():
    decisions = pd.DataFrame({L.C_QUERY: ["a", "b"], L.C_VERDICT: [L.V_GAP, L.V_GAP]})
    assert count_new_pages(decisions) == 2
