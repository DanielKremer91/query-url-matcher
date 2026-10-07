import pandas as pd

from qum import labels as L
from qum.gaps import COLUMNS, count_new_pages, find_gaps
from tests.conftest import make_result

U1, U2 = "https://a.de/1", "https://a.de/2"
QUERIES = ["passt", "knapp", "schwach", "leer"]
SCORES = [[0.9, 0.1], [0.78, 0.2], [0.3, 0.6], [0.1, 0.05]]


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


RANKINGS = _rankings([("knapp", U1, U1, 4.0), ("schwach", U2, U2, 35.0), ("passt", U1, U1, 1.0)])


def _gaps(rankings=None, topics=None, **kwargs):
    result = make_result(QUERIES, [U1, U2], SCORES)
    return find_gaps(result, result.lead("chunk"), 0.8, rankings, topics, **kwargs)


def test_columns():
    df = _gaps()
    assert list(df.columns) == COLUMNS == [L.C_QUERY, L.C_BEST_SCORE, L.C_BEST_URL, L.C_POSITION, L.C_RANK_URL, L.C_TOPIC]
    assert [L.C_BEST_SCORE, L.C_TOPIC] == ["Bester Score", "Thema"]


def test_default_uses_the_threshold_from_step_7b():
    df = _gaps(RANKINGS)
    assert df[L.C_QUERY].tolist() == ["knapp", "schwach", "leer"]
    assert df[L.C_BEST_SCORE].tolist() == [0.78, 0.6, 0.1]
    assert df[L.C_BEST_URL].tolist() == [U1, U2, U1]
    assert df[L.C_POSITION].tolist() == ["4", "35", ""]
    ranking_urls = df[L.C_RANK_URL].tolist()
    assert ranking_urls[2] == "" and all(ranking_urls[:2])


def test_own_score_limit_replaces_the_threshold():
    assert _gaps(below=0.7)[L.C_QUERY].tolist() == ["schwach", "leer"]
    assert _gaps(below=0.95)[L.C_QUERY].tolist() == QUERIES


def test_ranking_condition_keeps_only_queries_without_a_good_own_ranking():
    assert _gaps(RANKINGS, max_position=10)[L.C_QUERY].tolist() == ["schwach", "leer"]
    assert _gaps(RANKINGS, max_position=40)[L.C_QUERY].tolist() == ["leer"]
    assert _gaps(RANKINGS, max_position=0)[L.C_QUERY].tolist() == ["knapp", "schwach", "leer"]  # 0 = aus


def test_ranking_condition_is_ignored_without_rankings():
    df = _gaps(None, max_position=10)
    assert df[L.C_QUERY].tolist() == ["knapp", "schwach", "leer"]
    assert df[L.C_POSITION].tolist() == ["", "", ""]
    assert df[L.C_RANK_URL].tolist() == ["", "", ""]


def test_topic_is_the_serp_cluster_and_empty_without_cluster():
    df = _gaps(topics=[1, 2, 0, 2])
    assert df[L.C_TOPIC].tolist() == [2, pd.NA, 2]


def test_topic_is_empty_without_serps():
    assert _gaps()[L.C_TOPIC].isna().all()


def test_new_pages_one_per_topic_and_one_per_gap_without_topic():
    assert count_new_pages(_gaps(topics=[1, 2, 0, 2])) == 2
    assert count_new_pages(_gaps(topics=[0, 0, 0, 0])) == 3
    assert count_new_pages(_gaps(topics=[5, 7, 3, 9])) == 3
    assert count_new_pages(_gaps(below=0.01)) == 0
