import pandas as pd

from qum import labels as L
from qum.gaps import COLUMNS, count_new_pages, find_gaps
from qum.verdict import build_decisions
from tests.conftest import make_result

U1, U2 = "https://a.de/1", "https://a.de/2"
QUERIES = ["passt", "knapp", "schwach", "leer"]
SCORES = [[0.9, 0.1], [0.78, 0.2], [0.3, 0.6], [0.1, 0.05]]


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


RANKINGS = _rankings([("knapp", U1, U1, 4.0), ("schwach", U2, U2, 35.0), ("passt", U1, U1, 1.0)])


def _gaps(rankings=None, topics=None, threshold=0.8, **kwargs):
    result = make_result(QUERIES, [U1, U2], SCORES)
    return find_gaps(result, result.lead("chunk"), threshold, rankings, topics, **kwargs)


def test_columns():
    df = _gaps()
    assert list(df.columns) == COLUMNS == [
        L.C_QUERY, L.C_BEST_SCORE, L.C_TO_THRESHOLD, L.C_BEST_URL, L.C_POSITION, L.C_RANK_URL, L.C_TOPIC,
    ]
    assert [L.C_BEST_SCORE, L.C_TOPIC] == ["Bester Score", "Thema"]


def test_without_rankings_every_query_below_the_threshold_is_a_gap_surest_first():
    df = _gaps()
    assert df[L.C_QUERY].tolist() == ["leer", "schwach", "knapp"]
    assert df[L.C_BEST_SCORE].tolist() == [0.1, 0.6, 0.78]
    assert df[L.C_TO_THRESHOLD].tolist() == [-0.7, -0.2, -0.02]
    assert df[L.C_BEST_URL].tolist() == [U1, U2, U1]
    assert df[L.C_POSITION].tolist() == ["", "", ""] and df[L.C_RANK_URL].tolist() == ["", "", ""]


def test_a_good_ranking_is_never_a_gap():
    df = _gaps(RANKINGS)
    assert df[L.C_QUERY].tolist() == ["leer", "schwach"]  # "knapp" rankt auf 4: Rankt trotz schwachem Match
    assert df[L.C_POSITION].tolist() == ["", "35"]
    assert df[L.C_RANK_URL].tolist() == ["", U2]


def test_ranking_up_to_the_gap_position_is_no_gap():
    assert _gaps(RANKINGS, gap_position=40)[L.C_QUERY].tolist() == ["leer"]
    assert _gaps(RANKINGS, gap_position=20)[L.C_QUERY].tolist() == ["leer", "schwach"]
    assert _gaps(None, gap_position=40)[L.C_QUERY].tolist() == ["leer", "schwach", "knapp"]  # ohne Rankings wirkungslos


def test_the_sheet_lists_exactly_the_content_gap_verdicts():
    result = make_result(QUERIES, [U1, U2], SCORES)
    lead = result.lead("chunk")
    for rankings, gap_position in ((None, 0), (RANKINGS, 0), (RANKINGS, 20), (RANKINGS, 40)):
        decisions = build_decisions(result, lead, 0.8, rankings, gap_position=gap_position)
        verdict_gaps = set(decisions.loc[decisions[L.C_VERDICT] == L.V_GAP, L.C_QUERY])
        sheet = find_gaps(result, lead, 0.8, rankings, gap_position=gap_position)
        assert set(sheet[L.C_QUERY]) == verdict_gaps


def test_topic_is_the_serp_cluster_and_empty_without_cluster():
    df = _gaps(topics=[1, 2, 0, 2])
    assert df[L.C_QUERY].tolist() == ["leer", "schwach", "knapp"]
    assert df[L.C_TOPIC].tolist() == [2, pd.NA, 2]


def test_topic_is_empty_without_serps():
    assert _gaps()[L.C_TOPIC].isna().all()


def test_new_pages_one_per_topic_and_one_per_gap_without_topic():
    assert count_new_pages(_gaps(topics=[1, 2, 0, 2])) == 2
    assert count_new_pages(_gaps(topics=[0, 0, 0, 0])) == 3
    assert count_new_pages(_gaps(topics=[5, 7, 3, 9])) == 3
    assert count_new_pages(_gaps(threshold=0.01)) == 0
