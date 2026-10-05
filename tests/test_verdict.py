import pandas as pd

from qum import labels as L
from qum.verdict import ADVICE, build_decisions
from tests.conftest import make_result

U1, U2 = "https://a.de/1", "https://a.de/2"


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def test_without_rankings_match_or_gap():
    result = make_result(["passt", "fehlt"], [U1, U2], [[0.8, 0.1], [0.2, 0.3]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert df[L.C_VERDICT].tolist() == [L.V_MATCH, L.V_GAP]
    assert df[L.C_BEST_URL].tolist() == [U1, U2]
    assert df[L.C_RANK_URL].tolist() == ["", ""]


def test_columns():
    result = make_result(["q"], [U1], [[0.8]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert list(df.columns) == [
        L.C_QUERY, L.C_VERDICT, L.C_BEST_URL, L.C_CHUNK, L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI,
        L.C_RANK_URL, L.C_POSITION, L.C_NOTE, L.C_ADVICE,
    ]


def test_all_five_verdicts_with_rankings():
    queries = ["ok", "risiko", "beobachten", "nutzen", "luecke"]
    scores = [
        [0.9, 0.1],  # rankt gut mit U1, U1 ist bester Treffer
        [0.7, 0.9],  # rankt gut mit U1, U2 passt besser
        [0.2, 0.1],  # rankt gut mit U1, nichts passt
        [0.1, 0.8],  # rankt schwach, U2 passt
        [0.2, 0.3],  # kein Ranking, nichts passt
    ]
    rankings = _rankings(
        [
            ("ok", U1, U1, 2.0),
            ("risiko", U1, U1, 4.0),
            ("beobachten", U1, U1, 1.0),
            ("nutzen", U1, U1, 35.0),
        ]
    )
    result = make_result(queries, [U1, U2], scores)
    df = build_decisions(result, result.lead("chunk"), 0.6, rankings, good_position=10)
    assert df[L.C_VERDICT].tolist() == [L.V_OK, L.V_RISK, L.V_WATCH, L.V_USE, L.V_GAP]
    assert df[L.C_POSITION].tolist() == ["2", "4", "1", "35", ""]
    assert df[L.C_RANK_URL].tolist() == [U1, U1, U1, U1, ""]


def test_best_position_wins_when_several_urls_rank():
    rankings = _rankings([("q", U2, U2, 8.0), ("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2], [[0.9, 0.1]])
    df = build_decisions(result, result.lead("chunk"), 0.6, rankings)
    assert df.iloc[0][L.C_RANK_URL] == U1
    assert df.iloc[0][L.C_VERDICT] == L.V_OK


def test_ranking_url_missing_in_export_is_noted_and_treated_as_other_url():
    rankings = _rankings([("q", "https://a.de/alt", "https://a.de/alt", 2.0)])
    result = make_result(["q"], [U1], [[0.9]])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_VERDICT] == L.V_RISK
    assert row[L.C_NOTE] == L.NOTE_NOT_IN_EXPORT


def test_query_matching_is_case_insensitive():
    rankings = _rankings([("hunde futter", U1, U1, 2.0)])
    result = make_result(["Hunde  Futter"], [U1], [[0.9]])
    assert build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0][L.C_VERDICT] == L.V_OK


def test_advice_is_template_with_values():
    result = make_result(["q"], [U1, U2], [[0.1, 0.8]])
    rankings = _rankings([("q", U1, U1, 35.0)])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_ADVICE] == ADVICE[L.V_USE].format(best=U2, rank_url=U1, position="35")
    assert U2 in row[L.C_ADVICE]


def test_every_verdict_has_advice():
    for verdict in [L.V_MATCH, L.V_GAP, L.V_OK, L.V_RISK, L.V_WATCH, L.V_USE, L.V_CHECK]:
        assert verdict in ADVICE
