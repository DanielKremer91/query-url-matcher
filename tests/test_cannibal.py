import pandas as pd

from qum import labels as L
from qum.cannibal import find_cannibalization
from qum.verdict import build_decisions
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def _run(queries, scores, rankings=None, **kwargs):
    result = make_result(queries, [U1, U2, U3], scores)
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.6, rankings)
    return find_cannibalization(result, lead, 0.6, decisions, rankings, **kwargs)


def test_columns_and_empty_result():
    df = _run(["q"], [[0.9, 0.5, 0.1]])
    assert list(df.columns) == [L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING]
    assert df.empty


def test_two_urls_close_together_without_rankings():
    df = _run(["q"], [[0.80, 0.79, 0.3]])
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Score 0.8) | {U2} (Score 0.79)"


def test_close_but_below_threshold_is_not_flagged():
    assert _run(["q"], [[0.50, 0.49, 0.1]]).empty


def test_margin_is_adjustable():
    assert _run(["q"], [[0.80, 0.70, 0.1]]).empty
    assert len(_run(["q"], [[0.80, 0.70, 0.1]], margin=0.15)) == 1


def test_risk_verdict_is_listed_with_ranking_and_better_url():
    rankings = _rankings([("q", U1, U1, 4.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 4) | {U2} (Score 0.9)"


def test_several_own_urls_ranking_is_visible_stage():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0), ("q", U3, U3, 45.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_VISIBLE]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 3) | {U2} (Position 12)"


def test_both_stages_can_apply_to_one_query():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK, L.STAGE_VISIBLE]
