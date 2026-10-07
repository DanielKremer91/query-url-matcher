import pandas as pd
import pytest

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
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Score 0.8) | {U2} (Score 0.79)"


def test_close_but_below_threshold_is_not_flagged():
    assert _run(["q"], [[0.50, 0.49, 0.1]]).empty


def test_margin_is_adjustable():
    assert _run(["q"], [[0.80, 0.70, 0.1]]).empty
    assert len(_run(["q"], [[0.80, 0.70, 0.1]], margin=0.15)) == 1


def test_risk_verdict_is_listed_with_ranking_and_better_url():
    rankings = _rankings([("q", U1, U1, 4.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 4, Score 0.65) | {U2} (Score 0.9)"


def test_risk_row_shows_position_of_better_url_when_it_ranks():
    rankings = _rankings([("q", U1, U1, 4.0), ("q", U2, U2, 31.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 4, Score 0.65) | {U2} (Position 31, Score 0.9)"


def test_risk_row_for_ranking_url_outside_export_shows_position_only():
    alt = "https://a.de/alt"
    rankings = _rankings([("q", alt, alt, 2.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df.iloc[0][L.C_COMPETING] == f"{alt} (Position 2) | {U1} (Score 0.9)"
    assert df.iloc[0][L.C_REASON] == L.REASON_NOT_IN_EXPORT
    assert L.REASON_NOT_IN_EXPORT == "Rankende URL steht nicht im Frog-Export und wurde nicht verglichen"


def test_default_margin_is_one_hundredth():
    assert _run(["q"], [[0.80, 0.785, 0.1]]).empty
    assert len(_run(["q"], [[0.80, 0.79, 0.1]])) == 1


def test_margin_boundary_counts_as_close():
    df = _run(["q"], [[0.80, 0.78, 0.1]], margin=0.02)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Score 0.8) | {U2} (Score 0.78)"


def test_visible_stage_for_several_queries_uses_each_querys_rankings():
    rankings = _rankings(
        [("a", U1, U1, 3.0), ("a", U2, U2, 5.0), ("b", U1, U1, 2.0), ("b", U3, U3, 25.0), ("b", U2, U2, 7.0)]
    )
    df = _run(["a", "b"], [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]], rankings)
    assert df[L.C_QUERY].tolist() == ["a", "b"]
    assert df.iloc[1][L.C_COMPETING] == f"{U1} (Position 2, Score 0.9) | {U2} (Position 7, Score 0.1)"


def test_several_own_urls_ranking_is_visible_stage():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0), ("q", U3, U3, 45.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_VISIBLE]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 3, Score 0.9) | {U2} (Position 12, Score 0.1)"


def test_visible_stage_url_outside_export_shows_position_only():
    alt = "https://a.de/alt"
    rankings = _rankings([("q", U1, U1, 3.0), ("q", alt, alt, 8.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 3, Score 0.9) | {alt} (Position 8)"


def test_both_stages_can_apply_to_one_query():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER, L.STAGE_VISIBLE]


def test_reasons_come_from_labels():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    assert _run(["q"], [[0.65, 0.9, 0.1]], rankings)[L.C_REASON].tolist() == [L.REASON_BETTER, L.REASON_RANKING]
    assert _run(["q"], [[0.80, 0.79, 0.3]])[L.C_REASON].tolist() == [L.REASON_CLOSE]



def test_ok_row_with_an_almost_as_good_url_lists_the_ranking_url_first():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 31.0)])
    df = _run(["q"], [[0.842, 0.848, 0.1]], rankings)
    assert df[L.C_REASON].tolist() == [L.REASON_OK_CLOSE]
    assert L.REASON_OK_CLOSE == "Rankende Seite passt, eine weitere passt fast gleich gut"
    assert df.iloc[0][L.C_STAGE] == L.STAGE_DANGER
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 3, Score 0.842) | {U2} (Position 31, Score 0.848)"


def test_ok_row_without_a_close_url_has_no_cannibalisation_entry():
    rankings = _rankings([("q", U1, U1, 3.0)])
    assert _run(["q"], [[0.85, 0.80, 0.1]], rankings, margin=0.01).empty


def test_reason_better_says_clearly():
    assert L.REASON_BETTER == "Eine andere Seite passt deutlich besser als die rankende"


def test_negative_margin_is_rejected():
    result = make_result(["q"], [U1, U2, U3], [[0.9, 0.1, 0.1]])
    lead = result.lead("chunk")
    with pytest.raises(ValueError):
        find_cannibalization(result, lead, 0.6, build_decisions(result, lead, 0.6), margin=-0.01)


def test_decision_and_cannibalisation_sheet_agree():
    alt = "https://a.de/alt"
    queries = ["ok-nah", "ok-allein", "risiko", "nah-ohne-ranking", "knapp-unter-schwelle", "nicht-im-export"]
    scores = [[0.842, 0.848, 0.1], [0.9, 0.7, 0.1], [0.7, 0.9, 0.1], [0.81, 0.805, 0.1], [0.795, 0.803, 0.1], [0.9, 0.1, 0.1]]
    rankings = _rankings(
        [
            ("ok-nah", U1, U1, 3.0),
            ("ok-allein", U1, U1, 2.0),
            ("risiko", U1, U1, 4.0),
            ("knapp-unter-schwelle", U1, U1, 3.0),
            ("nicht-im-export", alt, alt, 2.0),
        ]
    )
    result = make_result(queries, [U1, U2, U3], scores)
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.8, rankings, margin=0.01)
    cannibal = find_cannibalization(result, lead, 0.8, decisions, rankings, margin=0.01)
    assert decisions[L.C_VERDICT].tolist() == [L.V_OK, L.V_OK, L.V_CANNIBAL, L.V_USE, L.V_CANNIBAL, L.V_CANNIBAL]
    reasons = dict(zip(cannibal[L.C_QUERY], cannibal[L.C_REASON]))
    assert reasons == {
        "ok-nah": L.REASON_OK_CLOSE,
        "risiko": L.REASON_BETTER,
        "nah-ohne-ranking": L.REASON_CLOSE,
        "knapp-unter-schwelle": L.REASON_BETTER_PLAIN,
        "nicht-im-export": L.REASON_NOT_IN_EXPORT,
    }
    assert reasons["nicht-im-export"] == L.REASON_NOT_IN_EXPORT



def test_risk_with_margin_zero_uses_the_plain_reason():
    rankings = _rankings([("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2, U3], [[0.842, 0.843, 0.1]])
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.6, rankings, margin=0)
    df = find_cannibalization(result, lead, 0.6, decisions, rankings, margin=0)
    assert df[L.C_REASON].tolist() == [L.REASON_BETTER_PLAIN]
    assert L.REASON_BETTER_PLAIN == "Eine andere Seite passt besser als die rankende"


def test_risk_with_ranking_url_below_threshold_uses_the_plain_reason():
    rankings = _rankings([("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2, U3], [[0.795, 0.803, 0.1]])
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.8, rankings)
    assert find_cannibalization(result, lead, 0.8, decisions, rankings)[L.C_REASON].tolist() == [L.REASON_BETTER_PLAIN]


def test_annotate_adds_the_cannibalisation_hint_to_every_decision_row():
    from qum.cannibal import annotate

    decisions = pd.DataFrame({L.C_QUERY: ["a", "b", "c"], L.C_VERDICT: [L.V_USE, L.V_OK, L.V_GAP]})
    cannibal = pd.DataFrame(
        [
            ("a", L.STAGE_DANGER, "x", "https://a.de/1 (Score 0.8) | https://a.de/2 (Score 0.79)"),
            ("b", L.STAGE_DANGER, "x", "https://a.de/3 (Score 0.9)"),
            ("b", L.STAGE_VISIBLE, "y", "https://a.de/3 (Position 2) | https://a.de/4 (Position 9)"),
        ],
        columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING],
    )
    out = annotate(decisions, cannibal)
    assert list(out.columns)[-1] == L.C_CANNIBAL
    assert out[L.C_CANNIBAL].tolist() == [
        "Gefahr: https://a.de/1 (Score 0.8) | https://a.de/2 (Score 0.79)",
        "Gefahr: https://a.de/3 (Score 0.9); Bereits sichtbar: https://a.de/3 (Position 2) | https://a.de/4 (Position 9)",
        "",
    ]
    assert L.C_CANNIBAL not in decisions.columns


def test_annotate_with_empty_cannibalisation_sheet():
    from qum.cannibal import annotate

    decisions = pd.DataFrame({L.C_QUERY: ["a"], L.C_VERDICT: [L.V_GAP]})
    out = annotate(decisions, pd.DataFrame(columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING]))
    assert out[L.C_CANNIBAL].tolist() == [""]


def test_one_word_for_cannibalisation():
    from qum import export

    assert L.V_CANNIBAL == "Kannibalisierungsgefahr"
    assert export.SHEET_CANNIBAL == "Kannibalisierungsgefahr"
    assert L.C_CANNIBAL == "Kannibalisierungsgefahr"
    assert [L.STAGE_DANGER, L.STAGE_VISIBLE] == ["Gefahr", "Bereits sichtbar"]
