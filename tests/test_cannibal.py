import pandas as pd
import pytest

from qum import labels as L
from qum.cannibal import COLUMNS, annotate, find_cannibalization
from qum.verdict import build_decisions
from tests.conftest import make_result

U1, U2, U3, U4, U5 = (f"https://a.de/{n}" for n in range(1, 6))
ALT = "https://a.de/alt"  # rankt, steht aber nicht im Frog-Export


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def _run(queries, scores, rankings=None, threshold=0.6, urls=(U1, U2, U3), **kwargs):
    result = make_result(queries, list(urls), scores)
    return find_cannibalization(result, result.lead("chunk"), threshold, rankings, **kwargs)


def _urls(row):
    """(URL, Score, Position) der bis zu drei konkurrierenden URLs, leere Plätze weggelassen."""
    out = []
    for url, score, position in ((L.C_URL_1, L.C_SCORE_1, L.C_POS_1), (L.C_URL_2, L.C_SCORE_2, L.C_POS_2),
                                 (L.C_URL_3, L.C_SCORE_3, L.C_POS_3)):
        if row[url]:
            out.append((row[url], None if pd.isna(row[score]) else row[score], row[position]))
    return out


def test_columns_and_empty_result():
    df = _run(["q"], [[0.9, 0.5, 0.1]])
    assert list(df.columns) == COLUMNS == [
        L.C_QUERY,
        L.C_URL_1, L.C_SCORE_1, L.C_POS_1, L.C_URL_2, L.C_SCORE_2, L.C_POS_2, L.C_URL_3, L.C_SCORE_3, L.C_POS_3,
        L.C_POSITION, L.C_RANK_URL, L.C_STAGE, L.C_REASON, L.C_PRIORITY,
    ]
    assert [L.C_URL_1, L.C_SCORE_2, L.C_POS_3] == ["URL 1", "Score 2", "Position 3"]
    assert df.empty


# --- Stufe Gefahr ------------------------------------------------------------------------------------------------


def test_several_close_pages_without_rankings_are_ordered_by_score():
    df = _run(["q"], [[0.79, 0.80, 0.3]])
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df[L.C_REASON].tolist() == [L.REASON_CLOSE]
    row = df.iloc[0]
    assert _urls(row) == [(U2, 0.8, ""), (U1, 0.79, "")]
    assert row[L.C_URL_3] == "" and pd.isna(row[L.C_SCORE_3]) and row[L.C_POS_3] == ""


def test_close_but_below_threshold_is_not_listed():
    assert _run(["q"], [[0.50, 0.49, 0.1]]).empty


def test_margin_is_adjustable_and_its_boundary_counts_as_close():
    assert _run(["q"], [[0.80, 0.785, 0.1]])[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE]
    assert _run(["q"], [[0.80, 0.79, 0.1]])[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert _run(["q"], [[0.80, 0.70, 0.1]], margin=0.15)[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert _urls(_run(["q"], [[0.80, 0.78, 0.1]], margin=0.02).iloc[0]) == [(U1, 0.8, ""), (U2, 0.78, "")]


def test_weak_ranking_with_several_close_pages_is_ordered_by_score_with_positions():
    rankings = _rankings([("q", U3, U3, 35.0)])
    df = _run(["q"], [[0.1, 0.845, 0.85]], rankings)
    assert df[L.C_REASON].tolist() == [L.REASON_CLOSE]
    assert _urls(df.iloc[0]) == [(U3, 0.85, "35"), (U2, 0.845, "")]


def test_ranking_page_clearly_worse_lists_the_ranking_url_first():
    rankings = _rankings([("q", U1, U1, 4.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df[L.C_REASON].tolist() == [L.REASON_BETTER]
    assert _urls(df.iloc[0]) == [(U1, 0.65, "4"), (U2, 0.9, "")]


def test_better_page_shows_its_own_position_when_it_ranks():
    rankings = _rankings([("q", U1, U1, 4.0), ("q", U2, U2, 31.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert _urls(df.iloc[0]) == [(U1, 0.65, "4"), (U2, 0.9, "31")]


def test_ranking_url_outside_the_export_has_no_score():
    rankings = _rankings([("q", ALT, ALT, 2.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_REASON].tolist() == [L.REASON_NOT_IN_EXPORT]
    assert _urls(df.iloc[0]) == [(ALT, None, "2"), (U1, 0.9, "")]
    assert L.REASON_NOT_IN_EXPORT == "Rankende URL steht nicht im Frog-Export und wurde nicht verglichen"


def test_plain_reason_with_margin_zero_or_ranking_page_below_threshold():
    rankings = _rankings([("q", U1, U1, 3.0)])
    assert _run(["q"], [[0.842, 0.843, 0.1]], rankings, margin=0)[L.C_REASON].tolist() == [L.REASON_BETTER_PLAIN]
    df = _run(["q"], [[0.795, 0.803, 0.1]], rankings, threshold=0.8)
    assert df[L.C_REASON].tolist() == [L.REASON_BETTER_PLAIN]
    assert _urls(df.iloc[0]) == [(U1, 0.795, "3"), (U2, 0.803, "")]
    assert L.REASON_BETTER == "Eine andere Seite passt deutlich besser als die rankende"
    assert L.REASON_BETTER_PLAIN == "Eine andere Seite passt besser als die rankende"


def test_ok_with_an_almost_as_good_page_lists_the_ranking_url_first():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 31.0)])
    df = _run(["q"], [[0.842, 0.848, 0.1]], rankings, threshold=0.8)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]
    assert df[L.C_REASON].tolist() == [L.REASON_OK_CLOSE]
    assert L.REASON_OK_CLOSE == "Rankende Seite passt, eine weitere passt fast gleich gut"
    assert _urls(df.iloc[0]) == [(U1, 0.842, "3"), (U2, 0.848, "31")]


def test_ok_without_a_close_page_is_only_a_possible_danger():
    rankings = _rankings([("q", U1, U1, 3.0)])
    assert _run(["q"], [[0.85, 0.80, 0.1]], rankings, threshold=0.8)[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE]


def test_more_than_three_urls_are_counted_in_the_reason():
    urls = (U1, U2, U3, U4, U5)
    df = _run(["q"], [[0.80, 0.805, 0.799, 0.801, 0.802]], urls=urls)
    row = df.iloc[0]
    assert row[L.C_REASON] == f"{L.REASON_CLOSE} … und 2 weitere"
    assert [url for url, _, _ in _urls(row)] == [U2, U5, U4]


# --- Stufe Bereits sichtbar --------------------------------------------------------------------------------------


def test_several_own_urls_ranking_are_ordered_by_position():
    rankings = _rankings([("q", U2, U2, 12.0), ("q", U1, U1, 3.0), ("q", U3, U3, 45.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_VISIBLE]
    assert df[L.C_REASON].tolist() == [L.REASON_RANKING]
    assert _urls(df.iloc[0]) == [(U1, 0.9, "3"), (U2, 0.1, "12")]


def test_visible_position_is_adjustable():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 25.0)])
    assert _run(["q"], [[0.9, 0.1, 0.1]], rankings).empty
    assert len(_run(["q"], [[0.9, 0.1, 0.1]], rankings, visible_position=30)) == 1


def test_visible_url_outside_the_export_has_no_score():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", ALT, ALT, 8.0)])
    assert _urls(_run(["q"], [[0.9, 0.1, 0.1]], rankings).iloc[0]) == [(U1, 0.9, "3"), (ALT, None, "8")]


def test_visible_stage_with_more_than_three_urls():
    rankings = _rankings([("q", u, u, float(p)) for p, u in enumerate((U1, U2, U3, U4), start=1)])
    row = _run(["q"], [[0.9, 0.1, 0.1, 0.1]], rankings, urls=(U1, U2, U3, U4)).iloc[0]
    assert row[L.C_REASON] == f"{L.REASON_RANKING} … und 1 weitere"
    assert [position for _, _, position in _urls(row)] == ["1", "2", "3"]


def test_visible_stage_for_several_queries_uses_each_querys_rankings():
    rankings = _rankings(
        [("a", U1, U1, 3.0), ("a", U2, U2, 5.0), ("b", U1, U1, 2.0), ("b", U3, U3, 25.0), ("b", U2, U2, 7.0)]
    )
    df = _run(["a", "b"], [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]], rankings)
    assert df[L.C_QUERY].tolist() == ["a", "b"]
    assert _urls(df.iloc[1]) == [(U1, 0.9, "2"), (U2, 0.1, "7")]


def test_both_stages_can_apply_to_one_query_danger_first():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER, L.STAGE_VISIBLE]
    assert df[L.C_REASON].tolist() == [L.REASON_BETTER, L.REASON_RANKING]


# --- Übereinstimmung mit den Urteilen ----------------------------------------------------------------------------


def test_every_cannibalisation_verdict_and_every_competing_page_at_good_ranking_is_listed():
    queries = ["ok-nah", "ok-allein", "deutlich", "nah-schwach", "nah-ohne-ranking", "knapp-unter", "nicht-im-export",
               "nutzen", "luecke"]
    scores = [[0.842, 0.848, 0.1], [0.9, 0.7, 0.1], [0.7, 0.9, 0.1], [0.81, 0.805, 0.1], [0.81, 0.805, 0.1],
              [0.795, 0.803, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.5, 0.1, 0.1]]
    rankings = _rankings(
        [("ok-nah", U1, U1, 3.0), ("ok-allein", U1, U1, 2.0), ("deutlich", U1, U1, 4.0), ("nah-schwach", U1, U1, 40.0),
         ("knapp-unter", U1, U1, 3.0), ("nicht-im-export", ALT, ALT, 2.0), ("nutzen", U1, U1, 50.0)]
    )
    result = make_result(queries, [U1, U2, U3], scores)
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.8, rankings)
    cannibal = find_cannibalization(result, lead, 0.8, rankings)
    assert decisions[L.C_VERDICT].tolist() == [
        L.V_OK, L.V_OK, L.V_WATCH, L.V_CANNIBAL, L.V_CANNIBAL, L.V_WATCH, L.V_OK, L.V_MATCH, L.V_GAP,
    ]
    # gutes Ranking: das Urteil folgt der rankenden Seite, die Gefahr steht trotzdem im Blatt
    assert annotate(decisions, cannibal)[L.C_CANNIBAL].tolist() == [
        L.YES, L.NO, L.YES, L.YES, L.YES, L.YES, L.YES, L.NO, L.NO,
    ]
    flagged = set(decisions.loc[decisions[L.C_VERDICT] == L.V_CANNIBAL, L.C_QUERY])
    assert flagged <= set(cannibal.loc[cannibal[L.C_STAGE] == L.STAGE_DANGER, L.C_QUERY])
    assert dict(zip(cannibal[L.C_QUERY], cannibal[L.C_REASON])) == {
        "ok-nah": L.REASON_OK_CLOSE,
        "deutlich": L.REASON_BETTER,
        "nah-schwach": L.REASON_CLOSE,
        "nah-ohne-ranking": L.REASON_CLOSE,
        "knapp-unter": L.REASON_BETTER_PLAIN,
        "nicht-im-export": L.REASON_NOT_IN_EXPORT,
    }


def test_settings_are_the_same_as_for_the_verdicts():
    rankings = _rankings([("q", U1, U1, 12.0)])
    result = make_result(["q"], [U1, U2, U3], [[0.70, 0.90, 0.1]])
    lead = result.lead("chunk")
    # mit rankt gut bis 15 ist es eine Gefahr wegen der rankenden Seite, sonst nur eine mögliche
    assert find_cannibalization(result, lead, 0.6, rankings)[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE]
    df = find_cannibalization(result, lead, 0.6, rankings, good_position=15)
    assert df[L.C_REASON].tolist() == [L.REASON_BETTER]


def test_negative_margin_is_rejected():
    result = make_result(["q"], [U1, U2, U3], [[0.9, 0.1, 0.1]])
    with pytest.raises(ValueError):
        find_cannibalization(result, result.lead("chunk"), 0.6, margin=-0.01)


# --- Spalte Kannibalisierungsgefahr in der Übersicht -------------------------------------------------------------


def test_annotate_says_yes_for_every_query_in_the_cannibalisation_sheet():
    decisions = pd.DataFrame({L.C_QUERY: ["a", "b", "c"], L.C_VERDICT: [L.V_OK, L.V_MATCH, L.V_CANNIBAL]})
    rankings = _rankings([("b", U1, U1, 3.0), ("b", U2, U2, 9.0)])
    result = make_result(["a", "b", "c"], [U1, U2, U3], [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.8, 0.8, 0.1]])
    cannibal = find_cannibalization(result, result.lead("chunk"), 0.6, rankings)
    out = annotate(decisions, cannibal)
    assert list(out.columns)[-1] == L.C_CANNIBAL
    assert out[L.C_CANNIBAL].tolist() == [L.NO, L.YES, L.YES]
    assert L.C_CANNIBAL not in decisions.columns


def test_annotate_with_empty_cannibalisation_sheet():
    decisions = pd.DataFrame({L.C_QUERY: ["a"], L.C_VERDICT: [L.V_GAP]})
    assert annotate(decisions, pd.DataFrame(columns=COLUMNS))[L.C_CANNIBAL].tolist() == [L.NO]


def test_one_word_for_cannibalisation():
    from qum import export

    assert L.V_CANNIBAL == "Kannibalisierungsgefahr"
    assert export.SHEET_CANNIBAL == "Kannibalisierungsgefahr"
    assert L.C_CANNIBAL == "Kannibalisierungsgefahr"
    assert [L.STAGE_DANGER, L.STAGE_POSSIBLE, L.STAGE_VISIBLE] == ["Gefahr", "Möglich", "Kannibalisierung bereits sichtbar"]


# --- Stufe Möglich: weitere passende Seiten, deutlich hinter der besten -----------------------------------------


def test_possible_stage_for_further_fitting_pages_far_behind_without_rankings():
    df = _run(["q"], [[0.9, 0.7, 0.65]])
    assert df[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE]
    assert df[L.C_REASON].tolist() == [L.REASON_FURTHER]
    assert _urls(df.iloc[0]) == [(U1, 0.9, ""), (U2, 0.7, ""), (U3, 0.65, "")]


def test_possible_stage_when_ranking_well_with_the_best_page_and_another_fits_far_behind():
    rankings = _rankings([("q", U1, U1, 3.0)])
    df = _run(["q"], [[0.9, 0.7, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE]
    assert _urls(df.iloc[0]) == [(U1, 0.9, "3"), (U2, 0.7, "")]


def test_no_possible_stage_with_one_fitting_page_or_when_already_in_danger():
    assert _run(["q"], [[0.9, 0.5, 0.1]]).empty
    df = _run(["q"], [[0.9, 0.895, 0.7]])
    assert df[L.C_STAGE].tolist() == [L.STAGE_DANGER]


def test_possible_and_visible_stages_can_both_apply():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    df = _run(["q"], [[0.9, 0.7, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_POSSIBLE, L.STAGE_VISIBLE]


def test_possible_stage_counts_further_pages_beyond_three():
    df = _run(["q"], [[0.9, 0.7, 0.7, 0.7, 0.7]], urls=(U1, U2, U3, U4, U5))
    assert df[L.C_REASON].tolist() == [f"{L.REASON_FURTHER} … und 2 weitere"]


def test_annotate_grades_yes_possible_no():
    rankings = _rankings([("sichtbar", U1, U1, 3.0), ("sichtbar", U2, U2, 9.0)])
    queries = ["gefahr", "moeglich", "nein", "sichtbar", "moeglich-und-sichtbar"]
    scores = [[0.8, 0.8, 0.1], [0.9, 0.7, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.7, 0.1]]
    rankings = pd.concat([rankings, _rankings([("moeglich-und-sichtbar", U1, U1, 2.0),
                                               ("moeglich-und-sichtbar", U3, U3, 8.0)])])
    result = make_result(queries, [U1, U2, U3], scores)
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.6, rankings)
    out = annotate(decisions, find_cannibalization(result, lead, 0.6, rankings))
    assert out[L.C_CANNIBAL].tolist() == [L.YES, L.MAYBE, L.NO, L.YES, L.YES]
    assert L.MAYBE == "möglich" and L.STAGE_POSSIBLE == "Möglich"


def test_every_row_shows_the_querys_best_own_ranking():
    rankings = _rankings([("q", U2, U2, 12.0), ("q", U1, U1, 4.0), ("ohne", U3, U3, 50.0)])
    df = _run(["q", "nicht-rankend"], [[0.65, 0.9, 0.1], [0.8, 0.8, 0.1]], rankings)
    rows = {(q, s): (p, u) for q, s, p, u in zip(df[L.C_QUERY], df[L.C_STAGE], df[L.C_POSITION], df[L.C_RANK_URL])}
    assert rows[("q", L.STAGE_DANGER)] == ("4", U1)
    assert rows[("q", L.STAGE_VISIBLE)] == ("4", U1)
    assert rows[("nicht-rankend", L.STAGE_DANGER)] == ("", "")


def test_ranking_columns_are_empty_without_rankings():
    df = _run(["q"], [[0.8, 0.8, 0.1]])
    assert df[L.C_POSITION].tolist() == [""] and df[L.C_RANK_URL].tolist() == [""]


# --- Einordnung: wie dringend, je nach eigenem Ranking ------------------------------------------------------------


def _priority(scores, ranking, **kwargs):
    rankings = None if ranking is None else _rankings([("q", url, url, float(pos)) for url, pos in ranking])
    if ranking == []:
        rankings = _rankings([("andere query", U1, U1, 1.0)])
    return _run(["q"], [scores], rankings, **kwargs)[[L.C_STAGE, L.C_PRIORITY]].values.tolist()


def test_priority_values():
    assert [L.PRIO_HIGH, L.PRIO_MID, L.PRIO_LOW, L.PRIO_OPEN] == [
        "hoch: kein Top-Ranking", "mittel: Top-Ranking mit anderer Seite", "niedrig: Top-Ranking mit passender Seite",
        "offen: ohne Rankings",
    ]


def test_priority_without_rankings_is_open():
    assert _priority([0.8, 0.8, 0.1], None) == [[L.STAGE_DANGER, L.PRIO_OPEN]]


def test_priority_without_a_top_ranking_is_high():
    assert _priority([0.8, 0.8, 0.1], [(U1, 25)]) == [[L.STAGE_DANGER, L.PRIO_HIGH]]
    assert _priority([0.8, 0.8, 0.1], []) == [[L.STAGE_DANGER, L.PRIO_HIGH]]  # Rankings geladen, Query rankt nicht


def test_priority_with_a_top_ranking_of_the_best_page_is_low():
    assert _priority([0.805, 0.8, 0.1], [(U1, 3)]) == [[L.STAGE_DANGER, L.PRIO_LOW]]
    assert _priority([0.9, 0.7, 0.1], [(U1, 3)]) == [[L.STAGE_POSSIBLE, L.PRIO_LOW]]


def test_priority_with_a_top_ranking_of_another_page_is_medium():
    assert _priority([0.8, 0.805, 0.1], [(U1, 3)]) == [[L.STAGE_DANGER, L.PRIO_MID]]  # knapp hinter der besten
    assert _priority([0.65, 0.9, 0.1], [(U1, 3)]) == [[L.STAGE_DANGER, L.PRIO_MID]]
    assert _priority([0.9, 0.1, 0.1], [(ALT, 2)]) == [[L.STAGE_DANGER, L.PRIO_MID]]  # nicht im Frog-Export


def test_priority_of_the_visible_stage_follows_the_best_own_ranking():
    assert _priority([0.9, 0.1, 0.1], [(U1, 3), (U2, 15)]) == [[L.STAGE_VISIBLE, L.PRIO_LOW]]
    assert _priority([0.9, 0.1, 0.1], [(U1, 14), (U2, 15)]) == [[L.STAGE_VISIBLE, L.PRIO_HIGH]]
