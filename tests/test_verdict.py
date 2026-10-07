import pandas as pd
import pytest

from qum import labels as L
from qum.match import MATCH_COLUMNS
from qum.verdict import OVERVIEW_COLUMNS, build_decisions
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"
ALT = "https://a.de/alt"  # rankt, steht aber nicht im Frog-Export


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def _row(scores, ranking=None, threshold=0.8, margin=0.01, good_position=10, urls=(U1, U2, U3)):
    """Eine Query "q". ranking: None (keine Rankings geladen), [] (geladen, aber nicht für q) oder [(url, position)]."""
    rankings = None if ranking is None else _rankings([("q", url, url, float(pos)) for url, pos in ranking])
    if ranking == []:
        rankings = _rankings([("andere query", U1, U1, 1.0)])
    result = make_result(["q"], list(urls), [scores])
    return build_decisions(result, result.lead("chunk"), threshold, rankings, good_position=good_position, margin=margin).iloc[0]


# --- Die Urteilstabelle, Zelle für Zelle -------------------------------------------------------------------------


# Rankt gut (beste eigene Position bis rankt_gut_bis_position)
def test_ranks_well_and_exactly_one_page_fits_clearly_is_ok():
    row = _row([0.9, 0.7, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_RANK_IS_BEST] == L.YES


def test_ranks_well_with_several_close_pages_and_the_ranking_page_best_is_ok():
    row = _row([0.85, 0.845, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_RANK_IS_BEST] == L.YES


def test_ranks_well_with_several_close_pages_and_the_ranking_page_within_the_margin_is_ok():
    row = _row([0.842, 0.848, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_RANK_IS_BEST] == L.NO  # knapp dahinter ist trotzdem nicht die beste
    assert row[L.C_BEST_URL] == U2  # immer die wirklich beste URL, keine Ersetzung durch die rankende


# Bei gutem Ranking zählt die rankende Seite: passt sie, ist es in Ordnung, sonst rankt sie trotz schwachem Match.
# Konkurriert eine andere Seite, steht das in der Spalte Kannibalisierungsgefahr (test_cannibal.py).
def test_ranks_well_and_another_page_clearly_better_is_ok_when_the_ranking_page_fits():
    row = _row([0.81, 0.86, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_RANK_IS_BEST] == L.NO


def test_ranks_well_and_the_ranking_page_below_the_threshold_while_another_fits_is_a_weak_match():
    row = _row([0.795, 0.803, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_WATCH
    assert row[L.C_RANK_IS_BEST] == L.NO


def test_ranks_well_with_the_ranking_url_outside_the_export_is_ok():
    row = _row([0.9, 0.1, 0.1], ranking=[(ALT, 2)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_RANK_IS_BEST] == L.CMP_NOT_IN_EXPORT
    assert row[L.C_RANK_URL] == ALT


def test_ranks_well_with_one_clear_page_that_is_not_the_ranking_page_is_a_weak_match():
    row = _row([0.3, 0.9, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_WATCH
    assert row[L.C_RANK_IS_BEST] == L.NO


def test_ranks_well_but_nothing_fits():
    row = _row([0.55, 0.555, 0.1], ranking=[(U1, 3)])
    assert row[L.C_VERDICT] == L.V_WATCH  # "nichts passt" wird zuerst geprüft


# Rankt schwach (schlechter als rankt_gut_bis_position)
def test_ranks_weakly_and_one_page_fits_clearly():
    assert _row([0.1, 0.9, 0.7], ranking=[(U1, 35)])[L.C_VERDICT] == L.V_MATCH


def test_ranks_weakly_and_several_pages_fit_close_together_is_cannibalisation():
    assert _row([0.1, 0.85, 0.845], ranking=[(U1, 35)])[L.C_VERDICT] == L.V_CANNIBAL


def test_ranks_weakly_and_nothing_fits_is_a_gap():
    assert _row([0.1, 0.5, 0.4], ranking=[(U1, 35)])[L.C_VERDICT] == L.V_GAP


# Rankings geladen, aber kein Ranking für die Query
def test_not_ranking_at_all_follows_the_weak_row():
    assert _row([0.1, 0.9, 0.7], ranking=[])[L.C_VERDICT] == L.V_MATCH
    assert _row([0.1, 0.85, 0.845], ranking=[])[L.C_VERDICT] == L.V_CANNIBAL
    assert _row([0.1, 0.5, 0.4], ranking=[])[L.C_VERDICT] == L.V_GAP


# Keine Rankings geladen
def test_without_rankings_one_page_fits_clearly():
    assert _row([0.9, 0.7, 0.1])[L.C_VERDICT] == L.V_MATCH


def test_without_rankings_several_pages_fit_close_together_is_cannibalisation():
    assert _row([0.85, 0.84, 0.3])[L.C_VERDICT] == L.V_CANNIBAL


def test_without_rankings_nothing_fits_is_a_gap():
    assert _row([0.5, 0.4, 0.3])[L.C_VERDICT] == L.V_GAP


# --- Grenzen der Regeln ------------------------------------------------------------------------------------------


def test_close_means_fitting_and_at_most_the_margin_below_the_top():
    assert _row([0.85, 0.84, 0.1])[L.C_VERDICT] == L.V_CANNIBAL  # genau auf dem Rand
    assert _row([0.85, 0.8399, 0.1])[L.C_VERDICT] == L.V_MATCH
    assert _row([0.805, 0.799, 0.1])[L.C_VERDICT] == L.V_MATCH  # nah, aber die zweite erreicht die Schwelle nicht


def test_the_margin_is_adjustable():
    assert _row([0.85, 0.80, 0.1], margin=0.05)[L.C_VERDICT] == L.V_CANNIBAL
    assert _row([0.80, 0.80, 0.1], margin=0)[L.C_VERDICT] == L.V_CANNIBAL
    assert _row([0.842, 0.843, 0.1], ranking=[(U1, 3)], margin=0)[L.C_VERDICT] == L.V_OK


def test_ranking_url_exactly_at_the_margin_is_almost_as_good():
    assert _row([0.80, 0.81, 0.1], ranking=[(U1, 3)])[L.C_VERDICT] == L.V_OK


def test_exact_tie_between_ranking_and_best_url_is_ok_but_not_the_best():
    row = _row([0.8, 0.8, 0.1], ranking=[(U2, 3)])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_BEST_URL] == U1
    assert row[L.C_RANK_IS_BEST] == L.NO


def test_good_position_is_adjustable():
    assert _row([0.9, 0.7, 0.1], ranking=[(U1, 12)])[L.C_VERDICT] == L.V_MATCH
    assert _row([0.9, 0.7, 0.1], ranking=[(U1, 12)], good_position=15)[L.C_VERDICT] == L.V_OK


def test_the_rules_use_the_chosen_lead_score():
    result = make_result(["q"], [U1, U2], [[0.9, 0.8]], full_scores=[[0.2, 0.85]])
    df = build_decisions(result, result.lead("full"), 0.8)
    assert df[L.C_BEST_URL].tolist() == [U2]
    assert df[L.C_VERDICT].tolist() == [L.V_MATCH]
    assert df[L.C_LEAD_GAP].tolist() == [0.65]


def test_negative_margin_is_rejected():
    result = make_result(["q"], [U1], [[0.9]])
    with pytest.raises(ValueError):
        build_decisions(result, result.lead("chunk"), 0.6, margin=-0.01)


# --- Spalten der Übersicht ---------------------------------------------------------------------------------------


def test_overview_columns_in_order():
    result = make_result(["q"], [U1], [[0.8]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert OVERVIEW_COLUMNS == MATCH_COLUMNS + [L.C_POSITION, L.C_RANK_URL, L.C_RANK_IS_BEST, L.C_VERDICT, L.C_CANNIBAL]
    assert list(df.columns) == OVERVIEW_COLUMNS[:-1]  # die Spalte Kannibalisierungsgefahr setzt cannibal.annotate
    assert [L.C_POSITION, L.C_RANK_URL, L.C_RANK_IS_BEST] == ["Rankingposition", "Rankende URL", "Rankende URL = beste URL?"]


def test_overview_match_columns_come_from_the_three_best_urls():
    row = _row([0.7, 0.9, 0.8], ranking=[(U1, 4)])
    assert [row[L.C_BEST_URL], row[L.C_SECOND_URL], row[L.C_THIRD_URL]] == [U2, U3, U1]
    assert row[L.C_CHUNK] == f"Text {U2}"
    assert [row[L.C_S_CHUNK], row[L.C_S_CHUNK_2], row[L.C_S_CHUNK_3]] == [0.9, 0.8, 0.7]
    assert row[L.C_LEAD_GAP] == 0.1


def test_overview_with_fewer_than_three_urls_leaves_cells_empty():
    row = _row([0.9, 0.7], urls=(U1, U2))
    assert row[L.C_SECOND_URL] == U2 and row[L.C_THIRD_URL] == "" and pd.isna(row[L.C_S_COMBI_3])
    single = _row([0.9], urls=(U1,))
    assert single[L.C_SECOND_URL] == "" and pd.isna(single[L.C_LEAD_GAP])
    assert single[L.C_VERDICT] == L.V_MATCH


def test_ranking_columns_show_the_best_own_ranking_at_any_position():
    row = _row([0.1, 0.9, 0.7], ranking=[(U2, 48), (U1, 35)])
    assert row[L.C_POSITION] == "35"
    assert row[L.C_RANK_URL] == U1
    assert row[L.C_RANK_IS_BEST] == L.NO


def test_ranking_columns_are_empty_without_ranking():
    for ranking, comparison in ((None, ""), ([], L.CMP_NOT_RANKING)):
        row = _row([0.9, 0.7, 0.1], ranking=ranking)
        assert row[L.C_POSITION] == "" and row[L.C_RANK_URL] == ""
        assert row[L.C_RANK_IS_BEST] == comparison


def test_comparison_values():
    assert [L.YES, L.NO, L.CMP_NOT_RANKING, L.CMP_NOT_IN_EXPORT] == ["ja", "nein", "rankt nicht", "nicht im Frog-Export"]
    assert not hasattr(L, "CMP_CLOSE")  # Ja-Nein-Frage: knapp hinter der besten heißt "nein"
    # die Position spielt keine Rolle
    assert _row([0.795, 0.803, 0.1], ranking=[(U1, 40)])[L.C_RANK_IS_BEST] == L.NO
    assert _row([0.842, 0.848, 0.1], ranking=[(U1, 40)])[L.C_RANK_IS_BEST] == L.NO
    assert _row([0.9, 0.1, 0.1], ranking=[(U1, 40)])[L.C_RANK_IS_BEST] == L.YES


def test_fractional_positions_keep_one_decimal():
    assert _row([0.9, 0.1, 0.1], ranking=[(U1, 4.3)])[L.C_POSITION] == "4.3"


def test_query_matching_is_case_insensitive():
    rankings = _rankings([("hunde futter", U1, U1, 2.0)])
    result = make_result(["Hunde  Futter"], [U1], [[0.9]])
    assert build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0][L.C_VERDICT] == L.V_OK
