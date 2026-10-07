import pandas as pd

from qum import labels as L
import pytest

from qum.verdict import ADVICE, ADVICE_NOT_IN_EXPORT, ADVICE_OK_CLOSE, ADVICE_RISK_PLAIN, build_decisions
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"


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
        L.C_LEAD_GAP, L.C_RANK_URL, L.C_POSITION, L.C_NOTE, L.C_ADVICE,
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
    assert df[L.C_VERDICT].tolist() == [L.V_OK, L.V_CANNIBAL, L.V_WATCH, L.V_USE, L.V_GAP]
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
    assert row[L.C_VERDICT] == L.V_CANNIBAL
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
    for verdict in [L.V_MATCH, L.V_GAP, L.V_OK, L.V_CANNIBAL, L.V_WATCH, L.V_USE, L.V_CHECK]:
        assert verdict in ADVICE


def test_exact_tie_between_ranking_and_best_url_is_ok():
    rankings = _rankings([("q", U2, U2, 3.0)])
    result = make_result(["q"], [U1, U2], [[0.8, 0.8]])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_BEST_URL] == U2  # die rankende Seite gehört zu den besten Treffern
    assert row[L.C_VERDICT] == L.V_OK


def _ranking_u1(scores, **kwargs):
    rankings = _rankings([("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2], [scores], full_scores=[[s - 0.1 for s in scores]])
    return build_decisions(result, result.lead("chunk"), 0.6, rankings, **kwargs).iloc[0]


def test_ranking_url_just_behind_the_top_is_ok_and_named_as_best():
    row = _ranking_u1([0.842, 0.843])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_BEST_URL] == U1
    assert row[L.C_CHUNK] == f"Text {U1}"
    assert row[L.C_S_CHUNK] == 0.842
    assert row[L.C_S_FULL] == 0.742
    assert row[L.C_S_COMBI] == round(0.7 * 0.842 + 0.3 * 0.742, 4)


def test_ranking_url_clearly_behind_the_top_is_risk():
    row = _ranking_u1([0.790, 0.860])
    assert row[L.C_VERDICT] == L.V_CANNIBAL
    assert row[L.C_BEST_URL] == U2
    assert row[L.C_S_CHUNK] == 0.86


def test_ranking_url_exactly_at_the_margin_is_ok():
    assert _ranking_u1([0.80, 0.81])[L.C_VERDICT] == L.V_OK
    assert _ranking_u1([0.80, 0.8101])[L.C_VERDICT] == L.V_CANNIBAL


def test_verdict_margin_is_adjustable():
    assert _ranking_u1([0.790, 0.860], margin=0.1)[L.C_VERDICT] == L.V_OK
    assert _ranking_u1([0.842, 0.843], margin=0)[L.C_VERDICT] == L.V_CANNIBAL


def test_nothing_fits_is_checked_before_the_margin():
    row = _ranking_u1([0.55, 0.555])
    assert row[L.C_VERDICT] == L.V_WATCH
    assert row[L.C_BEST_URL] == U2


def test_ranking_url_missing_in_export_gets_its_own_advice():
    alt = "https://a.de/alt"
    rankings = _rankings([("q", alt, alt, 2.0)])
    result = make_result(["q"], [U1], [[0.9]])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_VERDICT] == L.V_CANNIBAL
    assert row[L.C_ADVICE] == ADVICE_NOT_IN_EXPORT.format(best=U1, rank_url=alt, position="2")
    assert "besser" not in row[L.C_ADVICE]
    assert "nicht verglichen" in row[L.C_ADVICE]


def test_advice_texts_do_not_overclaim_serp_similarity():
    assert all("fast gleicher SERP" not in text for text in ADVICE.values())
    assert "stark überlappender SERP" in ADVICE[L.V_CHECK]


def test_advice_reads_as_a_hint_to_check():
    assert ADVICE == {
        L.V_MATCH: (
            "Es gibt bereits eine passende Seite: {best}. Prüfen, ob sie die Query abdeckt oder ausgebaut werden kann, "
            "bevor eine neue Seite entsteht."
        ),
        L.V_GAP: "Keine Seite erreicht die Schwelle. Kandidat für eine neue Seite, nach Prüfung der besten Treffer.",
        L.V_OK: "Die rankende Seite gehört semantisch zu den besten Treffern. Kein Hinweis auf Handlungsbedarf.",
        L.V_CANNIBAL: (
            "{rank_url} rankt auf Position {position}, semantisch passt {best} deutlich besser. "
            "Prüfen, welche Seite die Query bedienen soll."
        ),
        L.V_WATCH: (
            "{rank_url} rankt auf Position {position}, obwohl keine Seite die Schwelle erreicht. "
            "Beobachten und die Schwelle prüfen."
        ),
        L.V_USE: (
            "{best} passt semantisch zur Query, rankt aber nicht gut. Prüfen, ob diese Seite ausgebaut und intern "
            "gestärkt werden kann, statt eine neue zu bauen."
        ),
        L.V_CHECK: (
            "Vor Neuerstellung prüfen: {best} bedient bereits ein Keyword mit stark überlappender SERP. "
            "Lässt sich die Seite erweitern?"
        ),
    }
    assert ADVICE_NOT_IN_EXPORT.endswith("Prüfen, ob die rankende Seite im Export fehlt, bevor daraus Schlüsse gezogen werden.")



def _ok_row(scores, threshold=0.8, margin=0.01):
    rankings = _rankings([("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2, U3], [scores])
    return build_decisions(result, result.lead("chunk"), threshold, rankings, margin=margin).iloc[0]


def test_ok_names_another_fitting_url_that_is_almost_as_good():
    row = _ok_row([0.842, 0.848, 0.1])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_ADVICE] == ADVICE_OK_CLOSE.format(other=U2)
    assert ADVICE_OK_CLOSE == (
        "Die rankende Seite gehört semantisch zu den besten Treffern. {other} passt fast gleich gut, siehe Blatt Kannibalisierungsgefahr."
    )


def test_ok_close_looks_on_both_sides_and_names_the_best_other_url():
    assert _ok_row([0.85, 0.845, 0.1])[L.C_ADVICE] == ADVICE_OK_CLOSE.format(other=U2)
    assert _ok_row([0.85, 0.845, 0.848])[L.C_ADVICE] == ADVICE_OK_CLOSE.format(other=U3)


def test_ok_keeps_the_plain_advice_when_no_other_url_is_close_and_fitting():
    assert _ok_row([0.85, 0.83, 0.1])[L.C_ADVICE] == ADVICE[L.V_OK]
    # nah dran, aber unter der Schwelle
    assert _ok_row([0.842, 0.835, 0.1], threshold=0.84)[L.C_ADVICE] == ADVICE[L.V_OK]


def test_ranking_url_within_margin_but_below_threshold_is_risk():
    row = _ok_row([0.795, 0.803, 0.1], threshold=0.80)
    assert row[L.C_VERDICT] == L.V_CANNIBAL
    assert row[L.C_BEST_URL] == U2
    assert row[L.C_ADVICE] == ADVICE_RISK_PLAIN.format(rank_url=U1, position="3", best=U2)


def test_risk_advice_says_clearly_better_only_beyond_a_positive_margin():
    assert "deutlich besser" in _ranking_u1([0.790, 0.860])[L.C_ADVICE]
    plain = _ranking_u1([0.842, 0.843], margin=0)
    assert plain[L.C_VERDICT] == L.V_CANNIBAL
    assert plain[L.C_ADVICE] == ADVICE_RISK_PLAIN.format(rank_url=U1, position="3", best=U2)
    assert ADVICE_RISK_PLAIN == (
        "{rank_url} rankt auf Position {position}, semantisch passt {best} besser. "
        "Prüfen, welche Seite die Query bedienen soll."
    )


def test_negative_margin_is_rejected():
    result = make_result(["q"], [U1], [[0.9]])
    with pytest.raises(ValueError):
        build_decisions(result, result.lead("chunk"), 0.6, margin=-0.01)


def test_lead_gap_for_a_clear_winner():
    result = make_result(["q"], [U1, U2], [[0.9, 0.6]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert df[L.C_LEAD_GAP].tolist() == [0.3]


def test_lead_gap_is_negative_when_the_shown_ranking_url_is_slightly_behind():
    row = _ok_row([0.842, 0.848, 0.1])
    assert row[L.C_VERDICT] == L.V_OK
    assert row[L.C_BEST_URL] == U1
    assert row[L.C_LEAD_GAP] == -0.006


def test_lead_gap_is_empty_with_a_single_url():
    result = make_result(["q"], [U1], [[0.8]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert df[L.C_LEAD_GAP].tolist() == [None]


def test_lead_gap_uses_the_chosen_lead_not_the_chunk_score():
    result = make_result(["q"], [U1, U2], [[0.9, 0.8]], full_scores=[[0.2, 0.6]])
    df = build_decisions(result, result.lead("full"), 0.5)
    assert df[L.C_BEST_URL].tolist() == [U2]
    assert df[L.C_LEAD_GAP].tolist() == [0.4]
