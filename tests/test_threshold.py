import numpy as np
import pandas as pd

from qum import labels as L
from qum.threshold import MIN_PAIRS, calibrated_threshold, calibration_scores, examples_around, median_threshold
from tests.conftest import make_result


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def test_median_of_best_scores():
    result = make_result(["q1", "q2", "q3"], ["u1", "u2"], [[0.9, 0.1], [0.5, 0.2], [0.3, 0.7]])
    proposal = median_threshold(result, result.lead("chunk"))
    assert proposal.source == "median"
    assert np.isclose(proposal.value, 0.7)


def test_no_calibration_without_rankings():
    result = make_result(["q1"], ["u1"], [[0.9]])
    assert calibrated_threshold(result, result.lead("chunk"), None) is None
    assert calibration_scores(result, result.lead("chunk"), None) == []
    assert MIN_PAIRS == 20


def test_rankings_give_25th_percentile_of_top_pairs():
    queries = [f"q{i}" for i in range(4)]
    result = make_result(queries, ["https://a.de/1"], [[0.4], [0.6], [0.8], [1.0]])
    rankings = _rankings([(q, "https://a.de/1", "https://a.de/1", 3.0) for q in queries])
    proposal = calibrated_threshold(result, result.lead("chunk"), rankings, max_position=5, min_pairs=4)
    assert proposal.source == "rankings"
    assert proposal.n_pairs == 4
    assert np.isclose(proposal.value, 0.55)


def test_calibration_does_not_depend_on_the_median():
    queries = [f"q{i}" for i in range(4)]
    result = make_result(queries, ["https://a.de/1"], [[0.4], [0.6], [0.8], [1.0]])
    rankings = _rankings([(q, "https://a.de/1", "https://a.de/1", 3.0) for q in queries])
    lead = result.lead("chunk")
    assert calibrated_threshold(result, lead, rankings, min_pairs=4).value == 0.55
    assert median_threshold(result, lead).value == 0.7


def test_rankings_ignore_bad_positions_and_unknown_urls():
    result = make_result(["q0", "q1"], ["https://a.de/1"], [[0.4], [0.6]])
    rankings = _rankings(
        [
            ("q0", "https://a.de/1", "https://a.de/1", 9.0),
            ("q1", "https://a.de/x", "https://a.de/x", 1.0),
        ]
    )
    assert calibration_scores(result, result.lead("chunk"), rankings) == []
    assert calibrated_threshold(result, result.lead("chunk"), rankings, min_pairs=1) is None


def test_too_few_pairs_give_no_calibration_but_keep_the_count():
    result = make_result(["q0"], ["https://a.de/1"], [[0.4]])
    rankings = _rankings([("q0", "https://a.de/1", "https://a.de/1", 1.0)])
    assert calibrated_threshold(result, result.lead("chunk"), rankings) is None
    assert len(calibration_scores(result, result.lead("chunk"), rankings)) == 1


def test_examples_around_threshold():
    result = make_result(["q1", "q2", "q3", "q4"], ["u1"], [[0.9], [0.62], [0.58], [0.1]])
    df = examples_around(result, result.lead("chunk"), 0.6, n=1)
    assert list(df.columns) == [L.C_SIDE, L.C_QUERY, L.C_URL, L.C_CHUNK, L.C_SCORE]
    assert df[L.C_QUERY].tolist() == ["q2", "q3"]
    assert df[L.C_SIDE].tolist() == [L.SIDE_ABOVE, L.SIDE_BELOW]
    assert L.SIDE_ABOVE == "knapp über der Schwelle" and L.SIDE_BELOW == "knapp unter der Schwelle"


def test_proposal_is_rounded_once_to_four_decimals():
    result = make_result(["q1", "q2", "q3"], ["u1"], [[0.812345], [0.7], [0.9]])
    assert median_threshold(result, result.lead("chunk")).value == 0.8123
    rankings = _rankings([(q, "u1", "u1", 1.0) for q in ["q1", "q2", "q3"]])
    assert calibrated_threshold(result, result.lead("chunk"), rankings, min_pairs=3).value == 0.7562


def test_examples_split_on_unrounded_score():
    # 0.59996 wird als 0.6 angezeigt, liegt aber unter der Schwelle 0.6
    result = make_result(["drunter", "drueber"], ["u1"], [[0.59996], [0.61]])
    df = examples_around(result, result.lead("chunk"), 0.6, n=1)
    assert df[L.C_QUERY].tolist() == ["drueber", "drunter"]
    assert df[L.C_SIDE].tolist() == [L.SIDE_ABOVE, L.SIDE_BELOW]
    assert df[L.C_SCORE].tolist() == [0.61, 0.6]


def test_examples_sort_on_unrounded_score():
    result = make_result(["a", "b"], ["u1"], [[0.70004], [0.70001]])
    df = examples_around(result, result.lead("chunk"), 0.6, n=1)
    assert df[L.C_QUERY].tolist() == ["b"]


def test_duplicate_ranking_rows_count_once_with_best_position():
    queries = [f"q{i}" for i in range(3)]
    result = make_result(queries, ["https://a.de/1"], [[0.4], [0.6], [0.8]])
    u = "https://a.de/1"
    rankings = _rankings(
        [("q0", u, u, 3.0), ("q0", u, u, 4.0), ("q1", u, u, 9.0), ("q1", u, u, 2.0), ("q2", u, u, 1.0)]
    )
    assert len(calibration_scores(result, result.lead("chunk"), rankings, max_position=5)) == 3
    proposal = calibrated_threshold(result, result.lead("chunk"), rankings, max_position=5, min_pairs=3)
    assert proposal.n_pairs == 3
    assert proposal.source == "rankings"
    assert proposal.value == 0.5


def test_examples_put_score_on_the_same_side_as_the_verdict():
    from qum.verdict import build_decisions

    # float32(0.5002) liegt knapp unter 0.5002 als float64; das Urteil vergleicht im Typ der Scores
    result = make_result(["q"], ["u1"], [[0.5002]])
    lead = result.lead("chunk")
    fits = build_decisions(result, lead, 0.5002).iloc[0][L.C_VERDICT] == L.V_MATCH
    side = examples_around(result, lead, 0.5002).iloc[0][L.C_SIDE]
    assert side == (L.SIDE_ABOVE if fits else L.SIDE_BELOW)
