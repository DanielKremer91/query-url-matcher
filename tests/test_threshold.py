import numpy as np
import pandas as pd

from qum import labels as L
from qum.threshold import examples_around, propose_threshold
from tests.conftest import make_result


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def test_median_of_best_scores_without_rankings():
    result = make_result(["q1", "q2", "q3"], ["u1", "u2"], [[0.9, 0.1], [0.5, 0.2], [0.3, 0.7]])
    proposal = propose_threshold(result, result.lead("chunk"))
    assert proposal.source == "median"
    assert np.isclose(proposal.value, 0.7)


def test_rankings_give_25th_percentile_of_top_pairs():
    queries = [f"q{i}" for i in range(4)]
    result = make_result(queries, ["https://a.de/1"], [[0.4], [0.6], [0.8], [1.0]])
    rankings = _rankings([(q, "https://a.de/1", "https://a.de/1", 3.0) for q in queries])
    proposal = propose_threshold(result, result.lead("chunk"), rankings, max_position=5, min_pairs=4)
    assert proposal.source == "rankings"
    assert proposal.n_pairs == 4
    assert np.isclose(proposal.value, 0.55)


def test_rankings_ignore_bad_positions_and_unknown_urls():
    result = make_result(["q0", "q1"], ["https://a.de/1"], [[0.4], [0.6]])
    rankings = _rankings(
        [
            ("q0", "https://a.de/1", "https://a.de/1", 9.0),
            ("q1", "https://a.de/x", "https://a.de/x", 1.0),
        ]
    )
    proposal = propose_threshold(result, result.lead("chunk"), rankings, min_pairs=1)
    assert proposal.source == "median"
    assert proposal.n_pairs == 0


def test_too_few_pairs_fall_back_to_median():
    result = make_result(["q0"], ["https://a.de/1"], [[0.4]])
    rankings = _rankings([("q0", "https://a.de/1", "https://a.de/1", 1.0)])
    proposal = propose_threshold(result, result.lead("chunk"), rankings)
    assert proposal.source == "median"
    assert proposal.n_pairs == 1


def test_examples_around_threshold():
    result = make_result(["q1", "q2", "q3", "q4"], ["u1"], [[0.9], [0.62], [0.58], [0.1]])
    df = examples_around(result, result.lead("chunk"), 0.6, n=1)
    assert list(df.columns) == [L.C_SIDE, L.C_QUERY, L.C_URL, L.C_CHUNK, "Score"]
    assert df[L.C_QUERY].tolist() == ["q2", "q3"]
    assert df[L.C_SIDE].tolist() == ["knapp über der Schwelle", "knapp unter der Schwelle"]
