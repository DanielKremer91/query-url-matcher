import pytest

from qum.compare import CONFIGS, evaluate, spec_for
from tests.conftest import FakeEmbedder

URLS = ["https://a.de/hund", "https://a.de/katze", "https://a.de/vogel"]
CONTENTS = ["hundefutter getreidefrei trocken", "katzenfutter nass sorten", "vogelfutter koerner mischung"]


def test_evaluate_perfect_ranking():
    truth = [("hundefutter getreidefrei", URLS[0]), ("katzenfutter nass", URLS[1])]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert out == {"n": 2, "fehlend": 0, "hit1": 1.0, "hit3": 1.0, "mrr": 1.0}


def test_evaluate_wrong_expectation_lowers_scores():
    truth = [("hundefutter getreidefrei", URLS[1]), ("katzenfutter nass", URLS[1])]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert out["hit1"] == 0.5
    assert out["hit3"] == 1.0
    assert 0.5 < out["mrr"] < 1.0


def test_evaluate_counts_urls_missing_in_content():
    truth = [("hundefutter", "https://a.de/weg"), ("katzenfutter nass", URLS[1] + "/")]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert (out["n"], out["fehlend"]) == (1, 1)


def test_evaluate_without_usable_pairs_raises():
    with pytest.raises(ValueError, match="Wahrheitsliste"):
        evaluate([("x", "https://a.de/weg")], URLS, CONTENTS, FakeEmbedder(), 5, 1)


def test_configs_cover_the_five_variants():
    assert list(CONFIGS) == ["paraphrase", "e5", "e5-ohne-prefix", "openai", "openai-prefix"]


def test_e5_without_prefix_drops_the_trained_prefixes():
    trained, plain = spec_for("e5"), spec_for("e5-ohne-prefix")
    assert (trained.query_prefix, trained.passage_prefix) == ("query: ", "passage: ")
    assert (plain.query_prefix, plain.passage_prefix) == ("", "")
    assert plain.model_id == trained.model_id


def test_openai_prefix_variant_only_exists_here():
    plain, prefixed = spec_for("openai"), spec_for("openai-prefix")
    assert (plain.query_prefix, plain.passage_prefix) == ("", "")
    assert (prefixed.query_prefix, prefixed.passage_prefix) == ("query: ", "passage: ")
    assert prefixed.model_id == plain.model_id
    assert prefixed.key == "openai-prefix"
