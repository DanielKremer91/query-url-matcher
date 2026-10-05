import numpy as np
import pytest

from qum import labels as L
from qum.match import estimate_chunks, ranks, run_matching, top_hits
from tests.conftest import FakeEmbedder, make_result

URLS = ["https://a.de/hund", "https://a.de/katze"]
CONTENTS = [
    "hundefutter getreidefrei ist gut " + "fuellwort " * 8 + "napf reinigen tipps",
    "katzenfutter nass sorten",
]


def test_run_matching_shapes_and_best_url():
    result = run_matching(["hundefutter getreidefrei", "katzenfutter"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert result.chunk_scores.shape == (2, 2)
    assert result.chunk_scores[0].argmax() == 0
    assert result.chunk_scores[1].argmax() == 1


def test_best_chunk_is_the_matching_passage():
    result = run_matching(["napf reinigen"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert "napf reinigen" in result.best_chunk(0, 0)


def test_full_method_single_chunk_is_fulltext():
    result = run_matching(["x"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert result.full_method == [L.CHUNK_MEAN, L.FULLTEXT]


def test_full_method_uses_fulltext_when_it_fits():
    embedder = FakeEmbedder(max_words=100)
    result = run_matching(["x"], URLS, CONTENTS, embedder, 5, 1)
    assert result.full_method == [L.FULLTEXT, L.FULLTEXT]
    passages = [texts for texts, role in embedder.calls if role == "passage"]
    assert passages[-1] == [" ".join(CONTENTS[0].split())]


def test_chunk_mean_is_normalised():
    embedder = FakeEmbedder()
    result = run_matching(["hundefutter getreidefrei"], URLS, CONTENTS, embedder, 5, 1)

    # Manually embed query and chunks to verify normalization
    q = embedder.embed(["hundefutter getreidefrei"], "query")[0]
    chunk_vecs = embedder.embed(result.chunks[0], "passage")
    mean = chunk_vecs.mean(axis=0)
    normalized_mean = mean / np.linalg.norm(mean)

    # Full score should equal query dot normalized mean
    expected = q @ normalized_mean
    assert np.isclose(result.full_scores[0, 0], expected, atol=1e-5)

    # Should be strictly greater than un-normalized
    un_normalized = q @ mean
    assert result.full_scores[0, 0] > un_normalized


def test_full_scores_rows_for_mixed_fulltext_and_chunk_mean():
    # Three URLs: 0 and 2 short enough for FULLTEXT, 1 too long for CHUNK_MEAN
    urls = [
        "https://a.de/url0",
        "https://a.de/url1",
        "https://a.de/url2",
    ]
    contents = [
        "alpha beta gamma delta epsilon zeta eta theta",  # 8 words, fits FULLTEXT
        "aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn",  # 13 words, needs CHUNK_MEAN
        "iota kappa lambda mu nu xi omicron pi",  # 8 different words, fits FULLTEXT
    ]

    embedder = FakeEmbedder(max_words=10)
    result = run_matching(["alpha"], urls, contents, embedder, 5, 1)

    # Check methods
    assert result.full_method == [L.FULLTEXT, L.CHUNK_MEAN, L.FULLTEXT]

    # Manually compute expected scores
    q = embedder.embed(["alpha"], "query")[0]

    # URL 0: FULLTEXT - embed full content directly
    full_0 = embedder.embed([" ".join(contents[0].split())], "passage")[0]
    expected_0 = q @ full_0
    assert np.isclose(result.full_scores[0, 0], expected_0, atol=1e-5)

    # URL 1: CHUNK_MEAN - mean of chunks, then normalized
    chunks_1 = result.chunks[1]
    chunk_vecs_1 = embedder.embed(chunks_1, "passage")
    mean_1 = chunk_vecs_1.mean(axis=0)
    normalized_mean_1 = mean_1 / np.linalg.norm(mean_1)
    expected_1 = q @ normalized_mean_1
    assert np.isclose(result.full_scores[0, 1], expected_1, atol=1e-5)

    # URL 2: FULLTEXT - embed full content directly
    full_2 = embedder.embed([" ".join(contents[2].split())], "passage")[0]
    expected_2 = q @ full_2
    assert np.isclose(result.full_scores[0, 2], expected_2, atol=1e-5)


def test_combined_and_lead():
    result = make_result(["q"], ["u1", "u2"], [[0.8, 0.2]], [[0.4, 0.6]])
    assert np.allclose(result.combined(0.7), [[0.68, 0.32]])
    assert np.allclose(result.lead("chunk"), [[0.8, 0.2]])
    assert np.allclose(result.lead("full"), [[0.4, 0.6]])
    assert np.allclose(result.lead("combined", 0.5), [[0.6, 0.4]])
    with pytest.raises(ValueError):
        result.lead("other")


def test_ranks():
    assert ranks(np.array([[0.1, 0.9, 0.5]])).tolist() == [[3, 1, 2]]


def test_top_hits_columns_order_and_ranks():
    result = make_result(["q"], ["u1", "u2", "u3"], [[0.5, 0.9, 0.7]], [[0.9, 0.1, 0.5]])
    df = top_hits(result, basis="chunk", weight=0.7, top_n=2)
    assert list(df.columns) == [
        L.C_QUERY, L.C_URL, L.C_CHUNK, L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI,
        L.C_R_CHUNK, L.C_R_FULL, L.C_R_COMBI, L.C_METHOD,
    ]
    assert df[L.C_URL].tolist() == ["u2", "u3"]
    assert df[L.C_R_CHUNK].tolist() == [1, 2]
    assert df[L.C_R_FULL].tolist() == [3, 2]
    assert df[L.C_S_CHUNK].tolist() == [0.9, 0.7]


def test_top_hits_sorted_by_selected_basis():
    result = make_result(["q"], ["u1", "u2"], [[0.5, 0.9]], [[0.9, 0.1]])
    assert top_hits(result, basis="full", top_n=1)[L.C_URL].tolist() == ["u1"]


def test_estimate_chunks():
    assert estimate_chunks(CONTENTS, 5, 1) == 5
