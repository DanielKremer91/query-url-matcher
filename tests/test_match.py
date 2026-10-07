import numpy as np
import pandas as pd
import pytest

from qum import labels as L
from qum.match import estimate_chunks, ranks, run_matching
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


def test_estimate_chunks():
    assert estimate_chunks(CONTENTS, 5, 1) == 5


def test_run_matching_labels_each_embedding_pass(capsys):
    from qum.embeddings.cache import CachedEmbedder

    run_matching(["hundefutter getreidefrei"], URLS, CONTENTS, CachedEmbedder(FakeEmbedder(max_words=20)), 5, 1)
    out = capsys.readouterr().out
    assert "✅ Queries: 1 von 1 eingebettet" in out
    assert "✅ Chunks: 5 von 5 eingebettet" in out
    assert "✅ Ganze Seiten: 1 von 1 eingebettet" in out


def test_best_matches_columns_in_overview_order():
    from qum.match import MATCH_COLUMNS, best_matches

    result = make_result(["q"], ["u1", "u2", "u3"], [[0.5, 0.9, 0.7]])
    df = best_matches(result, result.lead("chunk"))
    assert list(df.columns) == MATCH_COLUMNS == [
        L.C_QUERY, L.C_BEST_URL, L.C_CHUNK, L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI, L.C_LEAD_GAP,
        L.C_SECOND_URL, L.C_S_CHUNK_2, L.C_S_FULL_2, L.C_S_COMBI_2,
        L.C_THIRD_URL, L.C_S_CHUNK_3, L.C_S_FULL_3, L.C_S_COMBI_3,
    ]
    assert [L.C_SECOND_URL, L.C_S_CHUNK_2, L.C_THIRD_URL, L.C_S_COMBI_3] == [
        "Zweitbeste URL", "Score Chunk 2", "Drittbeste URL", "Score Kombi 3",
    ]


def test_best_matches_orders_the_three_best_urls_by_the_lead_score():
    from qum.match import best_matches

    result = make_result(["q"], ["u1", "u2", "u3", "u4"], [[0.5, 0.9, 0.7, 0.1]], [[0.9, 0.1, 0.5, 0.2]])
    row = best_matches(result, result.lead("chunk"), weight=0.5).iloc[0]
    assert [row[L.C_BEST_URL], row[L.C_SECOND_URL], row[L.C_THIRD_URL]] == ["u2", "u3", "u1"]
    assert row[L.C_CHUNK] == "Text u2"
    assert [row[L.C_S_CHUNK], row[L.C_S_FULL], row[L.C_S_COMBI]] == [0.9, 0.1, 0.5]
    assert [row[L.C_S_CHUNK_2], row[L.C_S_FULL_2], row[L.C_S_COMBI_2]] == [0.7, 0.5, 0.6]
    assert [row[L.C_S_CHUNK_3], row[L.C_S_FULL_3], row[L.C_S_COMBI_3]] == [0.5, 0.9, 0.7]
    assert row[L.C_LEAD_GAP] == 0.2
    full = best_matches(result, result.lead("full"), weight=0.5).iloc[0]
    assert [full[L.C_BEST_URL], full[L.C_SECOND_URL], full[L.C_THIRD_URL]] == ["u1", "u3", "u4"]
    assert full[L.C_CHUNK] == "Text u1"
    assert full[L.C_LEAD_GAP] == 0.4


def test_best_matches_leaves_cells_empty_with_fewer_than_three_urls():
    from qum.match import best_matches

    two = best_matches(make_result(["q"], ["u1", "u2"], [[0.5, 0.9]]), np.array([[0.5, 0.9]])).iloc[0]
    assert two[L.C_SECOND_URL] == "u1" and two[L.C_THIRD_URL] == ""
    assert all(pd.isna(two[c]) for c in (L.C_S_CHUNK_3, L.C_S_FULL_3, L.C_S_COMBI_3))
    one = best_matches(make_result(["q"], ["u1"], [[0.5]]), np.array([[0.5]])).iloc[0]
    assert one[L.C_SECOND_URL] == "" and pd.isna(one[L.C_LEAD_GAP]) and pd.isna(one[L.C_S_CHUNK_2])


def test_best_matches_lead_gap_is_rounded_per_query():
    from qum.match import best_matches

    result = make_result(["a", "b"], ["u1", "u2"], [[0.91234, 0.6], [0.3, 0.2]])
    assert best_matches(result, result.lead("chunk"))[L.C_LEAD_GAP].tolist() == [0.3123, 0.1]


@pytest.mark.parametrize("basis, first, second", [("chunk", 0.9, 0.7), ("full", 0.9, 0.5), ("combined", 0.66, 0.62)])
def test_preview_shows_the_lead_score_of_the_best_and_second_best_url(basis, first, second):
    from qum.match import best_matches, preview

    result = make_result(["q"], ["u1", "u2", "u3"], [[0.5, 0.9, 0.7]], [[0.9, 0.1, 0.5]])
    df = preview(best_matches(result, result.lead(basis, 0.6), 0.6), basis)
    assert list(df.columns) == [L.C_QUERY, L.C_BEST_URL, L.C_SCORE, L.C_SECOND_URL, L.C_SCORE_2]
    assert [df.iloc[0][L.C_SCORE], df.iloc[0][L.C_SCORE_2]] == [first, second]


def test_top_hits_are_gone():
    import qum.match

    assert not hasattr(qum.match, "top_hits")
    for name in ("C_R_CHUNK", "C_R_FULL", "C_R_COMBI", "C_METHOD", "C_GAP_TO_BEST"):
        assert not hasattr(L, name), name
