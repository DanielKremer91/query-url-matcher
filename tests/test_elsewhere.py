import pandas as pd

from qum import labels as L
from qum.elsewhere import COLUMNS, chunk_elsewhere
from qum.normalize import normalize_url
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"
ALT = "https://a.de/alt"
# Chunk-Scores und Gesamt-URL-Scores je Query: "anders" hat den besten Chunk auf U1, die beste Seite insgesamt ist U2
QUERIES = ["anders", "gleich", "knapp-anders"]
CHUNK = [[0.90, 0.80, 0.1], [0.90, 0.80, 0.1], [0.70, 0.69, 0.1]]
FULL = [[0.60, 0.85, 0.1], [0.88, 0.70, 0.1], [0.50, 0.55, 0.1]]


def _rankings(rows):
    return pd.DataFrame([(q, u, normalize_url(u), float(p)) for q, u, p in rows],
                        columns=["query_norm", "url", "url_norm", "position"])


def _run(rankings=None, chunk=CHUNK, full=FULL):
    return chunk_elsewhere(make_result(QUERIES, [U1, U2, U3], chunk, full), rankings)


def test_columns():
    assert list(_run().columns) == COLUMNS == [
        L.C_QUERY, L.C_CHUNK_URL, L.C_BEST_CHUNK, L.C_SECTION, L.C_S_BEST_CHUNK, L.C_S_FULL_CHUNK_URL,
        L.C_OVERALL_URL, L.C_S_FULL_OVERALL, L.C_POSITION, L.C_RANK_URL, L.C_RANK_IS,
    ]
    assert [L.C_CHUNK_URL, L.C_OVERALL_URL, L.C_RANK_IS] == ["Seite mit bestem Chunk", "Beste Seite insgesamt", "Rankende URL ist"]


def test_lists_queries_whose_best_chunk_is_not_on_the_best_page_overall_strongest_chunk_first():
    df = _run()
    assert df[L.C_QUERY].tolist() == ["anders", "knapp-anders"]
    row = df.iloc[0]
    assert (row[L.C_CHUNK_URL], row[L.C_OVERALL_URL]) == (U1, U2)
    assert (row[L.C_S_BEST_CHUNK], row[L.C_S_FULL_CHUNK_URL], row[L.C_S_FULL_OVERALL]) == (0.9, 0.6, 0.85)
    assert row[L.C_BEST_CHUNK] == f"Text {U1}"


def test_ignores_the_threshold_and_low_scores():
    # "knapp-anders" passt nirgends gut, steht aber trotzdem drin: die Liste hängt an keiner Schwelle
    assert "knapp-anders" in _run()[L.C_QUERY].tolist()


def test_ranking_columns_and_which_page_ranks():
    rankings = _rankings([("anders", U1, 4), ("knapp-anders", ALT, 7)])
    df = _run(rankings)
    assert df[L.C_POSITION].tolist() == ["4", "7"]
    assert df[L.C_RANK_URL].tolist() == [U1, ALT]
    assert df[L.C_RANK_IS].tolist() == [L.RANK_IS_CHUNK, L.CMP_NOT_IN_EXPORT]
    assert _run(_rankings([("anders", U2, 4)])).iloc[0][L.C_RANK_IS] == L.RANK_IS_OVERALL
    assert _run(_rankings([("anders", U3, 4)])).iloc[0][L.C_RANK_IS] == L.RANK_IS_OTHER
    assert _run(_rankings([("gleich", U1, 1)])).iloc[0][L.C_RANK_IS] == L.CMP_NOT_RANKING


def test_ranking_columns_are_empty_without_rankings():
    df = _run()
    assert set(df[L.C_POSITION]) == {""} and set(df[L.C_RANK_URL]) == {""} and set(df[L.C_RANK_IS]) == {""}


def test_values():
    assert [L.RANK_IS_CHUNK, L.RANK_IS_OVERALL, L.RANK_IS_OTHER] == [
        "die Seite mit bestem Chunk", "die beste Seite insgesamt", "eine andere Seite",
    ]


def test_empty_when_chunk_and_page_agree():
    assert _run(full=CHUNK).empty
