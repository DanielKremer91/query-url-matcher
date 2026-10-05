import pandas as pd
import pytest

from qum import labels as L
from qum.ingest import IngestError
from qum.pairs import score_pairs, stale_columns
from tests.conftest import FakeEmbedder

URLS = ["https://a.de/hund", "https://a.de/katze"]
CONTENTS = ["hundefutter getreidefrei trocken", "katzenfutter nass sorten"]


def _run(df, **kwargs):
    return score_pairs(df, URLS, CONTENTS, FakeEmbedder(), 5, 1, **kwargs)


def test_scores_rank_and_best_url_keep_extra_columns():
    df = pd.DataFrame(
        {
            "Keyword": ["hundefutter getreidefrei", "katzenfutter nass"],
            "Current URL": ["https://a.de/katze/", "https://a.de/katze"],
            "Volume": [900, 400],
        }
    )
    out = _run(df)
    assert out["Volume"].tolist() == [900, 400]
    assert out[L.C_PAIR_RANK].tolist() == [2, 1]
    assert out[L.C_BEST_URL].tolist() == URLS
    assert out[L.C_S_CHUNK].iloc[1] > out[L.C_S_CHUNK].iloc[0]
    assert out[L.C_NOTE].tolist() == ["", ""]


def test_unknown_url_is_reported_not_dropped():
    df = pd.DataFrame({"Keyword": ["hundefutter"], "URL": ["https://a.de/weg"]})
    row = _run(df).iloc[0]
    assert row[L.C_NOTE] == L.NOTE_URL_MISSING
    assert pd.isna(row[L.C_S_CHUNK])
    assert row[L.C_BEST_URL] == URLS[0]


def test_rows_without_keyword_or_url_are_dropped():
    df = pd.DataFrame({"Keyword": ["hundefutter", None], "URL": [None, "https://a.de/hund"]})
    assert _run(df).empty


def test_missing_columns_raise():
    with pytest.raises(IngestError, match="Keyword"):
        _run(pd.DataFrame({"foo": ["x"], "URL": ["https://a.de/hund"]}))


def test_given_column_that_does_not_exist_lists_found_columns():
    df = pd.DataFrame({"Keyword": ["a"], "URL": ["https://a.de/hund"]})
    with pytest.raises(IngestError, match=r"Spalte 'Suchbegriff' gibt es nicht.*Keyword.*URL"):
        _run(df, keyword_col="Suchbegriff")
    with pytest.raises(IngestError, match=r"Spalte 'Seite' gibt es nicht"):
        _run(df, url_col="Seite")


def test_given_columns_are_used():
    df = pd.DataFrame({"Begriff": ["hundefutter getreidefrei"], "Ziel": ["https://a.de/hund"]})
    out = _run(df, keyword_col="Begriff", url_col="Ziel")
    assert out[L.C_PAIR_RANK].tolist() == [1]


def test_each_keyword_is_embedded_once():
    embedder = FakeEmbedder()
    df = pd.DataFrame({"Keyword": ["hundefutter", "hundefutter"], "URL": URLS})
    score_pairs(df, URLS, CONTENTS, embedder, 5, 1)
    queries = [texts for texts, role in embedder.calls if role == "query"]
    assert queries == [["hundefutter"]]


def test_output_columns_in_the_upload_are_replaced():
    df = pd.DataFrame(
        {
            "Keyword": ["hundefutter getreidefrei"],
            "URL": ["https://a.de/hund"],
            L.C_NOTE: ["alt"],
            L.C_S_CHUNK: [0.1],
            "Volume": [900],
        }
    )
    out = _run(df)
    assert list(out.columns).count(L.C_NOTE) == 1 and list(out.columns).count(L.C_S_CHUNK) == 1
    assert out[L.C_NOTE].tolist() == [""]
    assert out[L.C_S_CHUNK].iloc[0] > 0.1
    assert out["Volume"].tolist() == [900]
    assert stale_columns(df) == [L.C_S_CHUNK, L.C_NOTE]


def test_given_column_with_an_output_name_is_kept():
    df = pd.DataFrame({"Keyword": ["hundefutter"], L.C_BEST_URL: ["https://a.de/hund"]})
    assert stale_columns(df, keep=("Keyword", L.C_BEST_URL)) == []
    out = _run(df, url_col=L.C_BEST_URL)
    assert out[L.C_PAIR_RANK].tolist() == [1]
