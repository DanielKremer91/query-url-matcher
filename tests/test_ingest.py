import io

import pandas as pd
import pytest

from qum import ingest


def test_read_table_semicolon_csv():
    df = ingest.read_table("Address;Main Content\nhttps://a.de/1;Text eins\n".encode("utf-8"), "x.csv")
    assert list(df.columns) == ["Address", "Main Content"]
    assert df.iloc[0]["Main Content"] == "Text eins"


def test_read_table_utf16_tab_export():
    raw = "Keyword\tURL\tPosition\nfutter\thttps://a.de/1\t3\n".encode("utf-16")
    df = ingest.read_table(raw, "ahrefs.csv")
    assert list(df.columns) == ["Keyword", "URL", "Position"]


def test_read_table_single_column_keeps_commas():
    df = ingest.read_table("Query\nwas kostet futter, nass\n".encode("utf-8"), "q.csv")
    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == "was kostet futter, nass"


def test_read_table_excel():
    buf = io.BytesIO()
    pd.DataFrame({"url": ["https://a.de/1"], "text": ["Hallo"]}).to_excel(buf, index=False)
    df = ingest.read_table(buf.getvalue(), "x.xlsx")
    assert list(df.columns) == ["url", "text"]


def test_read_table_unknown_format():
    with pytest.raises(ingest.IngestError, match="Format"):
        ingest.read_table(b"x", "x.pdf")


def test_load_queries_without_header_keeps_first_line():
    df = ingest.read_table("hundefutter getreidefrei\nkatzenfutter\n".encode("utf-8"), "q.csv")
    assert ingest.load_queries(df) == ["hundefutter getreidefrei", "katzenfutter"]


def test_load_queries_with_header_drops_empty_and_duplicates():
    df = pd.DataFrame({"Keyword": ["Futter", " futter ", None, "", "Napf"]})
    assert ingest.load_queries(df) == ["Futter", "Napf"]


def test_load_content_keeps_rows_aligned_and_reports_skips():
    df = pd.DataFrame(
        {
            "Address": ["https://a.de/1", "https://a.de/2", "https://a.de/3", "https://a.de/3/"],
            "Extract Main Content 1": ["Text eins", None, "Text drei", "Doppelt"],
        }
    )
    table = ingest.load_content(df)
    assert table.urls == ["https://a.de/1", "https://a.de/3"]
    assert table.contents == ["Text eins", "Text drei"]
    assert table.skipped_empty == 1
    assert table.skipped_duplicate == 1


def test_load_content_missing_column_names_available_columns():
    with pytest.raises(ingest.IngestError, match="Gefundene Spalten"):
        ingest.load_content(pd.DataFrame({"foo": ["x"], "bar": ["y"]}))


def test_load_content_manual_columns():
    df = pd.DataFrame({"foo": ["https://a.de/1"], "bar": ["Text"]})
    table = ingest.load_content(df, url_col="foo", content_col="bar")
    assert table.urls == ["https://a.de/1"]


def test_load_rankings_normalises_and_drops_bad_rows():
    df = pd.DataFrame(
        {
            "Keyword": ["Futter ", "napf", "leer"],
            "Current URL": ["https://a.de/1/", "https://a.de/2", None],
            "Current position": ["3", "x", "5"],
        }
    )
    out = ingest.load_rankings(df)
    assert out.to_dict("records") == [
        {"query_norm": "futter", "url": "https://a.de/1/", "url_norm": "https://a.de/1", "position": 3.0}
    ]


def test_load_serps_keeps_only_organic():
    df = pd.DataFrame(
        {
            "Keyword": ["futter", "futter"],
            "URL": ["https://a.de/1", "https://b.de/ad"],
            "Position": [1, 2],
            "Type": ["Organic", "Paid top"],
        }
    )
    out = ingest.load_serps(df)
    assert out["url"].tolist() == ["https://a.de/1"]
    assert out["keyword"].tolist() == ["futter"]


def test_load_serps_without_type_column_uses_all_rows():
    df = pd.DataFrame({"Keyword": ["futter"], "URL": ["https://a.de/1"], "Position": [1]})
    assert len(ingest.load_serps(df)) == 1


def test_own_rankings_from_serps_filters_by_host():
    df = pd.DataFrame(
        {
            "Keyword": ["futter", "futter"],
            "URL": ["https://a.de/1", "https://b.de/x"],
            "Position": [4, 1],
        }
    )
    own = ingest.own_rankings_from_serps(ingest.load_serps(df), {"a.de"})
    assert own.to_dict("records") == [
        {"query_norm": "futter", "url": "https://a.de/1", "url_norm": "https://a.de/1", "position": 4.0}
    ]
