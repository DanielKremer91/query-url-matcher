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


def test_load_rankings_drops_blank_keywords():
    df = pd.DataFrame(
        {
            "Keyword": [None, "", "  ", "futter"],
            "URL": ["https://a.de/1", "https://a.de/2", "https://a.de/3", "https://a.de/4"],
            "Position": [1, 2, 3, 4],
        }
    )
    out = ingest.load_rankings(df)
    assert out.to_dict("records") == [
        {"query_norm": "futter", "url": "https://a.de/4", "url_norm": "https://a.de/4", "position": 4.0}
    ]


def test_load_serps_drops_blank_keywords():
    df = pd.DataFrame(
        {
            "Keyword": [None, "", "  ", "futter"],
            "URL": ["https://a.de/1", "https://a.de/2", "https://a.de/3", "https://a.de/4"],
            "Position": [1, 2, 3, 4],
        }
    )
    out = ingest.load_serps(df)
    assert out.to_dict("records") == [
        {"keyword": "futter", "query_norm": "futter", "url": "https://a.de/4", "url_norm": "https://a.de/4", "position": 4.0}
    ]


def test_load_rankings_accepts_decimal_commas():
    df = pd.DataFrame(
        {
            "Keyword": ["futter", "napf"],
            "URL": ["https://a.de/1", "https://a.de/2"],
            "Position": ["3,5", "12"],
        }
    )
    out = ingest.load_rankings(df)
    assert out["position"].tolist() == [3.5, 12.0]


def test_read_table_loads_csv_with_very_long_cell():
    long_text = "wort " * 40_000  # 200.000 Zeichen, mehr als das Feldlimit der python-Engine
    raw = f"Address;Extract Main Content 1\nhttps://a.de/1;{long_text}\nhttps://a.de/2;kurz\n".encode("utf-8")
    df = ingest.read_table(raw, "frog.csv")
    assert len(df.iloc[0]["Extract Main Content 1"]) == len(long_text)
    assert df.iloc[1]["Address"] == "https://a.de/2"


def test_read_table_detects_separator_from_first_non_blank_line():
    df = ingest.read_table("\n\nKeyword;URL\nfutter, nass;https://a.de/1\n".encode("utf-8"), "r.csv")
    assert list(df.columns) == ["Keyword", "URL"]
    assert df.iloc[0]["Keyword"] == "futter, nass"


@pytest.mark.parametrize("value", ["NA", "null", "nan", "None", "N/A"])
def test_read_table_keeps_na_like_text(value):
    df = ingest.read_table(f"Query;Notiz\n{value};x\nfutter;\n".encode("utf-8"), "q.csv")
    assert df.iloc[0]["Query"] == value
    assert pd.isna(df.iloc[1]["Notiz"])
    assert ingest.load_queries(df) == [value, "futter"]


def test_read_table_excel_keeps_na_like_text():
    buf = io.BytesIO()
    pd.DataFrame({"Query": ["NA", "null"]}).to_excel(buf, index=False)
    df = ingest.read_table(buf.getvalue(), "q.xlsx")
    assert df["Query"].tolist() == ["NA", "null"]


def test_load_rankings_header_only_file_raises_ingest_error():
    df = ingest.read_table(b"Keyword;URL;Position\n", "r.csv")
    with pytest.raises(ingest.IngestError, match="keine verwertbaren Zeilen"):
        ingest.load_rankings(df)


def test_load_rankings_without_usable_rows_raises_ingest_error():
    df = pd.DataFrame({"Keyword": ["futter"], "URL": ["https://a.de/1"], "Position": ["x"]})
    with pytest.raises(
        ingest.IngestError,
        match=r"^Die Datei enthält keine verwertbaren Zeilen \(Keyword, URL und Position müssen gefüllt sein\)\.$",
    ):
        ingest.load_rankings(df)


def test_load_serps_header_only_file_raises_ingest_error():
    df = ingest.read_table(b"Keyword;URL;Position;Type\n", "s.csv")
    with pytest.raises(ingest.IngestError, match="keine verwertbaren Zeilen"):
        ingest.load_serps(df)


def test_load_serps_without_organic_type_raises_own_message():
    df = pd.DataFrame(
        {"Keyword": ["futter"], "URL": ["https://a.de/ad"], "Position": [1], "Type": ["Paid top"]}
    )
    with pytest.raises(ingest.IngestError, match=r"^Die Spalte Type enthält keinen Wert 'organic'\."):
        ingest.load_serps(df)


@pytest.mark.parametrize("header", ["Häufigste Suchanfragen", "Suchbegriff", "Search Term", "Suchanfragen"])
def test_query_and_keyword_aliases_from_exports(header):
    df = pd.DataFrame({header: ["futter"], "Klicks": [3]})
    assert ingest.load_queries(df) == ["futter"]
    ranking = pd.DataFrame({header: ["futter"], "URL": ["https://a.de/1"], "Position": [2]})
    assert ingest.load_rankings(ranking)["query_norm"].tolist() == ["futter"]


@pytest.mark.parametrize("header", ["Die häufigsten Seiten", "Seiten", "Landingpage"])
def test_url_aliases_from_exports(header):
    df = pd.DataFrame({header: ["https://a.de/1"], "Main Content": ["Text"]})
    assert ingest.load_content(df).urls == ["https://a.de/1"]


def test_load_queries_multi_column_without_alias_asks_for_column():
    df = pd.DataFrame({"Begriff": ["futter"], "Klicks": [3]})
    with pytest.raises(ingest.IngestError, match=r"Gefundene Spalten: \['Begriff', 'Klicks'\].*Spaltennamen"):
        ingest.load_queries(df)


def _xlsx(rows, header):
    buf = io.BytesIO()
    pd.DataFrame({header: rows}).to_excel(buf, index=False)
    return buf.getvalue()


def test_read_table_splits_xlsx_with_semicolon_lines_in_one_column():
    data = _xlsx(["futter;https://a.example/1;3", "katze;https://a.example/2;7.5"], "Keyword;URL;Position")
    df = ingest.read_table(data, "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL", "Position"]
    assert df["Keyword"].tolist() == ["futter", "katze"]
    assert df["URL"].tolist() == ["https://a.example/1", "https://a.example/2"]
    assert ingest.load_rankings(df)["position"].tolist() == [3.0, 7.5]


def test_read_table_splits_xlsx_with_comma_lines_and_keeps_quoted_commas():
    data = _xlsx(
        ["futter,https://a.example/1,3", '"was kostet futter, nass",https://a.example/2,2', "katze,https://a.example/3,4"],
        "Keyword,URL,Position",
    )
    df = ingest.read_table(data, "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL", "Position"]
    assert df["Keyword"].tolist() == ["futter", "was kostet futter, nass", "katze"]
    assert df["Position"].tolist() == ["3", "2", "4"]


def test_read_table_splits_xlsx_with_tab_lines_in_one_column():
    df = ingest.read_table(_xlsx(["futter\thttps://a.example/1"], "Keyword\tURL"), "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL"]
    assert df.iloc[0].tolist() == ["futter", "https://a.example/1"]


def test_read_table_splits_csv_whose_lines_are_wrapped_in_quotes():
    raw = '"Keyword;URL;Position"\n"futter;https://a.example/1;3"\n"katze;https://a.example/2;4"\n'.encode("utf-8")
    df = ingest.read_table(raw, "excel.csv")
    assert list(df.columns) == ["Keyword", "URL", "Position"]
    assert df["URL"].tolist() == ["https://a.example/1", "https://a.example/2"]


def test_read_table_splits_csv_with_quoted_comma_lines_and_inner_quotes():
    raw = '"Keyword,URL"\n"futter,https://a.example/1"\n"""was, nass"",https://a.example/2"\n"katze,https://a.example/3"\n'
    df = ingest.read_table(raw.encode("utf-8"), "excel.csv")
    assert list(df.columns) == ["Keyword", "URL"]
    assert df["Keyword"].tolist() == ["futter", "was, nass", "katze"]


def test_read_table_prefers_semicolon_over_tab_and_comma():
    # jede Zerlegung trifft einen bekannten Spaltennamen (Keyword bzw. Type/Position), also entscheidet die Reihenfolge
    df = ingest.read_table(_xlsx(["a;b,c\td"], "Keyword;URL,Position\tType"), "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL,Position\tType"]
    assert df.iloc[0].tolist() == ["a", "b,c\td"]
    df = ingest.read_table(_xlsx(["a\tb,c"], "Keyword\tURL,Position"), "x.xlsx")  # Tab vor Komma
    assert list(df.columns) == ["Keyword", "URL,Position"]


def test_read_table_leaves_real_single_column_query_files_alone():
    xlsx = _xlsx(["was kostet futter, nass", "katze, jung, futter"], "Query")
    df = ingest.read_table(xlsx, "q.xlsx")
    assert df.shape == (2, 1) and df.iloc[0, 0] == "was kostet futter, nass"
    csv = ingest.read_table("Query\nwas kostet futter, nass\nkatze, jung\n".encode("utf-8"), "q.csv")
    assert csv.shape == (2, 1) and csv.iloc[1, 0] == "katze, jung"


def test_read_table_does_not_split_when_most_cells_lack_the_header_delimiter():
    # Kopfzeile mit Komma, aber nur 1 von 5 Zeilen passt: das ist eine Queryliste, keine zusammengeklebte Tabelle
    rows = ["futter", "katze", "hund", "was, nass", "maus"]
    df = ingest.read_table(_xlsx(rows, "Query, Suchbegriff"), "q.xlsx")
    assert df.shape == (5, 1)


def test_read_table_splits_when_at_least_80_percent_of_the_cells_match():
    rows = [f"k{i};https://a.example/{i}" for i in range(4)] + ["lose Zeile"]
    df = ingest.read_table(_xlsx(rows, "Keyword;URL"), "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL"]
    assert len(df) == 5
    assert df.iloc[4, 0] == "lose Zeile" and pd.isna(df.iloc[4, 1])


def test_read_table_splits_header_only_single_cell():
    df = ingest.read_table(_xlsx([], "Keyword;URL;Position"), "x.xlsx")
    assert list(df.columns) == ["Keyword", "URL", "Position"]
    with pytest.raises(ingest.IngestError, match="keine verwertbaren Zeilen"):
        ingest.load_rankings(df)


_SEPARATORS = [(";", "semikolon"), (",", "komma"), ("\t", "tab")]


def _csv(sep, header, rows):
    return "\n".join(sep.join(row) for row in [header] + rows).encode("utf-8")


@pytest.mark.parametrize("sep", [s for s, _ in _SEPARATORS], ids=[n for _, n in _SEPARATORS])
def test_csv_input_with_any_separator_works_for_all_four_loaders(sep):
    queries = ingest.load_queries(ingest.read_table(_csv(sep, ["Top queries", "Clicks"], [["futter", "3"]]), "q.csv"))
    assert queries == ["futter"]

    content = ingest.load_content(
        ingest.read_table(_csv(sep, ["Address", "Extract Main Content 1"], [["https://a.example/1", "Text eins"]]), "f.csv")
    )
    assert content.urls == ["https://a.example/1"] and content.contents == ["Text eins"]

    rankings = ingest.load_rankings(
        ingest.read_table(_csv(sep, ["Keyword", "URL", "Position"], [["futter", "https://a.example/1", "3"]]), "r.csv")
    )
    assert rankings["query_norm"].tolist() == ["futter"] and rankings["position"].tolist() == [3.0]

    serps = ingest.load_serps(
        ingest.read_table(
            _csv(sep, ["Keyword", "URL", "Position", "Type"], [["futter", "https://a.example/1", "3", "Organic"]]), "s.csv"
        )
    )
    assert serps["keyword"].tolist() == ["futter"] and serps["position"].tolist() == [3.0]


PROMPTS = [
    "Welches Futter ist gut, wenn mein Hund Allergien hat?",
    "Was kostet Nassfutter, wenn ich viel kaufe?",
    "Wie groß muss ein Kratzbaum sein, damit er stabil steht?",
]


def test_read_table_keeps_headerless_xlsx_prompts_with_a_comma_in_one_column():
    df = ingest.read_table(_xlsx(PROMPTS[1:], PROMPTS[0]), "prompts.xlsx")
    assert df.shape == (2, 1)
    assert ingest.load_queries(df) == PROMPTS


def test_read_table_keeps_headerless_csv_prompts_with_a_comma_in_one_column():
    raw = "".join(f'"{prompt}"\n' for prompt in PROMPTS).encode("utf-8")  # so speichert Excel solche Zellen
    df = ingest.read_table(raw, "prompts.csv")
    assert df.shape == (2, 1)
    assert ingest.load_queries(df) == PROMPTS


def test_read_table_splits_only_when_a_split_name_is_a_known_column():
    df = ingest.read_table(_xlsx(["a;b", "c;d"], "Spalte eins;Spalte zwei"), "x.xlsx")
    assert df.shape == (2, 1)
    df = ingest.read_table(_xlsx(["a;b", "c;d"], "Spalte eins; KEYWORD "), "x.xlsx")
    assert list(df.columns) == ["Spalte eins", "KEYWORD"]


ALL_IN_ONE_CELL = (
    "Die Datei scheint alle Spalten in einer Zelle zu enthalten. Benenne die Spalten in der ersten Zeile mit bekannten "
    "Namen (z. B. Keyword, URL, Position) oder speichere die Datei als CSV mit getrennten Spalten."
)


def test_unsplit_file_with_unknown_names_explains_how_to_fix_it():
    df = ingest.read_table(_xlsx(["a;https://a.example/1;3"], "Begriff;Adresse X;Rang"), "x.xlsx")
    assert df.shape == (1, 1)
    with pytest.raises(ingest.IngestError) as error:
        ingest.load_rankings(df)
    assert "Keyword-Spalte nicht erkannt" in str(error.value)
    assert str(error.value).endswith(ALL_IN_ONE_CELL)


def test_ordinary_missing_column_has_no_extra_sentence():
    df = pd.DataFrame({"Begriff": ["a"], "Adresse X": ["b"]})
    with pytest.raises(ingest.IngestError) as error:
        ingest.load_rankings(df)
    assert ALL_IN_ONE_CELL not in str(error.value)


def test_load_content_detects_custom_extraction_column():
    # Screaming Frog benennt Custom-Extraction-Spalten "<Name> 1"
    df = pd.DataFrame(
        {
            "Address": ["https://a.de/1"],
            "Status Code": ["200"],
            "Toom Main Content Extracotr 1": ["Text eins"],
        }
    )
    table = ingest.load_content(df)
    assert table.contents == ["Text eins"]


def test_load_content_custom_extraction_ambiguous_raises():
    df = pd.DataFrame({"Address": ["https://a.de/1"], "Content Teaser 1": ["a"], "Main Content Extractor 1": ["b"]})
    with pytest.raises(ingest.IngestError, match="Content"):
        ingest.load_content(df)
