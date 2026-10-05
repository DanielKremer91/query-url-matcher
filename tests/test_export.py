import io
import zipfile

import pandas as pd
from openpyxl import load_workbook

from qum import export
from qum import labels as L


def _decisions(with_cluster=False):
    df = pd.DataFrame(
        {
            L.C_QUERY: ["a", "b", "c"],
            L.C_VERDICT: [L.V_MATCH, L.V_GAP, L.V_CHECK],
            L.C_BEST_URL: ["u1", "u2", "u3"],
        }
    )
    if with_cluster:
        df[L.C_CLUSTER] = [1, 1, 0]
    return df


TOP = pd.DataFrame({L.C_QUERY: ["a"], L.C_URL: ["u1"]})
CANNIBAL = pd.DataFrame(columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING])
SETTINGS = {"Modell": "multilingual-e5-large", "Schwelle": "0.81"}


def test_content_gaps_contains_gap_and_check():
    assert export.content_gaps(_decisions())[L.C_QUERY].tolist() == ["b", "c"]


def test_sheets_without_serps_and_pairs():
    sheets = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS)
    assert list(sheets) == [
        export.SHEET_README, export.SHEET_DECISION, export.SHEET_TOP, export.SHEET_CANNIBAL, export.SHEET_GAPS,
    ]


def test_sheets_with_serps_and_pairs():
    pairs = pd.DataFrame({"Keyword": ["a"]})
    sheets = export.build_sheets(_decisions(with_cluster=True), TOP, CANNIBAL, SETTINGS, pairs=pairs)
    assert list(sheets)[-2:] == [export.SHEET_GAP_CLUSTERS, export.SHEET_PAIRS]


def test_readme_lists_present_sheets_settings_and_disclaimer():
    readme = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS)[export.SHEET_README]
    assert list(readme.columns) == ["Bereich", "Eintrag", "Erklärung"]
    entries = readme["Eintrag"].tolist()
    assert export.SHEET_GAPS in entries
    assert export.SHEET_PAIRS not in entries
    assert "Modell" in entries
    assert L.V_RISK in entries
    assert readme.iloc[0]["Erklärung"].startswith("Das Notebook sortiert vor")


def test_every_sheet_has_query_first_except_readme_and_summary():
    pairs = pd.DataFrame({"Volume": [10], "Keyword": ["a"], "URL": ["u1"]})
    sheets = export.build_sheets(_decisions(with_cluster=True), TOP, CANNIBAL, SETTINGS, pairs=pairs)
    assert export.SHEET_PAIRS in sheets
    for name, df in sheets.items():
        if name not in (export.SHEET_README, export.SHEET_GAP_CLUSTERS):
            assert df.columns[0] in (L.C_QUERY, "Keyword"), name


def test_pairs_keyword_column_moves_to_front_without_renaming():
    pairs = pd.DataFrame({"Volume": [10], "Keyword": ["a"], "URL": ["u1"]})
    sheet = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS, pairs=pairs)[export.SHEET_PAIRS]
    assert list(sheet.columns) == ["Keyword", "Volume", "URL"]


def test_pairs_without_keyword_column_stay_unchanged():
    pairs = pd.DataFrame({"Volume": [10], "URL": ["u1"]})
    sheet = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS, pairs=pairs)[export.SHEET_PAIRS]
    assert list(sheet.columns) == ["Volume", "URL"]


def test_every_label_column_has_a_help_text():
    columns = [v for k, v in vars(L).items() if k.startswith("C_") and k != "C_SIDE"]
    assert columns
    for column in columns:
        assert column in export._COLUMN_HELP, column


def test_readme_explains_candidate_columns_only_when_present():
    decisions = _decisions(with_cluster=True)
    decisions[L.C_CAND] = ["", "", "u1"]
    readme = export.build_sheets(decisions, TOP, CANNIBAL, SETTINGS)[export.SHEET_README]
    assert L.C_CAND in readme["Eintrag"].tolist()
    stages = readme[readme["Bereich"] == "Stufe"]["Eintrag"].tolist()
    assert stages == [L.STAGE_RISK, L.STAGE_VISIBLE]
    plain = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS)[export.SHEET_README]
    assert L.C_CAND not in plain["Eintrag"].tolist()
    assert L.C_CLUSTER not in plain["Eintrag"].tolist()


def test_readme_skips_columns_without_explanation():
    pairs = pd.DataFrame({"Keyword": ["a"], "Volume": [10]})
    readme = export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS, pairs=pairs)[export.SHEET_README]
    assert "Volume" not in readme["Eintrag"].tolist()


def test_write_excel_creates_sheets_and_colours_verdicts(tmp_path):
    path = tmp_path / "out.xlsx"
    export.write_excel(path, export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS))
    book = load_workbook(path)
    assert book.sheetnames[:2] == [export.SHEET_README, export.SHEET_DECISION]
    sheet = book[export.SHEET_DECISION]
    assert sheet["A1"].font.bold
    fills = [sheet.cell(row=r, column=2).fill.fgColor.rgb for r in (2, 3, 4)]
    assert [f[-6:] for f in fills] == ["C6EFCE", "FFC7CE", "FFEB9C"]


def test_write_csv_zip_has_one_file_per_sheet(tmp_path):
    path = tmp_path / "out.zip"
    export.write_csv_zip(path, export.build_sheets(_decisions(), TOP, CANNIBAL, SETTINGS))
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "lesehilfe.csv", "entscheidung.csv", "top_treffer.csv", "kannibalisierung.csv", "content_luecken.csv",
        ]
        df = pd.read_csv(io.BytesIO(archive.read("entscheidung.csv")), encoding="utf-8-sig")
    assert df[L.C_QUERY].tolist() == ["a", "b", "c"]


def test_write_excel_strips_control_characters_without_mutating_input(tmp_path):
    decisions = _decisions()
    decisions[L.C_CHUNK] = ["ok", "tab\x0bvertical", "unit\x1fsep"]
    path = tmp_path / "out.xlsx"
    export.write_excel(path, export.build_sheets(decisions, TOP, CANNIBAL, SETTINGS))
    sheet = load_workbook(path)[export.SHEET_DECISION]
    values = [sheet.cell(row=r, column=4).value for r in (2, 3, 4)]
    assert values == ["ok", "tab vertical", "unit sep"]
    assert decisions[L.C_CHUNK].tolist() == ["ok", "tab\x0bvertical", "unit\x1fsep"]


def test_candidate_position_help_refers_to_the_gap_query():
    text = export._COLUMN_HELP[L.C_CAND_POS]
    assert "Lücken-Query" in text
    assert "Nachbar-Keyword" not in text


def test_help_texts_do_not_overclaim_serp_similarity():
    texts = list(export._COLUMN_HELP.values()) + list(export._VERDICT_HELP.values())
    assert all("fast gleicher SERP" not in text for text in texts)
    assert "stark überlappender SERP" in export._VERDICT_HELP[L.V_CHECK]
