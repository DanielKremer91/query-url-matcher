import io
import zipfile

import pandas as pd
import pytest
from openpyxl import load_workbook

from qum import export
from qum import labels as L
from qum.cannibal import COLUMNS as CANNIBAL_COLUMNS
from qum.cannibal import annotate, find_cannibalization
from qum.gaps import COLUMNS as GAP_COLUMNS
from qum.gaps import find_gaps
from qum.verdict import OVERVIEW_COLUMNS, build_decisions
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"
SETTINGS = {"Modell": "multilingual-e5-large", "Schwelle": "0.8"}


def _frames(queries=("a", "b", "c"), scores=((0.9, 0.1, 0.1), (0.5, 0.2, 0.1), (0.85, 0.845, 0.1)), topics=None):
    """Übersicht, Blatt Kannibalisierungsgefahr und Blatt Lücken aus den echten Funktionen (Schwelle 0.8)."""
    result = make_result(list(queries), [U1, U2, U3], [list(s) for s in scores])
    lead = result.lead("chunk")
    cannibal = find_cannibalization(result, lead, 0.8)
    overview = annotate(build_decisions(result, lead, 0.8), cannibal)
    return overview, cannibal, find_gaps(result, lead, 0.8, topics=topics)


def _sheets(**kwargs):
    overview, cannibal, gaps = _frames()
    return export.build_sheets(overview, cannibal, gaps, SETTINGS, **kwargs)


def _readme(**kwargs):
    return _sheets(**kwargs)[export.SHEET_README]


def _entries(readme, area):
    return readme[readme[L.R_AREA] == area][L.R_ENTRY].tolist()


# --- Blätter -----------------------------------------------------------------------------------------------------


def test_four_sheets_in_order():
    assert list(_sheets()) == [export.SHEET_OVERVIEW, export.SHEET_CANNIBAL, export.SHEET_GAPS, export.SHEET_README]
    assert [export.SHEET_OVERVIEW, export.SHEET_CANNIBAL, export.SHEET_GAPS, export.SHEET_README] == [
        "Übersicht", "Kannibalisierungsgefahr", "Potentielle Content-Lücken", "Lesehilfe",
    ]
    for name in ("SHEET_PAIRS", "SHEET_TOP", "SHEET_GAP_CLUSTERS", "SHEET_DECISION"):
        assert not hasattr(export, name), name


def test_query_is_the_first_column_of_the_three_data_sheets():
    sheets = _sheets()
    for name in (export.SHEET_OVERVIEW, export.SHEET_CANNIBAL, export.SHEET_GAPS):
        assert sheets[name].columns[0] == L.C_QUERY, name


def test_sheets_keep_their_columns():
    sheets = _sheets()
    assert list(sheets[export.SHEET_OVERVIEW].columns) == OVERVIEW_COLUMNS
    assert list(sheets[export.SHEET_CANNIBAL].columns) == CANNIBAL_COLUMNS
    assert list(sheets[export.SHEET_GAPS].columns) == GAP_COLUMNS
    assert sheets[export.SHEET_OVERVIEW][L.C_CANNIBAL].tolist() == [L.NO, L.NO, L.YES]
    assert sheets[export.SHEET_GAPS][L.C_QUERY].tolist() == ["b"]


# --- Lesehilfe ---------------------------------------------------------------------------------------------------


def test_readme_columns_and_disclaimer_first():
    readme = _readme()
    assert list(readme.columns) == [L.R_AREA, L.R_ENTRY, L.R_TEXT] == ["Bereich", "Eintrag", "Erklärung"]
    assert readme.iloc[0][L.R_TEXT].startswith("Das Notebook sortiert vor")


def test_readme_lists_all_four_sheets_all_verdicts_and_both_stages():
    readme = _readme()
    assert _entries(readme, "Blatt") == [export.SHEET_OVERVIEW, export.SHEET_CANNIBAL, export.SHEET_GAPS, export.SHEET_README]
    assert _entries(readme, "Urteil") == [L.V_MATCH, L.V_GAP, L.V_OK, L.V_CANNIBAL, L.V_WATCH]
    assert _entries(readme, "Stufe") == [L.STAGE_DANGER, L.STAGE_VISIBLE]


def test_readme_explains_every_column_of_every_sheet_once_in_sheet_order():
    expected = list(dict.fromkeys(OVERVIEW_COLUMNS + CANNIBAL_COLUMNS + GAP_COLUMNS))
    assert _entries(_readme(), "Spalte") == expected
    assert set(export._COLUMN_HELP) == set(expected)


def test_every_column_label_is_explained():
    # C_URL, C_SIDE und C_SCORE gehören nur zu den Prüfbeispielen in Schritt 7a und 7b, nicht zum Export
    columns = [v for k, v in vars(L).items() if k.startswith("C_") and k not in ("C_URL", "C_SIDE", "C_SCORE")]
    assert columns
    for column in columns:
        assert column in export._COLUMN_HELP, column


def test_readme_lists_the_settings_of_the_run():
    readme = _readme()
    settings = readme[readme[L.R_AREA] == "Einstellung"]
    assert dict(zip(settings[L.R_ENTRY], settings[L.R_TEXT])) == SETTINGS


def test_gap_sheet_row_states_the_new_pages_with_serps():
    readme = _readme(new_pages=7)
    text = readme[(readme[L.R_AREA] == "Blatt") & (readme[L.R_ENTRY] == export.SHEET_GAPS)][L.R_TEXT].item()
    assert "Neue Seiten: 7 (eine je Thema" in text
    assert "eigenen Einstellungen" in text and f"'{L.V_GAP}'" in text
    plain = _readme()
    text = plain[(plain[L.R_AREA] == "Blatt") & (plain[L.R_ENTRY] == export.SHEET_GAPS)][L.R_TEXT].item()
    assert "Neue Seiten" not in text and "Ohne SERPs" in text


def _notes(readme):
    return readme[readme[L.R_AREA] == "Hinweis"][L.R_TEXT].tolist()


def test_readme_always_explains_the_score_band():
    notes = _notes(_readme())
    assert notes == [export.DISCLAIMER, export.CAVEAT_SCORES]
    assert export.CAVEAT_SCORES.startswith("Scores eines Modells liegen in einem engen Band (bei e5 etwa 0,7 bis 0,9).")


def test_readme_flags_uncalibrated_median_threshold():
    notes = _notes(_readme(threshold_source="median"))
    assert notes == [export.DISCLAIMER, export.CAVEAT_SCORES, export.CAVEAT_THRESHOLD["median"]]
    assert notes[2].startswith("Schwelle nicht kalibriert:")


def test_readme_explains_threshold_calibrated_from_rankings():
    notes = _notes(_readme(threshold_source="rankings"))
    assert notes[-1] == export.CAVEAT_THRESHOLD["rankings"]
    assert "75 % der gut rankenden Paare" in notes[-1] and "Rankt trotz schwachem Match" in notes[-1]


def test_readme_says_when_threshold_was_set_by_hand():
    assert _notes(_readme(threshold_source="manuell"))[-1] == export.CAVEAT_THRESHOLD["manuell"]
    assert "von Hand" in export.CAVEAT_THRESHOLD["manuell"]


def test_help_texts_describe_the_rules():
    assert "fast gleich" in export.VERDICT_HELP[L.V_OK]
    assert "deutlich besser" in export.VERDICT_HELP[L.V_CANNIBAL]
    assert "oder die rankende URL steht nicht im Frog-Export" in export.VERDICT_HELP[L.V_CANNIBAL]
    assert "Mehrere Seiten erreichen die Schwelle und passen fast gleich gut" in export.VERDICT_HELP[L.V_CANNIBAL]
    assert "fast gleich gut" in export._STAGE_HELP[L.STAGE_DANGER]
    assert "enau eine Seite passt klar" in export.VERDICT_HELP[L.V_MATCH]
    assert "keine Rankings geladen" in export.VERDICT_HELP[L.V_MATCH]
    assert not hasattr(L, "V_USE")
    comparison = export._COLUMN_HELP[L.C_RANK_IS_BEST]
    for value in (L.YES, L.CMP_CLOSE, L.NO, L.CMP_NOT_RANKING, L.CMP_NOT_IN_EXPORT):
        assert f"'{value}'" in comparison, value
    assert "weitere" in export._COLUMN_HELP[L.C_URL_1]
    assert "Schritt 6" in export._COLUMN_HELP[L.C_TOPIC]


# --- Excel -------------------------------------------------------------------------------------------------------


def test_write_excel_creates_sheets_and_colours_verdicts(tmp_path):
    path = tmp_path / "out.xlsx"
    export.write_excel(path, _sheets())
    book = load_workbook(path)
    assert book.sheetnames == [export.SHEET_OVERVIEW, export.SHEET_CANNIBAL, export.SHEET_GAPS, export.SHEET_README]
    sheet = book[export.SHEET_OVERVIEW]
    assert sheet["A1"].font.bold
    column = OVERVIEW_COLUMNS.index(L.C_VERDICT) + 1
    fills = [sheet.cell(row=r, column=column).fill.fgColor.rgb for r in (2, 3, 4)]
    assert [sheet.cell(row=r, column=column).value for r in (2, 3, 4)] == [L.V_MATCH, L.V_GAP, L.V_CANNIBAL]
    assert [f[-6:] for f in fills] == ["FFEB9C", "FFC7CE", "F8CBAD"]
    assert set(export._FILLS) == set(export.VERDICT_HELP) == {L.V_MATCH, L.V_GAP, L.V_OK, L.V_CANNIBAL, L.V_WATCH}


def test_write_excel_leaves_missing_scores_empty(tmp_path):
    overview, cannibal, gaps = _frames()
    export.write_excel(tmp_path / "out.xlsx", export.build_sheets(overview, cannibal, gaps, SETTINGS))
    sheet = load_workbook(tmp_path / "out.xlsx")[export.SHEET_CANNIBAL]
    header = [cell.value for cell in sheet[1]]
    assert sheet.cell(row=2, column=header.index(L.C_URL_3) + 1).value is None
    assert sheet.cell(row=2, column=header.index(L.C_SCORE_3) + 1).value is None


def test_write_excel_strips_control_characters_without_mutating_input(tmp_path):
    sheets = _sheets()
    overview = sheets[export.SHEET_OVERVIEW]
    overview[L.C_CHUNK] = ["ok", "tab\x0bvertical", "unit\x1fsep"]
    export.write_excel(tmp_path / "out.xlsx", sheets)
    sheet = load_workbook(tmp_path / "out.xlsx")[export.SHEET_OVERVIEW]
    values = [sheet.cell(row=r, column=3).value for r in (2, 3, 4)]
    assert values == ["ok", "tab vertical", "unit sep"]
    assert overview[L.C_CHUNK].tolist() == ["ok", "tab\x0bvertical", "unit\x1fsep"]


def test_write_excel_keeps_formula_like_text_as_text(tmp_path):
    sheets = _sheets()
    overview = sheets[export.SHEET_OVERVIEW]
    overview[L.C_QUERY] = ["=cmd|x", "+49 hotline", "@home"]
    overview[L.C_CHUNK] = ["= 5 Euro", "-20 % Rabatt", "normal"]
    export.write_excel(tmp_path / "out.xlsx", sheets)
    sheet = load_workbook(tmp_path / "out.xlsx")[export.SHEET_OVERVIEW]
    cells = [sheet.cell(row=r, column=c) for c in (1, 3) for r in (2, 3, 4)]
    assert [cell.value for cell in cells] == ["=cmd|x", "+49 hotline", "@home", "= 5 Euro", "-20 % Rabatt", "normal"]
    assert all(cell.data_type == "s" for cell in cells)


# --- CSV-ZIP -----------------------------------------------------------------------------------------------------


def test_write_csv_zip_has_one_file_per_sheet(tmp_path):
    path = tmp_path / "out.zip"
    export.write_csv_zip(path, _sheets())
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "uebersicht.csv", "kannibalisierungsgefahr.csv", "potentielle_content_luecken.csv", "lesehilfe.csv",
        ]
        df = pd.read_csv(io.BytesIO(archive.read("uebersicht.csv")), encoding="utf-8-sig", sep=";")
    assert df[L.C_QUERY].tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("sep", [";", ","])
def test_write_csv_zip_round_trips_with_either_separator(tmp_path, sep):
    path = tmp_path / "out.zip"
    sheets = _sheets()
    sheets[export.SHEET_OVERVIEW].loc[0, L.C_QUERY] = "futter; nass, 10 kg"  # beide Trennzeichen im Text
    export.write_csv_zip(path, sheets, sep=sep)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("uebersicht.csv").decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    assert raw.splitlines()[0].count(sep) == len(OVERVIEW_COLUMNS) - 1
    assert list(df.columns) == OVERVIEW_COLUMNS
    assert df[L.C_QUERY].tolist() == ["futter; nass, 10 kg", "b", "c"]


def test_write_csv_zip_defaults_to_semicolon(tmp_path):
    path = tmp_path / "out.zip"
    export.write_csv_zip(path, _sheets())
    with zipfile.ZipFile(path) as archive:
        header = archive.read("uebersicht.csv").decode("utf-8-sig").splitlines()[0]
    assert header.startswith(f"{L.C_QUERY};{L.C_BEST_URL};")


@pytest.mark.parametrize("sep, number", [(";", "0,9"), (",", "0.9")])
def test_write_csv_zip_writes_decimal_comma_only_with_semicolon(tmp_path, sep, number):
    path = tmp_path / "out.zip"
    export.write_csv_zip(path, _sheets(), sep=sep)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("uebersicht.csv").decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str, keep_default_na=False)
    assert df[L.C_S_CHUNK].tolist()[0] == number
    parsed = pd.read_csv(io.StringIO(raw), sep=sep, decimal="," if sep == ";" else ".")
    assert parsed[L.C_S_CHUNK].tolist() == [0.9, 0.5, 0.85]


def _text_number_sheets():
    overview, _, gaps = _frames()
    overview[L.C_POSITION] = ["4.3", "12", ""]
    gaps[L.C_POSITION] = ["7.5"]
    cannibal = pd.DataFrame(
        [["a", L.STAGE_DANGER, L.REASON_BETTER, "https://a.de/x.html", 0.842, "4.3", "https://a.de/y", -0.0123, "",
          "https://a.de/z", None, "17.5"]],
        columns=CANNIBAL_COLUMNS,
    )
    settings = {"Datum": "2026-10-05", "Modell": "multilingual-e5-large", "Schwelle": 0.8123, "Rankt gut bis Position": 10}
    return export.build_sheets(overview, cannibal, gaps, settings)


def _csv_frames(path, sep):
    with zipfile.ZipFile(path) as archive:
        return {
            name: pd.read_csv(io.BytesIO(archive.read(name)), encoding="utf-8-sig", sep=sep, dtype=str, keep_default_na=False)
            for name in archive.namelist()
        }


def test_semicolon_csv_uses_decimal_comma_in_positions_settings_and_scores(tmp_path):
    sheets = _text_number_sheets()
    export.write_csv_zip(tmp_path / "out.zip", sheets, sep=";")
    frames = _csv_frames(tmp_path / "out.zip", ";")
    assert frames["uebersicht.csv"][L.C_POSITION].tolist() == ["4,3", "12", ""]
    assert frames["potentielle_content_luecken.csv"][L.C_POSITION].tolist() == ["7,5"]
    row = frames["kannibalisierungsgefahr.csv"].iloc[0]
    assert [row[L.C_POS_1], row[L.C_SCORE_1], row[L.C_SCORE_2], row[L.C_POS_2], row[L.C_SCORE_3], row[L.C_POS_3]] == [
        "4,3", "0,842", "-0,0123", "", "", "17,5",
    ]
    readme = frames["lesehilfe.csv"].set_index(L.R_ENTRY)[L.R_TEXT]
    assert readme["Schwelle"] == "0,8123"
    assert readme["Datum"] == "2026-10-05" and readme["Modell"] == "multilingual-e5-large"
    assert readme["Rankt gut bis Position"] == "10"
    assert sheets[export.SHEET_OVERVIEW][L.C_POSITION].tolist() == ["4.3", "12", ""]  # Original unverändert


def test_comma_csv_and_excel_keep_the_decimal_point_in_text(tmp_path):
    sheets = _text_number_sheets()
    export.write_csv_zip(tmp_path / "out.zip", sheets, sep=",")
    frames = _csv_frames(tmp_path / "out.zip", ",")
    assert frames["uebersicht.csv"][L.C_POSITION].tolist() == ["4.3", "12", ""]
    row = frames["kannibalisierungsgefahr.csv"].iloc[0]
    assert [row[L.C_POS_1], row[L.C_SCORE_1], row[L.C_POS_3]] == ["4.3", "0.842", "17.5"]
    export.write_excel(tmp_path / "out.xlsx", sheets)
    book = load_workbook(tmp_path / "out.xlsx")
    position_col = OVERVIEW_COLUMNS.index(L.C_POSITION) + 1
    assert book[export.SHEET_OVERVIEW].cell(row=2, column=position_col).value == "4.3"


def test_topic_is_written_as_a_whole_number_or_left_empty(tmp_path):
    overview, cannibal, gaps = _frames(
        queries=("a", "b", "c"), scores=((0.5, 0.1, 0.1), (0.4, 0.2, 0.1), (0.3, 0.2, 0.1)), topics=[3, 0, 3]
    )
    sheets = export.build_sheets(overview, cannibal, gaps, SETTINGS)
    export.write_csv_zip(tmp_path / "out.zip", sheets)
    assert _csv_frames(tmp_path / "out.zip", ";")["potentielle_content_luecken.csv"][L.C_TOPIC].tolist() == ["3", "", "3"]
    export.write_excel(tmp_path / "out.xlsx", sheets)
    sheet = load_workbook(tmp_path / "out.xlsx")[export.SHEET_GAPS]
    assert [sheet.cell(row=r, column=5).value for r in (2, 3, 4)] == [3, None, 3]
