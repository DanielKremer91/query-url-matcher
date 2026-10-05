import zipfile

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from . import labels as L
from .serp import gap_summary

SHEET_README = "Lesehilfe"
SHEET_DECISION = "Entscheidung"
SHEET_TOP = "Top-Treffer"
SHEET_CANNIBAL = "Kannibalisierung"
SHEET_GAPS = "Content-Lücken"
SHEET_GAP_CLUSTERS = "Lücken je Cluster"
SHEET_PAIRS = "Paare"

_CSV_NAMES = {
    SHEET_README: "lesehilfe.csv",
    SHEET_DECISION: "entscheidung.csv",
    SHEET_TOP: "top_treffer.csv",
    SHEET_CANNIBAL: "kannibalisierung.csv",
    SHEET_GAPS: "content_luecken.csv",
    SHEET_GAP_CLUSTERS: "luecken_je_cluster.csv",
    SHEET_PAIRS: "paare.csv",
}

_SHEET_HELP = {
    SHEET_DECISION: "Eine Zeile je Query mit Urteil, bester URL, Passage, Scores und Empfehlung.",
    SHEET_TOP: "Die besten URLs je Query mit Chunk-, Gesamt-URL- und Kombi-Score samt Rängen.",
    SHEET_CANNIBAL: "Queries, bei denen mehrere eigene Seiten konkurrieren, getrennt nach Stufe.",
    SHEET_GAPS: "Queries ohne passende Seite. 'Vor Neuerstellung prüfen' nennt eine Kandidaten-Seite.",
    SHEET_GAP_CLUSTERS: "Lücken gebündelt nach SERP-Cluster: ein Cluster entspricht einer neuen Seite.",
    SHEET_PAIRS: "Bewertung vorhandener Keyword-URL-Paare aus dem Ranking-Export.",
}

_COLUMN_HELP = {
    L.C_S_CHUNK: "Cosinus-Ähnlichkeit zwischen Query und dem am besten passenden Textblock der Seite.",
    L.C_S_FULL: "Cosinus-Ähnlichkeit zwischen Query und dem gesamten Main Content der Seite.",
    L.C_S_COMBI: "Gewichtete Mischung aus Chunk-Score und Gesamt-URL-Score.",
    L.C_METHOD: f"'{L.FULLTEXT}': ganzer Text als ein Embedding. '{L.CHUNK_MEAN}': Text zu lang, Näherung.",
    L.C_CLUSTER: "Keywords mit gleicher Nummer haben stark überlappende Google-Ergebnisse. 0 = kein Cluster.",
    L.C_CAND: "Bestehende Seite, die ein Keyword mit fast gleicher SERP bereits bedient.",
}

_VERDICT_HELP = {
    L.V_MATCH: "Mindestens eine Seite erreicht die Schwelle.",
    L.V_GAP: "Keine Seite erreicht die Schwelle.",
    L.V_OK: "Die rankende Seite ist auch der beste semantische Treffer.",
    L.V_RISK: "Eine Seite rankt gut, eine andere eigene Seite passt semantisch besser.",
    L.V_WATCH: "Die Query rankt gut, obwohl keine Seite die Schwelle erreicht.",
    L.V_USE: "Kein gutes Ranking, aber eine passende Seite existiert.",
    L.V_CHECK: "Lücke, aber ein Keyword mit fast gleicher SERP hat bereits eine passende Seite.",
}

_FILLS = {
    L.V_MATCH: "C6EFCE",
    L.V_OK: "C6EFCE",
    L.V_USE: "FFEB9C",
    L.V_CHECK: "FFEB9C",
    L.V_RISK: "F8CBAD",
    L.V_GAP: "FFC7CE",
    L.V_WATCH: "D9D9D9",
}

DISCLAIMER = (
    "Das Notebook sortiert vor und begründet. Cosinus-Werte sind Hinweise, keine Urteile: "
    "die Entscheidung trifft ein Mensch nach Prüfung der Seite."
)


def content_gaps(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions[decisions[L.C_VERDICT].isin([L.V_GAP, L.V_CHECK])].reset_index(drop=True)


def _readme(sheet_names, settings) -> pd.DataFrame:
    rows = [("Hinweis", "Einordnung", DISCLAIMER)]
    rows += [("Blatt", name, _SHEET_HELP[name]) for name in sheet_names]
    rows += [("Urteil", verdict, text) for verdict, text in _VERDICT_HELP.items()]
    rows += [("Spalte", column, text) for column, text in _COLUMN_HELP.items()]
    rows += [("Einstellung", key, str(value)) for key, value in settings.items()]
    return pd.DataFrame(rows, columns=["Bereich", "Eintrag", "Erklärung"])


def build_sheets(decisions, top, cannibal, settings, pairs=None) -> dict:
    sheets = {
        SHEET_DECISION: decisions,
        SHEET_TOP: top,
        SHEET_CANNIBAL: cannibal,
        SHEET_GAPS: content_gaps(decisions),
    }
    if L.C_CLUSTER in decisions.columns:
        sheets[SHEET_GAP_CLUSTERS] = gap_summary(decisions)
    if pairs is not None:
        sheets[SHEET_PAIRS] = pairs
    return {SHEET_README: _readme(list(sheets), settings), **sheets}


def write_excel(path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for idx, column in enumerate(df.columns, start=1):
                longest = max([len(str(column))] + [len(str(v)) for v in df[column].head(200)])
                sheet.column_dimensions[get_column_letter(idx)].width = min(max(12, longest + 2), 70)
            if L.C_VERDICT in df.columns:
                col = list(df.columns).index(L.C_VERDICT) + 1
                for row, verdict in enumerate(df[L.C_VERDICT], start=2):
                    colour = _FILLS.get(verdict)
                    if colour:
                        sheet.cell(row=row, column=col).fill = PatternFill("solid", fgColor=colour)


def write_csv_zip(path, sheets: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, df in sheets.items():
            archive.writestr(_CSV_NAMES[name], df.to_csv(index=False).encode("utf-8-sig"))
