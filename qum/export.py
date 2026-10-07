import re
import zipfile

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from . import labels as L
from .serp import gap_summary

SHEET_README = "Lesehilfe"
SHEET_DECISION = "Entscheidung"
SHEET_TOP = "Top-Treffer"
SHEET_CANNIBAL = "Kannibalisierungsgefahr"
SHEET_GAPS = "Content-Lücken"
SHEET_GAP_CLUSTERS = "Lücken je Cluster"

_CSV_NAMES = {
    SHEET_README: "lesehilfe.csv",
    SHEET_DECISION: "entscheidung.csv",
    SHEET_TOP: "top_treffer.csv",
    SHEET_CANNIBAL: "kannibalisierungsgefahr.csv",
    SHEET_GAPS: "content_luecken.csv",
    SHEET_GAP_CLUSTERS: "luecken_je_cluster.csv",
}

_SHEET_HELP = {
    SHEET_DECISION: "Eine Zeile je Query mit Urteil, bester URL, Passage und Scores.",
    SHEET_TOP: "Die besten URLs je Query mit Chunk-, Gesamt-URL- und Kombi-Score samt Rängen.",
    SHEET_CANNIBAL: "Queries, bei denen mehrere eigene Seiten konkurrieren, getrennt nach Stufe.",
    SHEET_GAPS: "Queries ohne passende Seite.",
    SHEET_GAP_CLUSTERS: "Lücken gebündelt nach SERP-Cluster: ein Cluster entspricht einer neuen Seite.",
}

_COLUMN_HELP = {
    L.C_QUERY: "Die Suchanfrage, um die es in der Zeile geht.",
    L.C_URL: "Die eigene Seite, die zur Query passt.",
    L.C_CHUNK: "Der Textblock der Seite, der zur Query am besten passt.",
    L.C_S_CHUNK: "Cosinus-Ähnlichkeit zwischen Query und dem am besten passenden Textblock der Seite.",
    L.C_S_FULL: "Cosinus-Ähnlichkeit zwischen Query und dem gesamten Main Content der Seite.",
    L.C_S_COMBI: "Gewichtete Mischung aus Chunk-Score und Gesamt-URL-Score.",
    L.C_LEAD_GAP: (
        "Abstand des Scores der besten URL zum Score der nächstbesten Seite. "
        "Ein großer Vorsprung heißt: eine Seite sticht klar heraus, auch wenn der absolute Score niedrig ist. "
        "Negativ, wenn eine andere Seite knapp davor liegt."
    ),
    L.C_GAP_TO_BEST: "Abstand dieses Treffers zum besten Treffer der Query (0 = bester Treffer).",
    L.C_R_CHUNK: "Rang der Seite nach Chunk-Score für diese Query. 1 = beste URL für diese Query unter allen URLs.",
    L.C_R_FULL: "Rang der Seite nach Gesamt-URL-Score für diese Query. 1 = beste URL für diese Query unter allen URLs.",
    L.C_R_COMBI: "Rang der Seite nach Kombi-Score für diese Query. 1 = beste URL für diese Query unter allen URLs.",
    L.C_METHOD: f"'{L.FULLTEXT}': ganzer Text als ein Embedding. '{L.CHUNK_MEAN}': Text zu lang, Näherung.",
    L.C_VERDICT: "Einordnung der Query, siehe die Urteile weiter unten in dieser Lesehilfe.",
    L.C_BEST_URL: (
        "Die Seite mit dem höchsten Score für die Query, nach der gewählten Bewertungsgrundlage. "
        "Bei 'In Ordnung' die rankende Seite, wenn sie fast gleich gut passt."
    ),
    L.C_RANK_URL: "Die URL, die laut Ranking-Export am besten für die Query rankt.",
    L.C_POSITION: "Beste Position der rankenden URL für die Query im Ranking-Export.",
    L.C_NOTE: "Ergänzender Hinweis, zum Beispiel wenn eine URL im Frog-Export fehlt.",
    L.C_CANNIBAL: (
        "Hinweis auf Kannibalisierungsgefahr zu dieser Query, unabhängig vom Urteil: Stufe (Gefahr oder Bereits sichtbar) und die "
        "konkurrierenden eigenen URLs. Leer, wenn es keinen Hinweis gibt. Details im Blatt Kannibalisierungsgefahr."
    ),
    L.C_CLUSTER: "Keywords mit gleicher Nummer haben stark überlappende Google-Ergebnisse. 0 = kein Cluster.",
    L.C_STAGE: "Stufe der Kannibalisierungsgefahr, siehe die Stufen weiter unten in dieser Lesehilfe.",
    L.C_REASON: "Warum die Query im Blatt Kannibalisierungsgefahr steht.",
    L.C_COMPETING: "Die konkurrierenden eigenen URLs mit Score oder Position.",
    L.C_GAP_COUNT: "Anzahl der Content-Lücken-Queries in diesem Cluster.",
    L.C_NEW_PAGES: "Anzahl neuer Seiten, die dafür nötig wären: ein Cluster eine Seite, Queries ohne Cluster je eine.",
    L.C_GAP_QUERIES: "Die Content-Lücken-Queries dieser Zeile, getrennt durch senkrechte Striche.",
}

_STAGE_HELP = {
    L.STAGE_DANGER: "Eine Seite rankt gut, eine andere passt deutlich besser, oder mehrere Seiten passen fast gleich gut.",
    L.STAGE_VISIBLE: "Mehrere eigene Seiten ranken bereits für die Query.",
}

VERDICT_HELP = {
    L.V_MATCH: "Mindestens eine Seite erreicht die Schwelle.",
    L.V_GAP: "Keine Seite erreicht die Schwelle.",
    L.V_OK: (
        "Die rankende Seite erreicht die Schwelle und passt semantisch am besten oder fast gleich gut wie die beste "
        "(Abstand 'fast gleich')."
    ),
    L.V_CANNIBAL: (
        "Eine Seite rankt gut, eine andere eigene Seite erreicht die Schwelle und passt semantisch deutlich besser "
        "(mehr als der Abstand 'fast gleich'), die rankende erreicht die Schwelle nicht, "
        "oder die rankende URL steht nicht im Frog-Export."
    ),
    L.V_WATCH: "Die Query rankt gut, obwohl keine Seite die Schwelle erreicht.",
    L.V_USE: "Kein gutes Ranking, aber eine passende Seite existiert.",
}

_FILLS = {
    L.V_MATCH: "C6EFCE",
    L.V_OK: "C6EFCE",
    L.V_USE: "FFEB9C",
    L.V_CANNIBAL: "F8CBAD",
    L.V_GAP: "FFC7CE",
    L.V_WATCH: "D9D9D9",
}

DISCLAIMER = (
    "Das Notebook sortiert vor und begründet. Cosinus-Werte sind Hinweise, keine Urteile: "
    "die Entscheidung trifft ein Mensch nach Prüfung der Seite."
)

CAVEAT_SCORES = (
    "Scores eines Modells liegen in einem engen Band (bei e5 etwa 0,7 bis 0,9). "
    "Sie sind nur innerhalb eines Laufs vergleichbar, nicht zwischen Modellen."
)
# Herkunft der Schwelle -> fester Hinweis in der Lesehilfe
CAVEAT_THRESHOLD = {
    "median": (
        "Schwelle nicht kalibriert: Sie ist der mittlere beste Score, etwa die Hälfte der Queries liegt per "
        "Konstruktion darunter. Urteile und die Zahl neuer Seiten sind nur mit selbst geprüfter Schwelle belastbar."
    ),
    "rankings": (
        "Schwelle aus Rankings kalibriert: Sie ist so gewählt, dass 75 % der gut rankenden Paare sie erreichen. "
        "Bis zu ein Viertel der gut rankenden Queries erscheint daher als 'Rankt trotz schwachem Match'."
    ),
    "manuell": (
        "Schwelle von Hand gesetzt: Sie wurde im Notebook eingetragen und nicht aus den Daten abgeleitet. "
        "Urteile und die Zahl neuer Seiten hängen direkt von diesem Wert ab."
    ),
}


def content_gaps(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions[decisions[L.C_VERDICT] == L.V_GAP].reset_index(drop=True)


def _readme(sheet_names, settings, present_columns, threshold_source=None) -> pd.DataFrame:
    rows = [("Hinweis", "Einordnung", DISCLAIMER), ("Hinweis", "Scores", CAVEAT_SCORES)]
    if threshold_source in CAVEAT_THRESHOLD:
        rows.append(("Hinweis", "Schwelle", CAVEAT_THRESHOLD[threshold_source]))
    rows += [("Blatt", name, _SHEET_HELP[name]) for name in sheet_names]
    rows += [("Urteil", verdict, text) for verdict, text in VERDICT_HELP.items()]
    rows += [("Stufe", stage, text) for stage, text in _STAGE_HELP.items()]
    rows += [("Spalte", column, text) for column, text in _COLUMN_HELP.items() if column in present_columns]
    rows += [("Einstellung", key, str(value)) for key, value in settings.items()]
    return pd.DataFrame(rows, columns=[L.R_AREA, L.R_ENTRY, L.R_TEXT])


def build_sheets(decisions, top, cannibal, settings, threshold_source=None) -> dict:
    """threshold_source: "rankings", "median" oder "manuell" (Herkunft der Schwelle für die Lesehilfe)."""
    sheets = {
        SHEET_DECISION: decisions,
        SHEET_TOP: top,
        SHEET_CANNIBAL: cannibal,
        SHEET_GAPS: content_gaps(decisions),
    }
    if L.C_CLUSTER in decisions.columns:
        sheets[SHEET_GAP_CLUSTERS] = gap_summary(decisions)
    present = {column for df in sheets.values() for column in df.columns}
    return {SHEET_README: _readme(list(sheets), settings, present, threshold_source), **sheets}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Steuerzeichen entfernen, die openpyxl nicht schreiben kann. Das Original bleibt unverändert."""
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].map(lambda v: ILLEGAL_CHARACTERS_RE.sub(" ", v) if isinstance(v, str) else v)
    return out


_FORMULA_START = ("=", "+", "-", "@")


def write_excel(path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df = _clean(df)
            df.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for row in sheet.iter_rows():
                for cell in row:
                    # Text wie "=cmd|x" bleibt Text und wird nicht als Formel ausgeführt
                    if isinstance(cell.value, str) and cell.value.startswith(_FORMULA_START):
                        cell.data_type = "s"
            for idx, column in enumerate(df.columns, start=1):
                longest = max([len(str(column))] + [len(str(v)) for v in df[column].head(200)])
                sheet.column_dimensions[get_column_letter(idx)].width = min(max(12, longest + 2), 70)
            if L.C_VERDICT in df.columns:
                col = list(df.columns).index(L.C_VERDICT) + 1
                for row, verdict in enumerate(df[L.C_VERDICT], start=2):
                    colour = _FILLS.get(verdict)
                    if colour:
                        sheet.cell(row=row, column=col).fill = PatternFill("solid", fgColor=colour)


_PLAIN_NUMBER = re.compile(r"^-?\d+\.\d+$")
_NUMBER_IN_TEXT = re.compile(r"(Score|Position) (-?\d+)\.(\d+)")


def _comma_in_text(name, df: pd.DataFrame) -> pd.DataFrame:
    """Zahlen, die als Text vorliegen, mit Dezimalkomma (sonst liest deutsches Excel "4.3" als Datum)."""
    out = df.copy()

    def swap(value):
        return value.replace(".", ",") if isinstance(value, str) else value

    if L.C_POSITION in out.columns:
        out[L.C_POSITION] = out[L.C_POSITION].map(swap)
    for column in (L.C_COMPETING,):
        if column in out.columns:
            out[column] = out[column].map(lambda v: _NUMBER_IN_TEXT.sub(r"\1 \2,\3", v) if isinstance(v, str) else v)
    if name == SHEET_README:
        numeric = (out[L.R_AREA] == "Einstellung") & out[L.R_TEXT].map(lambda v: bool(_PLAIN_NUMBER.match(str(v))))
        out.loc[numeric, L.R_TEXT] = out.loc[numeric, L.R_TEXT].map(swap)
    return out


def write_csv_zip(path, sheets: dict, sep: str = ";") -> None:
    # Semikolon ist das deutsche Excel-Format: dort gehört das Komma in die Zahl
    decimal = "," if sep == ";" else "."
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, df in sheets.items():
            if decimal == ",":
                df = _comma_in_text(name, df)
            archive.writestr(_CSV_NAMES[name], df.to_csv(index=False, sep=sep, decimal=decimal).encode("utf-8-sig"))
