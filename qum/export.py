import re
import zipfile

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from . import labels as L
from .cannibal import COLUMNS as CANNIBAL_COLUMNS
from .gaps import COLUMNS as GAP_COLUMNS
from .verdict import OVERVIEW_COLUMNS

SHEET_OVERVIEW = "Übersicht"
SHEET_CANNIBAL = "Kannibalisierungsgefahr"
SHEET_GAPS = "Potentielle Content-Lücken"
SHEET_README = "Lesehilfe"

_CSV_NAMES = {
    SHEET_OVERVIEW: "uebersicht.csv",
    SHEET_CANNIBAL: "kannibalisierungsgefahr.csv",
    SHEET_GAPS: "potentielle_content_luecken.csv",
    SHEET_README: "lesehilfe.csv",
}

_SHEET_HELP = {
    SHEET_OVERVIEW: (
        "Eine Zeile je Query: die drei am besten passenden URLs mit ihren Scores, die beste eigene Rankingposition, "
        "das Urteil und ob die Query im Blatt Kannibalisierungsgefahr steht."
    ),
    SHEET_CANNIBAL: (
        "Queries, bei denen mehrere eigene Seiten konkurrieren, getrennt nach Stufe, mit bis zu drei URLs. "
        f"Enthält jede Query mit dem Urteil '{L.V_CANNIBAL}'."
    ),
    SHEET_GAPS: (
        "Queries, deren bester Score unter der Lücken-Schwelle liegt, mit den eigenen Einstellungen luecke_unter_score "
        "und luecke_nur_ohne_ranking_bis_position aus Schritt 7c. Wegen dieser eigenen Einstellungen kann die Zahl der "
        f"Zeilen von der Zahl der Urteile '{L.V_GAP}' abweichen."
    ),
    SHEET_README: "Diese Erklärungen: Hinweise, Blätter, Urteile, Stufen, Spalten und die Einstellungen des Laufs.",
}

_COLUMN_HELP = {
    # Übersicht
    L.C_QUERY: "Die Suchanfrage, um die es in der Zeile geht.",
    L.C_BEST_URL: "Die Seite mit dem höchsten Score für die Query, nach der gewählten Bewertungsgrundlage.",
    L.C_CHUNK: "Der Textblock der besten URL, der zur Query am besten passt.",
    L.C_S_CHUNK: "Cosinus-Ähnlichkeit zwischen Query und dem am besten passenden Textblock der besten URL.",
    L.C_S_FULL: "Cosinus-Ähnlichkeit zwischen Query und dem gesamten Main Content der besten URL.",
    L.C_S_COMBI: "Gewichtete Mischung aus Chunk-Score und Gesamt-URL-Score der besten URL.",
    L.C_LEAD_GAP: (
        "Score der besten URL minus Score der zweitbesten URL, nach der gewählten Bewertungsgrundlage. "
        "Ein großer Vorsprung heißt: eine Seite sticht klar heraus, auch wenn der absolute Score niedrig ist. "
        "Leer, wenn es nur eine URL gibt."
    ),
    L.C_SECOND_URL: "Die Seite mit dem zweithöchsten Score für die Query, nach der gewählten Bewertungsgrundlage.",
    L.C_S_CHUNK_2: "Chunk-Score der zweitbesten URL.",
    L.C_S_FULL_2: "Gesamt-URL-Score der zweitbesten URL.",
    L.C_S_COMBI_2: "Kombi-Score der zweitbesten URL.",
    L.C_THIRD_URL: "Die Seite mit dem dritthöchsten Score für die Query, nach der gewählten Bewertungsgrundlage.",
    L.C_S_CHUNK_3: "Chunk-Score der drittbesten URL.",
    L.C_S_FULL_3: "Gesamt-URL-Score der drittbesten URL.",
    L.C_S_COMBI_3: "Kombi-Score der drittbesten URL.",
    L.C_POSITION: "Beste eigene Position für die Query laut Rankings. Leer, wenn die Query nicht rankt oder keine Rankings geladen sind.",
    L.C_RANK_URL: "Die eigene URL, die laut Rankings am besten für die Query rankt, egal auf welcher Position.",
    L.C_RANK_IS_BEST: (
        f"'{L.YES}': die rankende URL ist die beste URL. '{L.NO}': eine andere Seite hat einen höheren Score, auch wenn "
        "sie nur knapp vorn liegt; wie knapp, zeigen die Scores und der Grund im Blatt Kannibalisierungsgefahr. "
        f"'{L.CMP_NOT_RANKING}': keine eigene URL rankt für die Query. '{L.CMP_NOT_IN_EXPORT}': die rankende URL "
        "steht nicht im Frog-Export und wurde nicht verglichen. Leer ohne Rankings."
    ),
    L.C_VERDICT: "Einordnung der Query, siehe die Urteile weiter oben in dieser Lesehilfe.",
    L.C_CANNIBAL: (
        f"'{L.YES}': die Query steht im Blatt Kannibalisierungsgefahr mit der Stufe Gefahr oder Kannibalisierung bereits sichtbar. "
        f"'{L.MAYBE}': nur mit der Stufe Möglich, weitere Seiten passen, liegen aber deutlich dahinter. "
        f"'{L.NO}': keine weitere Seite erreicht die Schwelle. Unabhängig vom Urteil."
    ),
    # Kannibalisierungsgefahr
    L.C_STAGE: "Stufe der Kannibalisierungsgefahr, siehe die Stufen weiter oben in dieser Lesehilfe.",
    L.C_REASON: "Warum die Query hier steht. Konkurrieren mehr als drei URLs, steht die Zahl der weiteren am Ende.",
    L.C_URL_1: (
        "Erste konkurrierende eigene URL. Stufe Gefahr: nach Score sortiert, die rankende URL zuerst, wenn der Grund sie "
        "betrifft. Stufe Kannibalisierung bereits sichtbar: nach Position sortiert. Mehr als drei URLs zählt die Spalte Grund als weitere."
    ),
    L.C_SCORE_1: "Score der URL 1 nach der gewählten Bewertungsgrundlage. Leer, wenn sie nicht im Frog-Export steht.",
    L.C_POS_1: "Eigene Position der URL 1 für die Query. Leer, wenn sie dafür nicht rankt.",
    L.C_URL_2: "Zweite konkurrierende eigene URL.",
    L.C_SCORE_2: "Score der URL 2 nach der gewählten Bewertungsgrundlage. Leer, wenn sie nicht im Frog-Export steht.",
    L.C_POS_2: "Eigene Position der URL 2 für die Query. Leer, wenn sie dafür nicht rankt.",
    L.C_URL_3: "Dritte konkurrierende eigene URL. Leer, wenn nur zwei URLs konkurrieren.",
    L.C_SCORE_3: "Score der URL 3 nach der gewählten Bewertungsgrundlage. Leer, wenn sie nicht im Frog-Export steht.",
    L.C_POS_3: "Eigene Position der URL 3 für die Query. Leer, wenn sie dafür nicht rankt.",
    # Potentielle Content-Lücken
    L.C_BEST_SCORE: "Score der besten URL nach der gewählten Bewertungsgrundlage. Er liegt unter der Lücken-Schwelle.",
    L.C_TOPIC: (
        "Nummer des SERP-Clusters aus Schritt 6: Queries mit gleicher Nummer haben stark überlappende Google-Ergebnisse "
        "und ergeben zusammen eine neue Seite. Leer ohne SERPs oder wenn die Query in keinem Cluster ist."
    ),
}

_STAGE_HELP = {
    L.STAGE_DANGER: (
        "Mehrere eigene Seiten passen semantisch fast gleich gut, oder bei gutem Ranking passt eine andere eigene Seite "
        "besser als die rankende (oder die rankende URL steht nicht im Frog-Export). In den Rankings ist das noch nicht "
        "sichtbar, Google kann die rankende Seite aber wechseln."
    ),
    L.STAGE_POSSIBLE: (
        "Weitere eigene Seiten erreichen die Schwelle, liegen aber deutlich hinter der besten (mehr als der Abstand "
        "'fast gleich'). Kein akuter Konflikt, aber Seiten, die man beim Ausbau im Blick behalten sollte."
    ),
    L.STAGE_VISIBLE: "Mehrere eigene Seiten ranken bereits für die Query (bis zur Position aus sichtbar_bis_position).",
}

# "passt klar": eine Seite erreicht die Schwelle und liegt mehr als den Abstand 'fast gleich' vor jeder anderen passenden
VERDICT_HELP = {
    L.V_MATCH: (
        "Genau eine Seite passt klar: Sie erreicht die Schwelle und liegt deutlich vor allen anderen. Die Query rankt "
        "schwach, gar nicht, oder es sind keine Rankings geladen. Diese Seite ausbauen und intern stärken statt neu bauen."
    ),
    L.V_GAP: "Keine Seite erreicht die Schwelle. Die Query rankt nicht gut, oder es sind keine Rankings geladen.",
    L.V_OK: (
        "Die Query rankt gut, und die rankende Seite erreicht die Schwelle. Steht die rankende URL nicht im Frog-Export, "
        "kann das Tool sie nicht prüfen und wertet ebenfalls 'In Ordnung'. Passt eine andere eigene Seite besser oder "
        "fast gleich gut, steht in der Spalte Kannibalisierungsgefahr 'ja': Heute rankt die richtige Seite, Google kann "
        "aber wechseln."
    ),
    L.V_CANNIBAL: (
        "Mehrere eigene Seiten erreichen die Schwelle und passen fast gleich gut, und die Query rankt schwach, gar nicht, "
        "oder es sind keine Rankings geladen. Details im Blatt Kannibalisierungsgefahr."
    ),
    L.V_WATCH: (
        "Die Query rankt gut, aber die rankende Seite erreicht die Schwelle nicht. Passt eine andere eigene Seite, steht "
        "in der Spalte Kannibalisierungsgefahr 'ja': prüfen, welche Seite die richtige ist. Passt gar keine Seite, die "
        "Schwelle prüfen."
    ),
}

_FILLS = {
    L.V_MATCH: "FFEB9C",
    L.V_OK: "C6EFCE",
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


def _gap_sheet_help(new_pages) -> str:
    if new_pages is None:
        return f"{_SHEET_HELP[SHEET_GAPS]} Ohne SERPs (Schritt 6) gibt es kein Thema, jede Query zählt einzeln."
    return f"{_SHEET_HELP[SHEET_GAPS]} Neue Seiten: {new_pages} (eine je Thema, Queries ohne Thema einzeln)."


def _readme(sheets, settings, threshold_source, new_pages) -> pd.DataFrame:
    rows = [("Hinweis", "Einordnung", DISCLAIMER), ("Hinweis", "Scores", CAVEAT_SCORES)]
    if threshold_source in CAVEAT_THRESHOLD:
        rows.append(("Hinweis", "Schwelle", CAVEAT_THRESHOLD[threshold_source]))
    for name in list(sheets) + [SHEET_README]:
        text = _gap_sheet_help(new_pages) if name == SHEET_GAPS else _SHEET_HELP[name]
        rows.append(("Blatt", name, text))
    rows += [("Urteil", verdict, text) for verdict, text in VERDICT_HELP.items()]
    rows += [("Stufe", stage, text) for stage, text in _STAGE_HELP.items()]
    columns = dict.fromkeys(column for df in sheets.values() for column in df.columns)
    rows += [("Spalte", column, _COLUMN_HELP[column]) for column in columns]
    rows += [("Einstellung", key, str(value)) for key, value in settings.items()]
    return pd.DataFrame(rows, columns=[L.R_AREA, L.R_ENTRY, L.R_TEXT])


def build_sheets(overview, cannibal, gaps, settings, threshold_source=None, new_pages=None) -> dict:
    """threshold_source: "rankings", "median" oder "manuell" (Herkunft der Schwelle für die Lesehilfe).
    new_pages: Zahl neuer Seiten aus den Lücken, nur mit SERPs (sonst None)."""
    sheets = {
        SHEET_OVERVIEW: overview[OVERVIEW_COLUMNS],
        SHEET_CANNIBAL: cannibal[CANNIBAL_COLUMNS],
        SHEET_GAPS: gaps[GAP_COLUMNS],
    }
    return {**sheets, SHEET_README: _readme(sheets, settings, threshold_source, new_pages)}


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
# Positionen liegen als Text vor ("4.3"), damit ganze Zahlen ohne ".0" erscheinen
_POSITION_COLUMNS = (L.C_POSITION, L.C_POS_1, L.C_POS_2, L.C_POS_3)


def _comma_in_text(name, df: pd.DataFrame) -> pd.DataFrame:
    """Zahlen, die als Text vorliegen, mit Dezimalkomma (sonst liest deutsches Excel "4.3" als Datum)."""
    out = df.copy()

    def swap(value):
        return value.replace(".", ",") if isinstance(value, str) else value

    for column in _POSITION_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(swap)
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
