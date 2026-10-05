import io
from dataclasses import dataclass

import pandas as pd

from .normalize import host_of, normalize_query, normalize_url

QUERY_ALIASES = [
    "query",
    "queries",
    "keyword",
    "keywords",
    "suchanfrage",
    "suchanfragen",
    "häufigste suchanfragen",
    "suchbegriff",
    "prompt",
    "top queries",
    "search query",
    "search term",
]
KEYWORD_ALIASES = [
    "keyword",
    "query",
    "top queries",
    "suchanfrage",
    "suchanfragen",
    "häufigste suchanfragen",
    "suchbegriff",
    "search query",
    "search term",
    "prompt",
]
URL_ALIASES = [
    "url",
    "address",
    "adresse",
    "page",
    "seite",
    "seiten",
    "die häufigsten seiten",
    "current url",
    "landing page",
    "landingpage",
    "target url",
]
CONTENT_ALIASES = [
    "extract main content 1",
    "main content",
    "content",
    "text",
    "body text",
    "seiteninhalt",
    "inhalt",
]
POSITION_ALIASES = ["position", "current position", "avg. position", "average position", "pos"]
TYPE_ALIASES = ["type"]


class IngestError(ValueError):
    """Fehler beim Einlesen, mit einer Meldung für den Nutzer."""


@dataclass
class ContentTable:
    urls: list
    contents: list
    skipped_empty: int
    skipped_duplicate: int


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


# "NA", "null", "nan" usw. bleiben Text, nur leere Zellen fehlen
_NA = {"keep_default_na": False, "na_values": [""]}


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), **_NA)
    if not name.endswith((".csv", ".tsv", ".txt")):
        raise IngestError(f"Format von '{filename}' wird nicht unterstützt. Erlaubt sind CSV, TSV und XLSX.")
    text = _decode(data)
    first = next((line for line in io.StringIO(text) if line.strip()), None)
    if first is None:
        raise IngestError(f"Die Datei '{filename}' ist leer.")
    sep = max(["\t", ";", ","], key=first.count)
    if first.count(sep) == 0:
        clean = [line.strip().strip('"') for line in text.splitlines() if line.strip()]
        return pd.DataFrame({clean[0]: clean[1:]})
    return pd.read_csv(io.StringIO(text), sep=sep, **_NA)


def find_column(df: pd.DataFrame, aliases: list) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def _require(df, given, aliases, what):
    if given:
        if given not in df.columns:
            raise IngestError(f"Spalte '{given}' gibt es nicht. Gefundene Spalten: {list(df.columns)}")
        return given
    found = find_column(df, aliases)
    if found is None:
        raise IngestError(
            f"{what}-Spalte nicht erkannt. Gefundene Spalten: {list(df.columns)}. "
            f"Trage den Spaltennamen im Formular ein."
        )
    return found


def _blank(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def load_queries(df: pd.DataFrame, column: str | None = None) -> list:
    col = _require(df, column, [], "Query") if column else find_column(df, QUERY_ALIASES)
    if col is None and len(df.columns) > 1:
        col = _require(df, None, QUERY_ALIASES, "Query")
    if col is None:
        # keine Kopfzeile: die erste Zeile ist selbst eine Query
        col = df.columns[0]
        values = [col] + df[col].tolist()
    else:
        values = df[col].tolist()
    seen, out = set(), []
    for value in values:
        if _blank(value):
            continue
        key = normalize_query(value)
        if key not in seen:
            seen.add(key)
            out.append(str(value).strip())
    return out


def load_content(df: pd.DataFrame, url_col=None, content_col=None) -> ContentTable:
    ucol = _require(df, url_col, URL_ALIASES, "URL")
    ccol = _require(df, content_col, CONTENT_ALIASES, "Content")
    urls, contents, seen = [], [], set()
    skipped_empty = skipped_duplicate = 0
    for url, content in zip(df[ucol], df[ccol]):
        if _blank(url):
            continue
        if _blank(content):
            skipped_empty += 1
            continue
        key = normalize_url(url)
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)
        urls.append(str(url).strip())
        contents.append(str(content))
    return ContentTable(urls, contents, skipped_empty, skipped_duplicate)


NO_USABLE_ROWS = "Die Datei enthält keine verwertbaren Zeilen (Keyword, URL und Position müssen gefüllt sein)."


def _ranking_frame(df, kcol, ucol, pcol, keep_keyword=False):
    # Zeilen ohne Keyword zuerst entfernen; astype(bool), weil die Maske bei leeren Tabellen object ist
    df = df[~df[kcol].map(_blank).astype(bool)]
    out = pd.DataFrame(
        {
            "keyword": df[kcol].astype(str).str.strip(),
            "query_norm": df[kcol].map(normalize_query),
            "url": df[ucol].map(lambda u: None if _blank(u) else str(u).strip()),
            "position": pd.to_numeric(df[pcol].astype(str).str.replace(",", "."), errors="coerce"),
        }
    )
    out = out.dropna(subset=["url", "position"])
    if out.empty:
        raise IngestError(NO_USABLE_ROWS)
    out["url_norm"] = out["url"].map(normalize_url)
    out["position"] = out["position"].astype(float)
    cols = ["query_norm", "url", "url_norm", "position"]
    if keep_keyword:
        cols = ["keyword"] + cols
    return out[cols].reset_index(drop=True)


def load_rankings(df, keyword_col=None, url_col=None, position_col=None) -> pd.DataFrame:
    kcol = _require(df, keyword_col, KEYWORD_ALIASES, "Keyword")
    ucol = _require(df, url_col, URL_ALIASES, "URL")
    pcol = _require(df, position_col, POSITION_ALIASES, "Position")
    return _ranking_frame(df, kcol, ucol, pcol)


def load_serps(df, keyword_col=None, url_col=None, position_col=None, type_col=None) -> pd.DataFrame:
    kcol = _require(df, keyword_col, KEYWORD_ALIASES, "Keyword")
    ucol = _require(df, url_col, URL_ALIASES, "URL")
    pcol = _require(df, position_col, POSITION_ALIASES, "Position")
    tcol = type_col or find_column(df, TYPE_ALIASES)
    if tcol is not None:
        organic = df[df[tcol].astype(str).str.lower().str.contains("organic", na=False)]
        if organic.empty and not df.empty:
            raise IngestError(
                f"Die Spalte {tcol} enthält keinen Wert 'organic'. Ausgewertet werden nur organische Ergebnisse. "
                "Prüfe, ob die Datei ein SERP-Export mit organischen Treffern ist."
            )
        df = organic
    return _ranking_frame(df, kcol, ucol, pcol, keep_keyword=True)


def own_rankings_from_serps(serps: pd.DataFrame, own_hosts: set) -> pd.DataFrame:
    mask = serps["url"].map(host_of).isin(own_hosts)
    return serps.loc[mask, ["query_norm", "url", "url_norm", "position"]].reset_index(drop=True)
