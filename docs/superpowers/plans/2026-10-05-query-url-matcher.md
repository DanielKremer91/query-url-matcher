# Query-URL Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein Colab-Notebook mit getestetem Python-Paket `qum`, das Queries per Embeddings gegen den Main Content eigener Seiten matcht und daraus Urteile zu Kannibalisierung und Content-Lücken ableitet.

**Architecture:** Die gesamte Logik liegt im Paket `qum` (ein Modul je Aufgabe, ohne Notebook und ohne echtes Modell testbar). Das Notebook `query_url_matcher.ipynb` wird aus `tools/build_notebook.py` erzeugt und enthält nur Formulare, Erklärtexte und Aufrufe. Embedding-Anbieter hängen hinter einer gemeinsamen Schnittstelle `Embedder`; Tests nutzen einen `FakeEmbedder` oder handgebaute Score-Matrizen.

**Tech Stack:** Python ≥ 3.10, pandas, numpy, openpyxl, httpx, optional sentence-transformers, pytest. Google Colab als Laufzeit.

**Spec:** `docs/superpowers/specs/2026-10-05-query-url-matcher-design.md`

## Global Constraints

- Alle Texte, die der Nutzer sieht (Notebook, Meldungen, Spaltennamen, Export), sind auf Deutsch. Bezeichner im Code sind englisch.
- Spaltennamen, Urteile und Stufen kommen ausschließlich aus `qum/labels.py`. Keine String-Literale dafür in anderen Modulen.
- Kein generierter Text: Empfehlungen sind feste Textbausteine aus `qum/verdict.py`.
- Tests laufen ohne Netzwerk und ohne Modell-Download: `.venv/bin/pytest -q`.
- Prefixe `query: ` / `passage: ` nur bei den e5-Modellen. OpenAI bekommt im Notebook reinen Text.
- Gemini-Modell ist `gemini-embedding-001`. `text-embedding-004` darf nirgends vorkommen.
- API-Keys stehen nie im Notebook oder im Repo. Quelle sind Colab-Secrets `OPENAI_API_KEY` und `GEMINI_API_KEY` bzw. Umgebungsvariablen.
- Defaults: Top-N 5, Kombi-Gewicht 0,7, Kalibrierung Position ≤ 5, mindestens 20 Paare, „rankt gut" Position ≤ 10, Kannibalisierungs-Abstand 0,02, „bereits sichtbar" Position ≤ 20, SERP-Überschneidung 50 %, Cluster-Dichte 50 %, Warnung ab 50.000 Chunks.
- Kein `git push`. Commits bleiben lokal, bis Daniel den Push freigibt.
- Commit-Nachrichten enden mit `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `pyproject.toml` | Paketdefinition, Abhängigkeiten, pytest-Konfiguration |
| `qum/labels.py` | alle Spaltennamen, Urteile, Stufen |
| `qum/normalize.py` | Queries und URLs vergleichbar machen |
| `qum/ingest.py` | Dateien lesen, Spalten erkennen, Zeilen bereinigen |
| `qum/chunk.py` | Wort-Chunks |
| `qum/models.py` | Modellregister |
| `qum/embeddings/base.py` | Schnittstelle, Prefix-Logik, Normalisierung, HTTP mit Wiederholung |
| `qum/embeddings/cache.py` | Zwischenspeicher |
| `qum/embeddings/local.py` | sentence-transformers |
| `qum/embeddings/openai.py`, `gemini.py` | API-Anbieter |
| `qum/embeddings/__init__.py` | Fabrik `make_embedder` |
| `qum/match.py` | drei Scores, Ränge, Top-N |
| `qum/threshold.py` | Schwellen-Vorschlag, Beispielpaare |
| `qum/verdict.py` | Urteile, Textbausteine |
| `qum/cannibal.py` | Kannibalisierungs-Stufen |
| `qum/serp.py` | Clustering, Nachbar-Hinweis, Lücken-Zusammenfassung |
| `qum/pairs.py` | Paare bewerten |
| `qum/export.py` | Excel, CSV-ZIP, Lesehilfe |
| `qum/compare.py` | Modellvergleich (Messung) |
| `qum/colab.py` | dünne Colab-Helfer (Upload, Secrets, Download, Fehleranzeige) |
| `tools/build_notebook.py` | erzeugt `query_url_matcher.ipynb` |
| `tests/` | ein Testmodul je Paketmodul, `conftest.py` mit Test-Helfern |

---

### Task 1: Gerüst, Labels, Normalisierung

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `qum/__init__.py`, `qum/labels.py`, `qum/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_query(q: str) -> str`, `normalize_url(u: str) -> str`, `host_of(u: str) -> str`; alle Konstanten aus `qum/labels.py`.

- [ ] **Step 1: Gerüst anlegen**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "qum"
version = "0.1.0"
description = "Query-URL Matcher: Neue Seite bauen oder Bestehendes nutzen?"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "pandas>=2.2,<4",
    "numpy>=1.26,<3",
    "openpyxl>=3.1,<4",
    "httpx>=0.27,<1",
]

[project.optional-dependencies]
local = ["sentence-transformers>=3,<6"]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["qum*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`LICENSE`: MIT-Lizenztext, Zeile `Copyright (c) 2026 Daniel Kremer`.

`qum/__init__.py`:

```python
__version__ = "0.1.0"
```

`qum/labels.py`:

```python
"""Alle Bezeichnungen, die der Nutzer im Notebook und im Export sieht."""

C_QUERY = "Query"
C_URL = "URL"
C_CHUNK = "Relevanter Chunk"
C_S_CHUNK = "Score Chunk"
C_S_FULL = "Score Gesamt-URL"
C_S_COMBI = "Score Kombi"
C_R_CHUNK = "Rang Chunk"
C_R_FULL = "Rang Gesamt-URL"
C_R_COMBI = "Rang Kombi"
C_METHOD = "Methode Gesamt-URL"

C_VERDICT = "Urteil"
C_BEST_URL = "Beste URL"
C_RANK_URL = "Rankende URL"
C_POSITION = "Position"
C_NOTE = "Hinweis"
C_ADVICE = "Empfehlung"

C_CLUSTER = "Cluster"
C_CAND = "Nachbar-Kandidat"
C_CAND_KW = "Nachbar-Keyword"
C_CAND_OVERLAP = "SERP-Überschneidung"
C_CAND_SCORE = "Score Kandidat"
C_CAND_CHUNK = "Passage Kandidat"
C_CAND_POS = "Position Kandidat"
C_CAND_MORE = "Weitere Kandidaten"

C_STAGE = "Stufe"
C_REASON = "Grund"
C_COMPETING = "Konkurrierende URLs"

C_PAIR_RANK = "Rang der URL"
C_SIDE = "Lage"

FULLTEXT = "Volltext"
CHUNK_MEAN = "Mittelwert der Chunks"

V_MATCH = "Passende Seite vorhanden"
V_GAP = "Content-Lücke"
V_OK = "In Ordnung"
V_RISK = "Kannibalisierungs-Risiko"
V_WATCH = "Rankt trotz schwachem Match"
V_USE = "Bestehende Seite nutzen"
V_CHECK = "Vor Neuerstellung prüfen"

STAGE_RISK = "Risiko"
STAGE_VISIBLE = "Bereits sichtbar"

NOTE_NOT_IN_EXPORT = "Rankende URL steht nicht im Frog-Export"
NOTE_URL_MISSING = "URL steht nicht im Frog-Export"
NO_CLUSTER = "ohne Cluster"

# Auswahl im Formular -> interner Schlüssel
BASIS = {"Chunk": "chunk", "Gesamt-URL": "full", "Kombi": "combined"}
```

- [ ] **Step 2: Failing Test schreiben**

`tests/test_normalize.py`:

```python
from qum.normalize import host_of, normalize_query, normalize_url


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  Hundefutter   Getreidefrei ") == "hundefutter getreidefrei"


def test_normalize_url_strips_fragment_trailing_slash_and_tracking():
    url = "HTTPS://Www.Example.de/Ratgeber/?utm_source=x&gclid=1&farbe=rot#abschnitt"
    assert normalize_url(url) == "https://www.example.de/Ratgeber?farbe=rot"


def test_normalize_url_root_with_and_without_slash_are_equal():
    assert normalize_url("https://example.de/") == normalize_url("https://example.de")


def test_normalize_url_keeps_path_case():
    assert normalize_url("https://example.de/A") != normalize_url("https://example.de/a")


def test_host_of():
    assert host_of("https://WWW.Example.de/x") == "www.example.de"
```

- [ ] **Step 3: Umgebung einrichten und Test fehlschlagen sehen**

Run: `python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]" && .venv/bin/pytest tests/test_normalize.py -q`
Expected: FAIL mit `ModuleNotFoundError: No module named 'qum.normalize'`

- [ ] **Step 4: Implementieren**

`qum/normalize.py`:

```python
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = {"gclid", "fbclid", "msclkid", "srsltid"}


def normalize_query(q) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def normalize_url(u) -> str:
    parts = urlsplit(str(u).strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING)
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), "")
    )


def host_of(u) -> str:
    return urlsplit(str(u).strip()).netloc.lower()
```

- [ ] **Step 5: Test bestehen sehen**

Run: `.venv/bin/pytest tests/test_normalize.py -q`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml LICENSE qum tests
git commit -m "Add package scaffold, labels and normalisation"
```

---

### Task 2: Dateien einlesen

**Files:**
- Create: `qum/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `normalize_query`, `normalize_url`, `host_of`
- Produces:
  - `class IngestError(ValueError)`
  - `read_table(data: bytes, filename: str) -> pd.DataFrame`
  - `find_column(df, aliases: list[str]) -> str | None`
  - `load_queries(df, column: str | None = None) -> list[str]`
  - `@dataclass ContentTable(urls: list[str], contents: list[str], skipped_empty: int, skipped_duplicate: int)`
  - `load_content(df, url_col=None, content_col=None) -> ContentTable`
  - `load_rankings(df, keyword_col=None, url_col=None, position_col=None) -> pd.DataFrame` mit Spalten `query_norm, url, url_norm, position`
  - `load_serps(df, keyword_col=None, url_col=None, position_col=None, type_col=None) -> pd.DataFrame` mit Spalten `keyword, query_norm, url, url_norm, position` (nur organische Zeilen)
  - `own_rankings_from_serps(serps, own_hosts: set[str]) -> pd.DataFrame` im Format von `load_rankings`
  - Aliaslisten `QUERY_ALIASES, KEYWORD_ALIASES, URL_ALIASES, CONTENT_ALIASES, POSITION_ALIASES, TYPE_ALIASES`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_ingest.py`:

```python
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
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_ingest.py -q`
Expected: FAIL mit `ImportError` / `AttributeError`

- [ ] **Step 3: Implementieren**

`qum/ingest.py`:

```python
import io
from dataclasses import dataclass

import pandas as pd

from .normalize import host_of, normalize_query, normalize_url

QUERY_ALIASES = ["query", "queries", "keyword", "keywords", "suchanfrage", "prompt", "top queries", "search query"]
KEYWORD_ALIASES = ["keyword", "query", "top queries", "suchanfrage", "search query", "prompt"]
URL_ALIASES = ["url", "address", "adresse", "page", "seite", "current url", "landing page", "target url"]
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


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    if not name.endswith((".csv", ".tsv", ".txt")):
        raise IngestError(f"Format von '{filename}' wird nicht unterstützt. Erlaubt sind CSV, TSV und XLSX.")
    text = _decode(data)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise IngestError(f"Die Datei '{filename}' ist leer.")
    sep = max(["\t", ";", ","], key=lines[0].count)
    if lines[0].count(sep) == 0:
        clean = [line.strip().strip('"') for line in lines]
        return pd.DataFrame({clean[0]: clean[1:]})
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")


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


def _ranking_frame(df, kcol, ucol, pcol, keep_keyword=False):
    out = pd.DataFrame(
        {
            "keyword": df[kcol].astype(str).str.strip(),
            "query_norm": df[kcol].map(normalize_query),
            "url": df[ucol].map(lambda u: None if _blank(u) else str(u).strip()),
            "position": pd.to_numeric(df[pcol], errors="coerce"),
        }
    )
    out = out.dropna(subset=["url", "position"])
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
        df = df[df[tcol].astype(str).str.lower().str.contains("organic", na=False)]
    return _ranking_frame(df, kcol, ucol, pcol, keep_keyword=True)


def own_rankings_from_serps(serps: pd.DataFrame, own_hosts: set) -> pd.DataFrame:
    mask = serps["url"].map(host_of).isin(own_hosts)
    return serps.loc[mask, ["query_norm", "url", "url_norm", "position"]].reset_index(drop=True)
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_ingest.py -q`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add qum/ingest.py tests/test_ingest.py
git commit -m "Add file ingestion with column detection"
```

---

### Task 3: Chunking und Modellregister

**Files:**
- Create: `qum/chunk.py`, `qum/models.py`
- Test: `tests/test_chunk.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `chunk_words(text: str, size: int, overlap: int) -> list[str]`
  - `@dataclass(frozen=True) ModelSpec(key, label, provider, model_id, chunk_size, chunk_overlap, query_prefix="", passage_prefix="", query_task=None, passage_task=None, fulltext_max_words=None, max_seq_length=None, comparison_only=False)`; `provider` ist `"local"`, `"openai"` oder `"gemini"`
  - `MODELS: dict[str, ModelSpec]`, `DEFAULT_MODEL = "e5-large"`, `get_model(key) -> ModelSpec`, `model_by_label(label) -> ModelSpec`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_chunk.py`:

```python
import pytest

from qum.chunk import chunk_words


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_words("a  b\nc", 5, 1) == ["a b c"]


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_words(words(10), 4, 1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_last_chunk_may_be_shorter():
    chunks = chunk_words(words(6), 4, 1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_words("a b c", 3, 3)
```

`tests/test_models.py`:

```python
from qum.models import DEFAULT_MODEL, MODELS, get_model, model_by_label


def test_default_is_e5_large_with_prefixes():
    spec = get_model(DEFAULT_MODEL)
    assert spec.model_id == "intfloat/multilingual-e5-large"
    assert (spec.query_prefix, spec.passage_prefix) == ("query: ", "passage: ")


def test_only_e5_models_have_prefixes():
    with_prefix = {k for k, s in MODELS.items() if s.query_prefix or s.passage_prefix}
    assert with_prefix == {"e5-large", "e5-base"}


def test_gemini_uses_current_model_and_task_types():
    spec = get_model("gemini")
    assert spec.model_id == "gemini-embedding-001"
    assert (spec.query_task, spec.passage_task) == ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")


def test_retired_gemini_model_is_gone():
    assert all("text-embedding-004" not in s.model_id for s in MODELS.values())


def test_only_paraphrase_model_is_comparison_only():
    assert [k for k, s in MODELS.items() if s.comparison_only] == ["paraphrase-mpnet"]


def test_chunk_defaults_follow_spec():
    got = {k: (s.chunk_size, s.chunk_overlap) for k, s in MODELS.items()}
    assert got == {
        "e5-large": (250, 40),
        "e5-base": (250, 40),
        "bge-m3": (1000, 150),
        "msmarco": (250, 40),
        "gemini": (800, 120),
        "openai": (1500, 200),
        "paraphrase-mpnet": (250, 40),
    }


def test_labels_are_unique_and_resolvable():
    labels = [s.label for s in MODELS.values()]
    assert len(labels) == len(set(labels))
    assert model_by_label(labels[0]).key == "e5-large"
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_chunk.py tests/test_models.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/chunk.py`:

```python
def chunk_words(text: str, size: int, overlap: int) -> list:
    """Zerlegt Text in überlappende Wort-Chunks. Gibt immer mindestens einen Chunk zurück."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk-Größe muss > 0 sein und der Overlap kleiner als die Chunk-Größe.")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    chunks, start, step = [], 0, size - overlap
    while True:
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            return chunks
        start += step
```

`qum/models.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str  # "local" | "openai" | "gemini"
    model_id: str
    chunk_size: int
    chunk_overlap: int
    query_prefix: str = ""
    passage_prefix: str = ""
    query_task: str | None = None
    passage_task: str | None = None
    # API-Modelle: Volltext-Embedding bis zu dieser Wortzahl. Lokale Modelle prüfen per Tokenizer.
    fulltext_max_words: int | None = None
    max_seq_length: int | None = None
    comparison_only: bool = False


_SPECS = [
    ModelSpec(
        "e5-large",
        "multilingual-e5-large · Deutsch, kostenlos (Empfehlung)",
        "local",
        "intfloat/multilingual-e5-large",
        250,
        40,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    ModelSpec(
        "e5-base",
        "multilingual-e5-base · Deutsch, kostenlos, schneller",
        "local",
        "intfloat/multilingual-e5-base",
        250,
        40,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    ModelSpec("bge-m3", "bge-m3 · Deutsch, kostenlos, lange Texte", "local", "BAAI/bge-m3", 1000, 150),
    ModelSpec(
        "msmarco",
        "msmarco-distilbert-base-v4 · nur englische Projekte",
        "local",
        "msmarco-distilbert-base-v4",
        250,
        40,
    ),
    ModelSpec(
        "gemini",
        "Gemini gemini-embedding-001 · API-Key nötig",
        "gemini",
        "gemini-embedding-001",
        800,
        120,
        query_task="RETRIEVAL_QUERY",
        passage_task="RETRIEVAL_DOCUMENT",
        fulltext_max_words=1000,
    ),
    ModelSpec(
        "openai",
        "OpenAI text-embedding-3-large · API-Key nötig",
        "openai",
        "text-embedding-3-large",
        1500,
        200,
        fulltext_max_words=4000,
    ),
    ModelSpec(
        "paraphrase-mpnet",
        "paraphrase-multilingual-mpnet-base-v2 · symmetrisch, NUR zum Vergleich",
        "local",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        250,
        40,
        max_seq_length=512,
        comparison_only=True,
    ),
]

MODELS = {spec.key: spec for spec in _SPECS}
DEFAULT_MODEL = "e5-large"


def get_model(key: str) -> ModelSpec:
    return MODELS[key]


def model_by_label(label: str) -> ModelSpec:
    for spec in _SPECS:
        if spec.label == label:
            return spec
    raise KeyError(label)
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_chunk.py tests/test_models.py -q`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add qum/chunk.py qum/models.py tests/test_chunk.py tests/test_models.py
git commit -m "Add word chunking and model registry"
```

---

### Task 4: Embedding-Schnittstelle, Cache, Test-Embedder

**Files:**
- Create: `qum/embeddings/__init__.py` (zunächst leer), `qum/embeddings/base.py`, `qum/embeddings/cache.py`, `tests/conftest.py`
- Test: `tests/test_embeddings_base.py`

**Interfaces:**
- Consumes: `ModelSpec`
- Produces:
  - `class EmbeddingError(RuntimeError)`
  - `class Embedder` mit Attribut `spec: ModelSpec`, `embed(texts: list[str], role: str) -> np.ndarray` (Rolle `"query"` oder `"passage"`, Rückgabe L2-normalisiert, `float32`, Form `(n, d)`), `fits_context(text: str) -> bool`
  - `prepare(spec, texts, role) -> list[str]`
  - `l2_normalize(m) -> np.ndarray`
  - `post_json(client, url, headers, payload, attempts=3, sleep=time.sleep) -> dict`
  - `class CachedEmbedder(Embedder)` mit `__init__(inner, cache_dir=None)`
  - Test-Helfer in `tests/conftest.py`: `FakeEmbedder(spec=None, max_words=None)` mit Attribut `calls`, und `make_result(queries, urls, chunk_scores, full_scores=None)` (wird in Task 7 ergänzt)

- [ ] **Step 1: Test-Embedder anlegen**

`tests/conftest.py`:

```python
import zlib

import numpy as np

from qum.embeddings.base import Embedder, l2_normalize
from qum.models import ModelSpec

FAKE_SPEC = ModelSpec("fake", "Fake", "local", "fake-model", 5, 1)


class FakeEmbedder(Embedder):
    """Bag-of-Words-Vektoren: Texte mit gemeinsamen Wörtern sind ähnlich."""

    DIMS = 512

    def __init__(self, spec=None, max_words=None):
        self.spec = spec or FAKE_SPEC
        self.max_words = max_words
        self.calls = []

    def embed(self, texts, role):
        self.calls.append((list(texts), role))
        m = np.zeros((len(texts), self.DIMS), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                m[i, zlib.crc32(word.encode()) % self.DIMS] += 1
        return l2_normalize(m)

    def fits_context(self, text):
        return self.max_words is not None and len(text.split()) <= self.max_words
```

- [ ] **Step 2: Failing Tests schreiben**

`tests/test_embeddings_base.py`:

```python
import httpx
import numpy as np
import pytest

from qum.embeddings.base import EmbeddingError, l2_normalize, post_json, prepare
from qum.embeddings.cache import CachedEmbedder
from qum.models import get_model
from tests.conftest import FakeEmbedder


def test_prepare_adds_prefix_only_for_e5():
    e5 = get_model("e5-large")
    assert prepare(e5, ["futter"], "query") == ["query: futter"]
    assert prepare(e5, ["Ein Text"], "passage") == ["passage: Ein Text"]
    assert prepare(get_model("openai"), ["futter"], "query") == ["futter"]
    assert prepare(get_model("bge-m3"), ["Ein Text"], "passage") == ["Ein Text"]


def test_prepare_rejects_unknown_role():
    with pytest.raises(ValueError):
        prepare(get_model("e5-large"), ["x"], "document")


def test_l2_normalize_handles_zero_rows():
    out = l2_normalize([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


def _client(responses):
    calls = iter(responses)

    def handler(request):
        status, body = next(calls)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_json_retries_then_succeeds():
    sleeps = []
    client = _client([(500, {"e": 1}), (429, {"e": 2}), (200, {"ok": True})])
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append) == {"ok": True}
    assert sleeps == [1, 2]


def test_post_json_gives_up_after_three_attempts():
    client = _client([(500, {}), (500, {}), (500, {})])
    with pytest.raises(EmbeddingError, match="500"):
        post_json(client, "https://x.test", {}, {}, sleep=lambda s: None)


def test_post_json_does_not_retry_auth_errors():
    sleeps = []
    client = _client([(401, {"error": "bad key"})])
    with pytest.raises(EmbeddingError, match="401"):
        post_json(client, "https://x.test", {}, {}, sleep=sleeps.append)
    assert sleeps == []


def test_cache_embeds_each_text_once_per_role():
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    first = cached.embed(["a b", "c d", "a b"], "passage")
    second = cached.embed(["c d", "e f"], "passage")
    cached.embed(["a b"], "query")
    assert inner.calls == [(["a b", "c d"], "passage"), (["e f"], "passage"), (["a b"], "query")]
    assert first.shape == (3, FakeEmbedder.DIMS)
    assert np.allclose(first[1], second[0])


def test_cache_persists_to_disk(tmp_path):
    CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path).embed(["a b"], "passage")
    inner = FakeEmbedder()
    out = CachedEmbedder(inner, cache_dir=tmp_path).embed(["a b"], "passage")
    assert inner.calls == []
    assert out.shape == (1, FakeEmbedder.DIMS)


def test_cache_delegates_fits_context():
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c") is True
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c d") is False
```

Lege zusätzlich eine leere Datei `tests/__init__.py` an, damit `from tests.conftest import …` funktioniert.

- [ ] **Step 3: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_embeddings_base.py -q`
Expected: FAIL mit `ModuleNotFoundError: No module named 'qum.embeddings'`

- [ ] **Step 4: Implementieren**

`qum/embeddings/__init__.py`: leere Datei.

`qum/embeddings/base.py`:

```python
import time

import httpx
import numpy as np

from ..models import ModelSpec

_NO_RETRY = {400, 401, 403, 404}


class EmbeddingError(RuntimeError):
    """Fehler beim Erzeugen von Embeddings, mit einer Meldung für den Nutzer."""


class Embedder:
    spec: ModelSpec

    def embed(self, texts: list, role: str) -> np.ndarray:
        raise NotImplementedError

    def fits_context(self, text: str) -> bool:
        limit = self.spec.fulltext_max_words
        return limit is not None and len(text.split()) <= limit


def prepare(spec: ModelSpec, texts: list, role: str) -> list:
    if role not in ("query", "passage"):
        raise ValueError(f"Unbekannte Rolle: {role}")
    prefix = spec.query_prefix if role == "query" else spec.passage_prefix
    return [prefix + text for text in texts]


def l2_normalize(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def post_json(client: httpx.Client, url: str, headers: dict, payload: dict, attempts: int = 3, sleep=time.sleep) -> dict:
    error = None
    for attempt in range(attempts):
        try:
            response = client.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()
            error = EmbeddingError(f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code in _NO_RETRY:
                break
        except httpx.HTTPError as exc:
            error = EmbeddingError(f"Verbindungsfehler: {exc}")
        if attempt < attempts - 1:
            sleep(2**attempt)
    raise error
```

`qum/embeddings/cache.py`:

```python
import hashlib
from pathlib import Path

import numpy as np

from .base import Embedder


class CachedEmbedder(Embedder):
    """Bettet jeden Text je Rolle nur einmal ein. Optional auf Platte gesichert."""

    def __init__(self, inner: Embedder, cache_dir=None):
        self.inner = inner
        self.spec = inner.spec
        self._store = {}
        self._path = Path(cache_dir) / f"{self.spec.key}.npz" if cache_dir else None
        if self._path is not None and self._path.exists():
            data = np.load(self._path)
            self._store = dict(zip(data["keys"].tolist(), data["vectors"]))

    def _key(self, text: str, role: str) -> str:
        spec = self.spec
        raw = "\x00".join([spec.model_id, spec.query_prefix, spec.passage_prefix, role, text])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def embed(self, texts, role):
        keys = [self._key(text, role) for text in texts]
        missing = {}
        for key, text in zip(keys, texts):
            if key not in self._store and key not in missing:
                missing[key] = text
        if missing:
            vectors = self.inner.embed(list(missing.values()), role)
            self._store.update(zip(missing.keys(), vectors))
            self._save()
        return np.vstack([self._store[key] for key in keys])

    def fits_context(self, text):
        return self.inner.fits_context(text)

    def _save(self):
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, keys=np.array(list(self._store)), vectors=np.vstack(list(self._store.values())))
```

- [ ] **Step 5: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_embeddings_base.py -q`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add qum/embeddings tests/__init__.py tests/conftest.py tests/test_embeddings_base.py
git commit -m "Add embedder interface, retrying HTTP helper and cache"
```

---

### Task 5: Anbieter (lokal, OpenAI, Gemini) und Fabrik

**Files:**
- Create: `qum/embeddings/local.py`, `qum/embeddings/openai.py`, `qum/embeddings/gemini.py`
- Modify: `qum/embeddings/__init__.py`
- Test: `tests/test_embeddings_providers.py`

**Interfaces:**
- Consumes: `Embedder`, `prepare`, `l2_normalize`, `post_json`, `EmbeddingError`, `CachedEmbedder`, `ModelSpec`, `get_model`
- Produces:
  - `LocalEmbedder(spec, model=None)` mit zusätzlich `truncated_share(texts: list[str]) -> float`
  - `OpenAIEmbedder(spec, api_key, client=None, sleep=time.sleep, batch_size=32)`
  - `GeminiEmbedder(spec, api_key, client=None, sleep=time.sleep, batch_size=50)`
  - `make_embedder(model: str | ModelSpec, api_key: str | None = None, cache_dir=None) -> CachedEmbedder`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_embeddings_providers.py`:

```python
import json

import httpx
import numpy as np
import pytest

from qum.embeddings import make_embedder
from qum.embeddings.base import EmbeddingError
from qum.embeddings.cache import CachedEmbedder
from qum.embeddings.gemini import GeminiEmbedder
from qum.embeddings.local import LocalEmbedder
from qum.embeddings.openai import OpenAIEmbedder
from qum.models import get_model


class FakeSentenceModel:
    max_seq_length = 6

    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.seen.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])

    def tokenizer(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": text.split()}


def test_local_embedder_prefixes_e5_and_normalises():
    model = FakeSentenceModel()
    out = LocalEmbedder(get_model("e5-large"), model=model).embed(["futter"], "query")
    assert model.seen == [["query: futter"]]
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_local_embedder_sets_max_seq_length_for_comparison_model():
    model = FakeSentenceModel()
    LocalEmbedder(get_model("paraphrase-mpnet"), model=model)
    assert model.max_seq_length == 512


def test_local_fits_context_counts_tokens_including_prefix():
    emb = LocalEmbedder(get_model("e5-large"), model=FakeSentenceModel())
    assert emb.fits_context("a b c d e") is True  # "passage:" + 5 Wörter = 6 Tokens
    assert emb.fits_context("a b c d e f") is False


def test_local_truncated_share():
    emb = LocalEmbedder(get_model("e5-large"), model=FakeSentenceModel())
    assert emb.truncated_share(["a b", "a b c d e f g", "a", "a b c d e f g h"]) == 0.5


def _recording_client(make_response):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=make_response(json.loads(request.content)))

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_openai_sends_plain_text_in_batches_and_keeps_order():
    def respond(body):
        # absichtlich in umgekehrter Reihenfolge
        items = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return {"data": items[::-1]}

    client, requests = _recording_client(respond)
    emb = OpenAIEmbedder(get_model("openai"), "sk-test", client=client, batch_size=2)
    out = emb.embed(["a", "bbb", "cc"], "query")
    bodies = [json.loads(r.content) for r in requests]
    assert [b["input"] for b in bodies] == [["a", "bbb"], ["cc"]]
    assert bodies[0]["model"] == "text-embedding-3-large"
    assert requests[0].headers["authorization"] == "Bearer sk-test"
    assert out.shape == (3, 2)
    assert out[1, 0] > out[0, 0]  # "bbb" ist länger als "a": Reihenfolge stimmt


def test_openai_without_key_raises():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(get_model("openai"), "")


def test_gemini_sends_task_type_per_role():
    def respond(body):
        return {"embeddings": [{"values": [1.0, 2.0]} for _ in body["requests"]]}

    client, requests = _recording_client(respond)
    emb = GeminiEmbedder(get_model("gemini"), "g-key", client=client)
    emb.embed(["futter"], "query")
    emb.embed(["Ein Text"], "passage")
    first, second = (json.loads(r.content)["requests"][0] for r in requests)
    assert first == {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": "futter"}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    assert second["taskType"] == "RETRIEVAL_DOCUMENT"
    assert requests[0].headers["x-goog-api-key"] == "g-key"
    assert str(requests[0].url).endswith("/models/gemini-embedding-001:batchEmbedContents")


def test_factory_wraps_in_cache_and_requires_key():
    emb = make_embedder("openai", api_key="sk-test")
    assert isinstance(emb, CachedEmbedder)
    assert isinstance(emb.inner, OpenAIEmbedder)
    with pytest.raises(EmbeddingError):
        make_embedder("gemini", api_key=None)
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_embeddings_providers.py -q`
Expected: FAIL mit `ImportError`

- [ ] **Step 3: Implementieren**

`qum/embeddings/local.py`:

```python
import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, prepare


class LocalEmbedder(Embedder):
    def __init__(self, spec, model=None):
        self.spec = spec
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers ist nicht installiert. Führe Schritt 2 (Installation) erneut aus."
                ) from exc
            model = SentenceTransformer(spec.model_id)
        if spec.max_seq_length:
            model.max_seq_length = spec.max_seq_length
        self.model = model

    def embed(self, texts, role):
        vectors = self.model.encode(
            prepare(self.spec, texts, role),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return l2_normalize(np.asarray(vectors))

    def _tokens(self, text: str) -> int:
        prepared = prepare(self.spec, [text], "passage")[0]
        return len(self.model.tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"])

    def fits_context(self, text):
        limit = self.model.max_seq_length
        # ein Wort ist mindestens ein Token: lange Texte ohne Tokenizer ausschließen
        if len(text.split()) > limit:
            return False
        return self._tokens(text) <= limit

    def truncated_share(self, texts) -> float:
        if not texts:
            return 0.0
        return sum(not self.fits_context(text) for text in texts) / len(texts)
```

`qum/embeddings/openai.py`:

```python
import time

import httpx
import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, post_json, prepare

URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbedder(Embedder):
    def __init__(self, spec, api_key, client=None, sleep=time.sleep, batch_size=32):
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY fehlt. Lege ihn in Colab unter Secrets (Schlüssel-Symbol) an.")
        self.spec = spec
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._batch_size = batch_size

    def embed(self, texts, role):
        prepared = prepare(self.spec, texts, role)
        vectors = []
        for start in range(0, len(prepared), self._batch_size):
            batch = prepared[start : start + self._batch_size]
            data = post_json(
                self._client, URL, self._headers, {"model": self.spec.model_id, "input": batch}, sleep=self._sleep
            )
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in rows)
        return l2_normalize(np.array(vectors))
```

`qum/embeddings/gemini.py`:

```python
import time

import httpx
import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, post_json, prepare

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiEmbedder(Embedder):
    def __init__(self, spec, api_key, client=None, sleep=time.sleep, batch_size=50):
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY fehlt. Lege ihn in Colab unter Secrets (Schlüssel-Symbol) an.")
        self.spec = spec
        self._headers = {"x-goog-api-key": api_key}
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._batch_size = batch_size

    def embed(self, texts, role):
        prepared = prepare(self.spec, texts, role)
        task = self.spec.query_task if role == "query" else self.spec.passage_task
        model = f"models/{self.spec.model_id}"
        url = f"{BASE}/{self.spec.model_id}:batchEmbedContents"
        vectors = []
        for start in range(0, len(prepared), self._batch_size):
            requests = [
                {"model": model, "content": {"parts": [{"text": text}]}, "taskType": task}
                for text in prepared[start : start + self._batch_size]
            ]
            data = post_json(self._client, url, self._headers, {"requests": requests}, sleep=self._sleep)
            vectors.extend(item["values"] for item in data["embeddings"])
        return l2_normalize(np.array(vectors))
```

`qum/embeddings/__init__.py`:

```python
from ..models import ModelSpec, get_model
from .base import Embedder, EmbeddingError
from .cache import CachedEmbedder


def make_embedder(model, api_key=None, cache_dir=None) -> CachedEmbedder:
    """Erzeugt den Embedder zu einem Modellschlüssel oder einer ModelSpec, immer mit Cache."""
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    if spec.provider == "openai":
        from .openai import OpenAIEmbedder

        inner = OpenAIEmbedder(spec, api_key)
    elif spec.provider == "gemini":
        from .gemini import GeminiEmbedder

        inner = GeminiEmbedder(spec, api_key)
    else:
        from .local import LocalEmbedder

        inner = LocalEmbedder(spec)
    return CachedEmbedder(inner, cache_dir=cache_dir)


__all__ = ["Embedder", "EmbeddingError", "CachedEmbedder", "make_embedder"]
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_embeddings_providers.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add qum/embeddings tests/test_embeddings_providers.py
git commit -m "Add local, OpenAI and Gemini embedders with factory"
```

---

### Task 6: Matching

**Files:**
- Create: `qum/match.py`
- Modify: `tests/conftest.py` (Helfer `make_result` ergänzen)
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `chunk_words`, `Embedder`, `l2_normalize`, Labels
- Produces:
  - `@dataclass MatchResult(queries: list[str], urls: list[str], chunks: list[list[str]], chunk_scores: np.ndarray, full_scores: np.ndarray, best_chunk_idx: np.ndarray, full_method: list[str])`; Matrizen haben die Form `(len(queries), len(urls))`
    - `.combined(weight: float) -> np.ndarray`
    - `.lead(basis: str, weight: float = 0.7) -> np.ndarray`, `basis` ist `"chunk"`, `"full"` oder `"combined"`
    - `.best_chunk(i: int, j: int) -> str`
  - `run_matching(queries, urls, contents, embedder, chunk_size, chunk_overlap) -> MatchResult`
  - `ranks(scores: np.ndarray) -> np.ndarray` (1 = bester Wert je Zeile)
  - `top_hits(result, basis="chunk", weight=0.7, top_n=5) -> pd.DataFrame`
  - `estimate_chunks(contents, chunk_size, chunk_overlap) -> int`
  - Test-Helfer `make_result(queries, urls, chunk_scores, full_scores=None) -> MatchResult`

- [ ] **Step 1: Test-Helfer ergänzen**

An `tests/conftest.py` anhängen:

```python
from qum import labels as L
from qum.match import MatchResult


def make_result(queries, urls, chunk_scores, full_scores=None):
    """MatchResult aus handgebauten Scores. Jede URL hat genau einen Chunk 'Text <url>'."""
    chunk = np.array(chunk_scores, dtype=np.float32)
    full = chunk if full_scores is None else np.array(full_scores, dtype=np.float32)
    return MatchResult(
        queries=list(queries),
        urls=list(urls),
        chunks=[[f"Text {u}"] for u in urls],
        chunk_scores=chunk,
        full_scores=full,
        best_chunk_idx=np.zeros(chunk.shape, dtype=int),
        full_method=[L.FULLTEXT] * len(urls),
    )
```

- [ ] **Step 2: Failing Tests schreiben**

`tests/test_match.py`:

```python
import numpy as np
import pytest

from qum import labels as L
from qum.match import estimate_chunks, ranks, run_matching, top_hits
from tests.conftest import FakeEmbedder, make_result

URLS = ["https://a.de/hund", "https://a.de/katze"]
CONTENTS = [
    "hundefutter getreidefrei ist gut " + "fuellwort " * 8 + "napf reinigen tipps",
    "katzenfutter nass sorten",
]


def test_run_matching_shapes_and_best_url():
    result = run_matching(["hundefutter getreidefrei", "katzenfutter"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert result.chunk_scores.shape == (2, 2)
    assert result.chunk_scores[0].argmax() == 0
    assert result.chunk_scores[1].argmax() == 1


def test_best_chunk_is_the_matching_passage():
    result = run_matching(["napf reinigen"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert "napf reinigen" in result.best_chunk(0, 0)


def test_full_method_single_chunk_is_fulltext():
    result = run_matching(["x"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert result.full_method == [L.CHUNK_MEAN, L.FULLTEXT]


def test_full_method_uses_fulltext_when_it_fits():
    embedder = FakeEmbedder(max_words=100)
    result = run_matching(["x"], URLS, CONTENTS, embedder, 5, 1)
    assert result.full_method == [L.FULLTEXT, L.FULLTEXT]
    passages = [texts for texts, role in embedder.calls if role == "passage"]
    assert passages[-1] == [" ".join(CONTENTS[0].split())]


def test_chunk_mean_is_normalised():
    result = run_matching(["hundefutter getreidefrei"], URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert -1.0001 <= result.full_scores[0, 0] <= 1.0001
    assert result.full_scores[0, 0] < result.chunk_scores[0, 0]


def test_combined_and_lead():
    result = make_result(["q"], ["u1", "u2"], [[0.8, 0.2]], [[0.4, 0.6]])
    assert np.allclose(result.combined(0.7), [[0.68, 0.32]])
    assert np.allclose(result.lead("chunk"), [[0.8, 0.2]])
    assert np.allclose(result.lead("full"), [[0.4, 0.6]])
    assert np.allclose(result.lead("combined", 0.5), [[0.6, 0.4]])
    with pytest.raises(ValueError):
        result.lead("other")


def test_ranks():
    assert ranks(np.array([[0.1, 0.9, 0.5]])).tolist() == [[3, 1, 2]]


def test_top_hits_columns_order_and_ranks():
    result = make_result(["q"], ["u1", "u2", "u3"], [[0.5, 0.9, 0.7]], [[0.9, 0.1, 0.5]])
    df = top_hits(result, basis="chunk", weight=0.7, top_n=2)
    assert list(df.columns) == [
        L.C_QUERY, L.C_URL, L.C_CHUNK, L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI,
        L.C_R_CHUNK, L.C_R_FULL, L.C_R_COMBI, L.C_METHOD,
    ]
    assert df[L.C_URL].tolist() == ["u2", "u3"]
    assert df[L.C_R_CHUNK].tolist() == [1, 2]
    assert df[L.C_R_FULL].tolist() == [3, 2]
    assert df[L.C_S_CHUNK].tolist() == [0.9, 0.7]


def test_top_hits_sorted_by_selected_basis():
    result = make_result(["q"], ["u1", "u2"], [[0.5, 0.9]], [[0.9, 0.1]])
    assert top_hits(result, basis="full", top_n=1)[L.C_URL].tolist() == ["u1"]


def test_estimate_chunks():
    assert estimate_chunks(CONTENTS, 5, 1) == 5
```

- [ ] **Step 3: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_match.py -q`
Expected: FAIL mit `ModuleNotFoundError: No module named 'qum.match'`

- [ ] **Step 4: Implementieren**

`qum/match.py`:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .chunk import chunk_words
from .embeddings.base import l2_normalize


@dataclass
class MatchResult:
    queries: list
    urls: list
    chunks: list  # je URL die Liste ihrer Chunks
    chunk_scores: np.ndarray
    full_scores: np.ndarray
    best_chunk_idx: np.ndarray
    full_method: list

    def combined(self, weight: float) -> np.ndarray:
        return weight * self.chunk_scores + (1 - weight) * self.full_scores

    def lead(self, basis: str, weight: float = 0.7) -> np.ndarray:
        if basis == "chunk":
            return self.chunk_scores
        if basis == "full":
            return self.full_scores
        if basis == "combined":
            return self.combined(weight)
        raise ValueError(f"Unbekannte Bewertungsgrundlage: {basis}")

    def best_chunk(self, i: int, j: int) -> str:
        return self.chunks[j][int(self.best_chunk_idx[i, j])]


def estimate_chunks(contents, chunk_size, chunk_overlap) -> int:
    return sum(len(chunk_words(c, chunk_size, chunk_overlap)) for c in contents)


def run_matching(queries, urls, contents, embedder, chunk_size, chunk_overlap) -> MatchResult:
    chunks = [chunk_words(c, chunk_size, chunk_overlap) for c in contents]
    flat = [chunk for per_url in chunks for chunk in per_url]
    offsets = np.cumsum([0] + [len(per_url) for per_url in chunks])

    q = embedder.embed(list(queries), "query")
    c = embedder.embed(flat, "passage")
    sims = q @ c.T

    n_q, n_u = len(queries), len(urls)
    chunk_scores = np.zeros((n_q, n_u), dtype=np.float32)
    best = np.zeros((n_q, n_u), dtype=int)
    full_vecs = np.zeros((n_u, c.shape[1]), dtype=np.float32)
    methods, as_fulltext = [], []
    for u in range(n_u):
        lo, hi = offsets[u], offsets[u + 1]
        block = sims[:, lo:hi]
        best[:, u] = block.argmax(axis=1)
        chunk_scores[:, u] = block.max(axis=1)
        if hi - lo == 1:
            full_vecs[u] = c[lo]
            methods.append(L.FULLTEXT)
        elif embedder.fits_context(contents[u]):
            as_fulltext.append(u)
            methods.append(L.FULLTEXT)
        else:
            full_vecs[u] = c[lo:hi].mean(axis=0)
            methods.append(L.CHUNK_MEAN)
    if as_fulltext:
        full_vecs[as_fulltext] = embedder.embed([" ".join(contents[u].split()) for u in as_fulltext], "passage")
    full_scores = q @ l2_normalize(full_vecs).T

    return MatchResult(list(queries), list(urls), chunks, chunk_scores, full_scores, best, methods)


def ranks(scores: np.ndarray) -> np.ndarray:
    return (-scores).argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1


def top_hits(result: MatchResult, basis: str = "chunk", weight: float = 0.7, top_n: int = 5) -> pd.DataFrame:
    combined = result.combined(weight)
    lead = result.lead(basis, weight)
    r_chunk, r_full, r_combi = ranks(result.chunk_scores), ranks(result.full_scores), ranks(combined)
    rows = []
    for i, query in enumerate(result.queries):
        for j in np.argsort(-lead[i], kind="stable")[:top_n]:
            rows.append(
                {
                    L.C_QUERY: query,
                    L.C_URL: result.urls[j],
                    L.C_CHUNK: result.best_chunk(i, j),
                    L.C_S_CHUNK: round(float(result.chunk_scores[i, j]), 4),
                    L.C_S_FULL: round(float(result.full_scores[i, j]), 4),
                    L.C_S_COMBI: round(float(combined[i, j]), 4),
                    L.C_R_CHUNK: int(r_chunk[i, j]),
                    L.C_R_FULL: int(r_full[i, j]),
                    L.C_R_COMBI: int(r_combi[i, j]),
                    L.C_METHOD: result.full_method[j],
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_match.py -q`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add qum/match.py tests/conftest.py tests/test_match.py
git commit -m "Add matching with chunk, whole-URL and combined scores"
```

---

### Task 7: Schwelle

**Files:**
- Create: `qum/threshold.py`
- Test: `tests/test_threshold.py`

**Interfaces:**
- Consumes: `MatchResult`, `normalize_query`, `normalize_url`, Labels; Ranking-DataFrame mit `query_norm, url, url_norm, position`
- Produces:
  - `@dataclass ThresholdProposal(value: float, source: str, n_pairs: int)`; `source` ist `"rankings"` oder `"median"`
  - `propose_threshold(result, lead, rankings=None, max_position=5, min_pairs=20) -> ThresholdProposal`
  - `examples_around(result, lead, threshold, n=5) -> pd.DataFrame` mit Spalten `Lage, Query, URL, Relevanter Chunk, Score`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_threshold.py`:

```python
import numpy as np
import pandas as pd

from qum import labels as L
from qum.threshold import examples_around, propose_threshold
from tests.conftest import make_result


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def test_median_of_best_scores_without_rankings():
    result = make_result(["q1", "q2", "q3"], ["u1", "u2"], [[0.9, 0.1], [0.5, 0.2], [0.3, 0.7]])
    proposal = propose_threshold(result, result.lead("chunk"))
    assert proposal.source == "median"
    assert np.isclose(proposal.value, 0.7)


def test_rankings_give_25th_percentile_of_top_pairs():
    queries = [f"q{i}" for i in range(4)]
    result = make_result(queries, ["https://a.de/1"], [[0.4], [0.6], [0.8], [1.0]])
    rankings = _rankings([(q, "https://a.de/1", "https://a.de/1", 3.0) for q in queries])
    proposal = propose_threshold(result, result.lead("chunk"), rankings, max_position=5, min_pairs=4)
    assert proposal.source == "rankings"
    assert proposal.n_pairs == 4
    assert np.isclose(proposal.value, 0.55)


def test_rankings_ignore_bad_positions_and_unknown_urls():
    result = make_result(["q0", "q1"], ["https://a.de/1"], [[0.4], [0.6]])
    rankings = _rankings(
        [
            ("q0", "https://a.de/1", "https://a.de/1", 9.0),
            ("q1", "https://a.de/x", "https://a.de/x", 1.0),
        ]
    )
    proposal = propose_threshold(result, result.lead("chunk"), rankings, min_pairs=1)
    assert proposal.source == "median"
    assert proposal.n_pairs == 0


def test_too_few_pairs_fall_back_to_median():
    result = make_result(["q0"], ["https://a.de/1"], [[0.4]])
    rankings = _rankings([("q0", "https://a.de/1", "https://a.de/1", 1.0)])
    proposal = propose_threshold(result, result.lead("chunk"), rankings)
    assert proposal.source == "median"
    assert proposal.n_pairs == 1


def test_examples_around_threshold():
    result = make_result(["q1", "q2", "q3", "q4"], ["u1"], [[0.9], [0.62], [0.58], [0.1]])
    df = examples_around(result, result.lead("chunk"), 0.6, n=1)
    assert list(df.columns) == [L.C_SIDE, L.C_QUERY, L.C_URL, L.C_CHUNK, "Score"]
    assert df[L.C_QUERY].tolist() == ["q2", "q3"]
    assert df[L.C_SIDE].tolist() == ["knapp über der Schwelle", "knapp unter der Schwelle"]
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_threshold.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/threshold.py`:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url


@dataclass
class ThresholdProposal:
    value: float
    source: str  # "rankings" | "median"
    n_pairs: int


def propose_threshold(result, lead, rankings=None, max_position=5, min_pairs=20) -> ThresholdProposal:
    scores = []
    if rankings is not None:
        q_index = {normalize_query(q): i for i, q in enumerate(result.queries)}
        u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
        for row in rankings[rankings["position"] <= max_position].itertuples():
            i, j = q_index.get(row.query_norm), u_index.get(row.url_norm)
            if i is not None and j is not None:
                scores.append(float(lead[i, j]))
    if len(scores) >= min_pairs:
        return ThresholdProposal(float(np.percentile(scores, 25)), "rankings", len(scores))
    return ThresholdProposal(float(np.median(lead.max(axis=1))), "median", len(scores))


def examples_around(result, lead, threshold, n=5) -> pd.DataFrame:
    best = lead.argmax(axis=1)
    rows = [
        {
            L.C_QUERY: query,
            L.C_URL: result.urls[j],
            L.C_CHUNK: result.best_chunk(i, j),
            "Score": round(float(lead[i, j]), 4),
        }
        for i, (query, j) in enumerate(zip(result.queries, best))
    ]
    df = pd.DataFrame(rows)
    above = df[df["Score"] >= threshold].sort_values("Score").head(n).copy()
    below = df[df["Score"] < threshold].sort_values("Score", ascending=False).head(n).copy()
    above.insert(0, L.C_SIDE, "knapp über der Schwelle")
    below.insert(0, L.C_SIDE, "knapp unter der Schwelle")
    return pd.concat([above, below], ignore_index=True)
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_threshold.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add qum/threshold.py tests/test_threshold.py
git commit -m "Add threshold proposal and boundary examples"
```

---

### Task 8: Urteile

**Files:**
- Create: `qum/verdict.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Consumes: `MatchResult`, `normalize_query`, `normalize_url`, Labels
- Produces:
  - `build_decisions(result, lead, threshold, rankings=None, good_position=10, weight=0.7) -> pd.DataFrame`: eine Zeile je Query in der Reihenfolge von `result.queries`, Spalten `Query, Urteil, Beste URL, Relevanter Chunk, Score Chunk, Score Gesamt-URL, Score Kombi, Rankende URL, Position, Hinweis, Empfehlung`
  - `ADVICE: dict[str, str]` (Textbausteine je Urteil)
  - `format_position(p) -> str`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_verdict.py`:

```python
import pandas as pd

from qum import labels as L
from qum.verdict import ADVICE, build_decisions
from tests.conftest import make_result

U1, U2 = "https://a.de/1", "https://a.de/2"


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def test_without_rankings_match_or_gap():
    result = make_result(["passt", "fehlt"], [U1, U2], [[0.8, 0.1], [0.2, 0.3]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert df[L.C_VERDICT].tolist() == [L.V_MATCH, L.V_GAP]
    assert df[L.C_BEST_URL].tolist() == [U1, U2]
    assert df[L.C_RANK_URL].tolist() == ["", ""]


def test_columns():
    result = make_result(["q"], [U1], [[0.8]])
    df = build_decisions(result, result.lead("chunk"), 0.6)
    assert list(df.columns) == [
        L.C_QUERY, L.C_VERDICT, L.C_BEST_URL, L.C_CHUNK, L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI,
        L.C_RANK_URL, L.C_POSITION, L.C_NOTE, L.C_ADVICE,
    ]


def test_all_five_verdicts_with_rankings():
    queries = ["ok", "risiko", "beobachten", "nutzen", "luecke"]
    scores = [
        [0.9, 0.1],  # rankt gut mit U1, U1 ist bester Treffer
        [0.7, 0.9],  # rankt gut mit U1, U2 passt besser
        [0.2, 0.1],  # rankt gut mit U1, nichts passt
        [0.1, 0.8],  # rankt schwach, U2 passt
        [0.2, 0.3],  # kein Ranking, nichts passt
    ]
    rankings = _rankings(
        [
            ("ok", U1, U1, 2.0),
            ("risiko", U1, U1, 4.0),
            ("beobachten", U1, U1, 1.0),
            ("nutzen", U1, U1, 35.0),
        ]
    )
    result = make_result(queries, [U1, U2], scores)
    df = build_decisions(result, result.lead("chunk"), 0.6, rankings, good_position=10)
    assert df[L.C_VERDICT].tolist() == [L.V_OK, L.V_RISK, L.V_WATCH, L.V_USE, L.V_GAP]
    assert df[L.C_POSITION].tolist() == ["2", "4", "1", "35", ""]
    assert df[L.C_RANK_URL].tolist() == [U1, U1, U1, U1, ""]


def test_best_position_wins_when_several_urls_rank():
    rankings = _rankings([("q", U2, U2, 8.0), ("q", U1, U1, 3.0)])
    result = make_result(["q"], [U1, U2], [[0.9, 0.1]])
    df = build_decisions(result, result.lead("chunk"), 0.6, rankings)
    assert df.iloc[0][L.C_RANK_URL] == U1
    assert df.iloc[0][L.C_VERDICT] == L.V_OK


def test_ranking_url_missing_in_export_is_noted_and_treated_as_other_url():
    rankings = _rankings([("q", "https://a.de/alt", "https://a.de/alt", 2.0)])
    result = make_result(["q"], [U1], [[0.9]])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_VERDICT] == L.V_RISK
    assert row[L.C_NOTE] == L.NOTE_NOT_IN_EXPORT


def test_query_matching_is_case_insensitive():
    rankings = _rankings([("hunde futter", U1, U1, 2.0)])
    result = make_result(["Hunde  Futter"], [U1], [[0.9]])
    assert build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0][L.C_VERDICT] == L.V_OK


def test_advice_is_template_with_values():
    result = make_result(["q"], [U1, U2], [[0.1, 0.8]])
    rankings = _rankings([("q", U1, U1, 35.0)])
    row = build_decisions(result, result.lead("chunk"), 0.6, rankings).iloc[0]
    assert row[L.C_ADVICE] == ADVICE[L.V_USE].format(best=U2, rank_url=U1, position="35")
    assert U2 in row[L.C_ADVICE]


def test_every_verdict_has_advice():
    for verdict in [L.V_MATCH, L.V_GAP, L.V_OK, L.V_RISK, L.V_WATCH, L.V_USE, L.V_CHECK]:
        assert verdict in ADVICE
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_verdict.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/verdict.py`:

```python
import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url

# Feste Textbausteine. Platzhalter: {best}, {rank_url}, {position}
ADVICE = {
    L.V_MATCH: "Es gibt bereits eine passende Seite: {best}. Vor einer neuen Seite prüfen, ob sie ausgebaut werden kann.",
    L.V_GAP: "Keine passende Seite gefunden. Kandidat für eine neue Seite.",
    L.V_OK: "Die rankende Seite ist auch semantisch der beste Treffer. Kein Handlungsbedarf.",
    L.V_RISK: (
        "{rank_url} rankt auf Position {position}, semantisch passt {best} besser. "
        "Prüfen, welche Seite die Query bedienen soll, und die andere abgrenzen."
    ),
    L.V_WATCH: "{rank_url} rankt auf Position {position}, obwohl keine Seite semantisch gut passt. Beobachten.",
    L.V_USE: "Keine neue Seite bauen: {best} passt bereits. Seite ausbauen und intern stärken.",
    L.V_CHECK: (
        "Vor Neuerstellung prüfen: {best} bedient bereits ein Keyword mit fast gleicher SERP. "
        "Seite erweitern statt neu bauen?"
    ),
}


def format_position(position) -> str:
    if position is None or pd.isna(position):
        return ""
    return str(int(position)) if float(position).is_integer() else f"{position:.1f}"


def build_decisions(result, lead, threshold, rankings=None, good_position=10, weight=0.7) -> pd.DataFrame:
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    best_ranking = {}
    if rankings is not None:
        ordered = rankings.sort_values("position", kind="stable")
        for row in ordered.itertuples():
            best_ranking.setdefault(row.query_norm, row)
    combined = result.combined(weight)

    rows = []
    for i, query in enumerate(result.queries):
        j = int(lead[i].argmax())
        fits = lead[i, j] >= threshold
        hit = best_ranking.get(normalize_query(query))
        rank_url = hit.url if hit is not None else ""
        position = format_position(hit.position) if hit is not None else ""
        note = ""
        if rankings is None:
            verdict = L.V_MATCH if fits else L.V_GAP
        elif hit is not None and hit.position <= good_position:
            ranking_j = u_index.get(hit.url_norm)
            if ranking_j is None:
                note = L.NOTE_NOT_IN_EXPORT
            if not fits:
                verdict = L.V_WATCH
            elif ranking_j == j:
                verdict = L.V_OK
            else:
                verdict = L.V_RISK
        else:
            verdict = L.V_USE if fits else L.V_GAP
        rows.append(
            {
                L.C_QUERY: query,
                L.C_VERDICT: verdict,
                L.C_BEST_URL: result.urls[j],
                L.C_CHUNK: result.best_chunk(i, j),
                L.C_S_CHUNK: round(float(result.chunk_scores[i, j]), 4),
                L.C_S_FULL: round(float(result.full_scores[i, j]), 4),
                L.C_S_COMBI: round(float(combined[i, j]), 4),
                L.C_RANK_URL: rank_url,
                L.C_POSITION: position,
                L.C_NOTE: note,
                L.C_ADVICE: ADVICE[verdict].format(best=result.urls[j], rank_url=rank_url, position=position),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_verdict.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add qum/verdict.py tests/test_verdict.py
git commit -m "Add verdicts with fixed advice templates"
```

---

### Task 9: Kannibalisierung

**Files:**
- Create: `qum/cannibal.py`
- Test: `tests/test_cannibal.py`

**Interfaces:**
- Consumes: `MatchResult`, Decisions-DataFrame aus `build_decisions` (gleiche Zeilenreihenfolge wie `result.queries`), `normalize_query`, `format_position`, Labels
- Produces: `find_cannibalization(result, lead, threshold, decisions, rankings=None, margin=0.02, visible_position=20) -> pd.DataFrame` mit Spalten `Query, Stufe, Grund, Konkurrierende URLs`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_cannibal.py`:

```python
import pandas as pd

from qum import labels as L
from qum.cannibal import find_cannibalization
from qum.verdict import build_decisions
from tests.conftest import make_result

U1, U2, U3 = "https://a.de/1", "https://a.de/2", "https://a.de/3"


def _rankings(rows):
    return pd.DataFrame(rows, columns=["query_norm", "url", "url_norm", "position"])


def _run(queries, scores, rankings=None, **kwargs):
    result = make_result(queries, [U1, U2, U3], scores)
    lead = result.lead("chunk")
    decisions = build_decisions(result, lead, 0.6, rankings)
    return find_cannibalization(result, lead, 0.6, decisions, rankings, **kwargs)


def test_columns_and_empty_result():
    df = _run(["q"], [[0.9, 0.5, 0.1]])
    assert list(df.columns) == [L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING]
    assert df.empty


def test_two_urls_close_together_without_rankings():
    df = _run(["q"], [[0.80, 0.79, 0.3]])
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Score 0.8) | {U2} (Score 0.79)"


def test_close_but_below_threshold_is_not_flagged():
    assert _run(["q"], [[0.50, 0.49, 0.1]]).empty


def test_margin_is_adjustable():
    assert _run(["q"], [[0.80, 0.70, 0.1]]).empty
    assert len(_run(["q"], [[0.80, 0.70, 0.1]], margin=0.15)) == 1


def test_risk_verdict_is_listed_with_ranking_and_better_url():
    rankings = _rankings([("q", U1, U1, 4.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 4) | {U2} (Score 0.9)"


def test_several_own_urls_ranking_is_visible_stage():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0), ("q", U3, U3, 45.0)])
    df = _run(["q"], [[0.9, 0.1, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_VISIBLE]
    assert df.iloc[0][L.C_COMPETING] == f"{U1} (Position 3) | {U2} (Position 12)"


def test_both_stages_can_apply_to_one_query():
    rankings = _rankings([("q", U1, U1, 3.0), ("q", U2, U2, 12.0)])
    df = _run(["q"], [[0.65, 0.9, 0.1]], rankings)
    assert df[L.C_STAGE].tolist() == [L.STAGE_RISK, L.STAGE_VISIBLE]
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_cannibal.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/cannibal.py`:

```python
import numpy as np
import pandas as pd

from . import labels as L
from .normalize import normalize_query
from .verdict import format_position

REASON_BETTER = "Eine andere Seite passt besser als die rankende"
REASON_CLOSE = "Mehrere Seiten passen fast gleich gut"
REASON_RANKING = "Mehrere eigene Seiten ranken für die Query"


def _score(value) -> str:
    return f"Score {round(float(value), 4)}"


def find_cannibalization(
    result, lead, threshold, decisions, rankings=None, margin=0.02, visible_position=20
) -> pd.DataFrame:
    rows = []
    for i, query in enumerate(result.queries):
        decision = decisions.iloc[i]
        order = np.argsort(-lead[i], kind="stable")
        top = lead[i, order[0]]
        close = [j for j in order if lead[i, j] >= threshold and top - lead[i, j] <= margin]
        if decision[L.C_VERDICT] == L.V_RISK:
            competing = [
                f"{decision[L.C_RANK_URL]} (Position {decision[L.C_POSITION]})",
                f"{decision[L.C_BEST_URL]} ({_score(top)})",
            ]
            rows.append((query, L.STAGE_RISK, REASON_BETTER, " | ".join(competing)))
        elif len(close) >= 2:
            competing = [f"{result.urls[j]} ({_score(lead[i, j])})" for j in close]
            rows.append((query, L.STAGE_RISK, REASON_CLOSE, " | ".join(competing)))
        if rankings is not None:
            own = rankings[
                (rankings["query_norm"] == normalize_query(query)) & (rankings["position"] <= visible_position)
            ]
            own = own.sort_values("position", kind="stable").drop_duplicates("url_norm")
            if len(own) >= 2:
                competing = [f"{row.url} (Position {format_position(row.position)})" for row in own.itertuples()]
                rows.append((query, L.STAGE_VISIBLE, REASON_RANKING, " | ".join(competing)))
    return pd.DataFrame(rows, columns=[L.C_QUERY, L.C_STAGE, L.C_REASON, L.C_COMPETING])
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_cannibal.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add qum/cannibal.py tests/test_cannibal.py
git commit -m "Add cannibalisation stages"
```

---

### Task 10: SERP-Clustering und Nachbar-Hinweis

**Files:**
- Create: `qum/serp.py`
- Test: `tests/test_serp.py`

**Interfaces:**
- Consumes: SERP-DataFrame aus `load_serps` (`keyword, query_norm, url, url_norm, position`), Decisions-DataFrame, `MatchResult`, `ADVICE`, `format_position`, Labels
- Produces:
  - `top_urls_per_keyword(serps) -> dict[str, list[str]]` (Schlüssel `query_norm`, höchstens 10 `url_norm`)
  - `overlap_edges(kw_urls, min_overlap=0.5) -> dict[tuple[str, str], float]` (Schlüssel ist das sortierte Paar)
  - `cluster_keywords(kw_urls, min_overlap=0.5, min_density=0.5) -> dict[str, int]` (0 = kein Cluster)
  - `apply_serp(decisions, result, lead, serps, rankings=None, min_overlap=0.5, min_density=0.5) -> pd.DataFrame`: Kopie der Decisions mit den Spalten `Cluster, Nachbar-Kandidat, Nachbar-Keyword, SERP-Überschneidung, Score Kandidat, Passage Kandidat, Position Kandidat, Weitere Kandidaten`; Lücken mit Kandidat bekommen das Urteil `V_CHECK`
  - `gap_summary(decisions) -> pd.DataFrame` mit Spalten `Cluster, Lücken-Queries, Neue Seiten, Queries`
  - `count_new_pages(decisions) -> int`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_serp.py`:

```python
import pandas as pd

from qum import labels as L
from qum.serp import (
    apply_serp,
    cluster_keywords,
    count_new_pages,
    gap_summary,
    overlap_edges,
    top_urls_per_keyword,
)
from qum.verdict import build_decisions
from tests.conftest import make_result


def _urls(*ids):
    return [f"https://s.de/{i}" for i in ids]


def _serps(kw_urls):
    rows = []
    for kw, urls in kw_urls.items():
        for pos, url in enumerate(urls, start=1):
            rows.append((kw, kw, url, url, float(pos)))
    return pd.DataFrame(rows, columns=["keyword", "query_norm", "url", "url_norm", "position"])


def test_top_urls_per_keyword_sorts_dedupes_and_limits():
    serps = pd.DataFrame(
        [("k", "k", f"https://s.de/{i}", f"https://s.de/{i}", float(i)) for i in range(12, 0, -1)]
        + [("k", "k", "https://s.de/1", "https://s.de/1", 99.0)],
        columns=["keyword", "query_norm", "url", "url_norm", "position"],
    )
    urls = top_urls_per_keyword(serps)["k"]
    assert urls == _urls(*range(1, 11))


def test_overlap_uses_smaller_list_as_denominator():
    edges = overlap_edges({"a": _urls(1, 2, 3, 4), "b": _urls(1, 2)}, min_overlap=0.5)
    assert edges == {("a", "b"): 1.0}


def test_overlap_below_threshold_is_no_edge():
    assert overlap_edges({"a": _urls(1, 2, 3, 4), "b": _urls(1, 5, 6, 7)}, min_overlap=0.5) == {}


def test_chain_is_broken_by_density():
    # a-b, b-c, c-d überschneiden sich jeweils zur Hälfte, a und d haben nichts gemeinsam
    kw = {
        "a": _urls(1, 2, 3, 4),
        "b": _urls(3, 4, 5, 6),
        "c": _urls(5, 6, 7, 8),
        "d": _urls(7, 8, 9, 10),
    }
    clusters = cluster_keywords(kw, 0.5, 0.5)
    assert clusters["a"] == 0 and clusters["d"] == 0
    assert clusters["b"] == clusters["c"] != 0


def test_single_link_into_clique_is_dropped():
    clique = {k: _urls(1, 2, 3, 4) for k in ["a", "b", "c", "d"]}
    clique["x"] = _urls(1, 2, 20, 21, 22, 23, 24, 25)  # nur 2 von 4 gemeinsam mit jedem: Kante zu allen
    clique["y"] = _urls(20, 21, 22, 23, 30, 31, 32, 33)  # hängt nur an x
    clusters = cluster_keywords(clique, 0.5, 0.5)
    assert len({clusters[k] for k in ["a", "b", "c", "d", "x"]}) == 1
    assert clusters["y"] == 0


def test_isolated_keyword_has_no_cluster():
    assert cluster_keywords({"a": _urls(1), "b": _urls(2)}) == {"a": 0, "b": 0}


OWN = ["https://a.de/getreidefrei", "https://a.de/anderes"]


def _decisions(queries, scores):
    result = make_result(queries, OWN, scores)
    lead = result.lead("chunk")
    return result, lead, build_decisions(result, lead, 0.6)


def test_neighbour_hint_only_for_direct_neighbours():
    queries = ["getreidefrei", "ohne getreide", "allergie"]
    result, lead, decisions = _decisions(queries, [[0.9, 0.1], [0.4, 0.1], [0.3, 0.1]])
    serps = _serps(
        {
            "getreidefrei": _urls(1, 2, 3, 4),
            "ohne getreide": _urls(1, 2, 3, 9),  # direkter Nachbar von "getreidefrei"
            "allergie": _urls(3, 9, 11, 12),  # Nachbar von "ohne getreide", nicht von "getreidefrei"
        }
    )
    out = apply_serp(decisions, result, lead, serps)
    assert out[L.C_VERDICT].tolist() == [L.V_MATCH, L.V_CHECK, L.V_GAP]
    row = out.iloc[1]
    assert row[L.C_CAND] == OWN[0]
    assert row[L.C_CAND_KW] == "getreidefrei"
    assert row[L.C_CAND_OVERLAP] == "75 %"
    assert row[L.C_CAND_SCORE] == 0.4
    assert row[L.C_CAND_CHUNK] == f"Text {OWN[0]}"
    assert OWN[0] in row[L.C_ADVICE]
    assert out.iloc[2][L.C_CAND] == ""


def test_cluster_column_is_added_and_original_untouched():
    result, lead, decisions = _decisions(["a", "b"], [[0.9, 0.1], [0.9, 0.1]])
    out = apply_serp(decisions, result, lead, _serps({"a": _urls(1, 2), "b": _urls(1, 2)}))
    assert out[L.C_CLUSTER].tolist() == [1, 1]
    assert L.C_CLUSTER not in decisions.columns


def test_query_missing_in_serps_gets_cluster_zero():
    result, lead, decisions = _decisions(["a"], [[0.9, 0.1]])
    out = apply_serp(decisions, result, lead, _serps({"z": _urls(1)}))
    assert out[L.C_CLUSTER].tolist() == [0]


def test_several_candidates_are_ordered_by_overlap():
    queries = ["luecke", "nachbar eins", "nachbar zwei"]
    result, lead, decisions = _decisions(queries, [[0.2, 0.3], [0.9, 0.1], [0.1, 0.9]])
    serps = _serps(
        {
            "luecke": _urls(1, 2, 3, 4),
            "nachbar eins": _urls(1, 2, 7, 8),  # 50 %
            "nachbar zwei": _urls(1, 2, 3, 9),  # 75 %
        }
    )
    row = apply_serp(decisions, result, lead, serps).iloc[0]
    assert row[L.C_CAND] == OWN[1]
    assert row[L.C_CAND_KW] == "nachbar zwei"
    assert row[L.C_CAND_MORE] == f"{OWN[0]} (Nachbar: nachbar eins, 50 %)"


def test_candidate_position_comes_from_rankings():
    queries = ["luecke", "nachbar"]
    result, lead, decisions = _decisions(queries, [[0.2, 0.1], [0.9, 0.1]])
    serps = _serps({"luecke": _urls(1, 2), "nachbar": _urls(1, 2)})
    rankings = pd.DataFrame(
        [("luecke", OWN[0], OWN[0], 44.0)], columns=["query_norm", "url", "url_norm", "position"]
    )
    row = apply_serp(decisions, result, lead, serps, rankings).iloc[0]
    assert row[L.C_CAND_POS] == "44"


def test_gap_summary_counts_one_page_per_cluster():
    decisions = pd.DataFrame(
        {
            L.C_QUERY: ["a", "b", "c", "d", "e"],
            L.C_VERDICT: [L.V_GAP, L.V_GAP, L.V_GAP, L.V_GAP, L.V_MATCH],
            L.C_CLUSTER: [1, 1, 0, 0, 1],
        }
    )
    summary = gap_summary(decisions)
    assert summary.to_dict("records") == [
        {L.C_CLUSTER: "1", "Lücken-Queries": 2, "Neue Seiten": 1, "Queries": "a | b"},
        {L.C_CLUSTER: L.NO_CLUSTER, "Lücken-Queries": 2, "Neue Seiten": 2, "Queries": "c | d"},
    ]
    assert count_new_pages(decisions) == 3


def test_count_new_pages_without_cluster_column():
    decisions = pd.DataFrame({L.C_QUERY: ["a", "b"], L.C_VERDICT: [L.V_GAP, L.V_GAP]})
    assert count_new_pages(decisions) == 2
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_serp.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/serp.py`:

```python
import math

import pandas as pd

from . import labels as L
from .normalize import normalize_query, normalize_url
from .verdict import ADVICE, format_position

# Urteile, bei denen "Beste URL" eine passende Seite ist
_HAS_PAGE = {L.V_MATCH, L.V_USE, L.V_OK, L.V_RISK}
_CANDIDATE_COLUMNS = [
    L.C_CAND, L.C_CAND_KW, L.C_CAND_OVERLAP, L.C_CAND_SCORE, L.C_CAND_CHUNK, L.C_CAND_POS, L.C_CAND_MORE,
]


def top_urls_per_keyword(serps: pd.DataFrame) -> dict:
    out = {}
    for keyword, group in serps.groupby("query_norm", sort=False):
        ordered = group.sort_values("position", kind="stable").drop_duplicates("url_norm")
        out[keyword] = ordered["url_norm"].head(10).tolist()
    return out


def overlap_edges(kw_urls: dict, min_overlap: float = 0.5) -> dict:
    keywords = sorted(kw_urls)
    sets = {k: set(kw_urls[k]) for k in keywords}
    edges = {}
    for a_idx, a in enumerate(keywords):
        for b in keywords[a_idx + 1 :]:
            denominator = min(len(sets[a]), len(sets[b]), 10)
            if denominator == 0:
                continue
            overlap = len(sets[a] & sets[b]) / denominator
            if overlap >= min_overlap:
                edges[(a, b)] = overlap
    return edges


def _neighbours(kw_urls, edges):
    neighbours = {k: set() for k in kw_urls}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    return neighbours


def _components(nodes: set, neighbours: dict) -> list:
    seen, out = set(), []
    for start in sorted(nodes):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend((neighbours[node] & nodes) - component)
        seen |= component
        out.append(component)
    return out


def cluster_keywords(kw_urls: dict, min_overlap: float = 0.5, min_density: float = 0.5) -> dict:
    neighbours = _neighbours(kw_urls, overlap_edges(kw_urls, min_overlap))
    assignments = {k: 0 for k in kw_urls}
    next_id = 1
    for component in _components(set(kw_urls), neighbours):
        nodes = set(component)
        # Dichte: wer mit zu wenigen Mitgliedern verbunden ist, fliegt raus (gegen den Ketteneffekt)
        while len(nodes) >= 2:
            needed = math.ceil(min_density * (len(nodes) - 1))
            weak = {n for n in nodes if len(neighbours[n] & nodes) < needed}
            if not weak:
                break
            nodes -= weak
        for sub in _components(nodes, neighbours):
            if len(sub) >= 2:
                for keyword in sub:
                    assignments[keyword] = next_id
                next_id += 1
    return assignments


def apply_serp(decisions, result, lead, serps, rankings=None, min_overlap=0.5, min_density=0.5) -> pd.DataFrame:
    kw_urls = top_urls_per_keyword(serps)
    edges = overlap_edges(kw_urls, min_overlap)
    neighbours = _neighbours(kw_urls, edges)
    clusters = cluster_keywords(kw_urls, min_overlap, min_density)

    out = decisions.copy()
    norms = [normalize_query(q) for q in result.queries]
    row_of = {norm: i for i, norm in enumerate(norms)}
    u_index = {normalize_url(u): j for j, u in enumerate(result.urls)}
    positions = {}
    if rankings is not None:
        for row in rankings.sort_values("position", kind="stable").itertuples():
            positions.setdefault((row.query_norm, row.url_norm), row.position)

    out[L.C_CLUSTER] = [clusters.get(norm, 0) for norm in norms]
    for column in _CANDIDATE_COLUMNS:
        # object-Spalte: nimmt später Text und Zahlen auf
        out[column] = pd.Series([""] * len(out), index=out.index, dtype=object)

    for i, norm in enumerate(norms):
        if decisions.iloc[i][L.C_VERDICT] != L.V_GAP:
            continue
        candidates = []
        for other in neighbours.get(norm, ()):
            k = row_of.get(other)
            if k is None or decisions.iloc[k][L.C_VERDICT] not in _HAS_PAGE:
                continue
            overlap = edges[tuple(sorted((norm, other)))]
            candidates.append((overlap, result.queries[k], decisions.iloc[k][L.C_BEST_URL]))
        if not candidates:
            continue
        candidates.sort(key=lambda c: (-c[0], c[1]))
        overlap, keyword, url = candidates[0]
        j = u_index[normalize_url(url)]
        label = out.index[i]
        out.loc[label, L.C_VERDICT] = L.V_CHECK
        out.loc[label, L.C_ADVICE] = ADVICE[L.V_CHECK].format(best=url, rank_url="", position="")
        out.loc[label, L.C_CAND] = url
        out.loc[label, L.C_CAND_KW] = keyword
        out.loc[label, L.C_CAND_OVERLAP] = f"{round(overlap * 100)} %"
        out.loc[label, L.C_CAND_SCORE] = round(float(lead[i, j]), 4)
        out.loc[label, L.C_CAND_CHUNK] = result.best_chunk(i, j)
        out.loc[label, L.C_CAND_POS] = format_position(positions.get((norm, normalize_url(url))))
        out.loc[label, L.C_CAND_MORE] = " | ".join(
            f"{u} (Nachbar: {kw}, {round(o * 100)} %)" for o, kw, u in candidates[1:] if u != url
        )
    return out


def gap_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    gaps = decisions[decisions[L.C_VERDICT] == L.V_GAP]
    clusters = gaps[L.C_CLUSTER] if L.C_CLUSTER in gaps.columns else pd.Series(0, index=gaps.index)
    rows = []
    for cluster in sorted(c for c in clusters.unique() if c != 0):
        queries = gaps.loc[clusters == cluster, L.C_QUERY].tolist()
        rows.append(
            {L.C_CLUSTER: str(cluster), "Lücken-Queries": len(queries), "Neue Seiten": 1, "Queries": " | ".join(queries)}
        )
    loose = gaps.loc[clusters == 0, L.C_QUERY].tolist()
    if loose:
        rows.append(
            {L.C_CLUSTER: L.NO_CLUSTER, "Lücken-Queries": len(loose), "Neue Seiten": len(loose), "Queries": " | ".join(loose)}
        )
    return pd.DataFrame(rows, columns=[L.C_CLUSTER, "Lücken-Queries", "Neue Seiten", "Queries"])


def count_new_pages(decisions: pd.DataFrame) -> int:
    return int(gap_summary(decisions)["Neue Seiten"].sum())
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_serp.py -q`
Expected: 13 passed

Hinweis für `test_single_link_into_clique_is_dropped`: `x` teilt 2 URLs mit jeder Clique-Liste (Nenner 4, also 50 %) und hat damit Kanten zu `a`–`d`. `y` teilt 4 URLs mit `x` (Nenner 8, also 50 %) und hängt nur an `x`. In der 6er-Komponente braucht jedes Mitglied `ceil(0,5 × 5) = 3` Kanten, `y` hat eine und fällt heraus.

- [ ] **Step 5: Commit**

```bash
git add qum/serp.py tests/test_serp.py
git commit -m "Add SERP clustering, neighbour hint and gap summary"
```

---

### Task 11: Paare bewerten

**Files:**
- Create: `qum/pairs.py`
- Test: `tests/test_pairs.py`

**Interfaces:**
- Consumes: `run_matching`, `ranks`, `find_column`, `IngestError`, `KEYWORD_ALIASES`, `URL_ALIASES`, `normalize_url`, Labels
- Produces: `score_pairs(df, urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7, keyword_col=None, url_col=None) -> pd.DataFrame`: alle Eingabespalten plus `Score Chunk, Score Gesamt-URL, Score Kombi, Rang der URL, Beste URL, Hinweis`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_pairs.py`:

```python
import pandas as pd
import pytest

from qum import labels as L
from qum.ingest import IngestError
from qum.pairs import score_pairs
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


def test_each_keyword_is_embedded_once():
    embedder = FakeEmbedder()
    df = pd.DataFrame({"Keyword": ["hundefutter", "hundefutter"], "URL": URLS})
    score_pairs(df, URLS, CONTENTS, embedder, 5, 1)
    queries = [texts for texts, role in embedder.calls if role == "query"]
    assert queries == [["hundefutter"]]
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_pairs.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/pairs.py`:

```python
import pandas as pd

from . import labels as L
from .ingest import KEYWORD_ALIASES, URL_ALIASES, IngestError, find_column
from .match import ranks, run_matching
from .normalize import normalize_url

_NEW_COLUMNS = [L.C_S_CHUNK, L.C_S_FULL, L.C_S_COMBI, L.C_PAIR_RANK, L.C_BEST_URL, L.C_NOTE]


def score_pairs(
    df, urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7, keyword_col=None, url_col=None
) -> pd.DataFrame:
    kcol = keyword_col or find_column(df, KEYWORD_ALIASES)
    ucol = url_col or find_column(df, URL_ALIASES)
    if kcol is None or ucol is None:
        missing = "Keyword" if kcol is None else "URL"
        raise IngestError(f"{missing}-Spalte nicht erkannt. Gefundene Spalten: {list(df.columns)}")

    work = df.dropna(subset=[kcol, ucol]).reset_index(drop=True)
    if work.empty:
        return work.assign(**{column: [] for column in _NEW_COLUMNS})

    keywords = list(dict.fromkeys(str(k).strip() for k in work[kcol]))
    result = run_matching(keywords, urls, contents, embedder, chunk_size, chunk_overlap)
    lead = result.lead(basis, weight)
    combined = result.combined(weight)
    rank = ranks(lead)
    k_index = {k: i for i, k in enumerate(keywords)}
    u_index = {normalize_url(u): j for j, u in enumerate(urls)}

    rows = []
    for keyword, url in zip(work[kcol], work[ucol]):
        i = k_index[str(keyword).strip()]
        j = u_index.get(normalize_url(url))
        best = result.urls[int(lead[i].argmax())]
        if j is None:
            rows.append((None, None, None, None, best, L.NOTE_URL_MISSING))
        else:
            rows.append(
                (
                    round(float(result.chunk_scores[i, j]), 4),
                    round(float(result.full_scores[i, j]), 4),
                    round(float(combined[i, j]), 4),
                    int(rank[i, j]),
                    best,
                    "",
                )
            )
    scored = pd.DataFrame(rows, columns=_NEW_COLUMNS)
    scored[L.C_PAIR_RANK] = scored[L.C_PAIR_RANK].astype("Int64")
    return pd.concat([work, scored], axis=1)
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_pairs.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add qum/pairs.py tests/test_pairs.py
git commit -m "Add scoring of existing keyword-URL pairs"
```

---

### Task 12: Export

**Files:**
- Create: `qum/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: Labels, `gap_summary`
- Produces:
  - Blattnamen `SHEET_README = "Lesehilfe"`, `SHEET_DECISION = "Entscheidung"`, `SHEET_TOP = "Top-Treffer"`, `SHEET_CANNIBAL = "Kannibalisierung"`, `SHEET_GAPS = "Content-Lücken"`, `SHEET_GAP_CLUSTERS = "Lücken je Cluster"`, `SHEET_PAIRS = "Paare"`
  - `content_gaps(decisions) -> pd.DataFrame` (Urteile `V_GAP` und `V_CHECK`)
  - `build_sheets(decisions, top, cannibal, settings: dict, pairs=None) -> dict[str, pd.DataFrame]` in fester Reihenfolge; `Lücken je Cluster` nur, wenn `decisions` eine Cluster-Spalte hat; `Paare` nur, wenn übergeben
  - `write_excel(path, sheets) -> None`
  - `write_csv_zip(path, sheets) -> None`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_export.py`:

```python
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
    sheets = export.build_sheets(_decisions(with_cluster=True), TOP, CANNIBAL, SETTINGS)
    for name, df in sheets.items():
        if name not in (export.SHEET_README, export.SHEET_GAP_CLUSTERS):
            assert df.columns[0] == L.C_QUERY, name


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
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_export.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/export.py`:

```python
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
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_export.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add qum/export.py tests/test_export.py
git commit -m "Add Excel and CSV export with reading guide"
```

---

### Task 13: Modellvergleich für die Messung

**Files:**
- Create: `qum/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `run_matching`, `ranks`, `normalize_url`, `get_model`, `make_embedder`, `read_table`, `load_content`, `find_column`, Aliaslisten
- Produces:
  - `evaluate(truth: list[tuple[str, str]], urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7) -> dict` mit Schlüsseln `n, fehlend, hit1, hit3, mrr`
  - `CONFIGS: dict[str, tuple[str, dict]]` mit den Schlüsseln `paraphrase, e5, openai, openai-prefix`
  - `spec_for(config: str) -> ModelSpec`
  - `main(argv=None) -> int`; Aufruf `python -m qum.compare --truth wahrheit.csv --content frog.csv`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_compare.py`:

```python
import pytest

from qum.compare import CONFIGS, evaluate, spec_for
from tests.conftest import FakeEmbedder

URLS = ["https://a.de/hund", "https://a.de/katze", "https://a.de/vogel"]
CONTENTS = ["hundefutter getreidefrei trocken", "katzenfutter nass sorten", "vogelfutter koerner mischung"]


def test_evaluate_perfect_ranking():
    truth = [("hundefutter getreidefrei", URLS[0]), ("katzenfutter nass", URLS[1])]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert out == {"n": 2, "fehlend": 0, "hit1": 1.0, "hit3": 1.0, "mrr": 1.0}


def test_evaluate_wrong_expectation_lowers_scores():
    truth = [("hundefutter getreidefrei", URLS[1]), ("katzenfutter nass", URLS[1])]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert out["hit1"] == 0.5
    assert out["hit3"] == 1.0
    assert 0.5 < out["mrr"] < 1.0


def test_evaluate_counts_urls_missing_in_content():
    truth = [("hundefutter", "https://a.de/weg"), ("katzenfutter nass", URLS[1] + "/")]
    out = evaluate(truth, URLS, CONTENTS, FakeEmbedder(), 5, 1)
    assert (out["n"], out["fehlend"]) == (1, 1)


def test_evaluate_without_usable_pairs_raises():
    with pytest.raises(ValueError, match="Wahrheitsliste"):
        evaluate([("x", "https://a.de/weg")], URLS, CONTENTS, FakeEmbedder(), 5, 1)


def test_configs_cover_the_four_variants():
    assert list(CONFIGS) == ["paraphrase", "e5", "openai", "openai-prefix"]


def test_openai_prefix_variant_only_exists_here():
    plain, prefixed = spec_for("openai"), spec_for("openai-prefix")
    assert (plain.query_prefix, plain.passage_prefix) == ("", "")
    assert (prefixed.query_prefix, prefixed.passage_prefix) == ("query: ", "passage: ")
    assert prefixed.model_id == plain.model_id
    assert prefixed.key == "openai-prefix"
```

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_compare.py -q`
Expected: FAIL mit `ModuleNotFoundError`

- [ ] **Step 3: Implementieren**

`qum/compare.py`:

```python
"""Modellvergleich an einer Wahrheitsliste: Wie oft landet die erwartete URL vorn?

Aufruf: python -m qum.compare --truth wahrheit.csv --content frog.csv
"""

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import make_embedder
from .ingest import KEYWORD_ALIASES, URL_ALIASES, IngestError, find_column, load_content, read_table
from .match import ranks, run_matching
from .models import ModelSpec, get_model
from .normalize import normalize_url

# Name -> (Modellschlüssel, Abweichungen von der Modelldefinition)
CONFIGS = {
    "paraphrase": ("paraphrase-mpnet", {}),
    "e5": ("e5-large", {}),
    "openai": ("openai", {}),
    "openai-prefix": ("openai", {"query_prefix": "query: ", "passage_prefix": "passage: "}),
}


def spec_for(config: str) -> ModelSpec:
    key, overrides = CONFIGS[config]
    return dataclasses.replace(get_model(key), key=config, **overrides)


def evaluate(truth, urls, contents, embedder, chunk_size, chunk_overlap, basis="chunk", weight=0.7) -> dict:
    u_index = {normalize_url(u): j for j, u in enumerate(urls)}
    pairs = [(query, u_index[normalize_url(url)]) for query, url in truth if normalize_url(url) in u_index]
    if not pairs:
        raise ValueError("Keine URL der Wahrheitsliste steht im Frog-Export.")
    result = run_matching([query for query, _ in pairs], urls, contents, embedder, chunk_size, chunk_overlap)
    rank = ranks(result.lead(basis, weight))
    positions = np.array([rank[i, j] for i, (_, j) in enumerate(pairs)])
    return {
        "n": len(pairs),
        "fehlend": len(truth) - len(pairs),
        "hit1": float((positions == 1).mean()),
        "hit3": float((positions <= 3).mean()),
        "mrr": float((1 / positions).mean()),
    }


def _load_truth(path: Path) -> list:
    df = read_table(path.read_bytes(), path.name)
    kcol, ucol = find_column(df, KEYWORD_ALIASES), find_column(df, URL_ALIASES)
    if kcol is None or ucol is None:
        raise IngestError(f"Wahrheitsliste braucht eine Query- und eine URL-Spalte. Gefunden: {list(df.columns)}")
    df = df.dropna(subset=[kcol, ucol])
    return [(str(q).strip(), str(u).strip()) for q, u in zip(df[kcol], df[ucol])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Modellvergleich an einer Wahrheitsliste")
    parser.add_argument("--truth", required=True, type=Path, help="CSV/XLSX mit Query und erwarteter URL")
    parser.add_argument("--content", required=True, type=Path, help="Frog-Export mit URL und Main Content")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--basis", default="chunk", choices=["chunk", "full", "combined"])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("modellvergleich.csv"))
    args = parser.parse_args(argv)

    truth = _load_truth(args.truth)
    content = load_content(read_table(args.content.read_bytes(), args.content.name))
    rows = []
    for config in args.configs:
        spec = spec_for(config)
        api_key = os.environ.get("OPENAI_API_KEY") if spec.provider == "openai" else None
        embedder = make_embedder(spec, api_key=api_key, cache_dir=args.cache_dir)
        # gleiche Chunks für alle Konfigurationen, damit nur das Modell den Unterschied macht
        metrics = evaluate(truth, content.urls, content.contents, embedder, args.chunk_size, args.overlap, args.basis)
        rows.append({"Konfiguration": config, "Modell": spec.model_id, **metrics})
        print(f"{config}: Rang 1 {metrics['hit1']:.0%}, Top 3 {metrics['hit3']:.0%}, MRR {metrics['mrr']:.3f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Gespeichert: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Tests bestehen sehen**

Run: `.venv/bin/pytest tests/test_compare.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add qum/compare.py tests/test_compare.py
git commit -m "Add model comparison against a truth list"
```

---

### Task 14: Notebook, Colab-Helfer, README

**Files:**
- Create: `qum/colab.py`, `tools/build_notebook.py`, `query_url_matcher.ipynb` (erzeugt), `README.md`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: alle öffentlichen Funktionen der Tasks 1 bis 12
- Produces:
  - `qum.colab.upload(title: str) -> tuple[str, bytes]`, `secret(name: str) -> str | None`, `download(path) -> None`, `guard()` (Kontextmanager, der `IngestError` und `EmbeddingError` als „❌ …" ausgibt und die Zelle ohne Traceback beendet), `class NotebookStop(Exception)`
  - `tools/build_notebook.py`: `CELLS: list[tuple[str, str]]` (`"markdown"` oder `"code"`, Quelltext), `build() -> dict`, `main()` schreibt `query_url_matcher.ipynb`

- [ ] **Step 1: Failing Tests schreiben**

`tests/test_notebook.py`:

```python
import ast
import json
import re
from pathlib import Path

import pytest

from qum import colab
from qum.embeddings.base import EmbeddingError
from qum.ingest import IngestError
from qum.models import MODELS
from tools.build_notebook import CELLS, build

ROOT = Path(__file__).resolve().parents[1]


def _code_cells():
    return [source for kind, source in CELLS if kind == "code"]


def test_committed_notebook_matches_builder():
    committed = json.loads((ROOT / "query_url_matcher.ipynb").read_text(encoding="utf-8"))
    assert committed == build()


def test_code_cells_are_valid_python():
    for source in _code_cells():
        python = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
        ast.parse(python)


def test_every_code_cell_is_a_titled_form():
    for source in _code_cells():
        assert source.startswith("#@title "), source[:40]


def test_model_dropdown_lists_exactly_the_registry_labels():
    source = next(s for s in _code_cells() if "modell = " in s)
    options = json.loads(re.search(r"modell = .*#@param (\[.*\])", source).group(1))
    assert options == [spec.label for spec in MODELS.values()]


def test_notebook_has_nine_steps_in_order():
    titles = [re.match(r"#@title (.*?)( \{|$)", s.splitlines()[0]).group(1) for s in _code_cells()]
    assert [t.split(":")[0].split(" (")[0] for t in titles] == [f"Schritt {n}" for n in range(2, 10)]
    assert any(kind == "markdown" and "Schritt 1" in source for kind, source in CELLS)


def test_no_keys_and_no_retired_model_in_notebook():
    text = (ROOT / "query_url_matcher.ipynb").read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "text-embedding-004" not in text


def test_guard_turns_known_errors_into_notebook_stop(capsys):
    for error in (IngestError("Spalte fehlt"), EmbeddingError("Key fehlt")):
        with pytest.raises(colab.NotebookStop):
            with colab.guard():
                raise error
    assert "❌ Spalte fehlt" in capsys.readouterr().out


def test_guard_lets_other_errors_through():
    with pytest.raises(ZeroDivisionError):
        with colab.guard():
            1 / 0


def test_notebook_stop_hides_traceback():
    assert colab.NotebookStop()._render_traceback_() == []
```

Lege eine leere Datei `tools/__init__.py` an.

- [ ] **Step 2: Tests fehlschlagen sehen**

Run: `.venv/bin/pytest tests/test_notebook.py -q`
Expected: FAIL mit `ModuleNotFoundError: No module named 'qum.colab'`

- [ ] **Step 3: Colab-Helfer implementieren**

`qum/colab.py`:

```python
"""Dünne Helfer für das Notebook. Alles Colab-Spezifische wird erst beim Aufruf importiert."""

from contextlib import contextmanager

from .embeddings.base import EmbeddingError
from .ingest import IngestError


class NotebookStop(Exception):
    """Beendet eine Zelle ohne Traceback."""

    def _render_traceback_(self):
        return []


@contextmanager
def guard():
    try:
        yield
    except (IngestError, EmbeddingError) as error:
        print(f"❌ {error}")
        raise NotebookStop() from None


def stop(message: str):
    print(f"❌ {message}")
    raise NotebookStop()


def upload(title: str):
    from google.colab import files

    print(f"📤 {title}")
    uploaded = files.upload()
    if not uploaded:
        stop("Keine Datei hochgeladen. Führe die Zelle erneut aus.")
    return next(iter(uploaded.items()))


def secret(name: str):
    from google.colab import userdata

    try:
        return userdata.get(name)
    except Exception:
        return None


def download(path) -> None:
    from google.colab import files

    files.download(str(path))
```

- [ ] **Step 4: Notebook-Builder implementieren**

`tools/build_notebook.py`:

```python
"""Erzeugt query_url_matcher.ipynb. Nach jeder Änderung ausführen: python -m tools.build_notebook"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = '''# Query-URL Matcher

**Neue Seite bauen oder Bestehendes nutzen?**

Dieses Notebook vergleicht deine Suchanfragen (oder Prompts) per Embeddings mit dem Inhalt deiner Seiten. Es zeigt je Query, welche Seite und welche Textstelle am besten passt, und leitet daraus Hinweise zu **Kannibalisierung** und **Content-Lücken** ab.

## Schritt 1: Was du brauchst

| Datei | Pflicht | Inhalt |
|---|---|---|
| Queries | ja | CSV oder Excel, eine Query pro Zeile |
| Screaming-Frog-Export | ja | Spalten URL und Main Content |
| Eigene Rankings | optional | Keyword, URL, Position (GSC, Ahrefs, SISTRIX) |
| Top-10-SERPs | optional | Ahrefs-Export mit Keyword, URL, Position, Type |

Ohne Ranking-Dateien bekommst du das reine Matching. Mit eigenen Rankings kommen die Urteile zu Kannibalisierung und Content-Lücke dazu, mit Top-10-SERPs zusätzlich die Bündelung der Lücken zu Themen.

## So gehst du vor

Führe die Zellen von oben nach unten aus (Play-Symbol links). Jede Zelle endet mit einer Zeile, die sagt, wie es weitergeht. Die Schritte 5 und 6 kannst du überspringen.

## Wichtig

Das Notebook sortiert vor und begründet. Cosinus-Werte sind Hinweise, keine Urteile: Die Entscheidung triffst du, nachdem du die Seite angesehen hast.
'''

STEP2 = '''#@title Schritt 2: Installation und Modellwahl { display-mode: "form" }
#@markdown Wähle das Embedding-Modell. Für deutschen Content ist **multilingual-e5-large** die Empfehlung.
#@markdown Die Modelle mit API-Key brauchen einen Eintrag unter Secrets (Schlüssel-Symbol links): `OPENAI_API_KEY` bzw. `GEMINI_API_KEY`.
modell = "multilingual-e5-large · Deutsch, kostenlos (Empfehlung)" #@param ["multilingual-e5-large · Deutsch, kostenlos (Empfehlung)", "multilingual-e5-base · Deutsch, kostenlos, schneller", "bge-m3 · Deutsch, kostenlos, lange Texte", "msmarco-distilbert-base-v4 · nur englische Projekte", "Gemini gemini-embedding-001 · API-Key nötig", "OpenAI text-embedding-3-large · API-Key nötig", "paraphrase-multilingual-mpnet-base-v2 · symmetrisch, NUR zum Vergleich"]

!pip install -q "qum[local] @ git+https://github.com/DanielKremer91/query-url-matcher"

from qum import colab
from qum.embeddings import make_embedder
from qum.models import model_by_label

spec = model_by_label(modell)
api_key = None
if spec.provider == "openai":
    api_key = colab.secret("OPENAI_API_KEY")
elif spec.provider == "gemini":
    api_key = colab.secret("GEMINI_API_KEY")
print(f"⏳ Lade {spec.model_id} (lokale Modelle: beim ersten Mal bis zu 2 GB Download) ...")
with colab.guard():
    embedder = make_embedder(spec, api_key=api_key, cache_dir="/content/qum_cache")
if spec.comparison_only:
    print("⚠️ Dieses Modell ist symmetrisch und nur als Gegenbeispiel gedacht. Nutze es nicht für die Auswertung.")
print(f"✅ Schritt 2 fertig: Modell {spec.model_id} ist bereit. Weiter mit Schritt 3.")
'''

STEP3 = '''#@title Schritt 3: Queries und Frog-Export hochladen { display-mode: "form" }
#@markdown Die Spalten werden automatisch erkannt. Nur wenn die Zelle eine Spalte nicht findet, trägst du den Namen hier ein.
query_spalte = "" #@param {type:"string"}
url_spalte = "" #@param {type:"string"}
content_spalte = "" #@param {type:"string"}

from qum import colab, ingest
from qum.normalize import host_of

with colab.guard():
    name, data = colab.upload("Queries-Datei (CSV oder Excel, eine Query pro Zeile)")
    queries = ingest.load_queries(ingest.read_table(data, name), column=query_spalte or None)
    name, data = colab.upload("Screaming-Frog-Export (URL und Main Content)")
    content = ingest.load_content(
        ingest.read_table(data, name), url_col=url_spalte or None, content_col=content_spalte or None
    )
if not queries or not content.urls:
    colab.stop("Es wurden keine Queries oder keine Seiten mit Content gefunden.")
own_hosts = {host_of(u) for u in content.urls}
rankings = None
serps = None
print(f"Beispiel-Query: {queries[0]}")
print(f"Beispiel-URL:   {content.urls[0]}")
print(f"Content-Anfang: {content.contents[0][:120]} ...")
if content.skipped_empty or content.skipped_duplicate:
    print(f"ℹ️ Übersprungen: {content.skipped_empty} Seiten ohne Content, {content.skipped_duplicate} doppelte URLs.")
print(f"✅ Schritt 3 fertig: {len(queries)} Queries und {len(content.urls)} URLs geladen. Weiter mit Schritt 4.")
'''

STEP4 = '''#@title Schritt 4: Matching { display-mode: "form" }
#@markdown Der Content wird automatisch in überlappende Textblöcke (Chunks) zerlegt. **0 = Empfehlung für das gewählte Modell.**
chunk_groesse = 0 #@param {type:"integer"}
chunk_overlap = 0 #@param {type:"integer"}
#@markdown Welcher Score entscheidet über "passend" und die Sortierung? Im Export stehen immer alle drei.
bewertungsgrundlage = "Chunk" #@param ["Chunk", "Gesamt-URL", "Kombi"]
#@markdown Anteil des Chunk-Scores im Kombi-Score (der Rest ist der Gesamt-URL-Score).
kombi_gewicht_chunk = 0.7 #@param {type:"slider", min:0, max:1, step:0.05}
top_n = 5 #@param {type:"integer"}

from qum import colab
from qum import labels as L
from qum.match import estimate_chunks, run_matching, top_hits

size = chunk_groesse or spec.chunk_size
overlap = chunk_overlap or spec.chunk_overlap
if overlap >= size:
    colab.stop("Der Overlap muss kleiner sein als die Chunk-Größe.")
basis = L.BASIS[bewertungsgrundlage]
n_chunks = estimate_chunks(content.contents, size, overlap)
print(f"ℹ️ {len(content.urls)} URLs ergeben {n_chunks} Chunks ({size} Wörter, Overlap {overlap}).")
if n_chunks > 50000:
    print("⚠️ Das sind sehr viele Chunks. Der Lauf kann lange dauern. Schränke den Frog-Export auf ein Verzeichnis ein.")
with colab.guard():
    result = run_matching(queries, content.urls, content.contents, embedder, size, overlap)
inner = embedder.inner
if hasattr(inner, "truncated_share"):
    flat = [c for per_url in result.chunks for c in per_url]
    share = inner.truncated_share(flat[:2000])
    if share > 0.05:
        print(f"⚠️ {share:.0%} der Chunks sind länger als das Modell lesen kann. Verkleinere die Chunk-Größe.")
lead = result.lead(basis, kombi_gewicht_chunk)
top = top_hits(result, basis, kombi_gewicht_chunk, top_n)
display(top.head(10))
print(f"✅ Schritt 4 fertig: {len(top)} Treffer berechnet. Optional weiter mit Schritt 5 und 6, sonst Schritt 7.")
'''

STEP5 = '''#@title Schritt 5 (optional): Eigene Rankings hochladen { display-mode: "form" }
#@markdown Export mit Keyword, URL und Position, zum Beispiel aus GSC, Ahrefs (Organic Keywords) oder SISTRIX.
#@markdown Schaltet die Urteile zu Kannibalisierung und Content-Lücke frei.
keyword_spalte = "" #@param {type:"string"}
url_spalte_ranking = "" #@param {type:"string"}
position_spalte = "" #@param {type:"string"}

from qum import colab, ingest

with colab.guard():
    name, data = colab.upload("Eigene Rankings (Keyword, URL, Position)")
    rankings = ingest.load_rankings(
        ingest.read_table(data, name),
        keyword_col=keyword_spalte or None,
        url_col=url_spalte_ranking or None,
        position_col=position_spalte or None,
    )
known = {ingest.normalize_query(q) for q in queries}
covered = rankings["query_norm"].isin(known).sum()
print(f"✅ Schritt 5 fertig: {len(rankings)} Ranking-Zeilen, davon {covered} zu deinen Queries. Weiter mit Schritt 6 oder 7.")
'''

STEP6 = '''#@title Schritt 6 (optional): Top-10-SERPs hochladen { display-mode: "form" }
#@markdown Ahrefs-Export mit Keyword, URL, Position und Type (die kompletten Top 10 je Keyword, inklusive Wettbewerber).
#@markdown Schaltet die Bündelung der Content-Lücken zu Themen frei.

from qum import colab, ingest

with colab.guard():
    name, data = colab.upload("Top-10-SERPs (Keyword, URL, Position, Type)")
    serps = ingest.load_serps(ingest.read_table(data, name))
if rankings is None:
    rankings = ingest.own_rankings_from_serps(serps, own_hosts)
    print(f"ℹ️ Keine eigene Ranking-Datei: {len(rankings)} eigene Rankings aus den SERPs übernommen.")
print(f"✅ Schritt 6 fertig: SERPs für {serps['query_norm'].nunique()} Keywords geladen. Weiter mit Schritt 7.")
'''

STEP7 = '''#@title Schritt 7: Schwelle prüfen und Urteile bilden { display-mode: "form" }
#@markdown **Ab welchem Score gilt eine Seite als "passend"?** Lass 0 stehen, um den Vorschlag zu übernehmen.
#@markdown Schau dir die Beispiele unter der Zelle an, trage bei Bedarf einen eigenen Wert ein und führe die Zelle erneut aus.
schwelle = 0 #@param {type:"number"}
#@markdown Kalibrierung: Paare bis zu dieser Position gelten als "rankt heute gut" und liefern den Vorschlag.
kalibrierung_bis_position = 5 #@param {type:"integer"}
#@markdown Ab welcher Position gilt eine Query als gut rankend?
rankt_gut_bis_position = 10 #@param {type:"integer"}
#@markdown Kannibalisierung: maximaler Score-Abstand für "fast gleich gut" und Positionsgrenze für "rankt ebenfalls".
abstand_fast_gleich = 0.02 #@param {type:"number"}
sichtbar_bis_position = 20 #@param {type:"integer"}
#@markdown Clustering (nur mit Schritt 6): Mindest-Überschneidung der Top 10 und Mindest-Dichte im Cluster, in Prozent.
serp_ueberschneidung = 50 #@param {type:"slider", min:10, max:100, step:5}
cluster_dichte = 50 #@param {type:"slider", min:10, max:100, step:5}

from qum import labels as L
from qum.cannibal import find_cannibalization
from qum.serp import apply_serp, count_new_pages
from qum.threshold import examples_around, propose_threshold
from qum.verdict import build_decisions

proposal = propose_threshold(result, lead, rankings, max_position=kalibrierung_bis_position)
if proposal.source == "rankings":
    print(f"Vorschlag {proposal.value:.4f}: Diesen Score erreichen 75 % der {proposal.n_pairs} Paare, die heute gut ranken.")
else:
    print(f"Vorschlag {proposal.value:.4f}: mittlerer Score der besten Treffer (zu wenige Ranking-Paare für eine Kalibrierung).")
threshold = schwelle or proposal.value
print(f"Verwendete Schwelle: {threshold:.4f}")
print("Diese Paare liegen knapp über und knapp unter der Schwelle. Passt die Grenze?")
display(examples_around(result, lead, threshold))

decisions = build_decisions(result, lead, threshold, rankings, rankt_gut_bis_position, kombi_gewicht_chunk)
cannibal = find_cannibalization(result, lead, threshold, decisions, rankings, abstand_fast_gleich, sichtbar_bis_position)
if serps is not None:
    decisions = apply_serp(decisions, result, lead, serps, rankings, serp_ueberschneidung / 100, cluster_dichte / 100)
counts = decisions[L.C_VERDICT].value_counts()
for verdict, count in counts.items():
    print(f"   {count:>5} × {verdict}")
print(f"   {cannibal[L.C_QUERY].nunique():>5} Queries mit Kannibalisierungs-Hinweis")
print(f"   {count_new_pages(decisions):>5} neue Seiten aus den Content-Lücken")
print("✅ Schritt 7 fertig. Weiter mit Schritt 8 (Export).")
'''

STEP8 = '''#@title Schritt 8: Export { display-mode: "form" }
#@markdown Die Excel-Datei enthält ein Blatt je Auswertung und eine Lesehilfe.
zusaetzlich_csv_zip = False #@param {type:"boolean"}

from datetime import date

from qum import colab, export

settings = {
    "Datum": date.today().isoformat(),
    "Modell": spec.model_id,
    "Chunk-Größe (Wörter)": size,
    "Overlap (Wörter)": overlap,
    "Bewertungsgrundlage": bewertungsgrundlage,
    "Kombi-Gewicht Chunk": kombi_gewicht_chunk,
    "Schwelle": round(threshold, 4),
    "Rankt gut bis Position": rankt_gut_bis_position,
    "Abstand fast gleich": abstand_fast_gleich,
    "Sichtbar bis Position": sichtbar_bis_position,
    "Eigene Rankings": "ja" if rankings is not None else "nein",
    "Top-10-SERPs": "ja" if serps is not None else "nein",
}
sheets = export.build_sheets(decisions, top, cannibal, settings)
export.write_excel("query_url_matcher.xlsx", sheets)
colab.download("query_url_matcher.xlsx")
if zusaetzlich_csv_zip:
    export.write_csv_zip("query_url_matcher_csv.zip", sheets)
    colab.download("query_url_matcher_csv.zip")
print(f"✅ Schritt 8 fertig: {len(sheets)} Blätter exportiert. Schritt 9 ist optional.")
'''

STEP9 = '''#@title Schritt 9 (nachgelagert): Vorhandene Keyword-URL-Paare bewerten { display-mode: "form" }
#@markdown Lade einen Export mit Keyword und URL hoch (zum Beispiel Ahrefs Organic Keywords). Je Paar siehst du, wie gut die
#@markdown rankende URL passt und welche deiner Seiten am besten passen würde. Die URLs müssen im Frog-Export aus Schritt 3 stehen.

from qum import colab, export, ingest
from qum import labels as L
from qum.pairs import score_pairs

with colab.guard():
    name, data = colab.upload("Keyword-URL-Paare (Keyword, URL)")
    pairs = score_pairs(
        ingest.read_table(data, name), content.urls, content.contents, embedder, size, overlap, basis, kombi_gewicht_chunk
    )
missing = (pairs[L.C_NOTE] != "").sum()
if missing:
    print(f"ℹ️ {missing} Paare haben eine URL, die nicht im Frog-Export steht.")
display(pairs.head(10))
sheets = export.build_sheets(decisions, top, cannibal, settings, pairs=pairs)
export.write_excel("query_url_matcher.xlsx", sheets)
colab.download("query_url_matcher.xlsx")
print(f"✅ Schritt 9 fertig: {len(pairs)} Paare bewertet, Export mit Blatt 'Paare' heruntergeladen.")
'''

CELLS = [
    ("markdown", INTRO),
    ("code", STEP2),
    ("code", STEP3),
    ("code", STEP4),
    ("code", STEP5),
    ("code", STEP6),
    ("code", STEP7),
    ("code", STEP8),
    ("code", STEP9),
]


def _cell(kind: str, source: str) -> dict:
    lines = source.rstrip("\n").split("\n")
    body = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {
        "cell_type": "code",
        "metadata": {"cellView": "form"},
        "execution_count": None,
        "outputs": [],
        "source": body,
    }


def build() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [_cell(kind, source) for kind, source in CELLS],
    }


def main() -> None:
    path = ROOT / "query_url_matcher.ipynb"
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Geschrieben: {path}")


if __name__ == "__main__":
    main()
```

In `qum/ingest.py` wird `normalize_query` bereits importiert; Schritt 5 des Notebooks greift über `ingest.normalize_query` darauf zu.

- [ ] **Step 5: Notebook erzeugen und Tests bestehen sehen**

Run: `.venv/bin/python -m tools.build_notebook && .venv/bin/pytest tests/test_notebook.py -q`
Expected: `Geschrieben: …/query_url_matcher.ipynb`, danach 9 passed

- [ ] **Step 6: README schreiben**

`README.md` (auf Deutsch) mit diesen Abschnitten und Inhalten:

```markdown
# Query-URL Matcher

**Neue Seite bauen oder Bestehendes nutzen?** Ein Colab-Notebook, das Suchanfragen per Embeddings mit dem Inhalt deiner Seiten vergleicht und Hinweise zu Kannibalisierung und Content-Lücken liefert.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielKremer91/query-url-matcher/blob/main/query_url_matcher.ipynb)

## Was es beantwortet

- Gibt es für eine Query schon eine passende Seite oder Textstelle?
- Passt eine andere Seite besser als die, die gerade rankt (Kannibalisierungs-Risiko)?
- Für welche Queries fehlt Content, und wie viele neue Seiten sind das wirklich?

Das Notebook sortiert vor und begründet. Die Entscheidung trifft ein Mensch.

## Was du brauchst

| Datei | Pflicht | Inhalt |
|---|---|---|
| Queries | ja | CSV oder Excel, eine Query pro Zeile |
| Screaming-Frog-Export | ja | URL und Main Content |
| Eigene Rankings | optional | Keyword, URL, Position |
| Top-10-SERPs | optional | Ahrefs-Export mit Keyword, URL, Position, Type |

## Modelle

| Modell | Einsatz | Kosten |
|---|---|---|
| multilingual-e5-large | Empfehlung für Deutsch | kostenlos |
| multilingual-e5-base | schneller bei vielen URLs | kostenlos |
| bge-m3 | Deutsch, lange Texte | kostenlos |
| msmarco-distilbert-base-v4 | nur englische Projekte | kostenlos |
| Gemini gemini-embedding-001 | API | Secret `GEMINI_API_KEY` |
| OpenAI text-embedding-3-large | API | Secret `OPENAI_API_KEY` |
| paraphrase-multilingual-mpnet-base-v2 | symmetrisches Gegenbeispiel, nur zum Vergleich | kostenlos |

Die Modelle werden automatisch so angesteuert, wie sie trainiert wurden: e5 mit den Prefixen `query: ` und `passage: `, Gemini mit `task_type`, alle anderen mit reinem Text.

## Symmetrisch und asymmetrisch

Eine kurze Query gegen einen langen Text zu matchen ist eine asymmetrische Aufgabe. Modelle für Satzähnlichkeit (symmetrisch) bewerten eher, ob ein Text wie die Query klingt, und weniger, ob er sie beantwortet. Drei Regeln:

1. Wähle ein Modell, das für Retrieval trainiert wurde.
2. Steuere es so an, wie es trainiert wurde. Ein Prefix macht ein Modell nicht asymmetrisch, wenn es ihn nie gelernt hat.
3. Vergleiche keine absoluten Scores zwischen Modellen. Es zählt, ob die richtige Seite vorn landet.

## Ergebnis

Eine Excel-Datei mit den Blättern Lesehilfe, Entscheidung, Top-Treffer, Kannibalisierung, Content-Lücken, mit SERPs zusätzlich Lücken je Cluster, optional Paare. Auf Wunsch zusätzlich ein ZIP mit einer CSV je Blatt.

## Grenzen

- Für einige tausend bis wenige zehntausend URLs gedacht. Schränke große Websites auf ein Verzeichnis ein.
- Der Content liegt während des Laufs in deiner Colab-Sitzung, bei den API-Modellen zusätzlich beim Anbieter.

## Entwicklung

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest -q
    .venv/bin/python -m tools.build_notebook   # nach Änderungen an tools/build_notebook.py

## Modellvergleich

    .venv/bin/pip install -e ".[local]"
    OPENAI_API_KEY=... .venv/bin/python -m qum.compare --truth wahrheit.csv --content frog.csv

`wahrheit.csv` enthält je Zeile eine Query und die erwartete URL. Ausgegeben werden je Konfiguration Treffer auf Rang 1, Treffer in den Top 3 und der mittlere reziproke Rang.

## Lizenz

MIT
```

- [ ] **Step 7: Gesamte Testsuite laufen lassen**

Run: `.venv/bin/pytest -q`
Expected: alle Tests grün (rund 110), keine Warnung über Netzwerkzugriffe

- [ ] **Step 8: Commit**

```bash
git add qum/colab.py tools tests/test_notebook.py query_url_matcher.ipynb README.md
git commit -m "Add Colab notebook, notebook builder and README"
```

---

### Task 15: Echter Durchlauf

**Files:**
- Create: `examples/queries.csv`, `examples/frog_export.csv`, `examples/rankings.csv`, `examples/serps.csv`
- Modify: `README.md` (Abschnitt „Beispieldaten")

**Interfaces:**
- Consumes: das gesamte Paket und das Notebook

- [ ] **Step 1: Beispieldaten anlegen**

Vier kleine, erfundene Dateien zu einem Tierbedarf-Shop (keine Kundendaten):

- `examples/frog_export.csv`: Spalten `Address;Extract Main Content 1`, 8 Seiten mit je 150 bis 600 Wörtern deutschem Text zu klar getrennten Themen (getreidefreies Hundefutter, Nassfutter Katze, Kratzbaum, Hundeleine, Aquarium einrichten, Vogelfutter, Welpenerziehung, Katzenklo). Zwei der Seiten überschneiden sich bewusst (getreidefreies Hundefutter als eigene Seite und als Abschnitt im Ratgeber „Hundefutter Arten").
- `examples/queries.csv`: Spalte `Query`, 12 Queries: je eine pro Thema, zwei Varianten zu getreidefreiem Hundefutter, zwei ohne passende Seite („hamsterkäfig größe", „pferdedecke winter").
- `examples/rankings.csv`: Spalten `Keyword;URL;Position` mit Rankings für 8 der Queries, darunter eine Query mit zwei rankenden eigenen URLs und eine mit Position 35.
- `examples/serps.csv`: Spalten `Keyword;URL;Position;Type`, je Query 10 Zeilen `Organic` mit erfundenen Wettbewerber-URLs; die Varianten zu getreidefreiem Hundefutter teilen 7 von 10 URLs.

- [ ] **Step 2: Lokaler Durchlauf mit echtem Modell**

Run:

```bash
.venv/bin/pip install -q -e ".[local]"
.venv/bin/python - <<'EOF'
from pathlib import Path
from qum import export, ingest
from qum.cannibal import find_cannibalization
from qum.embeddings import make_embedder
from qum.match import run_matching, top_hits
from qum.models import get_model
from qum.normalize import host_of
from qum.serp import apply_serp, count_new_pages
from qum.threshold import propose_threshold
from qum.verdict import build_decisions

def table(name):
    return ingest.read_table(Path("examples", name).read_bytes(), name)

spec = get_model("e5-base")
queries = ingest.load_queries(table("queries.csv"))
content = ingest.load_content(table("frog_export.csv"))
rankings = ingest.load_rankings(table("rankings.csv"))
serps = ingest.load_serps(table("serps.csv"))
embedder = make_embedder(spec, cache_dir=".cache")
result = run_matching(queries, content.urls, content.contents, embedder, spec.chunk_size, spec.chunk_overlap)
lead = result.lead("chunk")
proposal = propose_threshold(result, lead, rankings)
decisions = build_decisions(result, lead, proposal.value, rankings)
cannibal = find_cannibalization(result, lead, proposal.value, decisions, rankings)
decisions = apply_serp(decisions, result, lead, serps, rankings)
sheets = export.build_sheets(decisions, top_hits(result), cannibal, {"Modell": spec.model_id})
export.write_excel("/tmp/qum_probe.xlsx", sheets)
print(proposal)
print(decisions[["Query", "Urteil", "Beste URL"]].to_string())
print(cannibal.to_string())
print("Neue Seiten:", count_new_pages(decisions))
EOF
```

Expected: Der Lauf endet ohne Fehler. Jede thematische Query hat ihre Themenseite als „Beste URL". „hamsterkäfig größe" und „pferdedecke winter" haben die niedrigsten besten Scores. Die Query mit zwei rankenden URLs steht in der Kannibalisierungs-Tabelle mit Stufe „Bereits sichtbar". Weicht ein Ergebnis davon ab, erst die Ursache klären (systematic-debugging), nicht die Beispieldaten passend biegen.

- [ ] **Step 3: README ergänzen und committen**

Abschnitt „Beispieldaten" in `README.md`: ein Satz je Datei unter `examples/` und der Hinweis, dass die Inhalte erfunden sind.

```bash
git add examples README.md
git commit -m "Add example data and verify a real run"
```

- [ ] **Step 4: Colab-Durchlauf (braucht Daniels Freigabe zum Push)**

Das Notebook installiert das Paket von GitHub. Dieser Schritt geht erst, nachdem Daniel den Push freigegeben hat und das Repo öffentlich ist. Dann:

1. Notebook über den „Open in Colab"-Link öffnen.
2. Schritte 2 bis 9 mit den Dateien aus `examples/` und `multilingual-e5-large` ausführen.
3. Prüfen: jede Zelle endet mit ihrer ✅-Zeile, die Excel-Datei wird heruntergeladen und enthält die Blätter Lesehilfe, Entscheidung, Top-Treffer, Kannibalisierung, Content-Lücken, Lücken je Cluster, Paare.
4. Einmal absichtlich eine Datei ohne Content-Spalte hochladen: Die Zelle zeigt „❌ Content-Spalte nicht erkannt …" ohne Traceback.

Bis zur Freigabe diesen Schritt als offen melden, nicht als erledigt.

---

## Nach dem Plan (eigene Schritte mit Daniel)

Diese beiden Punkte aus der Spec (Abschnitt 15) brauchen Eingaben von Daniel und sind keine Code-Aufgaben:

1. **Messung:** Daniel liefert eine Wahrheitsliste (30 bis 50 Queries mit erwarteter URL) oder einen Ranking-Export als Näherung, dazu den Frog-Export und seinen OpenAI-Key als Umgebungsvariable. Lauf mit `python -m qum.compare`, Ergebnis ist `modellvergleich.csv`.
2. **Slide:** eine PowerPoint-Folie für Daniels bestehendes Deck, gebaut mit dem pptx-Skill nach der Messung. Dafür wird sein Deck oder eine Beispielfolie als Vorlage gebraucht.
