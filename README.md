# Query-URL Matcher

**Neue Seite bauen oder Bestehendes nutzen?** Ein Colab-Notebook, das Suchanfragen per Embeddings mit dem Inhalt deiner Seiten vergleicht und Hinweise zu Kannibalisierung und Content-Lücken liefert.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielKremer91/query-url-matcher/blob/main/query_url_matcher.ipynb)

## Was es beantwortet

- Gibt es für eine Query schon eine passende Seite oder Textstelle?
- Passt eine andere Seite deutlich besser als die, die gerade rankt (Kannibalisierungsgefahr)? Liegt die rankende Seite höchstens 0,01 hinter der besten (einstellbar in Schritt 7c), gilt sie als bester Treffer.
- Für welche Queries fehlt Content, und wie viele neue Seiten sind das wirklich?

Das Notebook sortiert vor und begründet. Die Entscheidung trifft ein Mensch.

## So startest du

1. Öffne das Notebook mit dem Colab-Button oben.
2. Für die Modelle mit API-Key (Gemini, OpenAI): Lege im Secrets-Panel (Schlüssel-Symbol links) das Secret `GEMINI_API_KEY` bzw. `OPENAI_API_KEY` an und aktiviere dort den Schalter "Notebook-Zugriff". Ohne diesen Schalter kann das Notebook den Key nicht lesen. Die lokalen Modelle brauchen kein Secret.
3. Führe die Zellen einzeln von oben nach unten aus, nicht mit "Alle ausführen": Die Zellen fragen nach Datei-Uploads.
4. Die Schritte 5 (eigene Rankings) und 6 (Top-10-SERPs) sind optional.
5. Schwelle und Urteile entstehen in drei Zellen: Schritt 7a zeigt Vorschläge für die Schwelle „passend" mit Beispielen und bildet noch keine Urteile. In Schritt 7b legst du die Schwelle fest (aus Rankings kalibriert, mittlerer bester Score oder eigener Wert). Schritt 7c bildet die Urteile. Erst danach gibt es den Export (Schritt 8). Willst du nur die Feineinstellungen der Urteile ändern, starte nur Schritt 7c erneut.
6. Für einen ersten Versuch liegen kleine Beispieldateien unter `examples/` (Queries, Frog-Export, Rankings, SERPs).

Führst du einen früheren Schritt erneut aus, setzt das Notebook alles zurück, was darauf aufbaut, und sagt dir, welchen Schritt du wiederholen musst.

## Was du brauchst

| Datei | Pflicht | Inhalt |
|---|---|---|
| Queries | ja | CSV oder Excel, eine Query pro Zeile |
| Screaming-Frog-Export | ja | URL und Main Content: Custom Extraction mit dem Hauptinhalt als Text (Configuration → Custom → Custom Extraction, Name mit „Content", z. B. „Main Content"), Export aus dem Tab „Custom Extraction" |
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

Eine Excel-Datei mit den Blättern Lesehilfe, Entscheidung, Kannibalisierungsgefahr, Content-Lücken, mit SERPs zusätzlich Lücken je Cluster. Auf Wunsch zusätzlich ein ZIP mit einer CSV je Blatt: mit Semikolon und Dezimalkomma für deutsches Excel oder mit Komma und Dezimalpunkt.

## Grenzen

- Für einige tausend bis wenige zehntausend URLs gedacht. Schränke große Websites auf ein Verzeichnis ein.
- Der Content liegt während des Laufs in deiner Colab-Sitzung, bei den API-Modellen zusätzlich beim Anbieter.

## Beispieldaten

Im Ordner `examples/` liegen vier kleine Dateien zu einem erfundenen Tierbedarf-Shop. Alle Inhalte, Domains und Rankings sind erfunden, es sind keine Kundendaten.

- `queries.csv`: 12 Queries, je eine zu acht Themen, zwei Varianten zu getreidefreiem Hundefutter und zwei Queries ohne passende Seite.
- `frog_export.csv`: Screaming-Frog-Export mit URL und Main Content von 9 Seiten, darunter der Ratgeber „Hundefutter Arten", der sich bewusst mit der Seite zu getreidefreiem Hundefutter überschneidet.
- `rankings.csv`: eigene Rankings für 9 Kombinationen aus Keyword und URL, mit einer Query, für die zwei eigene Seiten ranken, und einer auf Position 35.
- `serps.csv`: Top-10-Ergebnisse (Organic) mit erfundenen Wettbewerber-URLs, die drei Queries zu getreidefreiem Hundefutter teilen sich 7 von 10 URLs.

Alle Domains enden auf `.example`, eine für Beispiele reservierte Endung (der Shop heißt `www.tierbedarf.example`).

Was du beim Ausprobieren erwarten kannst:

- Das Beispiel ist zu klein, um die Schwelle zu kalibrieren: Es hat weniger als 20 Ranking-Paare. In Schritt 7b bleiben deshalb nur der mittlere beste Score (nicht kalibriert) und ein eigener Wert; die Vorgabe „Aus Rankings kalibriert" bricht mit einem Hinweis ab. Mit dem mittleren besten Score liegt etwa die Hälfte der Queries per Konstruktion unter der Schwelle und wird als Lücke oder schwacher Match markiert.
- Kurze Ein-Wort-Queries (zum Beispiel „kratzbaum") können auch gegen die richtige Seite niedrig scoren. Schau dir in Schritt 7a die Beispiele um die Schwelle an und setze in Schritt 7b bei Bedarf einen eigenen Wert.

## Entwicklung

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest -q
    .venv/bin/python -m tools.build_notebook   # nach Änderungen an tools/build_notebook.py

Das Notebook installiert das Paket vom Git-Tag seiner Version (`@v0.1.0`). So gibst du eine neue Version frei:

1. Version in `qum/__init__.py` und `pyproject.toml` erhöhen (beide gleich).
2. Notebook neu bauen: `.venv/bin/python -m tools.build_notebook`.
3. Committen.
4. Tag anlegen: `git tag v<version>`.
5. Mit Tag pushen: `git push && git push origin v<version>`.

## Modellvergleich

    .venv/bin/pip install -e ".[local]"
    OPENAI_API_KEY=... .venv/bin/python -m qum.compare --truth wahrheit.csv --content frog.csv

`wahrheit.csv` enthält je Zeile eine Query und die erwartete URL. Ausgegeben werden je Konfiguration Treffer auf Rang 1, Treffer in den Top 3 und der mittlere reziproke Rang.

## Lizenz

MIT
