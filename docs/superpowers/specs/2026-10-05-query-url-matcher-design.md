# Query-URL Matcher – Design

Stand: 2026-10-05. Abgestimmt mit Daniel im Brainstorming. Grundlage ist sein Colab-Skript „Query-URL Matcher v2.2".

## 1. Zweck

Das Notebook beantwortet je Suchanfrage oder Prompt die Frage: **Neue Seite bauen oder Bestehendes nutzen?**

Es vergleicht Queries per Embeddings mit dem Main Content der eigenen Seiten, auf Chunk-Ebene und auf Ebene der ganzen URL. Mit optionalen Ranking-Daten leitet es daraus Urteile ab. Im Vordergrund stehen zwei Anwendungsfälle:

- **Kannibalisierung:** vorbeugen (es gibt schon eine passende Seite, also keine neue bauen) und erkennen (mehrere eigene Seiten passen oder ranken für dieselbe Query).
- **Content-Lücke:** keine passende Seite vorhanden.

Das Notebook sortiert vor und begründet. Die Entscheidung trifft ein Mensch. Das steht in der Einführung und in der Lesehilfe des Exports.

## 2. Rahmen

- **Nur Google Colab.** Kein Streamlit.
- **Sprache:** Notebook, Meldungen und Export auf Deutsch.
- **Aufbau:** getestetes Python-Paket `qum` im GitHub-Repo `query-url-matcher`, dazu ein dünnes Notebook `query_url_matcher.ipynb`, das das Paket per `pip install git+https://github.com/DanielKremer91/query-url-matcher` lädt.
- **Lizenz:** MIT. README auf Deutsch mit „Open in Colab"-Button.
- **Größenordnung:** einige tausend bis wenige zehntausend URLs. Vor dem Einbetten schätzt das Notebook die Zahl der Chunks und warnt ab 50.000.
- **Kein generierter Text:** Empfehlungen sind feste Textbausteine mit eingesetzten Werten.
- **Push:** Das Repo wird erst auf Daniels Ansage nach GitHub gepusht. Bis dahin installiert das Notebook das Paket aus einem lokalen Stand.

Nicht im Umfang: eigener Crawler, Verarbeitung von Millionen Seiten, Ollama, eine Oberfläche außerhalb von Colab.

## 3. Eingaben

| Datei | Pflicht | Spalten | Schaltet frei |
|---|---|---|---|
| Queries | ja | eine Query oder ein Prompt je Zeile | Matching |
| Frog-Export | ja | URL, Main Content | Matching |
| Eigene Rankings | optional | Keyword, URL, Position | Urteile mit Ranking, Kannibalisierungs-Stufen, Kalibrierung der Schwelle |
| Top-10-SERPs | optional | Keyword, URL, Position, Type (Ahrefs) | Clustering, Nachbar-Hinweis |

Regeln:

- CSV (auch UTF-16 und Tab-getrennt) und Excel werden gelesen.
- Spalten werden über Aliaslisten erkannt (GSC, Ahrefs, SISTRIX, Semrush, Screaming Frog) und vor dem Rechnen als Vorschau gezeigt. Wird eine Pflichtspalte nicht erkannt, nennt die Fehlermeldung die gefundenen Spalten, und der Nutzer trägt den Spaltennamen im Formular ein. Fehlt in der SERP-Datei die Spalte Type, gelten alle Zeilen als organisch.
- URL und Content bleiben zeilenweise zusammen. Zeilen ohne Content und doppelte URLs werden übersprungen und gezählt gemeldet.
- Eigene Rankings und SERPs sind zwei getrennte Uploads. Fehlt die Ranking-Datei, zieht das Notebook die eigenen Rankings aus der SERP-Datei: Zeilen, deren Host im Frog-Export vorkommt. Liegen beide vor, gilt für Positionen die Ranking-Datei.
- Abgleich von Queries: Kleinschreibung, getrimmt, Mehrfach-Leerzeichen zusammengezogen.
- Abgleich von URLs: Schema und Host klein, Fragment entfernt, abschließender Schrägstrich vereinheitlicht, Tracking-Parameter (`utm_*`, `gclid`, `fbclid`, `msclkid`, `srsltid`) entfernt.

## 4. Modelle

| Modell | Rolle | Ansteuerung | Chunk-Default (Wörter / Overlap) |
|---|---|---|---|
| `intfloat/multilingual-e5-large` | Default, Deutsch | Prefix `query: ` / `passage: ` | 250 / 40 |
| `intfloat/multilingual-e5-base` | schneller | Prefix | 250 / 40 |
| `BAAI/bge-m3` | Deutsch, lange Texte | reiner Text | 1000 / 150 |
| `msmarco-distilbert-base-v4` | nur englische Projekte | reiner Text | 250 / 40 |
| Gemini `gemini-embedding-001` | API-Key | `task_type` `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | 800 / 120 |
| OpenAI `text-embedding-3-large` | API-Key | reiner Text, keine Prefixe | 1500 / 200 |
| `paraphrase-multilingual-mpnet-base-v2` | symmetrisches Gegenbeispiel, nur zum Vergleich | reiner Text | 250 / 40 |

- Die Ansteuerung geschieht automatisch. Texte im Export bleiben unverändert.
- `text-embedding-004` aus dem alten Skript ist seit 14.01.2026 abgeschaltet, Nachfolger ist `gemini-embedding-001`.
- Das Gegenbeispiel-Modell ist im Formular als „nur zum Vergleich, nicht für die Auswertung" beschriftet. Seine maximale Sequenzlänge wird auf 512 Tokens gesetzt, damit es dieselben Chunks wie e5 sieht.
- Lokale Modelle laufen über `sentence-transformers` (Extra `qum[local]`), die API-Modelle über `httpx` ohne SDK. Keys kommen aus den Colab-Secrets `OPENAI_API_KEY` und `GEMINI_API_KEY`.
- Bei lokalen Modellen prüft das Paket mit dem Tokenizer, wie viele Chunks die Sequenzlänge überschreiten, und meldet den Anteil.
- API-Aufrufe laufen gebündelt mit Wiederholung bei Fehlern (bis zu sechs Versuche mit Pausen von 2 bis 32 Sekunden, bei 429 und 503 nach dem Header Retry-After, höchstens 60 Sekunden). Fertige Teile landen sofort im Zwischenspeicher.

## 5. Matching

Je Query und URL entstehen immer drei Scores:

1. **Chunk-Score:** Der Content wird in überlappende Wort-Chunks zerlegt. Der Score ist der höchste Cosinus-Wert eines Chunks. Der beste Chunk steht im Export.
2. **Gesamt-URL-Score:** Passt der Main Content ins Kontextfenster, wird er als Ganzes eingebettet (Methode „Volltext"). Sonst gilt der normalisierte Mittelwert der Chunk-Vektoren (Methode „Mittelwert der Chunks"). Die Methode steht je URL im Export. Die Passung wird bei lokalen Modellen mit dem Tokenizer geprüft, bei OpenAI gilt die Grenze 3.000 Wörter, bei Gemini 1.000 Wörter.
3. **Kombi-Score:** `w × Chunk + (1 − w) × Gesamt-URL`, `w` per Schieberegler, Default 0,7.

Ein Formularfeld „Bewertungsgrundlage" (Chunk als Default, Gesamt-URL, Kombi) legt den **Leit-Score** fest. Er entscheidet über „passend" und die Sortierung. Alle drei Scores und Ränge stehen immer im Export.

Je Query werden die Top-N-URLs nach Leit-Score ausgegeben (Default 5, einstellbar).

Berechnete Vektoren werden in der Sitzung und auf der Colab-Platte zwischengespeichert, Schlüssel ist Modell, Rolle (Query oder Passage) und Text-Hash. Geänderte Schwellen oder ein wiederholter Schritt betten nichts neu ein. Die Ähnlichkeiten werden als Matrixoperation gerechnet.

## 6. Schwelle „passend"

Die Schwelle wird vorgeschlagen und vom Nutzer bestätigt.

- **Mit eigenen Rankings:** alle Paare aus Query und eigener URL mit Position ≤ 5 (einstellbar), bei denen die Query in der Query-Liste und die URL im Frog-Export steht. Vorschlag ist das 25. Perzentil ihrer Leit-Scores, also der Wert, den 75 % dieser Paare erreichen. Bei weniger als 20 Paaren gilt der Weg ohne Rankings.
- **Ohne Rankings:** Median der besten Leit-Scores aller Queries.
- **Prüfschritt:** Das Notebook zeigt fünf Paare knapp über und fünf knapp unter der Schwelle mit Query, URL und Passage. Die Schwelle wird per Schieberegler angepasst.

## 7. Urteile je Query

Eine URL „passt", wenn ihr Leit-Score mindestens die Schwelle erreicht. „Rankt gut" heißt: beste eigene Position ≤ 10 (einstellbar).

Ohne Rankings:

| Matching | Urteil |
|---|---|
| beste URL passt | passende Seite vorhanden |
| keine URL passt | keine passende Seite (Content-Lücke) |

Mit Rankings, in dieser Prüfreihenfolge:

| Ranking | Matching | Urteil |
|---|---|---|
| gut | keine URL passt | rankt trotz schwachem Match, beobachten |
| gut | rankende URL ist bester Treffer | in Ordnung |
| gut | eine andere URL ist bester Treffer und passt | Kannibalisierungs-Risiko |
| schwach oder keines | beste URL passt | bestehende Seite nutzen, nicht neu bauen |
| schwach oder keines | keine URL passt | Content-Lücke |

Steht die rankende URL nicht im Frog-Export, wird das in einer Hinweisspalte vermerkt und die Query wie „andere URL" behandelt.

## 8. Kannibalisierung

Eigene Auswertung mit zwei Stufen:

- **Risiko:** Urteil „Kannibalisierungs-Risiko" aus Abschnitt 7, oder zwei und mehr eigene URLs passen und liegen im Leit-Score höchstens 0,02 auseinander (einstellbar). Der zweite Fall funktioniert auch ohne Rankings.
- **Bereits sichtbar:** zwei oder mehr eigene URLs ranken für die Query mit Position ≤ 20 (einstellbar).

Ausgegeben werden die Query, die Stufe, die konkurrierenden URLs mit Scores und Positionen.

## 9. SERP-Clustering

Nur mit Top-10-SERPs. Übernommen aus dem alten Skript:

- nur Zeilen mit Type „organic", je Keyword die ersten 10 URLs nach Position
- Überschneidung zweier Keywords: gemeinsame URLs geteilt durch die kleinere Listenlänge, höchstens 10
- Kante ab Überschneidung ≥ 50 % (einstellbar)
- Cluster sind zusammenhängende Komponenten, bereinigt über die Dichte: Ein Keyword bleibt nur, wenn es mit mindestens 50 % (einstellbar) der übrigen Mitglieder eine Kante hat. Das verhindert den Ketteneffekt.

Verwendung:

- **Cluster-Spalte** an allen Ergebniszeilen.
- **Nachbar-Hinweis** bei Lücken-Queries: Hat ein direkter Nachbar (Kante, nicht bloß gleiches Cluster) eine passende Seite, erscheint sie als Kandidat mit Nachbar-Keyword, Überschneidung, eigenem Leit-Score der Lücken-Query gegen diese Seite, bester Passage und, falls vorhanden, Position. Bei mehreren Nachbarn mit verschiedenen Seiten werden alle gelistet, höchste Überschneidung zuerst. Das Urteil lautet dann „vor Neuerstellung prüfen".
- **Zusammenfassung** „Lücken je Cluster": Anzahl Lücken-Queries und daraus die Zahl der tatsächlich neuen Seiten (ein Cluster zählt als eine Seite, Queries ohne Cluster einzeln).

## 10. Nachgelagert: Paare bewerten

Letzter, optionaler Schritt. Eingabe ist ein Export mit Keyword und URL (weitere Spalten bleiben erhalten). Je Paar: die drei Scores, der Rang dieser URL unter allen URLs für das Keyword und die beste URL zum Vergleich. URLs müssen im Frog-Export stehen, fehlende werden gemeldet. Keywords außerhalb der Query-Liste werden nachträglich eingebettet.

## 11. Nutzerführung im Notebook

Schritt-Zellen mit Colab-Formularen (`#@param`), Code eingeklappt. Jede Zelle hat Überschrift und Kurzerklärung, prüft ihre Eingaben, meldet Fehler in klarem Deutsch mit Lösungshinweis und endet mit einer Statuszeile samt Verweis auf den nächsten Schritt.

1. Einführung: was das Notebook beantwortet, was man braucht, wie man die Ergebnisse liest
2. Installation und Modellwahl
3. Upload Queries und Frog-Export, Vorschau der erkannten Spalten
4. Matching (Chunk-Einstellungen, Bewertungsgrundlage, Kombi-Gewicht)
5. Optional: eigene Rankings
6. Optional: Top-10-SERPs
7. Schwelle prüfen und Urteile bilden
8. Export
9. Nachgelagert: Paare bewerten

Ohne die Schritte 5 und 6 arbeitet das Notebook wie das alte Skript, nur mit den Korrekturen.

## 12. Export

Standard ist eine Excel-Datei, per Häkchen zusätzlich ein ZIP mit einer CSV je Blatt.

| Blatt | Inhalt | Vorhanden |
|---|---|---|
| Lesehilfe | Bedeutung jedes Blatts und jeder Spalte, alle Einstellungen des Laufs | immer |
| Entscheidung | eine Zeile je Query: Urteil, beste URL, Passage, drei Scores, rankende URL mit Position, Cluster, Nachbar-Kandidat, Empfehlung | immer |
| Top-Treffer | Top-N-URLs je Query mit Scores, Rängen, Methode der Gesamt-URL | immer |
| Kannibalisierung | Abschnitt 8 | immer (Stufe „bereits sichtbar" nur mit Rankings) |
| Content-Lücken | Lücken-Queries, mit SERPs zusätzlich Cluster und Nachbar-Kandidat | immer |
| Lücken je Cluster | Zusammenfassung: Lücken je Cluster und Zahl der neuen Seiten | nur mit SERPs |
| Paare | Abschnitt 10 | nur wenn der Schritt lief |

Die Query steht in jedem Blatt in der ersten Spalte. Urteile sind farbig hinterlegt.

## 13. Paketaufbau

| Modul | Aufgabe |
|---|---|
| `qum/ingest.py` | Dateien lesen, Spalten erkennen, Zeilen bereinigen |
| `qum/normalize.py` | Queries und URLs vergleichbar machen |
| `qum/chunk.py` | Wort-Chunks |
| `qum/models.py` | Modellregister: Ansteuerung, Chunk-Defaults, Kontextgrenzen |
| `qum/embeddings/` | Basis-Schnittstelle, lokal, OpenAI, Gemini, Cache |
| `qum/match.py` | drei Scores, Ränge, Top-N |
| `qum/threshold.py` | Vorschlag der Schwelle, Beispielpaare |
| `qum/verdict.py` | Urteile, Textbausteine |
| `qum/cannibal.py` | Kannibalisierungs-Stufen |
| `qum/serp.py` | Clustering, Nachbarn, Lücken-Zusammenfassung |
| `qum/pairs.py` | Paare bewerten |
| `qum/export.py` | Excel und CSV-ZIP |
| `qum/compare.py` | Modellvergleich für die Messung (Abschnitt 15) |

Jedes Modul ist ohne Notebook und ohne echtes Modell testbar.

## 14. Tests

`pytest`, ohne Netzwerk und ohne Modell-Download (ein Test-Embedder liefert feste Vektoren). Abgedeckt: verrutschte Zeilen, Encodings, Spaltenerkennung, Chunk-Grenzen, Ansteuerung je Modell (Prefix nur bei e5), Methode der Gesamt-URL, Kombi-Gewicht, jede Zeile der Urteilstabellen, beide Kannibalisierungs-Stufen, Ketteneffekt und Dichte beim Clustering, Nachbar-Regel, Export-Blätter. Ein Test prüft, dass die Code-Zellen des Notebooks gültiges Python sind.

Vor der Übergabe: ein echter Durchlauf in Colab mit `multilingual-e5-large` und einem kleinen Datensatz.

## 15. Messung und Slide

**Messung:** `qum/compare.py` nimmt eine Wahrheitsliste (Query, erwartete URL) und den Frog-Export und gibt je Konfiguration Treffer auf Rang 1, Treffer in den Top 3 und den mittleren reziproken Rang aus. Konfigurationen: Paraphrasen-Modell, e5 mit Prefix, OpenAI ohne Prefix, OpenAI mit `query: `/`passage: `. Die Prefix-Variante für OpenAI gibt es nur hier, nicht im Notebook. Die Wahrheitsliste kommt von Daniel (30 bis 50 Queries) oder als Näherung aus einem Ranking-Export. Der Lauf braucht Daniels OpenAI-Key.

**Slide:** eine PowerPoint-Folie für Daniels bestehendes Deck, gebaut nach der Messung. Inhalt: symmetrisch gegen asymmetrisch an einem Beispiel, drei Warnungen (Modell passend zur Aufgabe wählen, so ansteuern wie trainiert, keine absoluten Scores zwischen Modellen vergleichen), das Messergebnis. Dafür wird Daniels Deck oder eine Beispielfolie als Vorlage gebraucht.

## 16. Reihenfolge

Plan, Paket mit Tests, Notebook, Colab-Durchlauf, Messung, Slide.
