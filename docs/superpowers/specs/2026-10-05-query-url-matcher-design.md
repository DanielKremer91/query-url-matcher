# Query-URL Matcher – Design

Stand: 2026-10-05, Export und Urteile überarbeitet für Version 0.2.0 (2026-10-07). Abgestimmt mit Daniel im Brainstorming. Grundlage ist sein Colab-Skript „Query-URL Matcher v2.2".

## 1. Zweck

Das Notebook beantwortet je Suchanfrage oder Prompt die Frage: **Neue Seite bauen oder Bestehendes nutzen?**

Es vergleicht Queries per Embeddings mit dem Main Content der eigenen Seiten, auf Chunk-Ebene und auf Ebene der ganzen URL. Mit optionalen Ranking-Daten leitet es daraus Urteile ab. Im Vordergrund stehen zwei Anwendungsfälle:

- **Kannibalisierung:** vorbeugen (es gibt schon eine passende Seite, also keine neue bauen) und erkennen (mehrere eigene Seiten passen oder ranken für dieselbe Query).
- **Content-Lücke:** keine passende Seite vorhanden.

Das Notebook sortiert vor und begründet. Die Entscheidung trifft ein Mensch. Das steht in der Einführung und in der Lesehilfe des Exports.

## 2. Rahmen

- **Nur Google Colab.** Kein Streamlit.
- **Sprache:** Notebook, Meldungen und Export auf Deutsch.
- **Aufbau:** getestetes Python-Paket `qum` im GitHub-Repo `query-url-matcher`, dazu ein dünnes Notebook `query_url_matcher.ipynb`, das das Paket per `pip install git+https://github.com/DanielKremer91/query-url-matcher@v<version>` lädt, festgelegt auf den Git-Tag der Paketversion (`qum.__version__`, gleich der Version in `pyproject.toml`).
- **Lizenz:** MIT. README auf Deutsch mit „Open in Colab"-Button.
- **Größenordnung:** einige tausend bis wenige zehntausend URLs. Vor dem Einbetten schätzt das Notebook die Zahl der Chunks und warnt ab 50.000.
- **Kein generierter Text:** Gründe und Erklärungen sind feste Textbausteine.
- **Push:** Das Repo wird erst auf Daniels Ansage nach GitHub gepusht. Bis dahin installiert das Notebook das Paket aus einem lokalen Stand.

Nicht im Umfang: eigener Crawler, Verarbeitung von Millionen Seiten, Ollama, eine Oberfläche außerhalb von Colab.

## 3. Eingaben

| Datei | Pflicht | Spalten | Schaltet frei |
|---|---|---|---|
| Queries | ja | eine Query oder ein Prompt je Zeile | Matching |
| Frog-Export | ja | URL, Main Content | Matching |
| Eigene Rankings | optional | Keyword, URL, Position | Urteile mit Ranking, Stufe „Kannibalisierung bereits sichtbar“, Rankingposition in den Blättern, Kalibrierung der Schwelle |
| Top-10-SERPs | optional | Keyword, URL, Position, Type (Ahrefs) | Thema der potentiellen Content-Lücken und Zahl neuer Seiten (Clustering) |

Regeln:

- CSV (auch UTF-16) und Excel werden gelesen. Das Trennzeichen (`;`, Tab oder `,`) wird an der ersten Zeile erkannt.
- Steht jede Zeile in einer einzigen Zelle (zum Beispiel eine Excel-Spalte mit „Keyword;URL;Position"), zerlegt das Notebook sie automatisch: nur wenn die Kopfzelle ein Trennzeichen enthält, mindestens ein Teil davon ein bekannter Spaltenname ist (Query, Keyword, URL, Content, Position oder Type, wie bei der Spaltenerkennung) und mindestens 80 % der Zeilen gleich viele Felder haben. Eine Liste von Prompts ohne Kopfzeile, die Kommas enthalten, bleibt so eine Spalte. Bleibt eine solche Datei mit unbekannten Spaltennamen unzerlegt, erklärt die Meldung „Spalte nicht erkannt" zusätzlich, wie man sie reparieren kann.
- Spalten werden über Aliaslisten erkannt (GSC, Ahrefs, SISTRIX, Semrush, Screaming Frog) und vor dem Rechnen als Vorschau gezeigt. Wird eine Pflichtspalte nicht erkannt, nennt die Fehlermeldung die gefundenen Spalten, und der Nutzer trägt den Spaltennamen im Formular ein. Fehlt in der SERP-Datei die Spalte Type, gelten alle Zeilen als organisch.
- URL und Content bleiben zeilenweise zusammen. Zeilen ohne Content und doppelte URLs werden übersprungen und gezählt gemeldet.
- Eigene Rankings und SERPs sind zwei getrennte Uploads. Fehlt die Ranking-Datei, zieht das Notebook die eigenen Rankings aus der SERP-Datei: Zeilen, deren Host im Frog-Export vorkommt. Liegen beide vor, gilt für Positionen die Ranking-Datei.
- Abgleich von Queries: Kleinschreibung, getrimmt, Mehrfach-Leerzeichen zusammengezogen.
- Abgleich von URLs: http und https gleich, Host klein und ohne „www.“ und Standard-Port (www.toom.de = toom.de; andere Subdomains bleiben verschieden), URLs ohne Protokoll werden als https gelesen, Fragment entfernt, abschließender Schrägstrich vereinheitlicht, Tracking-Parameter (`utm_*`, `gclid`, `fbclid`, `msclkid`, `srsltid`) entfernt.

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
- Lokale Modelle beginnen mit Stapelgröße 32. Meldet die Grafikkarte zu wenig Speicher, halbiert das Paket die Stapelgröße, meldet das in einer Zeile und wiederholt den Aufruf; die kleinere Größe bleibt für den Rest der Sitzung. Reicht auch Stapelgröße 1 nicht, bricht der Lauf mit einer deutschen Meldung ab (Chunk-Größe verkleinern oder kleineres Modell wählen).
- API-Aufrufe laufen gebündelt. Wiederholt wird nur bei 408, 429, 5xx und Verbindungsfehlern (bis zu sechs Versuche mit Pausen von 2 bis 32 Sekunden, bei 429 und 503 nach dem Header Retry-After, höchstens 60 Sekunden); andere Fehler brechen sofort mit Meldung ab. Ab 5 Sekunden Pause erscheint eine deutsche Meldung mit Grund und Versuch, immer auf einer eigenen Zeile.
- Fehlende Texte gehen in Scheiben von 256 an das Modell. Der Zwischenspeicher auf der Platte wird höchstens alle 60 Sekunden geschrieben, dazu am Ende und bei Fehler oder Abbruch, damit fertige Scheiben erhalten bleiben. Der Fortschritt steht in einer einzigen Zeile für den ganzen Lauf („⏳ 512 von 1.000 Texten eingebettet …", fertig „✅ 1.000 von 1.000 Texten eingebettet").

## 5. Matching

Je Query und URL entstehen immer drei Scores:

1. **Chunk-Score:** Der Content wird in überlappende Wort-Chunks zerlegt. Der Score ist der höchste Cosinus-Wert eines Chunks. Der beste Chunk steht im Export.
2. **Gesamt-URL-Score:** Passt der Main Content ins Kontextfenster, wird er als Ganzes eingebettet (Methode „Volltext"). Sonst gilt der Durchschnitt der Chunk-Scores (Methode „Mittelwert der Chunk-Scores"). Er liegt nie über dem besten Chunk. Bis 0.2.5 war es der neu normierte Mittelwert der Chunk-Vektoren; der behielt nur das gemeinsame Thema der Seite und lag deshalb oft über jedem einzelnen Chunk, sodass lange Seiten bei Gesamt-URL und Kombi bevorzugt wurden (Entscheidung Daniel, 2026-10-07). Die Methode wird nicht exportiert. Die Passung wird bei lokalen Modellen mit dem Tokenizer geprüft; bei bge-m3 zählt eine Seite nur bis 2.048 Tokens als passend, weil das Einbetten ganzer langer Seiten mehrere Sekunden je Seite und viel GPU-Speicher kostet (längere Seiten nehmen den Mittelwert der Chunks). Bei OpenAI gilt die Grenze 3.000 Wörter, bei Gemini 1.000 Wörter.
3. **Kombi-Score:** `w × Chunk + (1 − w) × Gesamt-URL`, `w` per Schieberegler, Default 0,7.

Ein Formularfeld „Bewertungsgrundlage" (Chunk als Default, Gesamt-URL, Kombi) legt den **Leit-Score** fest. Er entscheidet über „passend", über beste, zweitbeste und drittbeste URL und die Sortierung. Alle drei Scores stehen immer im Export.

Je Query gibt das Notebook die drei URLs mit dem höchsten Leit-Score aus, je mit allen drei Scores (Blatt Übersicht). Schritt 4 zeigt danach eine Vorschau der ersten 10 Queries mit bester und zweitbester URL und ihrem Leit-Score, gebaut aus derselben Funktion.

Berechnete Vektoren werden in der Sitzung und auf der Colab-Platte zwischengespeichert, Schlüssel ist Modell, Rolle (Query oder Passage) und Text-Hash. Geänderte Schwellen oder ein wiederholter Schritt betten nichts neu ein. Die Ähnlichkeiten werden als Matrixoperation gerechnet.

## 6. Schwelle „passend"

Die Schwelle ist die Cosinus-Ähnlichkeit (Leit-Score) zwischen Query und Seite, ab der eine Seite als passend gilt. Schritt 7 ist in drei Zellen geteilt: 7a zeigt die Vorschläge, 7b legt die Schwelle fest, 7c bildet die Urteile. Der Nutzer wählt in 7b ausdrücklich, wie die Schwelle bestimmt wird; nichts wird stillschweigend übernommen.

- **Schritt 7a (Vorschläge ansehen):** erklärt die Schwelle und zeigt die verfügbaren Vorschläge, je mit Wert, Bedeutung und Beispielpaaren. Bildet keine Urteile.
- **Schritt 7b (Schwelle festlegen):** Auswahl aus den drei folgenden Optionen, „Aus Rankings kalibriert" ist vorgewählt.
- **Aus Rankings kalibriert:** alle Paare aus Query und eigener URL mit Position ≤ 5 (einstellbar), bei denen die Query in der Query-Liste und die URL im Frog-Export steht, jedes Paar einmal. Wert ist das 25. Perzentil ihrer Leit-Scores, also der Wert, den 75 % dieser Paare erreichen. Angeboten nur ab 20 Paaren. Sonst nennt das Notebook den Grund (keine Rankings geladen oder nur n Paare bis Position k) und stoppt, wenn diese Option gewählt ist.
- **Mittlerer bester Score (nicht kalibriert):** Median der besten Leit-Scores aller Queries. Etwa die Hälfte der Queries liegt per Konstruktion darunter; das sagt das Notebook, die Zahl der neuen Seiten ist als „Schwelle nicht kalibriert" markiert.
- **Eigener Wert:** größer als 0 und höchstens 1.
- Der Abstand „fast gleich" (Abschnitt 7 und 8) muss mindestens 0 und kleiner als 0,1 sein; die Meldung sagt, dass er eine Differenz von Cosinus-Scores ist, kein Prozentwert.
- **Schritt 7c (Urteile bilden):** Feineinstellungen und Urteile; lässt sich allein neu starten, ohne die Schwelle neu zu wählen. Schritt 8 verlangt 7c. Wer eine Zelle überspringt, bekommt eine Meldung mit dem fehlenden Teilschritt. Bricht eine Zelle wegen einer ungültigen Eingabe ab, bleiben Urteile aus einem früheren Lauf stehen, und die Meldung sagt das.
- **Prüfschritt:** Zu jedem Vorschlag und zur verwendeten Schwelle zeigt das Notebook fünf Paare knapp über und fünf knapp unter dem Wert mit Query, URL und Passage.

Vorschläge werden einmal auf 4 Nachkommastellen gerundet. Die Wahl steht in den Einstellungen der Lesehilfe, dazu ein fester Hinweis je Herkunft (Rankings, Median, von Hand).

## 7. Urteile je Query

Eine URL „passt", wenn ihr Leit-Score mindestens die Schwelle erreicht. „Fast gleich gut" heißt: Mehrere URLs passen und liegen höchstens um den Abstand „fast gleich" (Default 0,01, einstellbar) unter dem besten Leit-Score. Verglichen werden die auf 4 Nachkommastellen gerundeten Leit-Scores; ein negativer Abstand ist ein Fehler. Passt genau eine URL und liegen alle anderen passenden weiter dahinter, passt sie „klar". „Rankt gut" heißt: beste eigene Position ≤ 10 (einstellbar).

Zwei Szenarien (so erklärt im Notebook vor Schritt 7a):

- **Ohne Rankings, und mit Rankings für Queries, die schwach oder gar nicht ranken (semantisch):** genau eine Seite passt klar → passende Seite vorhanden; mehrere passen fast gleich gut → Kannibalisierungsgefahr; keine passt → Content-Lücke.
- **Mit Rankings und gutem Ranking (von Google aus):** die rankende Seite erreicht die Schwelle oder fehlt im Frog-Export (nicht prüfbar) → in Ordnung; sonst → rankt trotz schwachem Match (auch wenn keine Seite passt).
- **Keine Content-Lücke bei Ranking bis Position 20:** Passt keine Seite, rankt aber eine eigene Seite bis `luecke_nur_ohne_ranking_bis_position` (Voreinstellung 20, 0 = aus), lautet das Urteil „rankt trotz schwachem Match" statt Content-Lücke. Dieselbe Einstellung gilt für das Lücken-Blatt (Entscheidung Daniel, 2026-10-07, nach „fenster streichen": Position 13 mit genau der passenden Seite).

Prüfreihenfolge: zuerst „keine Seite passt", dann gutes Ranking, dann die Zahl der fast gleich guten Seiten. Bei gutem Ranking richtet sich das Urteil nach der rankenden Seite (Entscheidung Daniel, 2026-10-07: bei TOOM waren die so gemeldeten Fälle Hub-Seiten, die das Modell knapp vorzieht, während Google die richtige Detailseite rankt). Passt eine andere eigene Seite besser oder fast gleich gut, steht trotzdem Kannibalisierungsgefahr „ja“, weil Google die rankende Seite wechseln kann. Passen weitere Seiten nur deutlich dahinter, steht „möglich“ (Entscheidung Daniel, 2026-10-07: auch dann einen Hinweis, aber abgestuft).

Die Übersicht nennt immer die wirklich beste URL nach Leit-Score, auch bei „in Ordnung"; die rankende URL wird nicht eingesetzt. Wie die rankende URL zur besten steht, zeigt die Spalte „Rankende URL = beste URL?" (Abschnitt 11).

## 8. Kannibalisierungsgefahr

Intern gibt es drei Arten von Fällen (Stufen); das Blatt zeigt sie seit 0.4.0 nicht mehr einzeln, sie bestimmen nur, welche URLs dastehen und wie die Einordnung ausfällt (Entscheidung Daniel, 2026-10-07: Ausgabe verschlanken, Stufe, Grund und die Übersichtsspalte ja / möglich / nein entfallen):

- **Gefahr:** jede Query mit dem Urteil „Kannibalisierungsgefahr" und jede gut rankende Query, bei der eine andere passende Seite besser oder fast gleich gut ist als die rankende (Urteil „in Ordnung" oder „rankt trotz schwachem Match"). Gründe: „deutlich besser" (die beste Seite liegt mehr als einen positiven Abstand vor der rankenden), „besser" (Abstand 0 oder die rankende Seite erreicht die Schwelle nicht), „nicht verglichen" (rankende URL fehlt im Frog-Export), „Rankende Seite passt, eine weitere passt fast gleich gut" (in Ordnung) und „Mehrere Seiten passen fast gleich gut" (schwaches, fehlendes oder kein Ranking). Funktioniert auch ohne Rankings.
- **Möglich:** keine Gefahr, aber zwei oder mehr Seiten erreichen die Schwelle; die weiteren liegen mehr als den Abstand „fast gleich" hinter der besten. Grund „Weitere Seiten erreichen die Schwelle, liegen aber deutlich hinter der besten". URLs nach Leit-Score sortiert.
- **Kannibalisierung bereits sichtbar:** zwei oder mehr eigene URLs ranken für die Query mit Position ≤ 20 (einstellbar).

Blatt im Langformat: eine Zeile je Query und konkurrierender URL, die Query steht in jeder Zeile. Alle URLs aus allen Fällen der Query, jede Seite einmal (www- und Frog-Schreibweise zusammengeführt), nach Leit-Score sortiert, Seiten ohne Score (nicht im Frog-Export) zuletzt nach Position. „Nr." zählt die URLs je Query. „Score der URL" ist der Leit-Score, „Abstand zur besten URL" der Abstand zur semantisch besten URL der Query, „Position der URL" die eigene Position (leer, wenn sie nicht rankt). Rankingposition, Rankende URL und Einordnung stehen in jeder Zeile; je Query gilt die dringendste Einordnung ihrer Fälle. Die Spalte Einordnung sagt, wie dringend der Fall ist (Entscheidung Daniel, 2026-10-07, Variante A): ohne Rankings „offen". Stufe Gefahr und Kannibalisierung bereits sichtbar nach dem besten eigenen Ranking: Top-Ranking (bis rankt_gut_bis_position) der besten URL „niedrig", einer anderen URL oder einer URL außerhalb des Frog-Exports „mittel"; danach bis sichtbar_bis_position (Voreinstellung 20) „sehr hoch" (fast oben, die Konkurrenz bremst vermutlich); schlechter oder kein Ranking „hoch". Stufe Möglich: „mittel", wenn eine andere als die beste URL rankt, egal wo, sonst „sehr niedrig". „Abstand zur besten URL" je Zeile: Leit-Score der semantisch besten URL der Query minus Leit-Score dieser URL (leer ohne Score).

## 9. SERP-Clustering und potentielle Content-Lücken

Clustering nur mit Top-10-SERPs. Übernommen aus dem alten Skript:

- nur Zeilen mit Type „organic", je Keyword die ersten 10 URLs nach Position
- Überschneidung zweier Keywords: gemeinsame URLs geteilt durch die kleinere Listenlänge, höchstens 10
- Kante ab Überschneidung ≥ 50 % (einstellbar)
- Cluster sind zusammenhängende Komponenten, bereinigt über die Dichte: Ein Keyword bleibt nur, wenn es mit mindestens 50 % (einstellbar) der übrigen Mitglieder eine Kante hat. Das verhindert den Ketteneffekt.

Das Clustering dient nur dem Blatt „Potentielle Content-Lücken": Die Cluster-Nummer steht dort in der Spalte Thema, und die Zahl neuer Seiten zählt eine Seite je Thema, Lücken ohne Thema einzeln.

Das Blatt enthält genau die Queries mit dem Urteil „Content-Lücke" (Abschnitt 7), sortiert nach „Abstand zur Schwelle", die sichersten Lücken zuerst. Spalten: Query · Bester Score · Abstand zur Schwelle · Beste URL · Rankingposition · Rankende URL · Thema. Bis 0.2.9 hatte das Blatt eine eigene Score-Grenze `luecke_unter_score`; sie fiel weg, weil Urteil und Blatt sonst auseinanderliefen (Entscheidung Daniel, 2026-10-07: „welcher boden im neuen haus" war Content-Lücke, fehlte aber im Blatt, weil 0.8 als absolute Grenze bei e5 fast wirkungslos ist). Wer nur deutliche Lücken will, filtert die Spalte Abstand zur Schwelle.

- `luecke_nur_ohne_ranking_bis_position` (Default 20, 0 = aus): Rankt eine eigene Seite bis zu dieser Position, ist die Query keine Content-Lücke (Abschnitt 7). Ohne Rankings wird der Wert ignoriert, mit einer Hinweiszeile nur, wenn er von 0 oder 20 abweicht.

Schritt 7c nennt die Zahl der potentiellen Lücken und, mit SERPs, die Zahl der neuen Seiten; die Lesehilfe nennt die Zahl der neuen Seiten in der Zeile zum Blatt.

## 10. Nutzerführung im Notebook

Schritt-Zellen mit Colab-Formularen (`#@param`), Code eingeklappt. Jede Zelle hat Überschrift und Kurzerklärung, prüft ihre Eingaben, meldet Fehler in klarem Deutsch mit Lösungshinweis und endet mit einer Statuszeile samt Verweis auf den nächsten Schritt.

1. Einführung: was das Notebook beantwortet, was man braucht, wie man die Ergebnisse liest
2. Installation und Modellwahl
3. Upload Queries und Frog-Export, Vorschau der erkannten Spalten
4. Matching (Chunk-Einstellungen, Bewertungsgrundlage, Kombi-Gewicht), Vorschau der ersten 10 Queries
5. Optional: eigene Rankings
6. Optional: Top-10-SERPs (nur für das Thema der potentiellen Content-Lücken und die Zahl neuer Seiten; ohne Ranking-Datei liefern sie auch die eigenen Rankings)
7. Schwelle und Urteile: 7a Vorschläge ansehen, 7b Schwelle festlegen, 7c Urteile bilden und potentielle Content-Lücken zusammenstellen
8. Export der vier Blätter

Ohne die Schritte 5 und 6 arbeitet das Notebook wie das alte Skript, nur mit den Korrekturen.

## 11. Export

Standard ist eine Excel-Datei, per Häkchen zusätzlich ein ZIP mit einer CSV je Blatt (UTF-8 mit BOM). Das Trennzeichen ist wählbar: Semikolon (Vorgabe, für deutsches Excel) schreibt Zahlen mit Dezimalkomma, Komma schreibt sie mit Dezimalpunkt. Beim Semikolon gilt das Dezimalkomma auch für Zahlen, die als Text vorliegen: Positionen und Zahlen-Einstellungen in der Lesehilfe. Die Excel-Datei bleibt unverändert.

Vier Blätter in dieser Reihenfolge:

| Blatt | Spalten |
|---|---|
| Übersicht | Query · Beste URL · Relevanter Chunk · Score Chunk · Score Gesamt-URL · Score Kombi · Abstand zur Schwelle · Vorsprung vor zweitbester URL · Zweitbeste URL · Score Chunk 2 · Score Gesamt-URL 2 · Score Kombi 2 · Drittbeste URL · Score Chunk 3 · Score Gesamt-URL 3 · Score Kombi 3 · Rankingposition · Rankende URL · Rankende URL = beste URL? · Urteil · Kannibalisierungsgefahr |
| Kannibalisierungsgefahr | Query · Nr. · Konkurrierende URL · Score der URL · Abstand zur besten URL · Position der URL · Rankingposition · Rankende URL · Einordnung (Abschnitt 8, Langformat) |
| Chunk auf anderer Seite | Query · Seite mit bestem Chunk · Bester Chunk · Score bester Chunk · Score Gesamt-URL der Chunk-Seite · Beste Seite insgesamt · Score Gesamt-URL der besten Seite · Rankingposition · Rankende URL · Rankende URL ist (die Seite mit bestem Chunk / die beste Seite insgesamt / eine andere Seite / rankt nicht / nicht im Frog-Export). Queries, bei denen die Seite mit dem höchsten Chunk-Score nicht die Seite mit dem höchsten Gesamt-URL-Score ist, unabhängig von Schwelle und Bewertungsgrundlage, stärkster Chunk zuerst (Entscheidung Daniel, 2026-10-07; bei TOOM 97 von 398) |
| Potentielle Content-Lücken | Query · Bester Score · Abstand zur Schwelle · Beste URL · Rankingposition · Rankende URL · Thema (Abschnitt 9) |
| Lesehilfe | Hinweise (Einordnung, Score-Band, Herkunft der Schwelle), alle Blätter, alle Urteile, jede Spalte, alle Einstellungen des Laufs |

Übersicht, eine Zeile je Query:

- Beste, zweitbeste und drittbeste URL nach Leit-Score; „Relevanter Chunk" ist der beste Chunk der besten URL. Gibt es weniger als zwei oder drei URLs, bleiben die Zellen leer.
- „Vorsprung vor zweitbester URL": Leit-Score der besten minus Leit-Score der zweitbesten URL, vier Nachkommastellen, leer bei nur einer URL. Er zeigt, wie klar die beste Seite vorn liegt, weil kurze Queries absolut niedrig scoren, und ändert kein Urteil.
- „Rankingposition" und „Rankende URL": bestes eigenes Ranking für die Query, auf jeder Position, leer ohne Ranking.
- „Vorsprung vor zweitbester URL" ist im Excel eingefärbt, gemessen am Abstand „fast gleich": rot bis zu diesem Abstand, gelb bis zum Doppelten, grün darüber (feste Grenzen statt Farbskala, damit dieselbe Zahl in jedem Lauf dieselbe Farbe hat).
- „Abstand zur Schwelle": bester Leit-Score minus Schwelle aus Schritt 7b, 4 Nachkommastellen; positiv heißt passend. Lesbarer als der Score selbst, weil Scores je nach Modell in ganz anderen Bereichen liegen (Entscheidung Daniel, 2026-10-07).
- „Rankende URL = beste URL?": „ja", „nein" (eine andere Seite hat einen höheren Leit-Score, auch knapp; Entscheidung Daniel, 2026-10-07: Ja-Nein-Frage, die Nähe zeigen Scores und Grund), „rankt nicht" (Rankings geladen, aber keines für die Query), „nicht im Frog-Export", leer ohne Rankings.

Die Query steht in den ersten drei Blättern in der ersten Spalte. Urteile sind farbig hinterlegt.

## 12. Paketaufbau

| Modul | Aufgabe |
|---|---|
| `qum/ingest.py` | Dateien lesen, Spalten erkennen, Zeilen bereinigen |
| `qum/normalize.py` | Queries und URLs vergleichbar machen |
| `qum/chunk.py` | Wort-Chunks |
| `qum/models.py` | Modellregister: Ansteuerung, Chunk-Defaults, Kontextgrenzen |
| `qum/embeddings/` | Basis-Schnittstelle, lokal, OpenAI, Gemini, Cache |
| `qum/match.py` | drei Scores, die drei besten URLs je Query und die Vorschau in Schritt 4, Ränge für den Modellvergleich |
| `qum/threshold.py` | Vorschläge der Schwelle (kalibriert, Median), Beispielpaare |
| `qum/verdict.py` | Urteile und die Ranking-Spalten der Übersicht |
| `qum/cannibal.py` | interne Fälle mit Stufe und Grund, daraus das Blatt Kannibalisierungsgefahr mit Einordnung |
| `qum/elsewhere.py` | Blatt Chunk auf anderer Seite |
| `qum/serp.py` | Clustering, Thema je Query |
| `qum/gaps.py` | Blatt Potentielle Content-Lücken, Zahl neuer Seiten |
| `qum/export.py` | vier Blätter, Lesehilfe, Excel und CSV-ZIP |
| `qum/compare.py` | Modellvergleich für die Messung (Abschnitt 14) |

Jedes Modul ist ohne Notebook und ohne echtes Modell testbar.

## 13. Tests

`pytest`, ohne Netzwerk und ohne Modell-Download (ein Test-Embedder liefert feste Vektoren). Abgedeckt: verrutschte Zeilen, Encodings, Spaltenerkennung, Chunk-Grenzen, Ansteuerung je Modell (Prefix nur bei e5), Methode der Gesamt-URL, Kombi-Gewicht, jede Zelle der Urteilstabelle, Spalten und Werte der Übersicht, alle drei Stufen der Kannibalisierungsgefahr, das Blatt der potentiellen Content-Lücken mit seinen Einstellungen, Ketteneffekt und Dichte beim Clustering, Export-Blätter und die Abdeckung jeder Spalte in der Lesehilfe. Ein Test prüft, dass die Code-Zellen des Notebooks gültiges Python sind.

Vor der Übergabe: ein echter Durchlauf in Colab mit `multilingual-e5-large` und einem kleinen Datensatz.

## 14. Messung und Slide

**Messung:** `qum/compare.py` nimmt eine Wahrheitsliste (Query, erwartete URL) und den Frog-Export und gibt je Konfiguration Treffer auf Rang 1, Treffer in den Top 3 und den mittleren reziproken Rang aus. Konfigurationen: Paraphrasen-Modell, e5 mit Prefix, OpenAI ohne Prefix, OpenAI mit `query: `/`passage: `. Die Prefix-Variante für OpenAI gibt es nur hier, nicht im Notebook. Die Wahrheitsliste kommt von Daniel (30 bis 50 Queries) oder als Näherung aus einem Ranking-Export. Der Lauf braucht Daniels OpenAI-Key.

**Slide:** eine PowerPoint-Folie für Daniels bestehendes Deck, gebaut nach der Messung. Inhalt: symmetrisch gegen asymmetrisch an einem Beispiel, drei Warnungen (Modell passend zur Aufgabe wählen, so ansteuern wie trainiert, keine absoluten Scores zwischen Modellen vergleichen), das Messergebnis. Dafür wird Daniels Deck oder eine Beispielfolie als Vorlage gebraucht.

## 15. Reihenfolge

Plan, Paket mit Tests, Notebook, Colab-Durchlauf, Messung, Slide.
