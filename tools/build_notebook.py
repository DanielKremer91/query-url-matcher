"""Erzeugt query_url_matcher.ipynb. Nach jeder Änderung ausführen: python -m tools.build_notebook"""

import json
from pathlib import Path

from qum import __version__

ROOT = Path(__file__).resolve().parents[1]

INTRO = '''# Query-URL Matcher

**Neue Seite bauen oder Bestehendes nutzen?**

Dieses Notebook vergleicht deine Suchanfragen (oder Prompts) per Embeddings mit dem Inhalt deiner Seiten. Es zeigt je Query, welche Seiten und welche Textstelle am besten passen, und leitet daraus Hinweise zu **Kannibalisierungsgefahr** und **Content-Lücken** ab.

## Schritt 1: Was du brauchst

| Datei | Pflicht | Inhalt |
|---|---|---|
| Queries | ja | CSV oder Excel, eine Query pro Zeile |
| Screaming-Frog-Export | ja | Spalten URL und Main Content |
| Eigene Rankings | optional | Keyword, URL, Position (GSC, Ahrefs, SISTRIX) |
| Top-10-SERPs | optional | Ahrefs-Export mit Keyword, URL, Position, Type |

Schon ohne Ranking-Dateien bekommst du das Matching mit Urteilen zu passender Seite, Kannibalisierungsgefahr (mehrere eigene Seiten passen fast gleich gut) und Content-Lücke. Eigene Rankings ergänzen die Urteile, die sich auf Rankings stützen (zum Beispiel "In Ordnung" und "Rankt trotz schwachem Match"), und ermöglichen eine Kalibrierung der Schwelle für "passend" (ab 20 gut rankenden Paaren, Wahl in Schritt 7b). Top-10-SERPs bündeln die potentiellen Content-Lücken zusätzlich zu Themen.

## So gehst du vor

Führe die Zellen einzeln von oben nach unten aus (Play-Symbol links), nicht mit "Alle ausführen": Die Zellen fragen nach Uploads. Jede Zelle endet mit einer Zeile, die sagt, wie es weitergeht. Die Schritte 5 und 6 sind optional.

Schwelle und Urteile entstehen in drei Zellen mit je einer Aufgabe: **Schritt 7a** zeigt Vorschläge für die Schwelle "passend" mit Beispielen, **Schritt 7b** legt die Schwelle fest, **Schritt 7c** bildet die Urteile. Willst du nur die Feineinstellungen der Urteile ändern, reicht es, Schritt 7c erneut zu starten.

Wenn du einen früheren Schritt erneut ausführst, setzt das Notebook alles zurück, was darauf aufbaut, und sagt dir, welchen Schritt du danach wiederholen musst.

Das Notebook fordert beim Start automatisch eine kostenlose T4-GPU an. Teilt Colab gerade keine zu, läuft es auf der CPU: Bei einigen hundert Seiten reicht das, bei mehreren tausend Seiten dauert das Einbetten deutlich länger.

Für die Modelle mit API-Key legst du im Secrets-Panel (Schlüssel-Symbol links) das Secret an und aktivierst dort den Schalter "Notebook-Zugriff".

## Ergebnis

Schritt 8 lädt eine Excel-Datei mit vier Blättern herunter, auf Wunsch zusätzlich ein ZIP mit einer CSV je Blatt:

- **Übersicht:** eine Zeile je Query mit den drei am besten passenden URLs und ihren Scores, der eigenen Rankingposition, dem Urteil und der Spalte Kannibalisierungsgefahr (ja oder nein).
- **Kannibalisierungsgefahr:** Queries, bei denen mehrere eigene Seiten konkurrieren, mit bis zu drei URLs samt Score und Position.
- **Potentielle Content-Lücken:** Queries ohne ausreichend passende Seite, mit SERPs nach Thema gebündelt.
- **Lesehilfe:** erklärt jedes Blatt, jede Spalte, jedes Urteil und nennt die Einstellungen des Laufs.

## Wichtig

Das Notebook sortiert vor und begründet. Cosinus-Werte sind Hinweise, keine Urteile: Die Entscheidung triffst du, nachdem du die Seite angesehen hast.
'''

STEP2 = '''#@title Schritt 2: Installation und Modellwahl { display-mode: "form" }
#@markdown Wähle das Embedding-Modell. Für deutschen Content ist **multilingual-e5-large** die Empfehlung.
#@markdown Die Modelle mit API-Key brauchen ein Secret (Schlüssel-Symbol links): `OPENAI_API_KEY` bzw. `GEMINI_API_KEY`.
#@markdown Lege es an **und** aktiviere dort den Schalter "Notebook-Zugriff", sonst kann das Notebook den Key nicht lesen.
modell = "multilingual-e5-large · Deutsch, kostenlos (Empfehlung)" #@param ["multilingual-e5-large · Deutsch, kostenlos (Empfehlung)", "multilingual-e5-base · Deutsch, kostenlos, schneller", "bge-m3 · Deutsch, kostenlos, lange Texte", "msmarco-distilbert-base-v4 · nur englische Projekte", "Gemini gemini-embedding-001 · API-Key nötig", "OpenAI text-embedding-3-large · API-Key nötig", "paraphrase-multilingual-mpnet-base-v2 · symmetrisch, NUR zum Vergleich"]

!pip install -q "qum[local] @ git+https://github.com/DanielKremer91/query-url-matcher@v__VERSION__"

import logging
import warnings

from qum import colab
from qum.embeddings import make_embedder
from qum.models import model_by_label

# Hinweis von Hugging Face zu Downloads ohne Token ausblenden: Die Modelle sind frei verfügbar, ein Token ist nicht nötig.
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

new_spec = model_by_label(modell)
api_key = None
secret_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}.get(new_spec.provider)
if secret_name:
    api_key = colab.secret(secret_name)
    if not api_key:
        colab.stop(
            f'Für {new_spec.model_id} fehlt der API-Key. Lege im Secrets-Panel (Schlüssel-Symbol links) das Secret '
            f'{secret_name} an und aktiviere dort den Schalter "Notebook-Zugriff". Führe die Zelle danach erneut aus.'
        )
print(f"⏳ Lade {new_spec.model_id} (lokale Modelle: beim ersten Mal bis zu 2 GB Download) ...")
try:
    with colab.guard():
        new_embedder = make_embedder(new_spec, api_key=api_key, cache_dir="/content/qum_cache")
except OSError as error:
    colab.stop(
        f"Das Modell {new_spec.model_id} konnte nicht geladen werden ({type(error).__name__}). "
        "Prüfe die Internetverbindung der Colab-Sitzung und führe die Zelle erneut aus."
    )
spec, embedder = new_spec, new_embedder
colab.invalidate(globals(), *colab.MATCH_STATE)
if spec.provider == "local":
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "⚠️ Colab hat keine GPU zugeteilt. Einige hundert Seiten laufen trotzdem, mehrere tausend dauern deutlich länger. "
                "Für eine GPU: Laufzeit → Laufzeittyp ändern → T4 GPU, dann ab Schritt 2 erneut ausführen."
            )
    except ImportError:
        pass
if spec.comparison_only:
    print("⚠️ Dieses Modell ist symmetrisch und nur als Gegenbeispiel gedacht. Nutze es nicht für die Auswertung.")
print(f"✅ Schritt 2 fertig: Modell {spec.model_id} ist bereit. Weiter mit Schritt 3 (Schritt 4 neu ausführen, falls du schon gematcht hattest).")
'''.replace("__VERSION__", __version__)  # installiert genau den Git-Tag der Paketversion

STEP3 = '''#@title Schritt 3: Queries und Frog-Export hochladen { display-mode: "form" }
#@markdown **▶ Klicke links auf das Play-Symbol. Unter der Zelle erscheint dann nacheinander der Knopf „Dateien auswählen": zuerst für die Queries, danach für den Frog-Export.**
#@markdown **Queries-Datei:** CSV oder Excel mit einer Spalte „Keyword" oder „Query", eine Suchanfrage (oder ein Prompt) pro Zeile. Eine Datei ohne Kopfzeile geht auch. Weitere Spalten, etwa Suchvolumen, stören nicht.
#@markdown **Frog-Export:** Screaming Frog liefert den Seitentext über eine Custom Extraction. ① Configuration → Custom → Custom Extraction: eine Regel anlegen, die den Hauptinhalt als Text ausliest (Extract Text, z. B. XPath `//main` oder `//article`, oder per Custom JavaScript), und ihr einen Namen mit „Content" geben, z. B. „Main Content". ② Seiten crawlen. ③ Im Tab „Custom Extraction" auf Export klicken (CSV oder Excel). Gebraucht werden nur die Spalten Address und die Content-Spalte.
#@markdown **Die Felder unten bleiben normalerweise leer.** Nur wenn die Zelle mit ❌ „…-Spalte nicht erkannt" abbricht, trägst du den Namen aus der Meldung ein und startest die Zelle erneut.
query_spalte = "" #@param {type:"string"}
url_spalte = "" #@param {type:"string"}
content_spalte = "" #@param {type:"string"}

from qum import colab, ingest

colab.require(globals(), 2, "spec", "embedder")
with colab.guard():
    name, data = colab.upload("Queries-Datei (CSV oder Excel, eine Query pro Zeile)")
    query_table = colab.read_table(data, name)
    new_queries = ingest.load_queries(query_table, column=query_spalte or None)
    name, data = colab.upload("Screaming-Frog-Export (URL und Main Content)")
    content_table = colab.read_table(data, name)
    new_content = ingest.load_content(content_table, url_col=url_spalte or None, content_col=content_spalte or None)
if not new_queries or not new_content.urls:
    colab.stop("Es wurden keine Queries oder keine Seiten mit Content gefunden.")

query_found = query_spalte or ingest.find_column(query_table, ingest.QUERY_ALIASES)
print(
    f"Erkannte Spalten: Query = {query_found or '(keine Kopfzeile)'}, "
    f"URL = {new_content.url_column}, Content = {new_content.content_column}"
)
if query_found is None:
    print("ℹ️ In der Queries-Datei wurde keine Kopfzeile erkannt. Die erste Zeile wird als Query behandelt.")

had_extras = globals().get("rankings") is not None or globals().get("serps") is not None
queries, content = new_queries, new_content
rankings = None
serps = None
rankings_source = None
colab.invalidate(globals(), *colab.MATCH_STATE)
if had_extras:
    print("ℹ️ Rankings und SERPs wurden zurückgesetzt. Führe Schritt 5 und 6 bei Bedarf erneut aus.")
print(f"Beispiel-Query: {queries[0]}")
print(f"Beispiel-URL:   {content.urls[0]}")
print(f"Content-Anfang: {content.contents[0][:120]} ...")
if content.skipped_empty or content.skipped_duplicate:
    print(f"ℹ️ Übersprungen: {content.skipped_empty} Seiten ohne Content, {content.skipped_duplicate} doppelte URLs.")
print(f"✅ Schritt 3 fertig: {len(queries)} Queries und {len(content.urls)} URLs geladen. Weiter mit Schritt 4.")
'''

STEP4 = '''#@title Schritt 4: Matching { display-mode: "form" }
#@markdown Chunking ist immer aktiv: Der Content wird in überlappende Textblöcke (Chunks) zerlegt, je Query zählt der beste Block.
#@markdown **0 heißt: empfohlene Größe für das gewählte Modell** (bei e5: 250 Wörter, Overlap 40). Nur ändern, wenn du bewusst andere Werte willst.
chunk_groesse = 0 #@param {type:"integer"}
chunk_overlap = 0 #@param {type:"integer"}
#@markdown **bewertungsgrundlage:** Je Query und Seite gibt es drei Cosinus-Werte. Der gewählte entscheidet, ob eine Seite als "passend" gilt (mit der Schwelle aus Schritt 7b), welche Seite die beste, zweitbeste und drittbeste ist und wie sortiert wird. Im Export stehen immer alle drei.
#@markdown **Chunk** (Empfehlung): Query gegen den besten Textabschnitt der Seite. Findet Seiten, die die Query in einem Abschnitt beantworten, auch wenn die Seite breiter ist.
#@markdown **Gesamt-URL:** Query gegen die Seite als Ganzes. Bevorzugt Seiten, die sich komplett um das Thema drehen.
#@markdown **Kombi:** Mischung aus beiden, der Anteil des Chunk-Werts steht im Regler darunter (0.7 = 70 % Chunk, 30 % Gesamt-URL).
bewertungsgrundlage = "Chunk" #@param ["Chunk", "Gesamt-URL", "Kombi"]
kombi_gewicht_chunk = 0.7 #@param {type:"slider", min:0, max:1, step:0.05}

from qum import colab
from qum import labels as L
from qum.match import best_matches, estimate_chunks, preview, run_matching

colab.require(globals(), 2, "spec", "embedder")
colab.require(globals(), 3, "queries", "content")
if chunk_groesse < 0 or chunk_overlap < 0:
    colab.stop("Chunk-Größe und Overlap dürfen nicht negativ sein. 0 bedeutet: Standardwert des Modells.")
# erst prüfen, übernommen wird erst nach erfolgreichem Matching (zusammen mit dem Ergebnis)
new_size = chunk_groesse or spec.chunk_size
new_overlap = chunk_overlap or spec.chunk_overlap
if new_overlap >= new_size:
    if chunk_overlap == 0:
        colab.stop(
            f"Der Standard-Overlap des Modells ({new_overlap}) ist nicht kleiner als die gewählte Chunk-Größe "
            f"({new_size}). Trage einen kleineren Overlap oder eine größere Chunk-Größe ein."
        )
    colab.stop(
        f"Der Overlap ({new_overlap}) muss kleiner sein als die Chunk-Größe ({new_size}). "
        "Trage 0 ein, um den Standardwert des Modells zu nutzen."
    )
new_basis = L.BASIS[bewertungsgrundlage]
n_chunks = estimate_chunks(content.contents, new_size, new_overlap)
print(f"ℹ️ {len(content.urls)} URLs ergeben {n_chunks} Chunks ({new_size} Wörter, Overlap {new_overlap}).")
if n_chunks >= 50000:
    print("⚠️ Das sind sehr viele Chunks (ab 50.000). Der Lauf wird trotzdem fortgesetzt und kann lange dauern. Schränke den Frog-Export auf ein Verzeichnis ein, wenn dir das zu lange ist.")
colab.invalidate(globals(), *colab.MATCH_STATE)
with colab.guard():
    new_result = run_matching(queries, content.urls, content.contents, embedder, new_size, new_overlap)
inner = embedder.inner
if hasattr(inner, "truncated_share"):
    flat = [c for per_url in new_result.chunks for c in per_url]
    share = inner.truncated_share(flat[:2000])
    if share > 0.05:
        print(f"⚠️ {share:.0%} der Chunks sind länger als das Modell lesen kann. Verkleinere die Chunk-Größe.")
size, overlap, basis = new_size, new_overlap, new_basis
weight, basis_label = kombi_gewicht_chunk, bewertungsgrundlage
result = new_result
lead = result.lead(basis, weight)
print(f"Vorschau der ersten 10 Queries: beste und zweitbeste URL mit Score ({basis_label}). Alle Spalten stehen im Export im Blatt Übersicht.")
display(preview(best_matches(result, lead, weight).head(10), basis))
print(f"✅ Schritt 4 fertig: {len(result.queries)} Queries gematcht. Optional weiter mit Schritt 5 und 6, sonst Schritt 7a.")
'''

STEP5 = '''#@title Schritt 5 (optional): Eigene Rankings hochladen { display-mode: "form" }
#@markdown **▶ Klicke links auf das Play-Symbol. Unter der Zelle erscheint dann der Knopf „Dateien auswählen".**
#@markdown Export mit Keyword, URL und Position, zum Beispiel aus GSC, Ahrefs (Organic Keywords) oder SISTRIX.
#@markdown Ergänzt die Urteile, die sich auf Rankings stützen, und ermöglicht eine Kalibrierung der Schwelle für "passend"
#@markdown (ab 20 gut rankenden Paaren, Wahl in Schritt 7b).
#@markdown Kannibalisierungsgefahr und Content-Lücken gibt es auch ohne Rankings.
#@markdown **Die Felder unten bleiben normalerweise leer.** Nur wenn die Zelle mit ❌ „…-Spalte nicht erkannt" abbricht, trägst du den Namen aus der Meldung ein und startest die Zelle erneut.
keyword_spalte = "" #@param {type:"string"}
url_spalte_ranking = "" #@param {type:"string"}
position_spalte = "" #@param {type:"string"}

from qum import colab, ingest
from qum.normalize import normalize_url

colab.require(globals(), 3, "queries", "content")
with colab.guard():
    name, data = colab.upload("Eigene Rankings (Keyword, URL, Position)")
    ranking_table = colab.read_table(data, name)
    loaded = ingest.load_rankings(
        ranking_table,
        keyword_col=keyword_spalte or None,
        url_col=url_spalte_ranking or None,
        position_col=position_spalte or None,
    )
print(
    "Erkannte Spalten: "
    f"Keyword = {keyword_spalte or ingest.find_column(ranking_table, ingest.KEYWORD_ALIASES)}, "
    f"URL = {url_spalte_ranking or ingest.find_column(ranking_table, ingest.URL_ALIASES)}, "
    f"Position = {position_spalte or ingest.find_column(ranking_table, ingest.POSITION_ALIASES)}"
)
known_queries = {ingest.normalize_query(q) for q in queries}
known_urls = {normalize_url(u) for u in content.urls}
covered = int(loaded["query_norm"].isin(known_queries).sum())
colab.invalidate(globals(), *colab.VERDICT_STATE)
if covered == 0:
    print("⚠️ Keine der Ranking-Zeilen passt zu deinen Queries. Prüfe, ob die Datei zu diesem Projekt gehört.")
    rankings, rankings_source = None, None
elif not loaded["url_norm"].isin(known_urls).any():
    print("⚠️ Keine der Ranking-URLs steht im Frog-Export. Prüfe, ob die Datei zu diesem Projekt gehört.")
    rankings, rankings_source = None, None
else:
    rankings, rankings_source = loaded, "Datei"
    print(f"✅ Schritt 5 fertig: {len(rankings)} Ranking-Zeilen, davon {covered} zu deinen Queries. Weiter mit Schritt 6 oder 7a.")
if rankings is None:
    print("ℹ️ Die Ranking-Datei wurde nicht übernommen, das Notebook bleibt im reinen Matching. Weiter mit Schritt 7a oder lade eine andere Datei hoch.")
'''

STEP6 = '''#@title Schritt 6 (optional): Top-10-SERPs hochladen { display-mode: "form" }
#@markdown **▶ Klicke links auf das Play-Symbol. Unter der Zelle erscheint dann der Knopf „Dateien auswählen".**
#@markdown Ahrefs-Export mit Keyword, URL, Position und Type (die kompletten Top 10 je Keyword, inklusive Wettbewerber).
#@markdown Die SERPs werden nur genutzt, um die potentiellen Content-Lücken zu Themen zu bündeln (Spalte Thema im Blatt „Potentielle Content-Lücken“) und daraus die Zahl neuer Seiten zu zählen. Ohne Ranking-Datei aus Schritt 5 übernimmt das Notebook außerdem deine eigenen Rankings aus den SERPs.
#@markdown **Die Felder unten bleiben normalerweise leer.** Nur wenn die Zelle mit ❌ „…-Spalte nicht erkannt" abbricht, trägst du den Namen aus der Meldung ein und startest die Zelle erneut.
serp_keyword_spalte = "" #@param {type:"string"}
serp_url_spalte = "" #@param {type:"string"}
serp_position_spalte = "" #@param {type:"string"}
serp_type_spalte = "" #@param {type:"string"}

from qum import colab, ingest
from qum.normalize import host_of

colab.require(globals(), 3, "queries", "content")
with colab.guard():
    name, data = colab.upload("Top-10-SERPs (Keyword, URL, Position, Type)")
    serp_table = colab.read_table(data, name)
    loaded = ingest.load_serps(
        serp_table,
        keyword_col=serp_keyword_spalte or None,
        url_col=serp_url_spalte or None,
        position_col=serp_position_spalte or None,
        type_col=serp_type_spalte or None,
    )
serp_type_found = serp_type_spalte or ingest.find_column(serp_table, ingest.TYPE_ALIASES)
print(
    "Erkannte Spalten: "
    f"Keyword = {serp_keyword_spalte or ingest.find_column(serp_table, ingest.KEYWORD_ALIASES)}, "
    f"URL = {serp_url_spalte or ingest.find_column(serp_table, ingest.URL_ALIASES)}, "
    f"Position = {serp_position_spalte or ingest.find_column(serp_table, ingest.POSITION_ALIASES)}, "
    f"Type = {serp_type_found or '(keine Spalte, alle Zeilen gelten als organisch)'}"
)
known_queries = {ingest.normalize_query(q) for q in queries}
colab.invalidate(globals(), *colab.VERDICT_STATE)
if not loaded["query_norm"].isin(known_queries).any():
    print("⚠️ Kein SERP-Keyword passt zu deinen Queries. Prüfe, ob die Datei zu diesem Projekt gehört.")
    serps = None
else:
    serps = loaded
if rankings_source == "SERPs":
    # abgeleitete Rankings stammen aus den alten SERPs und werden neu gebildet
    rankings, rankings_source = None, None
if serps is not None and rankings_source != "Datei":
    derived = ingest.own_rankings_from_serps(serps, {host_of(u) for u in content.urls})
    if derived.empty:
        print("ℹ️ In den SERPs steht keine URL deiner Domain. Eigene Rankings gibt es daraus nicht; lade sie in Schritt 5 hoch.")
    else:
        rankings, rankings_source = derived, "SERPs"
        print(f"ℹ️ Keine eigene Ranking-Datei: {len(rankings)} eigene Rankings aus den SERPs übernommen.")
if serps is not None:
    per_keyword = serps.groupby("query_norm")["url_norm"].nunique().median()
    if per_keyword < 5:
        n = f"{per_keyword:g}".replace(".", ",")
        print(
            f"⚠️ Im Schnitt nur {n} URLs je Keyword. Für das Clustering werden die kompletten Top 10 je Keyword "
            "gebraucht, mit eigenen Rankings allein ist es nicht aussagekräftig."
        )
if serps is None:
    print("ℹ️ Die SERPs wurden nicht übernommen. Weiter mit Schritt 7a oder lade eine andere Datei hoch.")
else:
    print(f"✅ Schritt 6 fertig: SERPs für {serps['query_norm'].nunique()} Keywords geladen. Weiter mit Schritt 7a.")
'''

STEP7A = '''#@title Schritt 7a: Vorschläge für die Schwelle ansehen { display-mode: "form" }
#@markdown **▶ Einfach starten, hier ist nichts einzutragen.**
#@markdown **Was ist die Schwelle?** Der Cosinus-Score, ab dem eine Seite für eine Query als "passend" gilt. Diese Zelle zeigt nur Vorschläge für die Schwelle und bildet noch keine Urteile. Zu jedem Vorschlag siehst du Beispielpaare: Oberhalb der Schwelle sollten passende Seiten stehen, unterhalb eher unpassende.
#@markdown **kalibrierung_bis_position** (optionale Feineinstellung, nur für "Aus Rankings kalibriert"): Paare aus Query und eigener URL, die bis zu dieser Position ranken, gelten als Beispiele für "die Seite passt sicher". Die Schwelle wird so gesetzt, dass 75 % dieser Beispiele sie erreichen. Höherer Wert: mehr Beispiele, meist etwas niedrigere Schwelle.
kalibrierung_bis_position = 5 #@param {type:"integer"}

from qum import colab
from qum.threshold import MIN_PAIRS, calibrated_threshold, calibration_scores, examples_around, median_threshold

colab.require(globals(), 4, "result", "lead")
# bei einem Abbruch bleiben frühere Urteile stehen: das soll die Meldung sagen
kept = " Bis dahin gelten die Urteile aus dem letzten Lauf von Schritt 7c." if globals().get("decisions") is not None else ""
if kalibrierung_bis_position < 1:
    colab.stop(f"kalibrierung_bis_position muss mindestens 1 sein.{kept}")
colab.invalidate(globals(), *colab.VERDICT_STATE)
new_calibrated = calibrated_threshold(result, lead, rankings, max_position=kalibrierung_bis_position)
new_median = median_threshold(result, lead)
if rankings is None:
    new_reason = "Es sind keine Rankings geladen (Schritt 5 oder 6)."
else:
    n_pairs = len(calibration_scores(result, lead, rankings, max_position=kalibrierung_bis_position))
    new_reason = f"Nur {n_pairs} Ranking-Paare bis Position {kalibrierung_bis_position}, für eine Kalibrierung sind {MIN_PAIRS} nötig."

print("Die Schwelle ist die Cosinus-Ähnlichkeit zwischen Query und Seite, ab der eine Seite als passend gilt.")
print()
if new_calibrated is None:
    print(f"Aus Rankings kalibriert: nicht verfügbar. {new_reason}")
else:
    print(
        f"Aus Rankings kalibriert: {new_calibrated.value:.4f}. Diesen Score erreichen 75 % der {new_calibrated.n_pairs} "
        f"Paare aus Query und eigener Seite, die heute bis Position {kalibrierung_bis_position} ranken."
    )
    print("Paare knapp über und knapp unter diesem Wert:")
    display(examples_around(result, lead, new_calibrated.value))
print()
print(f"Mittlerer bester Score (nicht kalibriert): {new_median.value:.4f}. Das ist der mittlere Score der besten Treffer je Query.")
print('⚠️ Dieser Wert ist kein Maßstab für "passend": Etwa die Hälfte deiner Queries liegt per Konstruktion darunter.')
print("Paare knapp über und knapp unter diesem Wert:")
display(examples_around(result, lead, new_median.value))
print()
available = ['"Aus Rankings kalibriert"'] if new_calibrated is not None else []
available.append('"Mittlerer bester Score (nicht kalibriert)"')
print(
    f'ℹ️ Noch keine Urteile. Wähle in Schritt 7b bei schwelle_waehlen {", ".join(available)} oder "Eigener Wert" '
    "(Wert bei eigene_schwelle eintragen)."
)
calibrated, median, no_calibration = new_calibrated, new_median, new_reason
calibration_position = kalibrierung_bis_position
print("✅ Schritt 7a fertig. Weiter mit Schritt 7b: Schwelle festlegen.")
'''

STEP7B = '''#@title Schritt 7b: Schwelle festlegen { display-mode: "form" }
#@markdown Wähle, welche Schwelle gilt. Die Vorschläge und Beispielpaare dazu hat Schritt 7a gezeigt.
#@markdown **schwelle_waehlen:**
#@markdown **Aus Rankings kalibriert** (Vorgabe): leitet die Schwelle aus deinen heute gut rankenden Seiten ab. Das ist am verlässlichsten und braucht Rankings aus Schritt 5 oder 6.
#@markdown **Mittlerer bester Score (nicht kalibriert):** nur ein Behelf ohne Rankings. Etwa die Hälfte der Queries liegt per Konstruktion darunter.
#@markdown **Eigener Wert:** nimmt die Zahl aus eigene_schwelle. Orientiere dich an den Beispielen aus Schritt 7a.
schwelle_waehlen = "Aus Rankings kalibriert" #@param ["Aus Rankings kalibriert", "Mittlerer bester Score (nicht kalibriert)", "Eigener Wert"]
#@markdown **eigene_schwelle** (nur für "Eigener Wert"): größer als 0 und höchstens 1, zum Beispiel 0.82.
eigene_schwelle = 0.0 #@param {type:"number"}

from qum import colab
from qum import labels as L
from qum.threshold import examples_around

colab.require(globals(), 4, "result", "lead")
colab.require(globals(), "7a", "median", "calibration_position")
choice = L.THRESHOLD_CHOICE[schwelle_waehlen]
# bei einem Abbruch bleiben frühere Urteile stehen: das soll die Meldung sagen
kept = " Bis dahin gelten die Urteile aus dem letzten Lauf von Schritt 7c." if globals().get("decisions") is not None else ""
if choice == "manuell" and not 0 < eigene_schwelle <= 1:
    colab.stop(
        f"Die eigene Schwelle muss größer als 0 und höchstens 1 sein. Trage sie bei eigene_schwelle ein, zum Beispiel 0.82.{kept}"
    )
if choice == "rankings" and calibrated is None:
    colab.stop(
        f"Die Kalibrierung aus Rankings ist nicht verfügbar. {no_calibration} "
        f"Wähle bei schwelle_waehlen eine andere Option und starte die Zelle erneut.{kept}"
    )
colab.invalidate(globals(), *colab.DECISION_STATE)
if choice == "rankings":
    threshold = calibrated.value
    print(
        f"Verwendete Schwelle {threshold:.4f}, aus Rankings kalibriert. Diesen Score erreichen 75 % der {calibrated.n_pairs} "
        f"Paare aus Query und eigener Seite, die heute bis Position {calibration_position} ranken."
    )
elif choice == "median":
    threshold = median.value
    print(f"Verwendete Schwelle {threshold:.4f}, mittlerer bester Score (nicht kalibriert). Das ist der mittlere Score der besten Treffer je Query.")
    print('⚠️ Dieser Wert ist kein Maßstab für "passend": Etwa die Hälfte deiner Queries liegt per Konstruktion darunter.')
else:
    threshold = float(eigene_schwelle)
    print(f"Verwendete Schwelle {threshold:.4f}, eigener Wert.")
threshold_source, threshold_label = choice, schwelle_waehlen
print("Diese Paare liegen knapp über und knapp unter der Schwelle. Passt die Grenze?")
display(examples_around(result, lead, threshold))
print(f"✅ Schritt 7b fertig: Schwelle {threshold:.4f}. Weiter mit Schritt 7c: Urteile bilden.")
'''

STEP7C = '''#@title Schritt 7c: Urteile bilden { display-mode: "form" }
#@markdown **▶ Starten. Die Werte darunter kannst du für den ersten Lauf auf den Voreinstellungen lassen.**
#@markdown Die Zelle entscheidet mit der Schwelle aus Schritt 7b je Query, ob eine oder mehrere deiner Seiten passen, und bildet daraus die Urteile (passende Seite, Kannibalisierungsgefahr, Content-Lücke). Dazu stellt sie die potentiellen Content-Lücken zusammen. Willst du nur die Feineinstellungen ändern, starte nur diese Zelle erneut. Welches Urteil wann entsteht, zeigt die Übersicht über Schritt 7a.
#@markdown **rankt_gut_bis_position:** Bis zu dieser Position gilt eine Query als gut rankend. Davon hängt das Urteil ab: "In Ordnung" oder "Kannibalisierungsgefahr" bei gutem Ranking, "Bestehende Seite nutzen", "Kannibalisierungsgefahr" oder "Content-Lücke" bei schwachem.
rankt_gut_bis_position = 10 #@param {type:"integer"}
#@markdown **abstand_fast_gleich:** Score-Unterschied, bis zu dem zwei Seiten als gleich gut gelten (0.01 = ein Hundertstel). Liegt die rankende Seite höchstens so weit hinter der besten, lautet das Urteil "In Ordnung". Liegen mehrere passende Seiten so nah beieinander, gibt es Kannibalisierungsgefahr.
abstand_fast_gleich = 0.01 #@param {type:"number"}
#@markdown **sichtbar_bis_position:** Ranken zwei eigene URLs für dieselbe Query bis zu dieser Position, steht die Query im Blatt Kannibalisierungsgefahr mit der Stufe "Bereits sichtbar".
sichtbar_bis_position = 20 #@param {type:"integer"}
#@markdown **Potentielle Content-Lücken:** Das Blatt „Potentielle Content-Lücken“ hat eigene Einstellungen. Seine Zahl kann deshalb von der Zahl der Urteile „Content-Lücke“ abweichen.
#@markdown **luecke_unter_score:** Eine Query gilt als potentielle Lücke, wenn der Score ihrer besten Seite unter diesem Wert liegt. 0 = Schwelle aus Schritt 7b. Sonst ein Wert zwischen 0 und 1, zum Beispiel 0.8 für eine strengere oder 0.75 für eine mildere Auswahl.
luecke_unter_score = 0.0 #@param {type:"number"}
#@markdown **luecke_nur_ohne_ranking_bis_position:** 0 = aus. Bei zum Beispiel 10 zählen nur Queries, für die keine eigene Seite bis Position 10 rankt (kein Ranking oder schlechter als Position 10). Ohne Rankings wird der Wert ignoriert.
luecke_nur_ohne_ranking_bis_position = 0 #@param {type:"integer"}
#@markdown **Clustering (nur mit Schritt 6):** Das Tool fasst Keywords mit ähnlichen Google-Ergebnissen zu Themen zusammen. Die beiden Regler ändern nur die Spalte Thema im Blatt „Potentielle Content-Lücken“: Lücken mit gleichem Thema zählen als eine neue Seite, nicht als mehrere.
#@markdown **serp_ueberschneidung:** Ab wie viel Prozent gleicher Top-10-URLs zwei Keywords als verwandt gelten. Beispiel: „fassade streichen" und „hausfassade streichen" teilen 6 von 10 URLs, also 60 %. Bei 50 % sind sie verwandt, Google behandelt sie als dieselbe Suchintention. Hat ein Keyword weniger als 10 Treffer, zählt die kürzere Liste.
serp_ueberschneidung = 50 #@param {type:"slider", min:10, max:100, step:5}
#@markdown **cluster_dichte:** Mit wie viel Prozent der anderen Mitglieder ein Keyword verwandt sein muss, um im Cluster zu bleiben. Das verhindert Ketten: Ist A mit B verwandt, B mit C, C mit D und D mit E, aber A hat mit E nichts gemeinsam, sollen nicht alle fünf in einem Cluster landen. Bei 50 % muss jedes Keyword mit mindestens 2 der 4 anderen verwandt sein. A und E fallen raus, B, C und D bleiben.
cluster_dichte = 50 #@param {type:"slider", min:10, max:100, step:5}
#@markdown Höhere Werte bei beiden Reglern ergeben kleinere, engere Themen und damit eher mehr neue Seiten, niedrigere Werte größere Themen und weniger neue Seiten.

from datetime import date

from qum import colab, export
from qum import labels as L
from qum.cannibal import annotate, find_cannibalization
from qum.gaps import count_new_pages, find_gaps
from qum.serp import topics
from qum.verdict import build_decisions

colab.require(globals(), 4, "result", "lead", "size", "overlap", "weight", "basis_label")
colab.require(globals(), "7b", "threshold", "threshold_source", "threshold_label")
# bei einem Abbruch bleiben frühere Urteile stehen: das soll die Meldung sagen
kept = " Bis dahin gelten die Urteile aus dem letzten Lauf von Schritt 7c." if globals().get("decisions") is not None else ""
if min(rankt_gut_bis_position, sichtbar_bis_position) < 1:
    colab.stop(f"Die Positionen (rankt gut, sichtbar) müssen mindestens 1 sein.{kept}")
if not 0 <= abstand_fast_gleich < 0.1:
    colab.stop(
        'Der Abstand für "fast gleich" muss mindestens 0 und kleiner als 0.1 sein. Er ist eine Differenz von '
        f"Cosinus-Scores (zum Beispiel 0.01), kein Prozentwert.{kept}"
    )
if not 0 <= luecke_unter_score <= 1:
    colab.stop(
        "luecke_unter_score muss 0 (Schwelle aus Schritt 7b) oder ein Wert zwischen 0 und 1 sein, zum Beispiel 0.8."
        f"{kept}"
    )
if luecke_nur_ohne_ranking_bis_position < 0:
    colab.stop(f"luecke_nur_ohne_ranking_bis_position darf nicht negativ sein. 0 schaltet die Bedingung aus.{kept}")
colab.invalidate(globals(), *colab.DECISION_STATE)
print(f"Schwelle aus Schritt 7b: {threshold:.4f} ({threshold_label}).")
new_decisions = build_decisions(result, lead, threshold, rankings, rankt_gut_bis_position, weight, abstand_fast_gleich)
new_cannibal = find_cannibalization(
    result, lead, threshold, rankings, rankt_gut_bis_position, abstand_fast_gleich, sichtbar_bis_position
)
new_decisions = annotate(new_decisions, new_cannibal)
gap_position = luecke_nur_ohne_ranking_bis_position if rankings is not None else 0
if luecke_nur_ohne_ranking_bis_position > 0 and rankings is None:
    print("ℹ️ luecke_nur_ohne_ranking_bis_position wird ignoriert: Es sind keine Rankings geladen.")
new_topics = None if serps is None else topics(result.queries, serps, serp_ueberschneidung / 100, cluster_dichte / 100)
new_gaps = find_gaps(result, lead, threshold, rankings, new_topics, luecke_unter_score, gap_position)
if luecke_nur_ohne_ranking_bis_position == 0:
    gap_position_setting = "aus"
elif rankings is None:
    gap_position_setting = "ignoriert, keine Rankings"
else:
    gap_position_setting = luecke_nur_ohne_ranking_bis_position
settings = {
    "Datum": date.today().isoformat(),
    "Modell": spec.model_id,
    "Nur zum Vergleich (symmetrisches Modell)": "ja" if spec.comparison_only else "nein",
    "Chunk-Größe (Wörter)": size,
    "Overlap (Wörter)": overlap,
    "Bewertungsgrundlage": basis_label,
    "Kombi-Gewicht Chunk": weight,
    "Schwelle": round(threshold, 4),
    "Schwelle aus": threshold_label,
    "Kalibrierung bis Position": calibration_position,
    "Rankt gut bis Position": rankt_gut_bis_position,
    "Abstand fast gleich": abstand_fast_gleich,
    "Sichtbar bis Position": sichtbar_bis_position,
    "Lücke unter Score": luecke_unter_score or "Schwelle aus Schritt 7b",
    "Lücke nur ohne Ranking bis Position": gap_position_setting,
    "SERP-Überschneidung (%)": serp_ueberschneidung,
    "Cluster-Dichte (%)": cluster_dichte,
    "Eigene Rankings": {"Datei": "ja, aus Datei", "SERPs": "ja, aus den SERPs abgeleitet"}.get(rankings_source, "nein"),
    "Top-10-SERPs": "ja" if serps is not None else "nein",
}
decisions, cannibal, gaps = new_decisions, new_cannibal, new_gaps
new_pages = None if serps is None else count_new_pages(gaps)
counts = decisions[L.C_VERDICT].value_counts()
for verdict, count in counts.items():
    print(f"   {count:>5} × {verdict}")
print(f"   {cannibal[L.C_QUERY].nunique():>5} Queries im Blatt Kannibalisierungsgefahr (über alle Urteile, Spalte „Kannibalisierungsgefahr“ = ja)")
uncalibrated = " (Schwelle nicht kalibriert)" if threshold_source == "median" and not luecke_unter_score else ""
pages = "" if new_pages is None else f", zusammen {new_pages} neue Seiten (eine je Thema, Lücken ohne Thema einzeln)"
print(f"   {len(gaps):>5} potentielle Content-Lücken{uncalibrated}{pages}")
print(
    "ℹ️ Das Blatt „Potentielle Content-Lücken“ folgt seinen eigenen Einstellungen (luecke_unter_score, "
    "luecke_nur_ohne_ranking_bis_position). Seine Zahl kann deshalb von der Zahl der Urteile „Content-Lücke“ abweichen."
)
print()
print("Was die Urteile bedeuten:")
for verdict in counts.index:
    print(f"   {verdict}: {export.VERDICT_HELP[verdict]}")
print("Alle Werte stehen im Export (Schritt 8), die Lesehilfe erklärt jede Spalte.")
print("✅ Schritt 7c fertig. Weiter mit Schritt 8 (Export).")
'''

STEP8 = '''#@title Schritt 8: Export { display-mode: "form" }
#@markdown Die Excel-Datei hat vier Blätter:
#@markdown **Übersicht:** eine Zeile je Query mit den drei besten URLs, ihren Scores, der Rankingposition, dem Urteil und Kannibalisierungsgefahr ja oder nein.
#@markdown **Kannibalisierungsgefahr:** Queries, bei denen mehrere eigene Seiten konkurrieren, mit bis zu drei URLs samt Score und Position.
#@markdown **Potentielle Content-Lücken:** Queries, deren beste Seite unter der Lücken-Schwelle aus Schritt 7c liegt, mit SERPs nach Thema gebündelt.
#@markdown **Lesehilfe:** erklärt Blätter, Spalten, Urteile und Stufen und nennt die Einstellungen des Laufs.
#@markdown Das Trennzeichen gilt nur für die zusätzliche CSV-ZIP (eine CSV je Blatt).
zusaetzlich_csv_zip = False #@param {type:"boolean"}
csv_trennzeichen = "Semikolon (für deutsches Excel)" #@param ["Semikolon (für deutsches Excel)", "Komma"]

from qum import colab, export

colab.require(globals(), "7c", "decisions", "cannibal", "gaps", "settings", "threshold_source")
sheets = export.build_sheets(decisions, cannibal, gaps, settings, threshold_source=threshold_source, new_pages=new_pages)
export.write_excel("query_url_matcher.xlsx", sheets)
colab.download("query_url_matcher.xlsx")
if zusaetzlich_csv_zip:
    separator = "," if csv_trennzeichen == "Komma" else ";"
    export.write_csv_zip("query_url_matcher_csv.zip", sheets, sep=separator)
    colab.download("query_url_matcher_csv.zip")
print(f"✅ Schritt 8 fertig: {len(sheets)} Blätter exportiert.")
'''

VERDICT_GUIDE = '''## So entstehen die Urteile (Schritt 7a bis 7c)

Für jede Query prüft das Tool zwei Dinge: Wie viele deiner Seiten passen semantisch, erreichen also mit ihrem Score die Schwelle aus Schritt 7b? Und rankt die Query heute schon gut (Rankings aus Schritt 5 oder 6)? „Fast gleich gut“ heißt: Mehrere Seiten passen und liegen höchstens um den Abstand „fast gleich“ (Voreinstellung 0.01) unter dem besten Score. Sonst passt genau eine Seite klar.

| | Genau eine Seite passt klar | Mehrere Seiten passen fast gleich gut | Keine Seite passt |
|---|---|---|---|
| **Rankt gut** (Voreinstellung: bis Position 10) | **In Ordnung** | **In Ordnung**, wenn die rankende Seite die beste ist oder fast gleich gut passt (in der Übersicht steht dann Kannibalisierungsgefahr „ja“). **Kannibalisierungsgefahr**, wenn eine andere eigene Seite deutlich besser passt, die rankende Seite die Schwelle nicht erreicht oder die rankende URL nicht im Frog-Export steht. | **Rankt trotz schwachem Match**: beobachten und die Schwelle prüfen |
| **Rankt schwach oder gar nicht** | **Bestehende Seite nutzen**: ausbauen und intern stärken statt neu bauen | **Kannibalisierungsgefahr**: mehrere eigene Seiten konkurrieren, bevor eine davon rankt | **Content-Lücke**: Kandidat für eine neue Seite |
| **Keine Rankings geladen** | **Passende Seite vorhanden** | **Kannibalisierungsgefahr** | **Content-Lücke** |

- Bei gutem Ranking gelten die Bedingungen der mittleren Spalte auch, wenn nur eine Seite klar passt: Ist das nicht die rankende Seite, lautet das Urteil **Kannibalisierungsgefahr**.
- **In jedem Feld** steht in der Übersicht neben dem Urteil die Spalte „Kannibalisierungsgefahr“: „ja“, wenn die Query im Blatt „Kannibalisierungsgefahr“ steht. Das ist auch der Fall, wenn mehrere eigene Seiten für dieselbe Query ranken (Stufe „Bereits sichtbar“).
- Das Blatt „Potentielle Content-Lücken“ hat eigene Einstellungen in Schritt 7c. Seine Zahl kann deshalb von der Zahl der Urteile „Content-Lücke“ abweichen.

Alle Urteile sind Hinweise zum Prüfen, keine Entscheidungen.
'''

CELLS = [
    ("markdown", INTRO),
    ("code", STEP2),
    ("code", STEP3),
    ("code", STEP4),
    ("code", STEP5),
    ("code", STEP6),
    ("markdown", VERDICT_GUIDE),
    ("code", STEP7A),
    ("code", STEP7B),
    ("code", STEP7C),
    ("code", STEP8),
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
            "accelerator": "GPU",  # Colab startet die Laufzeit damit direkt mit GPU
            "colab": {"provenance": [], "gpuType": "T4"},
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
