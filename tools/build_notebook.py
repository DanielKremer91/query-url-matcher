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

Schon ohne Ranking-Dateien bekommst du das Matching mit Urteilen zu passender Seite und Content-Lücke und den Hinweis, wenn mehrere Seiten fast gleich gut passen (Kannibalisierung). Eigene Rankings ergänzen die Urteile, die sich auf Rankings stützen (zum Beispiel "Kannibalisierungs-Risiko" und "Rankt trotz schwachem Match"), und kalibrieren die Schwelle für "passend". Top-10-SERPs bündeln die Lücken zusätzlich zu Themen.

## So gehst du vor

Führe die Zellen einzeln von oben nach unten aus (Play-Symbol links), nicht mit "Alle ausführen": Die Zellen fragen nach Uploads. Jede Zelle endet mit einer Zeile, die sagt, wie es weitergeht. Die Schritte 5, 6 und 9 sind optional. Wenn du einen früheren Schritt erneut ausführst, setzt das Notebook alles zurück, was darauf aufbaut, und sagt dir, welchen Schritt du danach wiederholen musst.

Für die Modelle mit API-Key legst du im Secrets-Panel (Schlüssel-Symbol links) das Secret an und aktivierst dort den Schalter "Notebook-Zugriff".

## Wichtig

Das Notebook sortiert vor und begründet. Cosinus-Werte sind Hinweise, keine Urteile: Die Entscheidung triffst du, nachdem du die Seite angesehen hast.
'''

STEP2 = '''#@title Schritt 2: Installation und Modellwahl { display-mode: "form" }
#@markdown Wähle das Embedding-Modell. Für deutschen Content ist **multilingual-e5-large** die Empfehlung.
#@markdown Die Modelle mit API-Key brauchen ein Secret (Schlüssel-Symbol links): `OPENAI_API_KEY` bzw. `GEMINI_API_KEY`.
#@markdown Lege es an **und** aktiviere dort den Schalter "Notebook-Zugriff", sonst kann das Notebook den Key nicht lesen.
modell = "multilingual-e5-large · Deutsch, kostenlos (Empfehlung)" #@param ["multilingual-e5-large · Deutsch, kostenlos (Empfehlung)", "multilingual-e5-base · Deutsch, kostenlos, schneller", "bge-m3 · Deutsch, kostenlos, lange Texte", "msmarco-distilbert-base-v4 · nur englische Projekte", "Gemini gemini-embedding-001 · API-Key nötig", "OpenAI text-embedding-3-large · API-Key nötig", "paraphrase-multilingual-mpnet-base-v2 · symmetrisch, NUR zum Vergleich"]

!pip install -q "qum[local] @ git+https://github.com/DanielKremer91/query-url-matcher"

from qum import colab
from qum.embeddings import make_embedder
from qum.models import model_by_label

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
if spec.comparison_only:
    print("⚠️ Dieses Modell ist symmetrisch und nur als Gegenbeispiel gedacht. Nutze es nicht für die Auswertung.")
print(f"✅ Schritt 2 fertig: Modell {spec.model_id} ist bereit. Weiter mit Schritt 3 (Schritt 4 neu ausführen, falls du schon gematcht hattest).")
'''

STEP3 = '''#@title Schritt 3: Queries und Frog-Export hochladen { display-mode: "form" }
#@markdown Die Spalten werden automatisch erkannt. Nur wenn die Zelle eine Spalte nicht findet, trägst du den Namen hier ein.
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
url_found = url_spalte or ingest.find_column(content_table, ingest.URL_ALIASES)
content_found = content_spalte or ingest.find_column(content_table, ingest.CONTENT_ALIASES)
print(f"Erkannte Spalten: Query = {query_found or '(keine Kopfzeile)'}, URL = {url_found}, Content = {content_found}")
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

colab.require(globals(), 2, "spec", "embedder")
colab.require(globals(), 3, "queries", "content")
if chunk_groesse < 0 or chunk_overlap < 0:
    colab.stop("Chunk-Größe und Overlap dürfen nicht negativ sein. 0 bedeutet: Standardwert des Modells.")
if top_n < 1:
    colab.stop("Die Zahl der Treffer je Query (top_n) muss mindestens 1 sein.")
size = chunk_groesse or spec.chunk_size
overlap = chunk_overlap or spec.chunk_overlap
if overlap >= size:
    colab.stop(
        f"Der Overlap ({overlap}) muss kleiner sein als die Chunk-Größe ({size}). "
        "Trage 0 ein, um den Standardwert des Modells zu nutzen."
    )
basis = L.BASIS[bewertungsgrundlage]
n_chunks = estimate_chunks(content.contents, size, overlap)
print(f"ℹ️ {len(content.urls)} URLs ergeben {n_chunks} Chunks ({size} Wörter, Overlap {overlap}).")
if n_chunks >= 50000:
    print("⚠️ Das sind sehr viele Chunks (ab 50.000). Der Lauf wird trotzdem fortgesetzt und kann lange dauern. Schränke den Frog-Export auf ein Verzeichnis ein, wenn dir das zu lange ist.")
colab.invalidate(globals(), *colab.MATCH_STATE)
with colab.guard():
    new_result = run_matching(queries, content.urls, content.contents, embedder, size, overlap)
inner = embedder.inner
if hasattr(inner, "truncated_share"):
    flat = [c for per_url in new_result.chunks for c in per_url]
    share = inner.truncated_share(flat[:2000])
    if share > 0.05:
        print(f"⚠️ {share:.0%} der Chunks sind länger als das Modell lesen kann. Verkleinere die Chunk-Größe.")
weight, basis_label, n_top = kombi_gewicht_chunk, bewertungsgrundlage, top_n
result = new_result
lead = result.lead(basis, weight)
top = top_hits(result, basis, weight, n_top)
display(top.head(10))
print(f"✅ Schritt 4 fertig: {len(top)} Treffer berechnet. Optional weiter mit Schritt 5 und 6, sonst Schritt 7.")
'''

STEP5 = '''#@title Schritt 5 (optional): Eigene Rankings hochladen { display-mode: "form" }
#@markdown Export mit Keyword, URL und Position, zum Beispiel aus GSC, Ahrefs (Organic Keywords) oder SISTRIX.
#@markdown Ergänzt die Urteile, die sich auf Rankings stützen, und kalibriert die Schwelle für "passend".
#@markdown Kannibalisierung und Content-Lücken gibt es auch ohne Rankings.
keyword_spalte = "" #@param {type:"string"}
url_spalte_ranking = "" #@param {type:"string"}
position_spalte = "" #@param {type:"string"}

from qum import colab, ingest
from qum.normalize import normalize_url

colab.require(globals(), 3, "queries", "content")
with colab.guard():
    name, data = colab.upload("Eigene Rankings (Keyword, URL, Position)")
    loaded = ingest.load_rankings(
        colab.read_table(data, name),
        keyword_col=keyword_spalte or None,
        url_col=url_spalte_ranking or None,
        position_col=position_spalte or None,
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
    print(f"✅ Schritt 5 fertig: {len(rankings)} Ranking-Zeilen, davon {covered} zu deinen Queries. Weiter mit Schritt 6 oder 7.")
if rankings is None:
    print("ℹ️ Die Ranking-Datei wurde nicht übernommen, das Notebook bleibt im reinen Matching. Weiter mit Schritt 7 oder lade eine andere Datei hoch.")
'''

STEP6 = '''#@title Schritt 6 (optional): Top-10-SERPs hochladen { display-mode: "form" }
#@markdown Ahrefs-Export mit Keyword, URL, Position und Type (die kompletten Top 10 je Keyword, inklusive Wettbewerber).
#@markdown Schaltet die Bündelung der Content-Lücken zu Themen frei. Die Spaltennamen trägst du nur ein, wenn sie nicht erkannt werden.
serp_keyword_spalte = "" #@param {type:"string"}
serp_url_spalte = "" #@param {type:"string"}
serp_position_spalte = "" #@param {type:"string"}
serp_type_spalte = "" #@param {type:"string"}

from qum import colab, ingest
from qum.normalize import host_of

colab.require(globals(), 3, "queries", "content")
with colab.guard():
    name, data = colab.upload("Top-10-SERPs (Keyword, URL, Position, Type)")
    loaded = ingest.load_serps(
        colab.read_table(data, name),
        keyword_col=serp_keyword_spalte or None,
        url_col=serp_url_spalte or None,
        position_col=serp_position_spalte or None,
        type_col=serp_type_spalte or None,
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
if serps is None:
    print("ℹ️ Die SERPs wurden nicht übernommen. Weiter mit Schritt 7 oder lade eine andere Datei hoch.")
else:
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

from datetime import date

from qum import colab
from qum import labels as L
from qum.cannibal import find_cannibalization
from qum.serp import apply_serp, count_new_pages
from qum.threshold import examples_around, propose_threshold
from qum.verdict import build_decisions

colab.require(globals(), 4, "result", "lead", "top", "size", "overlap", "weight", "basis_label", "n_top")
if not 0 <= schwelle <= 1:
    colab.stop("Die Schwelle muss zwischen 0 und 1 liegen (0 = Vorschlag übernehmen).")
if min(kalibrierung_bis_position, rankt_gut_bis_position, sichtbar_bis_position) < 1:
    colab.stop("Alle Positionen (Kalibrierung, rankt gut, sichtbar) müssen mindestens 1 sein.")
colab.invalidate(globals(), *colab.VERDICT_STATE)

proposal = propose_threshold(result, lead, rankings, max_position=kalibrierung_bis_position)
no_benchmark = (
    'Ohne Rankings gibt es keinen Maßstab für "passend". Der Vorschlag ist nur der mittlere beste Score: '
    "Etwa die Hälfte deiner Queries liegt darunter. Prüfe die Beispiele und trage eine eigene Schwelle ein."
)
if proposal.source == "rankings":
    print(f"Vorschlag {proposal.value:.4f}: Diesen Score erreichen 75 % der {proposal.n_pairs} Paare, die heute gut ranken.")
else:
    print(f"Vorschlag {proposal.value:.4f}: mittlerer Score der besten Treffer.")
    if rankings is None:
        print(f"⚠️ {no_benchmark}")
    else:
        print(f"⚠️ Nur {proposal.n_pairs} Ranking-Paare bis Position {kalibrierung_bis_position}, für eine Kalibrierung sind 20 nötig. {no_benchmark}")
threshold = schwelle or proposal.value
print(f"Verwendete Schwelle: {threshold:.4f}")
print("Diese Paare liegen knapp über und knapp unter der Schwelle. Passt die Grenze?")
display(examples_around(result, lead, threshold))

new_decisions = build_decisions(result, lead, threshold, rankings, rankt_gut_bis_position, weight)
new_cannibal = find_cannibalization(result, lead, threshold, new_decisions, rankings, abstand_fast_gleich, sichtbar_bis_position)
if serps is not None:
    new_decisions = apply_serp(new_decisions, result, lead, serps, rankings, serp_ueberschneidung / 100, cluster_dichte / 100)
settings = {
    "Datum": date.today().isoformat(),
    "Modell": spec.model_id,
    "Nur zum Vergleich (symmetrisches Modell)": "ja" if spec.comparison_only else "nein",
    "Chunk-Größe (Wörter)": size,
    "Overlap (Wörter)": overlap,
    "Bewertungsgrundlage": basis_label,
    "Kombi-Gewicht Chunk": weight,
    "Treffer je Query (Top-N)": n_top,
    "Schwelle": round(threshold, 4),
    "Schwelle aus": "eigener Wert" if schwelle else ("Rankings" if proposal.source == "rankings" else "Median der besten Scores"),
    "Kalibrierung bis Position": kalibrierung_bis_position,
    "Rankt gut bis Position": rankt_gut_bis_position,
    "Abstand fast gleich": abstand_fast_gleich,
    "Sichtbar bis Position": sichtbar_bis_position,
    "SERP-Überschneidung (%)": serp_ueberschneidung,
    "Cluster-Dichte (%)": cluster_dichte,
    "Eigene Rankings": {"Datei": "ja, aus Datei", "SERPs": "ja, aus den SERPs abgeleitet"}.get(rankings_source, "nein"),
    "Top-10-SERPs": "ja" if serps is not None else "nein",
}
decisions, cannibal = new_decisions, new_cannibal
threshold_source = "manuell" if schwelle else proposal.source
counts = decisions[L.C_VERDICT].value_counts()
for verdict, count in counts.items():
    print(f"   {count:>5} × {verdict}")
print(f"   {cannibal[L.C_QUERY].nunique():>5} Queries mit Kannibalisierungs-Hinweis")
uncalibrated = " (Schwelle nicht kalibriert)" if threshold_source == "median" else ""
print(f"   {count_new_pages(decisions):>5} neue Seiten aus den Content-Lücken{uncalibrated}")
print("✅ Schritt 7 fertig. Weiter mit Schritt 8 (Export).")
'''

STEP8 = '''#@title Schritt 8: Export { display-mode: "form" }
#@markdown Die Excel-Datei enthält ein Blatt je Auswertung und eine Lesehilfe.
zusaetzlich_csv_zip = False #@param {type:"boolean"}

from qum import colab, export

colab.require(globals(), 7, "decisions", "top", "cannibal", "settings", "threshold_source")
sheets = export.build_sheets(decisions, top, cannibal, settings, threshold_source=threshold_source)
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
#@markdown Die Spaltennamen trägst du nur ein, wenn sie nicht erkannt werden.
paare_keyword_spalte = "" #@param {type:"string"}
paare_url_spalte = "" #@param {type:"string"}

from qum import colab, export
from qum import labels as L
from qum.pairs import score_pairs

colab.require(globals(), 2, "spec", "embedder")
colab.require(globals(), 3, "content")
colab.require(globals(), 4, "result", "size", "overlap", "basis", "weight")
colab.require(globals(), 7, "decisions", "top", "cannibal", "settings", "threshold_source")
with colab.guard():
    name, data = colab.upload("Keyword-URL-Paare (Keyword, URL)")
    pairs = score_pairs(
        colab.read_table(data, name),
        content.urls,
        content.contents,
        embedder,
        size,
        overlap,
        basis,
        weight,
        keyword_col=paare_keyword_spalte or None,
        url_col=paare_url_spalte or None,
    )
missing = (pairs[L.C_NOTE] != "").sum()
if missing:
    print(f"ℹ️ {missing} Paare haben eine URL, die nicht im Frog-Export steht.")
display(pairs.head(10))
sheets = export.build_sheets(decisions, top, cannibal, settings, pairs=pairs, threshold_source=threshold_source)
export.write_excel("query_url_matcher_mit_paaren.xlsx", sheets)
colab.download("query_url_matcher_mit_paaren.xlsx")
print(f"✅ Schritt 9 fertig: {len(pairs)} Paare bewertet, Export mit Blatt 'Paare' als query_url_matcher_mit_paaren.xlsx heruntergeladen.")
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
