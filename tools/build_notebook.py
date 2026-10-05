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
