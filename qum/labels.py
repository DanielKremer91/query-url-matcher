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
C_GAP_COUNT = "Lücken-Queries"
C_NEW_PAGES = "Neue Seiten"
C_GAP_QUERIES = "Queries"

C_STAGE = "Stufe"
C_REASON = "Grund"
C_COMPETING = "Konkurrierende URLs"

C_PAIR_RANK = "Rang der URL"

# Prüfbeispiele um die Schwelle (Schritt 7, nicht im Export)
C_SIDE = "Lage"
C_SCORE = "Score"
SIDE_ABOVE = "knapp über der Schwelle"
SIDE_BELOW = "knapp unter der Schwelle"

# Spalten der Lesehilfe
R_AREA = "Bereich"
R_ENTRY = "Eintrag"
R_TEXT = "Erklärung"

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

REASON_BETTER = "Eine andere Seite passt deutlich besser als die rankende"
REASON_BETTER_PLAIN = "Eine andere Seite passt besser als die rankende"
REASON_NOT_IN_EXPORT = "Rankende URL steht nicht im Frog-Export und wurde nicht verglichen"
REASON_OK_CLOSE = "Rankende Seite passt, eine weitere passt fast gleich gut"
REASON_CLOSE = "Mehrere Seiten passen fast gleich gut"
REASON_RANKING = "Mehrere eigene Seiten ranken für die Query"

NOTE_NOT_IN_EXPORT = "Rankende URL steht nicht im Frog-Export"
NOTE_URL_MISSING = "URL steht nicht im Frog-Export"
NO_CLUSTER = "ohne Cluster"

# Auswahl im Formular -> interner Schlüssel
BASIS = {"Chunk": "chunk", "Gesamt-URL": "full", "Kombi": "combined"}
# Wahl der Schwelle in Schritt 7 -> Herkunft der Schwelle (None: nur die Vorschläge zeigen, keine Urteile)
THRESHOLD_CHOICE = {
    "Erst Vorschläge ansehen": None,
    "Aus Rankings kalibriert": "rankings",
    "Mittlerer bester Score (nicht kalibriert)": "median",
    "Eigener Wert": "manuell",
}
