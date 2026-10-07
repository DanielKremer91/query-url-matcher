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
C_GAP_TO_BEST = "Abstand zur besten URL"

C_VERDICT = "Urteil"
C_BEST_URL = "Beste URL"
C_LEAD_GAP = "Vorsprung vor zweitbester URL"
C_SECOND_URL = "Zweitbeste URL"
C_S_CHUNK_2 = "Score Chunk 2"
C_S_FULL_2 = "Score Gesamt-URL 2"
C_S_COMBI_2 = "Score Kombi 2"
C_THIRD_URL = "Drittbeste URL"
C_S_CHUNK_3 = "Score Chunk 3"
C_S_FULL_3 = "Score Gesamt-URL 3"
C_S_COMBI_3 = "Score Kombi 3"
C_POSITION = "Rankingposition"
C_RANK_URL = "Rankende URL"
C_RANK_IS_BEST = "Rankende URL = beste URL?"

C_CLUSTER = "Cluster"
C_GAP_COUNT = "Lücken-Queries"
C_NEW_PAGES = "Neue Seiten"
C_GAP_QUERIES = "Queries"

C_CANNIBAL = "Kannibalisierungsgefahr"

# Blatt Kannibalisierungsgefahr: bis zu drei konkurrierende URLs mit Leit-Score und eigener Position
C_STAGE = "Stufe"
C_REASON = "Grund"
C_URL_1 = "URL 1"
C_SCORE_1 = "Score 1"
C_POS_1 = "Position 1"
C_URL_2 = "URL 2"
C_SCORE_2 = "Score 2"
C_POS_2 = "Position 2"
C_URL_3 = "URL 3"
C_SCORE_3 = "Score 3"
C_POS_3 = "Position 3"

# Prüfbeispiele um die Schwelle (Schritt 7a und 7b, nicht im Export)
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
V_CANNIBAL = "Kannibalisierungsgefahr"
V_WATCH = "Rankt trotz schwachem Match"
V_USE = "Bestehende Seite nutzen"

STAGE_DANGER = "Gefahr"
STAGE_VISIBLE = "Bereits sichtbar"

REASON_BETTER = "Eine andere Seite passt deutlich besser als die rankende"
REASON_BETTER_PLAIN = "Eine andere Seite passt besser als die rankende"
REASON_NOT_IN_EXPORT = "Rankende URL steht nicht im Frog-Export und wurde nicht verglichen"
REASON_OK_CLOSE = "Rankende Seite passt, eine weitere passt fast gleich gut"
REASON_CLOSE = "Mehrere Seiten passen fast gleich gut"
REASON_RANKING = "Mehrere eigene Seiten ranken für die Query"

# Werte der Spalten "Rankende URL = beste URL?" und Kannibalisierungsgefahr
YES = "ja"
NO = "nein"
CMP_CLOSE = "fast gleich gut"
CMP_NOT_RANKING = "rankt nicht"
CMP_NOT_IN_EXPORT = "nicht im Frog-Export"

NO_CLUSTER = "ohne Cluster"

# Auswahl im Formular -> interner Schlüssel
BASIS = {"Chunk": "chunk", "Gesamt-URL": "full", "Kombi": "combined"}
# Wahl der Schwelle in Schritt 7b -> Herkunft der Schwelle
THRESHOLD_CHOICE = {
    "Aus Rankings kalibriert": "rankings",
    "Mittlerer bester Score (nicht kalibriert)": "median",
    "Eigener Wert": "manuell",
}

# Beschriftung der Fortschrittszeilen beim Einbetten (Schritt 4)
P_QUERIES = "Queries"
P_CHUNKS = "Chunks"
P_PAGES = "Ganze Seiten"
