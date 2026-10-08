"""Alle Bezeichnungen, die der Nutzer im Notebook und im Export sieht."""

C_QUERY = "Query"
C_URL = "URL"
C_CHUNK = "Relevanter Chunk"
C_S_CHUNK = "Score Chunk"
C_S_FULL = "Score Gesamt-URL"
C_S_COMBI = "Score Kombi"

C_VERDICT = "Urteil"
C_BEST_URL = "Beste URL"
C_LEAD_GAP = "Vorsprung vor zweitbester URL"
C_TO_THRESHOLD = "Abstand zur Schwelle"
C_THRESHOLD = "Schwelle"
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

# Blatt Potentielle Content-Lücken
C_BEST_SCORE = "Bester Score"
C_TOPIC = "Thema"


# Blatt Kannibalisierungsgefahr (Langformat): je konkurrierender URL eine Zeile mit Leit-Score und eigener Position
C_STAGE = "Stufe"
C_REASON = "Grund"
C_SCORE_2 = "Score 2"  # Vorschau in Schritt 4
C_NO = "Nr."
C_COMP_URL = "Konkurrierende URL"
C_COMP_SCORE = "Score der URL"
C_GAP_TO_BEST = "Abstand zur besten URL"
C_COMP_CHUNK = "Relevanter Chunk der URL"
C_COMP_POS = "Position der URL"

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
CHUNK_MEAN = "Mittelwert der Chunk-Scores"

V_MATCH = "Passende Seite vorhanden"
V_GAP = "Content-Lücke"
V_OK = "In Ordnung"
V_CANNIBAL = "Kannibalisierungsgefahr"
V_WATCH = "Rankt trotz schwachem Match"

STAGE_DANGER = "Gefahr"
C_PRIORITY = "Einordnung"
PRIO_VERY_HIGH = "sehr hoch"
PRIO_HIGH = "hoch"
PRIO_MID = "mittel"
PRIO_LOW = "niedrig"
PRIO_VERY_LOW = "sehr niedrig"
PRIO_OPEN = "offen"
STAGE_POSSIBLE = "Möglich"
STAGE_VISIBLE = "Kannibalisierung bereits sichtbar"

REASON_BETTER = "Eine andere Seite passt deutlich besser als die rankende"
REASON_BETTER_PLAIN = "Eine andere Seite passt besser als die rankende"
REASON_NOT_IN_EXPORT = "Rankende URL steht nicht im Frog-Export und wurde nicht verglichen"
REASON_OK_CLOSE = "Rankende Seite passt, eine weitere passt fast gleich gut"
REASON_CLOSE = "Mehrere Seiten passen fast gleich gut"
REASON_FURTHER = "Weitere Seiten erreichen die Schwelle, liegen aber deutlich hinter der besten"
REASON_RANKING = "Mehrere eigene Seiten ranken für die Query"

# Werte der Spalten "Rankende URL = beste URL?" und Kannibalisierungsgefahr
YES = "ja"
NO = "nein"
CMP_NOT_RANKING = "rankt nicht"
CMP_NOT_IN_EXPORT = "nicht im Frog-Export"

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

# Blatt Chunk auf anderer Seite: bester Chunk und beste Seite insgesamt (Gesamt-URL-Score) gehören zu verschiedenen URLs
C_CHUNK_URL = "Seite mit bestem Chunk"
C_BEST_CHUNK = "Bester Chunk"
C_S_BEST_CHUNK = "Score bester Chunk"
C_S_FULL_CHUNK_URL = "Score Gesamt-URL der Chunk-Seite"
C_OVERALL_URL = "Beste Seite insgesamt"
C_S_FULL_OVERALL = "Score Gesamt-URL der besten Seite"
C_RANK_IS = "Rankende URL ist"
RANK_IS_CHUNK = "die Seite mit bestem Chunk"
RANK_IS_OVERALL = "die beste Seite insgesamt"
RANK_IS_OTHER = "eine andere Seite"

# Leitfrage der Übersicht: Neue Seite bauen?
C_BUILD = "Neue Seite bauen?"
C_USE_INSTEAD = "Stattdessen nutzen"
BUILD_YES = "Ja, Content-Lücke"
BUILD_NO_MATCH = "Nein, Seite vorhanden"
BUILD_NO_RANKS = "Nein, rankt bereits"
BUILD_NO_CONFLICT = "Nein, erst Konkurrenz klären"
BUILD_NO_WEAK = "Nein, rankende Seite ausbauen"
# Blatt Kannibalisierungsgefahr: bestehende Konkurrenz oder Vorbeugung (eine neue Seite würde konkurrieren)
C_KIND = "Art"
KIND_EXISTING = "Bestehende Seiten konkurrieren"
KIND_PREVENT = "Vorbeugung: keine neue Seite bauen"
