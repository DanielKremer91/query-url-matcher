"""Dünne Helfer für das Notebook. Alles Colab-Spezifische wird erst beim Aufruf importiert."""

import zipfile
from contextlib import contextmanager

import pandas as pd

from .embeddings.base import EmbeddingError
from .ingest import IngestError, read_table as _read_table

# Schritt, der die Namen liefert -> Bezeichnung für die Meldung "Bitte zuerst Schritt N (...) ausführen."
STEP_NAMES = {
    2: "Modellwahl",
    3: "Dateien hochladen",
    4: "Matching",
    "7a": "Vorschläge ansehen",
    "7b": "Schwelle festlegen",
    "7c": "Urteile bilden",
}
# Was ungültig wird, wenn sich der Stand davor ändert
DECISION_STATE = ("decisions", "cannibal", "settings", "pairs")  # aus Schritt 7c und 9; Schritt 7b setzt sie zurück
VERDICT_STATE = (
    ("calibrated", "median", "no_calibration", "calibration_position")  # Schritt 7a
    + ("threshold", "threshold_source", "threshold_label")  # Schritt 7b
    + DECISION_STATE
)
MATCH_STATE = ("result", "lead", "top") + VERDICT_STATE


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


def require(namespace, step: int | str, *names):
    """Stoppt mit einem deutschen Hinweis, wenn ein Name aus einem früheren Schritt fehlt oder ungültig ist."""
    if any(namespace.get(name) is None for name in names):
        stop(f"Bitte zuerst Schritt {step} ({STEP_NAMES[step]}) ausführen.")


def invalidate(namespace, *names):
    for name in names:
        namespace[name] = None


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """ingest.read_table, aber kaputte Dateien enden in einer verständlichen Meldung."""
    from openpyxl.utils.exceptions import InvalidFileException

    try:
        return _read_table(data, filename)
    except IngestError:
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError, zipfile.BadZipFile, InvalidFileException, ValueError) as error:
        raise IngestError(
            f"Die Datei '{filename}' konnte nicht gelesen werden ({type(error).__name__}). "
            f"Prüfe, ob es eine gültige CSV- oder Excel-Datei ist, und lade sie erneut hoch."
        ) from None


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
    except userdata.SecretNotFoundError:
        return None
    except userdata.NotebookAccessError:
        stop(
            f'Das Secret {name} existiert, aber das Notebook darf es nicht lesen. Öffne links das Secrets-Panel '
            f'(Schlüssel-Symbol), aktiviere bei {name} den Schalter "Notebook-Zugriff" und führe die Zelle erneut aus.'
        )


def download(path) -> None:
    from google.colab import files

    files.download(str(path))
