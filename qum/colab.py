"""Dünne Helfer für das Notebook. Alles Colab-Spezifische wird erst beim Aufruf importiert."""

from contextlib import contextmanager

from .embeddings.base import EmbeddingError
from .ingest import IngestError


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
    except Exception:
        return None


def download(path) -> None:
    from google.colab import files

    files.download(str(path))
