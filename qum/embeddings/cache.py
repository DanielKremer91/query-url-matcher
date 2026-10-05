import hashlib
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from .base import CONSOLE, Embedder, EmbeddingError


class ProgressLine:
    """Eine Fortschrittszeile, die an Ort und Stelle überschrieben wird; fertig endet sie ohne Auslassungszeichen."""

    def __call__(self, done: int, total: int) -> None:
        shown = f"{done:,} von {total:,}".replace(",", ".")
        CONSOLE.progress(f"✅ {shown} Texten eingebettet" if done >= total else f"⏳ {shown} Texten eingebettet …")

    def close(self) -> None:
        CONSOLE.end_line()


class CachedEmbedder(Embedder):
    """Bettet jeden Text je Rolle nur einmal ein. Optional auf Platte gesichert."""

    # fehlende Texte gehen in Scheiben an den Anbieter, damit fertige Scheiben einen Abbruch überleben
    SLICE = 256
    # die Cache-Datei wird während eines Aufrufs höchstens so oft geschrieben (Sekunden)
    SAVE_INTERVAL = 60

    def __init__(self, inner: Embedder, cache_dir=None, notify=None, clock=time.monotonic):
        self.inner = inner
        self.spec = inner.spec
        self._notify = ProgressLine() if notify is None else notify
        self._clock = clock
        self._store = {}
        self._path = Path(cache_dir) / f"{self.spec.key}.npz" if cache_dir else None
        if self._path is not None and self._path.exists():
            try:
                with np.load(self._path) as data:
                    self._store = dict(zip(data["keys"].tolist(), data["vectors"]))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                print(f"⚠️ Zwischenspeicher {self._path} war unlesbar und wird neu aufgebaut.")

    def _key(self, text: str, role: str) -> str:
        spec = self.spec
        raw = "\x00".join([spec.model_id, spec.query_prefix, spec.passage_prefix, role, text])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def embed(self, texts, role):
        if not texts:
            return self.inner.embed([], role)

        keys = [self._key(text, role) for text in texts]
        missing = {}
        for key, text in zip(keys, texts):
            if key not in self._store and key not in missing:
                missing[key] = text
        todo = list(missing)
        show_progress = len(todo) > self.SLICE
        last_save = self._clock()
        unsaved = False
        try:
            for start in range(0, len(todo), self.SLICE):
                part = todo[start : start + self.SLICE]
                vectors = self.inner.embed([missing[key] for key in part], role)
                if len(vectors) != len(part):
                    raise EmbeddingError(f"Der Anbieter hat {len(vectors)} statt {len(part)} Embeddings geliefert.")
                self._store.update(zip(part, vectors))
                unsaved = True
                if show_progress:
                    self._notify(start + len(part), len(todo))
                now = self._clock()
                if now - last_save >= self.SAVE_INTERVAL:
                    self._save()
                    last_save, unsaved = now, False
        finally:
            # fertige Scheiben überleben Fehler und Abbruch
            try:
                if unsaved:
                    self._save()
            finally:
                close = getattr(self._notify, "close", None)
                if close is not None:
                    close()
        return np.vstack([self._store[key] for key in keys])

    def fits_context(self, text):
        return self.inner.fits_context(text)

    def _save(self):
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(self._path.stem + ".tmp.npz")
        np.savez(tmp_path, keys=np.array(list(self._store)), vectors=np.vstack(list(self._store.values())))
        os.replace(tmp_path, self._path)
