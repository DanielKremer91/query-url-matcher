import hashlib
from pathlib import Path

import numpy as np

from .base import Embedder


class CachedEmbedder(Embedder):
    """Bettet jeden Text je Rolle nur einmal ein. Optional auf Platte gesichert."""

    def __init__(self, inner: Embedder, cache_dir=None):
        self.inner = inner
        self.spec = inner.spec
        self._store = {}
        self._path = Path(cache_dir) / f"{self.spec.key}.npz" if cache_dir else None
        if self._path is not None and self._path.exists():
            data = np.load(self._path)
            self._store = dict(zip(data["keys"].tolist(), data["vectors"]))

    def _key(self, text: str, role: str) -> str:
        spec = self.spec
        raw = "\x00".join([spec.model_id, spec.query_prefix, spec.passage_prefix, role, text])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def embed(self, texts, role):
        keys = [self._key(text, role) for text in texts]
        missing = {}
        for key, text in zip(keys, texts):
            if key not in self._store and key not in missing:
                missing[key] = text
        if missing:
            vectors = self.inner.embed(list(missing.values()), role)
            self._store.update(zip(missing.keys(), vectors))
            self._save()
        return np.vstack([self._store[key] for key in keys])

    def fits_context(self, text):
        return self.inner.fits_context(text)

    def _save(self):
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, keys=np.array(list(self._store)), vectors=np.vstack(list(self._store.values())))
