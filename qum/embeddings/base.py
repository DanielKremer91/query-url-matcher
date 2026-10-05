import time

import httpx
import numpy as np

from ..models import ModelSpec

_NO_RETRY = {400, 401, 403, 404}


class EmbeddingError(RuntimeError):
    """Fehler beim Erzeugen von Embeddings, mit einer Meldung für den Nutzer."""


class Embedder:
    spec: ModelSpec

    def embed(self, texts: list, role: str) -> np.ndarray:
        raise NotImplementedError

    def fits_context(self, text: str) -> bool:
        limit = self.spec.fulltext_max_words
        return limit is not None and len(text.split()) <= limit


def prepare(spec: ModelSpec, texts: list, role: str) -> list:
    if role not in ("query", "passage"):
        raise ValueError(f"Unbekannte Rolle: {role}")
    prefix = spec.query_prefix if role == "query" else spec.passage_prefix
    return [prefix + text for text in texts]


def l2_normalize(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def post_json(client: httpx.Client, url: str, headers: dict, payload: dict, attempts: int = 3, sleep=time.sleep) -> dict:
    error = None
    for attempt in range(attempts):
        try:
            response = client.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()
            error = EmbeddingError(f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code in _NO_RETRY:
                break
        except httpx.HTTPError as exc:
            error = EmbeddingError(f"Verbindungsfehler: {exc}")
        if attempt < attempts - 1:
            sleep(2**attempt)
    raise error
