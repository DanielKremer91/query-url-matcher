import math
import time

import httpx
import numpy as np

from ..models import ModelSpec

_NO_RETRY = {400, 401, 403, 404}
_RETRY_AFTER = {429, 503}
_MAX_RETRY_AFTER = 60


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


def unexpected_response(provider: str, detail: str) -> EmbeddingError:
    return EmbeddingError(f"{provider} hat eine unerwartete Antwort geliefert ({detail}). Führe den Schritt erneut aus.")


def _retry_after(response: httpx.Response):
    """Wartezeit aus dem Header Retry-After in Sekunden (nur bei 429 und 503), höchstens 60."""
    if response.status_code not in _RETRY_AFTER:
        return None
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER) if math.isfinite(seconds) else None


def post_json(client: httpx.Client, url: str, headers: dict, payload: dict, attempts: int = 6, sleep=time.sleep) -> dict:
    """POST mit Wiederholung: Retry-After bei 429/503, sonst Pausen von 2, 4, 8, 16, 32 Sekunden."""
    error = None
    for attempt in range(attempts):
        wait = None
        try:
            response = client.post(url, headers=headers, json=payload, timeout=120)
        except httpx.HTTPError as exc:
            error = EmbeddingError(f"Verbindungsfehler: {exc}")
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise EmbeddingError(
                        f"Die Antwort des Anbieters ist kein gültiges JSON: {response.text[:200]}"
                    ) from None
            error = EmbeddingError(f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code in _NO_RETRY:
                break
            wait = _retry_after(response)
        if attempt < attempts - 1:
            sleep(2 ** (attempt + 1) if wait is None else wait)
    raise error
