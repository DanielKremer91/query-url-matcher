import time

import httpx
import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, post_json, prepare

URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbedder(Embedder):
    def __init__(self, spec, api_key, client=None, sleep=time.sleep, batch_size=32):
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY fehlt. Lege ihn in Colab unter Secrets (Schlüssel-Symbol) an.")
        self.spec = spec
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._batch_size = batch_size

    def embed(self, texts, role):
        prepared = prepare(self.spec, texts, role)
        vectors = []
        for start in range(0, len(prepared), self._batch_size):
            batch = prepared[start : start + self._batch_size]
            data = post_json(
                self._client, URL, self._headers, {"model": self.spec.model_id, "input": batch}, sleep=self._sleep
            )
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in rows)
        return l2_normalize(np.array(vectors))
