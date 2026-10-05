import time

import httpx
import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, post_json, prepare

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiEmbedder(Embedder):
    def __init__(self, spec, api_key, client=None, sleep=time.sleep, batch_size=50):
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY fehlt. Lege ihn in Colab unter Secrets (Schlüssel-Symbol) an.")
        self.spec = spec
        self._headers = {"x-goog-api-key": api_key}
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._batch_size = batch_size

    def embed(self, texts, role):
        prepared = prepare(self.spec, texts, role)
        task = self.spec.query_task if role == "query" else self.spec.passage_task
        model = f"models/{self.spec.model_id}"
        url = f"{BASE}/{self.spec.model_id}:batchEmbedContents"
        vectors = []
        for start in range(0, len(prepared), self._batch_size):
            requests = [
                {"model": model, "content": {"parts": [{"text": text}]}, "taskType": task}
                for text in prepared[start : start + self._batch_size]
            ]
            data = post_json(self._client, url, self._headers, {"requests": requests}, sleep=self._sleep)
            vectors.extend(item["values"] for item in data["embeddings"])
        return l2_normalize(np.array(vectors))
