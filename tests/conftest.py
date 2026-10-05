import zlib

import numpy as np

from qum.embeddings.base import Embedder, l2_normalize
from qum.models import ModelSpec

FAKE_SPEC = ModelSpec("fake", "Fake", "local", "fake-model", 5, 1)


class FakeEmbedder(Embedder):
    """Bag-of-Words-Vektoren: Texte mit gemeinsamen Wörtern sind ähnlich."""

    DIMS = 512

    def __init__(self, spec=None, max_words=None):
        self.spec = spec or FAKE_SPEC
        self.max_words = max_words
        self.calls = []

    def embed(self, texts, role):
        self.calls.append((list(texts), role))
        m = np.zeros((len(texts), self.DIMS), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                m[i, zlib.crc32(word.encode()) % self.DIMS] += 1
        return l2_normalize(m)

    def fits_context(self, text):
        return self.max_words is not None and len(text.split()) <= self.max_words
