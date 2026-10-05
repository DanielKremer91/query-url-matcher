import numpy as np

from .base import Embedder, EmbeddingError, l2_normalize, prepare


class LocalEmbedder(Embedder):
    def __init__(self, spec, model=None):
        self.spec = spec
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers ist nicht installiert. Führe Schritt 2 (Installation) erneut aus."
                ) from exc
            model = SentenceTransformer(spec.model_id)
        if spec.max_seq_length:
            model.max_seq_length = spec.max_seq_length
        self.model = model

    def embed(self, texts, role):
        vectors = self.model.encode(
            prepare(self.spec, texts, role),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,  # den Gesamtfortschritt meldet CachedEmbedder
        )
        return l2_normalize(np.asarray(vectors))

    def _tokens(self, text: str) -> int:
        prepared = prepare(self.spec, [text], "passage")[0]
        return len(self.model.tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"])

    def fits_context(self, text):
        limit = self.model.max_seq_length
        # ein Wort ist mindestens ein Token: lange Texte ohne Tokenizer ausschließen
        if len(text.split()) > limit:
            return False
        return self._tokens(text) <= limit

    def truncated_share(self, texts) -> float:
        if not texts:
            return 0.0
        return sum(not self.fits_context(text) for text in texts) / len(texts)
