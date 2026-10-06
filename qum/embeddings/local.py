import numpy as np

from .base import CONSOLE, Embedder, EmbeddingError, l2_normalize, prepare

_START_BATCH_SIZE = 32


def _is_out_of_memory(exc: Exception) -> bool:
    """CUDA-Speicherfehler erkennen, ohne torch zu importieren."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_gpu_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        self.batch_size = _START_BATCH_SIZE

    def embed(self, texts, role):
        prepared = prepare(self.spec, texts, role)
        while True:
            try:
                vectors = self.model.encode(
                    prepared,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,  # den Gesamtfortschritt meldet CachedEmbedder
                )
                break
            except Exception as exc:
                if not _is_out_of_memory(exc):
                    raise
                _free_gpu_cache()
                if self.batch_size == 1:
                    raise EmbeddingError(
                        "Die Grafikkarte hat nicht genug Speicher für diese Texte. "
                        "Verkleinere die Chunk-Größe in Schritt 4 oder wähle ein kleineres Modell."
                    ) from exc
                self.batch_size //= 2
                CONSOLE.say(f"⚠️ GPU-Speicher knapp, Stapelgröße auf {self.batch_size} reduziert.")
        return l2_normalize(np.asarray(vectors))

    def _tokens(self, text: str) -> int:
        prepared = prepare(self.spec, [text], "passage")[0]
        return len(self.model.tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"])

    def fits_context(self, text):
        limit = self.model.max_seq_length
        if self.spec.fulltext_max_tokens:
            limit = min(limit, self.spec.fulltext_max_tokens)
        # ein Wort ist mindestens ein Token: lange Texte ohne Tokenizer ausschließen
        if len(text.split()) > limit:
            return False
        return self._tokens(text) <= limit

    def truncated_share(self, texts) -> float:
        if not texts:
            return 0.0
        return sum(not self.fits_context(text) for text in texts) / len(texts)
