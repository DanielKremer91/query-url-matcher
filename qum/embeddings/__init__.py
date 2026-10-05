from ..models import ModelSpec, get_model
from .base import Embedder, EmbeddingError
from .cache import CachedEmbedder


def make_embedder(model, api_key=None, cache_dir=None) -> CachedEmbedder:
    """Erzeugt den Embedder zu einem Modellschlüssel oder einer ModelSpec, immer mit Cache."""
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    if spec.provider == "openai":
        from .openai import OpenAIEmbedder

        inner = OpenAIEmbedder(spec, api_key)
    elif spec.provider == "gemini":
        from .gemini import GeminiEmbedder

        inner = GeminiEmbedder(spec, api_key)
    else:
        from .local import LocalEmbedder

        inner = LocalEmbedder(spec)
    return CachedEmbedder(inner, cache_dir=cache_dir)


__all__ = ["Embedder", "EmbeddingError", "CachedEmbedder", "make_embedder"]
