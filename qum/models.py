from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str  # "local" | "openai" | "gemini"
    model_id: str
    chunk_size: int
    chunk_overlap: int
    query_prefix: str = ""
    passage_prefix: str = ""
    query_task: str | None = None
    passage_task: str | None = None
    # API-Modelle: Volltext-Embedding bis zu dieser Wortzahl. Lokale Modelle prüfen per Tokenizer.
    fulltext_max_words: int | None = None
    max_seq_length: int | None = None
    comparison_only: bool = False


_SPECS = [
    ModelSpec(
        "e5-large",
        "multilingual-e5-large · Deutsch, kostenlos (Empfehlung)",
        "local",
        "intfloat/multilingual-e5-large",
        250,
        40,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    ModelSpec(
        "e5-base",
        "multilingual-e5-base · Deutsch, kostenlos, schneller",
        "local",
        "intfloat/multilingual-e5-base",
        250,
        40,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    ModelSpec("bge-m3", "bge-m3 · Deutsch, kostenlos, lange Texte", "local", "BAAI/bge-m3", 1000, 150),
    ModelSpec(
        "msmarco",
        "msmarco-distilbert-base-v4 · nur englische Projekte",
        "local",
        "msmarco-distilbert-base-v4",
        250,
        40,
    ),
    ModelSpec(
        "gemini",
        "Gemini gemini-embedding-001 · API-Key nötig",
        "gemini",
        "gemini-embedding-001",
        800,
        120,
        query_task="RETRIEVAL_QUERY",
        passage_task="RETRIEVAL_DOCUMENT",
        fulltext_max_words=1000,
    ),
    ModelSpec(
        "openai",
        "OpenAI text-embedding-3-large · API-Key nötig",
        "openai",
        "text-embedding-3-large",
        1500,
        200,
        fulltext_max_words=3000,
    ),
    ModelSpec(
        "paraphrase-mpnet",
        "paraphrase-multilingual-mpnet-base-v2 · symmetrisch, NUR zum Vergleich",
        "local",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        250,
        40,
        max_seq_length=512,
        comparison_only=True,
    ),
]

MODELS = {spec.key: spec for spec in _SPECS}
DEFAULT_MODEL = "e5-large"


def get_model(key: str) -> ModelSpec:
    return MODELS[key]


def model_by_label(label: str) -> ModelSpec:
    for spec in _SPECS:
        if spec.label == label:
            return spec
    raise KeyError(label)
