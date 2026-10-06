from qum.models import DEFAULT_MODEL, MODELS, get_model, model_by_label


def test_default_is_e5_large_with_prefixes():
    spec = get_model(DEFAULT_MODEL)
    assert spec.model_id == "intfloat/multilingual-e5-large"
    assert (spec.query_prefix, spec.passage_prefix) == ("query: ", "passage: ")


def test_only_e5_models_have_prefixes():
    with_prefix = {k for k, s in MODELS.items() if s.query_prefix or s.passage_prefix}
    assert with_prefix == {"e5-large", "e5-base"}


def test_gemini_uses_current_model_and_task_types():
    spec = get_model("gemini")
    assert spec.model_id == "gemini-embedding-001"
    assert (spec.query_task, spec.passage_task) == ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")


def test_retired_gemini_model_is_gone():
    assert all("text-embedding-004" not in s.model_id for s in MODELS.values())


def test_only_paraphrase_model_is_comparison_only():
    assert [k for k, s in MODELS.items() if s.comparison_only] == ["paraphrase-mpnet"]


def test_chunk_defaults_follow_spec():
    got = {k: (s.chunk_size, s.chunk_overlap) for k, s in MODELS.items()}
    assert got == {
        "e5-large": (250, 40),
        "e5-base": (250, 40),
        "bge-m3": (1000, 150),
        "msmarco": (250, 40),
        "gemini": (800, 120),
        "openai": (1500, 200),
        "paraphrase-mpnet": (250, 40),
    }


def test_labels_are_unique_and_resolvable():
    labels = [s.label for s in MODELS.values()]
    assert len(labels) == len(set(labels))
    assert model_by_label(labels[0]).key == "e5-large"


def test_api_fulltext_limits_stay_below_the_token_limits():
    # OpenAI: 8.191 Tokens; 4.000 deutsche Wörter können darüber liegen
    assert get_model("openai").fulltext_max_words == 3000
    assert get_model("gemini").fulltext_max_words == 1000


def test_only_bge_m3_caps_whole_page_embedding():
    assert {k: s.fulltext_max_tokens for k, s in MODELS.items() if s.fulltext_max_tokens} == {"bge-m3": 2048}
