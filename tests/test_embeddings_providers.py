import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from qum.embeddings import make_embedder
from qum.embeddings.base import EmbeddingError
from qum.embeddings.cache import CachedEmbedder
from qum.embeddings.gemini import GeminiEmbedder
from qum.embeddings.local import LocalEmbedder
from qum.embeddings.openai import OpenAIEmbedder
from qum.models import get_model


class FakeSentenceModel:
    max_seq_length = 6

    def __init__(self):
        self.seen = []
        self.progress_flags = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.seen.append(list(texts))
        self.progress_flags.append(show_progress_bar)
        return np.array([[float(len(t)), 1.0] for t in texts])

    def tokenizer(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": text.split()}


def test_local_embedder_prefixes_e5_and_normalises():
    model = FakeSentenceModel()
    out = LocalEmbedder(get_model("e5-large"), model=model).embed(["futter"], "query")
    assert model.seen == [["query: futter"]]
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_local_embedder_never_shows_its_own_progress_bar():
    model = FakeSentenceModel()
    emb = LocalEmbedder(get_model("e5-large"), model=model)
    emb.embed(["futter"] * 10, "query")
    emb.embed(["futter"] * 500, "passage")
    assert model.progress_flags == [False, False]


def test_local_embedder_sets_max_seq_length_for_comparison_model():
    model = FakeSentenceModel()
    LocalEmbedder(get_model("paraphrase-mpnet"), model=model)
    assert model.max_seq_length == 512


def test_local_fits_context_counts_tokens_including_prefix():
    emb = LocalEmbedder(get_model("e5-large"), model=FakeSentenceModel())
    assert emb.fits_context("a b c d e") is True  # "passage:" + 5 Wörter = 6 Tokens
    assert emb.fits_context("a b c d e f") is False


def test_local_truncated_share():
    emb = LocalEmbedder(get_model("e5-large"), model=FakeSentenceModel())
    assert emb.truncated_share(["a b", "a b c d e f g", "a", "a b c d e f g h"]) == 0.5


class OutOfMemoryError(Exception):
    """Gleicher Klassenname wie torch.cuda.OutOfMemoryError."""


class OomModel(FakeSentenceModel):
    def __init__(self, fail_above=None, error=None):
        super().__init__()
        self.fail_above = fail_above
        self.error = error or OutOfMemoryError("CUDA out of memory. Tried to allocate 5.5 GiB")
        self.batch_sizes = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.batch_sizes.append(batch_size)
        if self.fail_above is None or batch_size > self.fail_above:
            raise self.error
        return super().encode(texts, batch_size, normalize_embeddings, show_progress_bar)


def test_local_embedder_halves_batch_size_on_out_of_memory(capsys):
    model = OomModel(fail_above=4)
    emb = LocalEmbedder(get_model("bge-m3"), model=model)
    out = emb.embed(["a", "bb", "ccc"], "passage")
    assert model.batch_sizes == [32, 16, 8, 4]
    assert emb.batch_size == 4
    assert out.shape == (3, 2)
    assert model.seen == [["a", "bb", "ccc"]]
    printed = capsys.readouterr().out
    assert printed.splitlines() == [f"⚠️ GPU-Speicher knapp, Stapelgröße auf {n} reduziert." for n in (16, 8, 4)]


def test_local_embedder_keeps_reduced_batch_size_for_later_calls(capsys):
    model = OomModel(fail_above=4)
    emb = LocalEmbedder(get_model("bge-m3"), model=model)
    emb.embed(["a"], "passage")
    capsys.readouterr()
    emb.embed(["b"], "passage")
    assert model.batch_sizes[-1] == 4
    assert capsys.readouterr().out == ""


def test_local_embedder_detects_runtime_error_with_out_of_memory_message():
    model = OomModel(fail_above=16, error=RuntimeError("CUDA error: out of memory"))
    emb = LocalEmbedder(get_model("bge-m3"), model=model)
    assert emb.embed(["a"], "passage").shape == (1, 2)
    assert emb.batch_size == 16


def test_local_embedder_raises_german_error_when_batch_size_one_still_fails():
    model = OomModel()
    emb = LocalEmbedder(get_model("bge-m3"), model=model)
    with pytest.raises(EmbeddingError, match="Die Grafikkarte hat nicht genug Speicher für diese Texte"):
        emb.embed(["a"], "passage")
    assert model.batch_sizes == [32, 16, 8, 4, 2, 1]


def test_local_embedder_error_message_names_the_remedy():
    emb = LocalEmbedder(get_model("bge-m3"), model=OomModel())
    with pytest.raises(EmbeddingError) as info:
        emb.embed(["a"], "passage")
    assert str(info.value) == (
        "Die Grafikkarte hat nicht genug Speicher für diese Texte. "
        "Verkleinere die Chunk-Größe in Schritt 4 oder wähle ein kleineres Modell."
    )


@pytest.mark.parametrize("error", [ValueError("kaputt"), RuntimeError("anderer Fehler")])
def test_local_embedder_propagates_other_errors_unchanged(error):
    model = OomModel(error=error)
    emb = LocalEmbedder(get_model("bge-m3"), model=model)
    with pytest.raises(type(error)) as info:
        emb.embed(["a"], "passage")
    assert info.value is error
    assert model.batch_sizes == [32]


class LongContextModel(FakeSentenceModel):
    max_seq_length = 8192


def test_local_fits_context_is_capped_by_fulltext_max_tokens():
    emb = LocalEmbedder(get_model("bge-m3"), model=LongContextModel())
    assert emb.fits_context(" ".join(["w"] * 2048)) is True
    assert emb.fits_context(" ".join(["w"] * 2049)) is False
    assert emb.truncated_share([" ".join(["w"] * 3000), "w"]) == 0.5


def test_local_fits_context_without_cap_uses_model_limit():
    emb = LocalEmbedder(get_model("e5-large"), model=LongContextModel())
    assert emb.fits_context(" ".join(["w"] * 3000)) is True


def test_local_fits_context_keeps_smaller_model_limit_when_cap_is_larger():
    spec = replace(get_model("bge-m3"), fulltext_max_tokens=100)
    emb = LocalEmbedder(spec, model=FakeSentenceModel())  # max_seq_length 6
    assert emb.fits_context("a b c d e f") is True
    assert emb.fits_context("a b c d e f g") is False


def _recording_client(make_response):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=make_response(json.loads(request.content)))

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_openai_sends_plain_text_in_batches_and_keeps_order():
    def respond(body):
        # absichtlich in umgekehrter Reihenfolge
        items = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return {"data": items[::-1]}

    client, requests = _recording_client(respond)
    emb = OpenAIEmbedder(get_model("openai"), "sk-test", client=client, batch_size=2)
    out = emb.embed(["a", "bbb", "cc"], "query")
    bodies = [json.loads(r.content) for r in requests]
    assert [b["input"] for b in bodies] == [["a", "bbb"], ["cc"]]
    assert bodies[0]["model"] == "text-embedding-3-large"
    assert requests[0].headers["authorization"] == "Bearer sk-test"
    assert out.shape == (3, 2)
    assert out[1, 0] > out[0, 0]  # "bbb" ist länger als "a": Reihenfolge stimmt


def test_openai_without_key_raises():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(get_model("openai"), "")


def test_gemini_sends_task_type_per_role():
    def respond(body):
        return {"embeddings": [{"values": [1.0, 2.0]} for _ in body["requests"]]}

    client, requests = _recording_client(respond)
    emb = GeminiEmbedder(get_model("gemini"), "g-key", client=client)
    emb.embed(["futter"], "query")
    emb.embed(["Ein Text"], "passage")
    first, second = (json.loads(r.content)["requests"][0] for r in requests)
    assert first == {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": "futter"}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    assert second["taskType"] == "RETRIEVAL_DOCUMENT"
    assert requests[0].headers["x-goog-api-key"] == "g-key"
    assert str(requests[0].url).endswith("/models/gemini-embedding-001:batchEmbedContents")


def test_factory_wraps_in_cache_and_requires_key():
    emb = make_embedder("openai", api_key="sk-test")
    assert isinstance(emb, CachedEmbedder)
    assert isinstance(emb.inner, OpenAIEmbedder)
    with pytest.raises(EmbeddingError):
        make_embedder("gemini", api_key=None)


def test_gemini_batches_and_keeps_order():
    def respond(body):
        return {"embeddings": [{"values": [float(len(r["content"]["parts"][0]["text"])), 1.0]} for r in body["requests"]]}

    client, requests = _recording_client(respond)
    emb = GeminiEmbedder(get_model("gemini"), "g-key", client=client, batch_size=2)
    out = emb.embed(["a", "bbb", "cc"], "passage")
    texts = [[r["content"]["parts"][0]["text"] for r in json.loads(req.content)["requests"]] for req in requests]
    assert texts == [["a", "bbb"], ["cc"]]
    assert out.shape == (3, 2)
    assert out[0, 0] < out[2, 0] < out[1, 0]  # Längen 1 < 2 < 3: Reihenfolge stimmt


@pytest.mark.parametrize(
    "response",
    [{"error": "x"}, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}],
    ids=["ohne data", "zu wenige Zeilen"],
)
def test_openai_malformed_response_raises_german_error(response):
    client, _ = _recording_client(lambda body: response)
    emb = OpenAIEmbedder(get_model("openai"), "sk-test", client=client)
    with pytest.raises(EmbeddingError, match="OpenAI hat eine unerwartete Antwort geliefert"):
        emb.embed(["a", "b"], "query")


@pytest.mark.parametrize(
    "response",
    [{"error": "x"}, {"embeddings": [{"values": [1.0, 0.0]}]}],
    ids=["ohne embeddings", "zu wenige Zeilen"],
)
def test_gemini_malformed_response_raises_german_error(response):
    client, _ = _recording_client(lambda body: response)
    emb = GeminiEmbedder(get_model("gemini"), "g-key", client=client)
    with pytest.raises(EmbeddingError, match="Gemini hat eine unerwartete Antwort geliefert"):
        emb.embed(["a", "b"], "query")
