import json

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
