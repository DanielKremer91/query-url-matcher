import httpx
import numpy as np
import pytest

from qum.embeddings.base import EmbeddingError, l2_normalize, post_json, prepare
from qum.embeddings.cache import CachedEmbedder
from qum.models import get_model
from tests.conftest import FakeEmbedder


def test_prepare_adds_prefix_only_for_e5():
    e5 = get_model("e5-large")
    assert prepare(e5, ["futter"], "query") == ["query: futter"]
    assert prepare(e5, ["Ein Text"], "passage") == ["passage: Ein Text"]
    assert prepare(get_model("openai"), ["futter"], "query") == ["futter"]
    assert prepare(get_model("bge-m3"), ["Ein Text"], "passage") == ["Ein Text"]


def test_prepare_rejects_unknown_role():
    with pytest.raises(ValueError):
        prepare(get_model("e5-large"), ["x"], "document")


def test_l2_normalize_handles_zero_rows():
    out = l2_normalize([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


def _client(responses):
    calls = iter(responses)

    def handler(request):
        status, body = next(calls)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_json_retries_then_succeeds():
    sleeps = []
    client = _client([(500, {"e": 1}), (429, {"e": 2}), (200, {"ok": True})])
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append) == {"ok": True}
    assert sleeps == [1, 2]


def test_post_json_gives_up_after_three_attempts():
    client = _client([(500, {}), (500, {}), (500, {})])
    with pytest.raises(EmbeddingError, match="500"):
        post_json(client, "https://x.test", {}, {}, sleep=lambda s: None)


def test_post_json_does_not_retry_auth_errors():
    sleeps = []
    client = _client([(401, {"error": "bad key"})])
    with pytest.raises(EmbeddingError, match="401"):
        post_json(client, "https://x.test", {}, {}, sleep=sleeps.append)
    assert sleeps == []


def test_cache_embeds_each_text_once_per_role():
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    first = cached.embed(["a b", "c d", "a b"], "passage")
    second = cached.embed(["c d", "e f"], "passage")
    cached.embed(["a b"], "query")
    assert inner.calls == [(["a b", "c d"], "passage"), (["e f"], "passage"), (["a b"], "query")]
    assert first.shape == (3, FakeEmbedder.DIMS)
    assert np.allclose(first[1], second[0])


def test_cache_persists_to_disk(tmp_path):
    CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path).embed(["a b"], "passage")
    inner = FakeEmbedder()
    out = CachedEmbedder(inner, cache_dir=tmp_path).embed(["a b"], "passage")
    assert inner.calls == []
    assert out.shape == (1, FakeEmbedder.DIMS)


def test_cache_delegates_fits_context():
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c") is True
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c d") is False
