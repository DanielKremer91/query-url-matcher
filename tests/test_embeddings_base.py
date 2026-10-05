import httpx
import numpy as np
import pytest

from qum.embeddings.base import EmbeddingError, l2_normalize, post_json, prepare
from qum.embeddings.cache import CachedEmbedder
from qum.models import get_model
from tests.conftest import FakeEmbedder


def test_cache_recovers_from_corrupted_file(tmp_path, capsys):
    """Corrupted cache file is discarded; cache is rebuilt."""
    # Write garbage to cache file
    cache_file = tmp_path / "fake.npz"
    cache_file.write_bytes(b"this is not a valid npz file")

    # Create CachedEmbedder on corrupted file
    cached = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path)

    # Should work despite corruption
    result = cached.embed(["a b"], "passage")
    assert result.shape == (1, FakeEmbedder.DIMS)

    # Should have called inner embedder (cache was empty)
    assert len(cached.inner.calls) == 1

    # Should have printed warning
    captured = capsys.readouterr()
    assert "Zwischenspeicher" in captured.out
    assert "unlesbar" in captured.out

    # New CachedEmbedder on same dir should load the valid cache
    inner2 = FakeEmbedder()
    cached2 = CachedEmbedder(inner2, cache_dir=tmp_path)
    result2 = cached2.embed(["a b"], "passage")
    assert inner2.calls == []  # Should not call inner embedder
    assert np.allclose(result, result2)


def test_cache_embed_empty_list():
    """Embedding empty list returns empty array with correct shape."""
    cached = CachedEmbedder(FakeEmbedder())
    result = cached.embed([], "passage")
    assert result.shape[0] == 0


def test_cache_detects_wrong_result_count():
    """Cache raises error if embedder returns wrong number of vectors."""
    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts, role):
            # Return one fewer vector than requested
            vectors = super().embed(texts, role)
            return vectors[:-1] if len(vectors) > 0 else vectors

    cached = CachedEmbedder(BrokenEmbedder())
    with pytest.raises(EmbeddingError, match="Anbieter"):
        cached.embed(["a b", "c d"], "passage")


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
    assert sleeps == [2, 4]


def test_post_json_gives_up_after_six_attempts_with_exponential_backoff():
    sleeps = []
    client = _client([(500, {})] * 6)
    with pytest.raises(EmbeddingError, match="500"):
        post_json(client, "https://x.test", {}, {}, sleep=sleeps.append)
    assert sleeps == [2, 4, 8, 16, 32]


def test_post_json_retries_connection_errors():
    sleeps, calls = [], []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ConnectError("weg")
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append) == {"ok": True}
    assert sleeps == [2, 4]


def _client_with_headers(responses):
    calls = iter(responses)

    def handler(request):
        status, headers = next(calls)
        return httpx.Response(status, json={}, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_json_honours_retry_after_on_429_and_503_capped_at_60():
    sleeps = []
    client = _client_with_headers(
        [
            (429, {"Retry-After": "7"}),
            (503, {"Retry-After": "120"}),
            (500, {"Retry-After": "5"}),  # nur 429 und 503 zählen
            (429, {"Retry-After": "bald"}),  # unlesbar: normale Pause
            (200, {}),
        ]
    )
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append) == {}
    assert sleeps == [7, 60, 8, 16]


def test_post_json_non_json_success_raises_embedding_error():
    def handler(request):
        return httpx.Response(200, text="<html>Wartung</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="kein gültiges JSON"):
        post_json(client, "https://x.test", {}, {}, sleep=lambda s: None)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_post_json_does_not_retry_client_errors(status):
    sleeps = []
    client = _client([(status, {"error": "bad key"})])
    with pytest.raises(EmbeddingError, match=str(status)):
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
    first_result = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path).embed(["a b"], "passage")
    inner = FakeEmbedder()
    out = CachedEmbedder(inner, cache_dir=tmp_path).embed(["a b"], "passage")
    assert inner.calls == []
    assert out.shape == (1, FakeEmbedder.DIMS)
    assert np.allclose(first_result, out)


def test_cache_delegates_fits_context():
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c") is True
    assert CachedEmbedder(FakeEmbedder(max_words=3)).fits_context("a b c d") is False


def test_cache_recovers_from_zero_byte_file(tmp_path, capsys):
    (tmp_path / "fake.npz").write_bytes(b"")
    cached = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path)
    assert cached.embed(["a b"], "passage").shape == (1, FakeEmbedder.DIMS)
    assert "unlesbar" in capsys.readouterr().out


class FailingOnThirdCall(FakeEmbedder):
    def embed(self, texts, role):
        if len(self.calls) == 2:
            self.calls.append((list(texts), role))
            raise EmbeddingError("HTTP 500: kaputt")
        return super().embed(texts, role)


def test_cache_embeds_in_slices_and_keeps_finished_slices_on_failure(tmp_path):
    texts = [f"wort{i} text" for i in range(600)]
    inner = FailingOnThirdCall()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    with pytest.raises(EmbeddingError):
        cached.embed(texts, "passage")
    assert [len(batch) for batch, _ in inner.calls] == [256, 256, 88]

    fresh_inner = FakeEmbedder()
    fresh = CachedEmbedder(fresh_inner, cache_dir=tmp_path)
    out = fresh.embed(texts, "passage")
    assert fresh_inner.calls == [(texts[512:], "passage")]  # die ersten zwei Scheiben kamen aus dem Cache
    assert np.allclose(out, FakeEmbedder().embed(texts, "passage"))
