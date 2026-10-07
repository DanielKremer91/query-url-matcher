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
        post_json(client, "https://x.test", {}, {}, sleep=sleeps.append, notify=lambda m: None)
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
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append, notify=lambda m: None) == {}
    assert sleeps == [7, 60, 8, 16]


def test_post_json_non_json_success_raises_embedding_error():
    def handler(request):
        return httpx.Response(200, text="<html>Wartung</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="kein gültiges JSON"):
        post_json(client, "https://x.test", {}, {}, sleep=lambda s: None)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422])
def test_post_json_does_not_retry_client_errors(status):
    sleeps = []
    client = _client([(status, {"error": "bad key"})])
    with pytest.raises(EmbeddingError, match=str(status)):
        post_json(client, "https://x.test", {}, {}, sleep=sleeps.append)
    assert sleeps == []


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_post_json_retries_timeouts_rate_limits_and_server_errors(status):
    sleeps = []
    client = _client([(status, {}), (200, {"ok": True})])
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append, notify=lambda m: None) == {"ok": True}
    assert sleeps == [2]


def test_post_json_notifies_before_waits_of_five_seconds_or_longer():
    sleeps, messages = [], []
    client = _client([(500, {})] * 4 + [(200, {"ok": True})])
    out = post_json(client, "https://x.test", {}, {}, sleep=sleeps.append, notify=messages.append)
    assert out == {"ok": True}
    assert sleeps == [2, 4, 8, 16]
    # die erste Pause (2 s) bleibt still, jede längere wird angekündigt
    assert messages == [
        "⏳ Anbieter antwortet mit HTTP 500, Versuch 3 von 6, nächster in 8 s.",
        "⏳ Anbieter antwortet mit HTTP 500, Versuch 4 von 6, nächster in 16 s.",
    ]


def test_post_json_notifies_about_connection_errors_and_retry_after_waits():
    sleeps, messages, calls = [], [], []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("weg")
        if len(calls) == 2:
            return httpx.Response(429, json={}, headers={"Retry-After": "30"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert post_json(client, "https://x.test", {}, {}, sleep=sleeps.append, notify=messages.append) == {"ok": True}
    assert sleeps == [2, 30]
    assert messages == ["⏳ Anbieter antwortet mit HTTP 429, Versuch 2 von 6, nächster in 30 s."]


def test_post_json_names_connection_errors_in_the_notice():
    messages = []

    def handler(request):
        raise httpx.ConnectError("weg")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="Verbindungsfehler"):
        post_json(client, "https://x.test", {}, {}, attempts=4, sleep=lambda s: None, notify=messages.append)
    assert messages == ["⏳ Verbindungsfehler zum Anbieter, Versuch 3 von 4, nächster in 8 s."]


def test_post_json_does_not_notify_without_a_wait():
    messages = []
    client = _client([(400, {})])
    with pytest.raises(EmbeddingError):
        post_json(client, "https://x.test", {}, {}, sleep=lambda s: None, notify=messages.append)
    assert messages == []


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


def _texts(n):
    return [f"wort{i} text" for i in range(n)]


def test_cache_reports_overall_progress_for_several_slices():
    progress = []
    cached = CachedEmbedder(FakeEmbedder(), notify=lambda done, total: progress.append((done, total)))
    cached.embed(_texts(600), "passage")
    assert progress == [(256, 600), (512, 600), (600, 600)]


def test_cache_progress_counts_only_missing_texts():
    progress = []
    cached = CachedEmbedder(FakeEmbedder(), notify=lambda done, total: progress.append((done, total)))
    cached.embed(_texts(100), "passage")
    cached.embed(_texts(400), "passage")  # 100 kommen aus dem Cache, 300 fehlen
    assert progress == [(256, 300), (300, 300)]


def test_cache_stays_quiet_for_a_single_slice():
    progress = []
    cached = CachedEmbedder(FakeEmbedder(), notify=lambda done, total: progress.append((done, total)))
    cached.embed(_texts(CachedEmbedder.SLICE), "passage")
    assert progress == []


def test_default_progress_line_is_updated_in_place_and_closed(capsys):
    cached = CachedEmbedder(FakeEmbedder())
    cached.embed(_texts(600), "passage")
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.endswith("\n")
    assert out.split("\r")[1:] == [
        "⏳ 256 von 600 Texten eingebettet …",
        "⏳ 512 von 600 Texten eingebettet …",
        "✅ 600 von 600 Texten eingebettet\n",
    ]
    cached.embed(_texts(20), "passage")  # alles im Cache: keine neue Zeile
    assert capsys.readouterr().out == ""


def test_default_progress_line_uses_german_thousands_separator(capsys):
    from qum.embeddings.cache import ProgressLine

    line = ProgressLine()
    line(1280, 20000)
    line.close()
    assert capsys.readouterr().out == "\r⏳ 1.280 von 20.000 Texten eingebettet …\n"


def test_default_progress_line_is_closed_when_a_slice_fails(capsys):
    cached = CachedEmbedder(FailingOnThirdCall())
    with pytest.raises(EmbeddingError):
        cached.embed(_texts(600), "passage")
    out = capsys.readouterr().out
    assert out.endswith("\n")  # die offene Zeile ist beendet, die Fehlermeldung beginnt sauber


class _Clock:
    def __init__(self, step):
        self.now, self.step = -step, step

    def __call__(self):
        self.now += self.step
        return self.now


def _count_saves(cached):
    saves = []
    original = cached._save
    cached._save = lambda: (saves.append(len(cached._store)), original())
    return saves


def test_cache_saves_at_most_every_60_seconds_and_once_at_the_end(tmp_path):
    cached = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path, notify=lambda d, t: None, clock=_Clock(25))
    saves = _count_saves(cached)
    cached.embed(_texts(1000), "passage")  # vier Scheiben; Uhr: Start 0, danach 25, 50, 75, 100
    assert saves == [768, 1000]  # nach der dritten Scheibe (75 s) und am Ende
    fresh_inner = FakeEmbedder()
    CachedEmbedder(fresh_inner, cache_dir=tmp_path).embed(_texts(1000), "passage")
    assert fresh_inner.calls == []


def test_cache_saves_only_once_when_everything_is_fast(tmp_path):
    cached = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path, notify=lambda d, t: None, clock=lambda: 0.0)
    saves = _count_saves(cached)
    cached.embed(_texts(1000), "passage")
    assert saves == [1000]


def test_cache_saves_before_the_error_propagates(tmp_path):
    cached = CachedEmbedder(FailingOnThirdCall(), cache_dir=tmp_path, notify=lambda d, t: None, clock=lambda: 0.0)
    saves = _count_saves(cached)
    with pytest.raises(EmbeddingError):
        cached.embed(_texts(600), "passage")
    assert saves == [512]


def test_cache_saves_on_keyboard_interrupt(tmp_path):
    class Interrupted(FakeEmbedder):
        def embed(self, texts, role):
            if len(self.calls) == 2:
                raise KeyboardInterrupt
            return super().embed(texts, role)

    cached = CachedEmbedder(Interrupted(), cache_dir=tmp_path, notify=lambda d, t: None, clock=lambda: 0.0)
    with pytest.raises(KeyboardInterrupt):
        cached.embed(_texts(600), "passage")
    fresh_inner = FakeEmbedder()
    CachedEmbedder(fresh_inner, cache_dir=tmp_path).embed(_texts(600), "passage")
    assert fresh_inner.calls == [(_texts(600)[512:], "passage")]


def test_cache_does_not_rewrite_the_file_when_nothing_was_missing(tmp_path):
    cached = CachedEmbedder(FakeEmbedder(), cache_dir=tmp_path, notify=lambda d, t: None)
    cached.embed(_texts(10), "passage")
    saves = _count_saves(cached)
    cached.embed(_texts(10), "passage")
    assert saves == []


class NoticeOnSecondCall(FakeEmbedder):
    def embed(self, texts, role):
        from qum.embeddings.base import CONSOLE

        if len(self.calls) == 1:
            CONSOLE.say("⏳ Anbieter antwortet mit HTTP 429, Versuch 1 von 6, nächster in 30 s.")
        return super().embed(texts, role)


def test_retry_notice_starts_on_its_own_line_while_progress_is_shown(capsys):
    CachedEmbedder(NoticeOnSecondCall()).embed(_texts(600), "passage")
    out = capsys.readouterr().out
    assert out == (
        "\r⏳ 256 von 600 Texten eingebettet …\n"
        "⏳ Anbieter antwortet mit HTTP 429, Versuch 1 von 6, nächster in 30 s.\n"
        "\r⏳ 512 von 600 Texten eingebettet …"
        "\r✅ 600 von 600 Texten eingebettet\n"
    )


def test_post_json_notice_breaks_an_open_progress_line_by_default(capsys):
    from qum.embeddings.cache import ProgressLine

    line = ProgressLine()
    line(1000, 2000)
    client = _client([(500, {})] * 3 + [(200, {"ok": True})])
    assert post_json(client, "https://x.test", {}, {}, sleep=lambda s: None) == {"ok": True}
    line(2000, 2000)
    line.close()
    assert capsys.readouterr().out == (
        "\r⏳ 1.000 von 2.000 Texten eingebettet …\n"
        "⏳ Anbieter antwortet mit HTTP 500, Versuch 3 von 6, nächster in 8 s.\n"
        "\r✅ 2.000 von 2.000 Texten eingebettet\n"
    )


def test_notice_without_open_progress_line_adds_no_blank_line(capsys):
    from qum.embeddings.base import CONSOLE

    CONSOLE.say("Hinweis")
    assert capsys.readouterr().out == "Hinweis\n"


def test_progress_line_with_label(capsys):
    from qum.embeddings.cache import ProgressLine

    line = ProgressLine("Queries")
    line(394, 394)
    line.close()
    assert capsys.readouterr().out == "\r✅ Queries: 394 von 394 eingebettet\n"


def test_labelled_embed_reports_even_small_batches(capsys):
    CachedEmbedder(FakeEmbedder()).embed(["a b", "c d", "e f"], "query", label="Queries")
    assert capsys.readouterr().out == "\r✅ Queries: 3 von 3 eingebettet\n"


def test_labelled_embed_is_quiet_when_everything_is_cached(capsys):
    cached = CachedEmbedder(FakeEmbedder())
    cached.embed(["a b"], "query", label="Queries")
    capsys.readouterr()
    cached.embed(["a b"], "query", label="Queries")
    assert capsys.readouterr().out == ""
