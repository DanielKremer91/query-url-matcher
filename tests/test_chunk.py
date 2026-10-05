import pytest

from qum.chunk import chunk_words


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_words("a  b\nc", 5, 1) == ["a b c"]


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_words(words(10), 4, 1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_last_chunk_may_be_shorter():
    chunks = chunk_words(words(6), 4, 1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_words("a b c", 3, 3)
