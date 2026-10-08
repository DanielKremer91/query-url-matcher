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


# --- Abschnitt: unter welcher Zwischenüberschrift ein Chunk überwiegend steht --------------------------------------

from qum.chunk import chunk_sections, is_heading  # noqa: E402


@pytest.mark.parametrize("line, heading", [
    ("Schritt 1: vorbereitende Arbeiten", True),
    ("Lack oder Lasur: Welche Farbe eignet sich zum Streichen von Holzfenstern?", True),
    ("Holzfenster richtig streichen", True),
    ("Tipp: Holzfenster lassen sich einfacher abschleifen, wenn du sie aushängst.", False),
    ("Ein langer Absatz ohne Punkt am Ende mit sehr vielen Wörtern die einfach weiter gehen und weiter und weiter", False),
    ("", False),
])
def test_heading_detection(line, heading):
    assert is_heading(line) is heading


def test_sections_follow_the_chunks_and_take_the_dominant_heading():
    text = "\n".join([
        "Titel der Seite",
        " ".join(f"a{i}" for i in range(6)) + ".",
        "Zweiter Abschnitt",
        " ".join(f"b{i}" for i in range(10)) + ".",
    ])
    chunks = chunk_words(text, 6, 1)
    sections = chunk_sections(text, 6, 1)
    assert len(sections) == len(chunks)
    assert sections[0] == "Titel der Seite"
    assert sections[-1] == "Zweiter Abschnitt"


def test_text_without_line_breaks_has_no_sections():
    assert chunk_sections(" ".join(f"w{i}" for i in range(30)) + ".", 10, 2) == ["", "", "", ""]
