"""Texte, die nach dem Umbau des Exports nirgends mehr stehen dürfen."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(ROOT.glob("qum/**/*.py")) + [
    ROOT / "tools/build_notebook.py",
    ROOT / "query_url_matcher.ipynb",
    ROOT / "README.md",
    ROOT / "docs/superpowers/specs/2026-10-05-query-url-matcher-design.md",
]
# "Paare" bleibt als Wort für Query-URL-Paare der Kalibrierung erlaubt, verboten ist das Blatt "Paare"
FORBIDDEN = ["Kannibalisierungs-Risiko", "Schritt 9", '"Paare"', "„Paare", "'Paare'", "Blatt Paare"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_retired_wording(path):
    text = path.read_text(encoding="utf-8")
    for word in FORBIDDEN:
        assert word not in text, word
