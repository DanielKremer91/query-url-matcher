import ast
import json
import re
from pathlib import Path

import pytest

from qum import colab
from qum.embeddings.base import EmbeddingError
from qum.ingest import IngestError
from qum.models import MODELS
from tools.build_notebook import CELLS, build

ROOT = Path(__file__).resolve().parents[1]


def _code_cells():
    return [source for kind, source in CELLS if kind == "code"]


def test_committed_notebook_matches_builder():
    committed = json.loads((ROOT / "query_url_matcher.ipynb").read_text(encoding="utf-8"))
    assert committed == build()


def test_code_cells_are_valid_python():
    for source in _code_cells():
        python = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
        ast.parse(python)


def test_every_code_cell_is_a_titled_form():
    for source in _code_cells():
        assert source.startswith("#@title "), source[:40]


def test_model_dropdown_lists_exactly_the_registry_labels():
    source = next(s for s in _code_cells() if "modell = " in s)
    options = json.loads(re.search(r"modell = .*#@param (\[.*\])", source).group(1))
    assert options == [spec.label for spec in MODELS.values()]


def test_notebook_has_nine_steps_in_order():
    titles = [re.match(r"#@title (.*?)( \{|$)", s.splitlines()[0]).group(1) for s in _code_cells()]
    assert [t.split(":")[0].split(" (")[0] for t in titles] == [f"Schritt {n}" for n in range(2, 10)]
    assert any(kind == "markdown" and "Schritt 1" in source for kind, source in CELLS)


def test_no_keys_and_no_retired_model_in_notebook():
    text = (ROOT / "query_url_matcher.ipynb").read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "text-embedding-004" not in text


def test_guard_turns_known_errors_into_notebook_stop(capsys):
    for error in (IngestError("Spalte fehlt"), EmbeddingError("Key fehlt")):
        with pytest.raises(colab.NotebookStop):
            with colab.guard():
                raise error
    assert "❌ Spalte fehlt" in capsys.readouterr().out


def test_guard_lets_other_errors_through():
    with pytest.raises(ZeroDivisionError):
        with colab.guard():
            1 / 0


def test_notebook_stop_hides_traceback():
    assert colab.NotebookStop()._render_traceback_() == []
