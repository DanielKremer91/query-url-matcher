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


def test_require_stops_with_german_hint_for_missing_or_none(capsys):
    colab.require({"result": 1}, 4, "result")
    for namespace in ({}, {"result": None}):
        with pytest.raises(colab.NotebookStop):
            colab.require(namespace, 4, "result")
    assert "❌ Bitte zuerst Schritt 4 (Matching) ausführen." in capsys.readouterr().out


def test_invalidate_sets_names_to_none():
    namespace = {"decisions": 1, "top": 2}
    colab.invalidate(namespace, "decisions", "settings")
    assert namespace == {"decisions": None, "top": 2, "settings": None}


def _fake_userdata(monkeypatch, get):
    import sys
    import types

    class SecretNotFoundError(Exception):
        pass

    class NotebookAccessError(Exception):
        pass

    userdata = types.ModuleType("google.colab.userdata")
    userdata.SecretNotFoundError = SecretNotFoundError
    userdata.NotebookAccessError = NotebookAccessError
    userdata.get = get(SecretNotFoundError, NotebookAccessError)
    package = types.ModuleType("google.colab")
    package.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", package)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)


def test_secret_returns_value_or_none_when_missing(monkeypatch):
    def get(not_found, no_access):
        def inner(name):
            if name == "SET":
                return "key"
            raise not_found(name)

        return inner

    _fake_userdata(monkeypatch, get)
    assert colab.secret("SET") == "key"
    assert colab.secret("OTHER") is None


def test_secret_without_notebook_access_explains_the_switch(monkeypatch, capsys):
    def get(not_found, no_access):
        def inner(name):
            raise no_access(name)

        return inner

    _fake_userdata(monkeypatch, get)
    with pytest.raises(colab.NotebookStop):
        colab.secret("OPENAI_API_KEY")
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out and "Notebook-Zugriff" in out


@pytest.mark.parametrize("name, data", [("kaputt.xlsx", b"das ist keine Excel-Datei"), ("kaputt.csv", b'a,b\n"1,2\n3')])
def test_read_table_turns_parse_errors_into_german_hint(name, data):
    with pytest.raises(IngestError, match=f"'{name}' konnte nicht gelesen werden"):
        colab.read_table(data, name)


def test_read_table_passes_ingest_errors_through():
    with pytest.raises(IngestError, match="nicht unterstützt"):
        colab.read_table(b"x", "bild.png")


# --- Rauchtest: die Zellen der Schritte 3 bis 9 laufen in einem gemeinsamen Namensraum ---

EXAMPLES = ROOT / "examples"


def _cell_code(step: int) -> str:
    source = next(s for s in _code_cells() if s.startswith(f"#@title Schritt {step}"))
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))


class Notebook:
    def __init__(self, monkeypatch, tmp_path):
        from qum.embeddings.cache import CachedEmbedder
        from tests.conftest import FAKE_SPEC, FakeEmbedder

        self.uploads, self.downloads = [], []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(colab, "upload", lambda title: self.uploads.pop(0))
        monkeypatch.setattr(colab, "download", lambda path: self.downloads.append(str(path)))
        monkeypatch.setattr(colab, "secret", lambda name: None)
        self.ns = {"display": lambda frame: None, "spec": FAKE_SPEC, "embedder": CachedEmbedder(FakeEmbedder())}

    def run(self, step, *files, **form):
        self.uploads += [(name, (EXAMPLES / name).read_bytes()) for name in files]
        code = _cell_code(step)
        for name, value in form.items():  # Formularwerte ersetzen die Vorgaben der #@param-Zeilen
            code, n = re.subn(rf"^{name} = .*?(?= #@param)", f"{name} = {value!r}", code, flags=re.M)
            assert n == 1, name
        exec(compile(code, f"<schritt {step}>", "exec"), self.ns)


@pytest.fixture
def nb(monkeypatch, tmp_path):
    return Notebook(monkeypatch, tmp_path)


def _load(nb):
    nb.run(3, "queries.csv", "frog_export.csv")
    nb.run(4)


def test_smoke_full_flow(nb, tmp_path, capsys):
    _load(nb)
    nb.run(5, "rankings.csv")
    nb.run(6, "serps.csv")
    nb.run(7)
    nb.run(8)
    nb.run(9, "rankings.csv")
    out = capsys.readouterr().out
    assert "✅ Schritt 9 fertig" in out
    assert nb.downloads == ["query_url_matcher.xlsx", "query_url_matcher_mit_paaren.xlsx"]
    assert (tmp_path / "query_url_matcher.xlsx").exists() and (tmp_path / "query_url_matcher_mit_paaren.xlsx").exists()
    settings = nb.ns["settings"]
    assert settings["Eigene Rankings"] == "ja, aus Datei"
    for key in ("Kalibrierung bis Position", "SERP-Überschneidung (%)", "Cluster-Dichte (%)", "Treffer je Query (Top-N)"):
        assert key in settings


def test_smoke_minimal_flow_without_rankings_and_serps(nb, capsys):
    _load(nb)
    nb.run(7)
    nb.run(8)
    out = capsys.readouterr().out
    assert "Ohne Rankings gibt es keinen Maßstab" in out
    assert nb.downloads == ["query_url_matcher.xlsx"]
    assert nb.ns["settings"]["Eigene Rankings"] == "nein"


def test_smoke_serps_alone_derive_own_rankings_and_refresh(nb, capsys):
    _load(nb)
    own = "getreidefreies hundefutter;https://www.tierbedarf.example/hundefutter/getreidefreies-hundefutter;5;Organic\n"
    serps_with_own = (EXAMPLES / "serps.csv").read_bytes() + own.encode()
    nb.uploads.append(("serps.csv", serps_with_own))
    nb.run(6)
    assert nb.ns["rankings_source"] == "SERPs" and len(nb.ns["rankings"]) == 1
    nb.uploads.append(("serps.csv", (EXAMPLES / "serps.csv").read_bytes()))
    nb.run(6)  # neue SERPs ohne eigene URL: die alten abgeleiteten Rankings dürfen nicht bleiben
    assert nb.ns["rankings"] is None and nb.ns["rankings_source"] is None
    nb.uploads.append(("serps.csv", serps_with_own))
    nb.run(6)
    assert nb.ns["rankings_source"] == "SERPs"
    nb.run(5, "rankings.csv")
    nb.run(6, "serps.csv")
    assert nb.ns["rankings_source"] == "Datei"


def test_smoke_step_8_before_step_7_points_to_step_7(nb, capsys):
    _load(nb)
    with pytest.raises(colab.NotebookStop):
        nb.run(8)
    assert "❌ Bitte zuerst Schritt 7 (Schwelle und Urteile) ausführen." in capsys.readouterr().out


def test_smoke_step_9_checks_prerequisites_before_asking_for_upload(nb, capsys):
    _load(nb)
    nb.uploads.append(("rankings.csv", b"x"))
    with pytest.raises(colab.NotebookStop):
        nb.run(9)
    assert len(nb.uploads) == 1  # nichts wurde abgefragt
    assert "Bitte zuerst Schritt 7" in capsys.readouterr().out


def test_smoke_step_before_model_or_files_points_back(nb, capsys):
    nb.ns["embedder"] = None
    with pytest.raises(colab.NotebookStop):
        nb.run(3, "queries.csv", "frog_export.csv")
    assert "Bitte zuerst Schritt 2" in capsys.readouterr().out
    nb2 = nb
    nb2.ns["embedder"] = object()
    with pytest.raises(colab.NotebookStop):
        nb2.run(4)
    assert "Bitte zuerst Schritt 3" in capsys.readouterr().out


def test_smoke_rerunning_step_4_invalidates_verdicts(nb, capsys):
    _load(nb)
    nb.run(7)
    assert nb.ns["decisions"] is not None
    nb.run(4, top_n=3)
    assert nb.ns["decisions"] is None and nb.ns["settings"] is None
    with pytest.raises(colab.NotebookStop):
        nb.run(8)
    assert "Bitte zuerst Schritt 7" in capsys.readouterr().out


def test_smoke_new_files_reset_rankings_and_serps_with_note(nb, capsys):
    _load(nb)
    nb.run(5, "rankings.csv")
    nb.run(6, "serps.csv")
    nb.run(7)
    capsys.readouterr()
    nb.run(3, "queries.csv", "frog_export.csv")
    out = capsys.readouterr().out
    assert "ℹ️ Rankings und SERPs wurden zurückgesetzt. Führe Schritt 5 und 6 bei Bedarf erneut aus." in out
    assert nb.ns["rankings"] is None and nb.ns["serps"] is None and nb.ns["result"] is None
    with pytest.raises(colab.NotebookStop):
        nb.run(7)


def test_smoke_ranking_file_of_another_project_keeps_plain_matching(nb, capsys):
    _load(nb)
    foreign = "Keyword;URL;Position\nganz anderes thema;https://fremd.example/x;3\n"
    nb.uploads.append(("fremd.csv", foreign.encode()))
    nb.run(5)
    out = capsys.readouterr().out
    assert "⚠️ Keine der Ranking-Zeilen passt zu deinen Queries. Prüfe, ob die Datei zu diesem Projekt gehört." in out
    assert nb.ns["rankings"] is None
    nb.run(7)
    assert nb.ns["settings"]["Eigene Rankings"] == "nein"


def test_smoke_ranking_urls_outside_the_export_are_flagged(nb, capsys):
    _load(nb)
    known = "Keyword;URL;Position\ngetreidefreies hundefutter;https://fremd.example/x;3\n"
    nb.uploads.append(("fremd.csv", known.encode()))
    nb.run(5)
    assert "Keine der Ranking-URLs steht im Frog-Export" in capsys.readouterr().out
    assert nb.ns["rankings"] is None


def test_smoke_serps_of_another_project_are_dropped(nb, capsys):
    _load(nb)
    nb.uploads.append(("fremd.csv", b"Keyword;URL;Position;Type\nganz anderes thema;https://x.example/a;1;Organic\n"))
    nb.run(6)
    assert "Kein SERP-Keyword passt zu deinen Queries" in capsys.readouterr().out
    assert nb.ns["serps"] is None


def test_smoke_numeric_inputs_are_validated(nb, capsys):
    nb.run(3, "queries.csv", "frog_export.csv")
    for form, hint in (
        ({"chunk_groesse": -1}, "nicht negativ"),
        ({"chunk_groesse": 10, "chunk_overlap": 10}, "Overlap"),
        ({"chunk_groesse": 0, "chunk_overlap": 0, "top_n": 0}, "mindestens 1"),
    ):
        with pytest.raises(colab.NotebookStop):
            nb.run(4, **form)
        assert hint in capsys.readouterr().out
    nb.run(4, chunk_groesse=0, chunk_overlap=0, top_n=5)
    for form, hint in (({"schwelle": 1.5}, "zwischen 0 und 1"), ({"schwelle": 0, "rankt_gut_bis_position": 0}, "mindestens 1")):
        with pytest.raises(colab.NotebookStop):
            nb.run(7, **form)
        assert hint in capsys.readouterr().out


def test_smoke_median_proposal_warning_with_too_few_pairs(nb, capsys):
    _load(nb)
    nb.run(5, "rankings.csv")
    nb.run(7)
    out = capsys.readouterr().out
    assert "Ranking-Paare bis Position 5, für eine Kalibrierung sind 20 nötig." in out
    assert "Ohne Rankings gibt es keinen Maßstab" in out


def test_smoke_header_only_ranking_file_stops_with_hint(nb, capsys):
    _load(nb)
    nb.uploads.append(("leer.csv", b"Keyword;URL;Position\n"))
    with pytest.raises(colab.NotebookStop):
        nb.run(5)
    assert "❌ Die Datei enthält keine verwertbaren Zeilen" in capsys.readouterr().out
    assert nb.ns.get("rankings") is None


def _readme_notes(path):
    import pandas as pd

    from qum import labels as L

    readme = pd.read_excel(path, sheet_name="Lesehilfe")
    return readme[readme[L.R_AREA] == "Hinweis"][L.R_TEXT].tolist()


def test_smoke_uncalibrated_threshold_is_flagged_in_counts_and_export(nb, tmp_path, capsys):
    from qum import export

    _load(nb)
    nb.run(7)
    assert "neue Seiten aus den Content-Lücken (Schwelle nicht kalibriert)" in capsys.readouterr().out
    nb.run(8)
    assert export.CAVEAT_THRESHOLD["median"] in _readme_notes(tmp_path / "query_url_matcher.xlsx")


def test_smoke_threshold_typed_by_hand_is_named_in_export(nb, tmp_path, capsys):
    from qum import export

    _load(nb)
    nb.run(7, schwelle=0.5)
    out = capsys.readouterr().out
    assert "neue Seiten aus den Content-Lücken" in out and "(Schwelle nicht kalibriert)" not in out
    nb.run(8)
    notes = _readme_notes(tmp_path / "query_url_matcher.xlsx")
    assert export.CAVEAT_THRESHOLD["manuell"] in notes and export.CAVEAT_THRESHOLD["median"] not in notes


def test_notebook_does_not_claim_rankings_unlock_cannibalisation_or_gaps():
    text = "\n".join(source for _, source in CELLS)
    assert "Schaltet die Urteile zu Kannibalisierung und Content-Lücke frei" not in text
    assert "Mit eigenen Rankings kommen die Urteile zu Kannibalisierung und Content-Lücke dazu" not in text
    step5 = next(s for s in _code_cells() if s.startswith("#@title Schritt 5"))
    assert "kalibriert" in step5


def test_example_files_use_reserved_example_domains():
    from urllib.parse import urlsplit

    import pandas as pd

    for name, column in (("frog_export.csv", "Address"), ("rankings.csv", "URL"), ("serps.csv", "URL")):
        hosts = pd.read_csv(EXAMPLES / name, sep=";")[column].map(lambda u: urlsplit(u).netloc)
        assert hosts.str.endswith(".example").all(), name
    assert "www.tierbedarf.example" in (EXAMPLES / "frog_export.csv").read_text(encoding="utf-8")
