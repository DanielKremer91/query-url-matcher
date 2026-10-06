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
CHOICES = ["Erst Vorschläge ansehen", "Aus Rankings kalibriert", "Mittlerer bester Score (nicht kalibriert)", "Eigener Wert"]
SHOW, CALIBRATED, MEDIAN, OWN = CHOICES
NO_VERDICTS_YET = (
    '❌ Schritt 7 hat noch keine Urteile gebildet. Wähle bei "schwelle_bestimmen" eine der Optionen '
    "und führe Schritt 7 erneut aus."
)
KEPT = "Bis dahin gelten die Urteile aus dem letzten Lauf von Schritt 7."


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
    nb.run(7, schwelle_bestimmen=MEDIAN)
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
    assert "Aus Rankings kalibriert: nicht verfügbar. Es sind keine Rankings geladen (Schritt 5 oder 6)." in capsys.readouterr().out
    nb.run(7, schwelle_bestimmen=MEDIAN)
    nb.run(8)
    out = capsys.readouterr().out
    assert "per Konstruktion darunter" in out
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
    nb.run(7, schwelle_bestimmen=MEDIAN)
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
    nb.run(7, schwelle_bestimmen=MEDIAN)
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
    nb.run(7, schwelle_bestimmen=MEDIAN)
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
    with pytest.raises(colab.NotebookStop):
        nb.run(7, rankt_gut_bis_position=0)
    assert "mindestens 1" in capsys.readouterr().out


def test_smoke_proposals_explain_why_calibration_is_missing(nb, capsys):
    _load(nb)
    nb.run(5, "rankings.csv")
    nb.run(7)
    out = capsys.readouterr().out
    assert "Aus Rankings kalibriert: nicht verfügbar. Nur 4 Ranking-Paare bis Position 5, für eine Kalibrierung sind 20 nötig." in out
    assert "per Konstruktion darunter" in out


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
    nb.run(7, schwelle_bestimmen=MEDIAN)
    assert "neue Seiten aus den Content-Lücken (Schwelle nicht kalibriert)" in capsys.readouterr().out
    nb.run(8)
    assert export.CAVEAT_THRESHOLD["median"] in _readme_notes(tmp_path / "query_url_matcher.xlsx")


def _zip_header(path):
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return archive.read("entscheidung.csv").decode("utf-8-sig").splitlines()[0]


def test_smoke_step_8_csv_zip_uses_semicolon_by_default_and_comma_on_request(nb, tmp_path):
    _load(nb)
    nb.run(7, schwelle_bestimmen=MEDIAN)
    nb.run(8, zusaetzlich_csv_zip=True)
    assert ";" in _zip_header(tmp_path / "query_url_matcher_csv.zip")
    nb.run(8, zusaetzlich_csv_zip=True, csv_trennzeichen="Komma")
    header = _zip_header(tmp_path / "query_url_matcher_csv.zip")
    assert "," in header and ";" not in header


def test_step_8_separator_dropdown_defaults_to_semicolon():
    source = next(s for s in _code_cells() if s.startswith("#@title Schritt 8"))
    match = re.search(r'^csv_trennzeichen = (".*?") #@param (\[.*\])$', source, flags=re.M)
    assert json.loads(match.group(1)) == "Semikolon (für deutsches Excel)"
    assert json.loads(match.group(2)) == ["Semikolon (für deutsches Excel)", "Komma"]


def test_smoke_threshold_typed_by_hand_is_named_in_export(nb, tmp_path, capsys):
    from qum import export

    _load(nb)
    nb.run(7, schwelle_bestimmen=OWN, eigene_schwelle=0.5)
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
    assert "ermöglicht eine Kalibrierung der Schwelle" in step5 and "ab 20 gut rankenden Paaren" in step5
    intro = next(source for kind, source in CELLS if kind == "markdown")
    assert "ermöglichen eine Kalibrierung der Schwelle" in intro and "kalibrieren die Schwelle" not in intro


def test_example_files_use_reserved_example_domains():
    from urllib.parse import urlsplit

    import pandas as pd

    for name, column in (("frog_export.csv", "Address"), ("rankings.csv", "URL"), ("serps.csv", "URL")):
        hosts = pd.read_csv(EXAMPLES / name, sep=";")[column].map(lambda u: urlsplit(u).netloc)
        assert hosts.str.endswith(".example").all(), name
    assert "www.tierbedarf.example" in (EXAMPLES / "frog_export.csv").read_text(encoding="utf-8")


class _FakeCuda:
    def __init__(self, available):
        self.available, self.calls = available, 0

    def is_available(self):
        self.calls += 1
        return self.available


def _run_step_2(nb, monkeypatch, label, gpu):
    import sys
    import types

    import qum.embeddings
    from qum.embeddings.cache import CachedEmbedder
    from tests.conftest import FakeEmbedder

    torch = types.ModuleType("torch")
    torch.cuda = _FakeCuda(gpu)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(qum.embeddings, "make_embedder", lambda spec, **kw: CachedEmbedder(FakeEmbedder(spec)))
    monkeypatch.setattr(colab, "secret", lambda name: "test-key")
    nb.run(2, modell=label)
    return torch.cuda


def test_smoke_step_2_warns_without_gpu_for_local_models(nb, monkeypatch, capsys):
    cuda = _run_step_2(nb, monkeypatch, MODELS["e5-large"].label, gpu=False)
    out = capsys.readouterr().out
    assert cuda.calls == 1
    assert "⚠️ Colab hat keine GPU zugeteilt" in out and "T4 GPU" in out
    assert "✅ Schritt 2 fertig" in out


def test_smoke_step_2_stays_quiet_with_gpu(nb, monkeypatch, capsys):
    _run_step_2(nb, monkeypatch, MODELS["e5-large"].label, gpu=True)
    assert "keine GPU zugeteilt" not in capsys.readouterr().out


def test_smoke_step_2_does_not_check_gpu_for_api_models(nb, monkeypatch, capsys):
    cuda = _run_step_2(nb, monkeypatch, MODELS["openai"].label, gpu=False)
    assert cuda.calls == 0
    assert "keine GPU zugeteilt" not in capsys.readouterr().out


def test_intro_says_the_gpu_is_requested_automatically():
    intro = next(source for kind, source in CELLS if kind == "markdown")
    assert "automatisch eine kostenlose T4-GPU" in intro


def test_smoke_rejected_rerun_of_step_4_keeps_previous_settings(nb, capsys):
    nb.run(3, "queries.csv", "frog_export.csv")
    nb.run(4)
    before = {name: nb.ns[name] for name in ("size", "overlap", "basis", "result")}
    with pytest.raises(colab.NotebookStop):
        nb.run(4, chunk_groesse=10, chunk_overlap=10, bewertungsgrundlage="Kombi")
    assert {name: nb.ns[name] for name in before} == before


def test_smoke_default_overlap_not_below_chunk_size_explains_itself(nb, capsys):
    nb.run(3, "queries.csv", "frog_export.csv")
    capsys.readouterr()
    with pytest.raises(colab.NotebookStop):
        nb.run(4, chunk_groesse=1, chunk_overlap=0)  # Standard-Overlap des Test-Modells ist 1
    out = capsys.readouterr().out
    assert "Standard-Overlap des Modells (1)" in out and "Trage 0 ein" not in out


def test_smoke_serps_with_few_urls_per_keyword_warn(nb, capsys):
    _load(nb)
    own_only = (
        "Keyword;URL;Position;Type\n"
        "getreidefreies hundefutter;https://www.tierbedarf.example/hundefutter/getreidefreies-hundefutter;5;Organic\n"
    )
    nb.uploads.append(("serps.csv", own_only.encode()))
    nb.run(6)
    assert "⚠️ Im Schnitt nur 1 URLs je Keyword." in capsys.readouterr().out
    nb.run(6, "serps.csv")
    assert "Im Schnitt nur" not in capsys.readouterr().out


def test_smoke_step_9_accepts_an_earlier_pairs_export(nb, capsys):
    _load(nb)
    nb.run(7, schwelle_bestimmen=MEDIAN)
    lines = (EXAMPLES / "rankings.csv").read_text(encoding="utf-8").splitlines()
    earlier = "\n".join([lines[0] + ";Hinweis;Score Chunk"] + [line + ";alt;0.1" for line in lines[1:]]) + "\n"
    nb.uploads.append(("paare.csv", earlier.encode()))
    nb.run(9)
    out = capsys.readouterr().out
    assert "ℹ️ Die Datei enthält schon Ergebnisspalten (Score Chunk, Hinweis)" in out
    assert "✅ Schritt 9 fertig" in out
    assert list(nb.ns["pairs"].columns).count("Hinweis") == 1


def test_step_7_margin_defaults_to_one_hundredth():
    source = next(s for s in _code_cells() if s.startswith("#@title Schritt 7"))
    assert re.search(r"^abstand_fast_gleich = 0\.01 #@param", source, flags=re.M)


def test_smoke_step_7_margin_reaches_the_verdicts(nb):
    from qum import labels as L

    _load(nb)
    # die Kratzbaum-Seite rankt, erreicht die Schwelle, liegt aber 0,04 hinter der besten Seite
    weak = "Keyword;URL;Position\ngetreidefreies trockenfutter hund;https://www.tierbedarf.example/katzenzubehoer/kratzbaum;3\n"
    nb.uploads.append(("rankings.csv", weak.encode()))
    nb.run(5)
    nb.run(7, schwelle_bestimmen=OWN, eigene_schwelle=0.2)
    assert nb.ns["decisions"][L.C_VERDICT].tolist().count(L.V_RISK) == 1
    nb.run(7, schwelle_bestimmen=OWN, eigene_schwelle=0.2, abstand_fast_gleich=0.05)
    assert L.V_RISK not in nb.ns["decisions"][L.C_VERDICT].tolist()


def test_step_7_threshold_dropdown_offers_the_four_choices():
    from qum import labels as L

    source = next(s for s in _code_cells() if s.startswith("#@title Schritt 7"))
    match = re.search(r'^schwelle_bestimmen = (".*?") #@param (\[.*\])$', source, flags=re.M)
    assert json.loads(match.group(1)) == SHOW
    assert json.loads(match.group(2)) == CHOICES == list(L.THRESHOLD_CHOICE)
    assert [L.THRESHOLD_CHOICE[c] for c in CHOICES] == [None, "rankings", "median", "manuell"]


def _many_rankings():
    """24 Paare bis Position 2: genug für die Kalibrierung."""
    queries = (EXAMPLES / "queries.csv").read_text(encoding="utf-8").splitlines()[1:]
    urls = [line.split(";")[0] for line in (EXAMPLES / "frog_export.csv").read_text(encoding="utf-8").splitlines()[1:3]]
    lines = ["Keyword;URL;Position"] + [f"{q};{u};{n}" for q in queries for n, u in enumerate(urls, start=1)]
    return ("rankings.csv", ("\n".join(lines) + "\n").encode())


def test_smoke_step_7_default_shows_proposals_and_builds_no_verdicts(nb, capsys):
    shown = []
    nb.ns["display"] = shown.append
    _load(nb)
    nb.uploads.append(_many_rankings())
    nb.run(5)
    nb.run(7, schwelle_bestimmen=MEDIAN)
    shown.clear()
    capsys.readouterr()
    nb.run(7)
    out = capsys.readouterr().out
    assert "Die Schwelle ist die Cosinus-Ähnlichkeit zwischen Query und Seite, ab der eine Seite als passend gilt." in out
    assert re.search(r"Aus Rankings kalibriert: 0\.\d{4}\. Diesen Score erreichen 75 % der 24 Paare", out)
    assert re.search(r"Mittlerer bester Score \(nicht kalibriert\): 0\.\d{4}\.", out)
    assert "per Konstruktion darunter" in out
    assert "führe die Zelle erneut aus" in out
    assert "✅ Schritt 7 fertig" not in out
    assert len(shown) == 2  # Beispielpaare um beide Vorschläge
    for name in ("decisions", "cannibal", "settings", "threshold_source"):
        assert nb.ns[name] is None, name
    with pytest.raises(colab.NotebookStop):
        nb.run(8)
    assert NO_VERDICTS_YET in capsys.readouterr().out


@pytest.mark.parametrize(
    "choice, source, form",
    [(CALIBRATED, "rankings", {}), (MEDIAN, "median", {}), (OWN, "manuell", {"eigene_schwelle": 0.42})],
)
def test_smoke_step_7_each_choice_builds_verdicts(nb, capsys, choice, source, form):
    from qum.threshold import calibrated_threshold, median_threshold

    _load(nb)
    nb.uploads.append(_many_rankings())
    nb.run(5)
    nb.run(7, schwelle_bestimmen=choice, **form)
    out = capsys.readouterr().out
    expected = {
        "rankings": lambda: calibrated_threshold(nb.ns["result"], nb.ns["lead"], nb.ns["rankings"]).value,
        "median": lambda: median_threshold(nb.ns["result"], nb.ns["lead"]).value,
        "manuell": lambda: 0.42,
    }[source]()
    assert nb.ns["decisions"] is not None and nb.ns["cannibal"] is not None
    assert nb.ns["threshold_source"] == source
    assert nb.ns["settings"]["Schwelle aus"] == choice
    assert nb.ns["settings"]["Schwelle"] == expected
    assert f"Verwendete Schwelle {expected:.4f}" in out
    assert "✅ Schritt 7 fertig" in out


def test_smoke_calibration_with_too_few_pairs_stops(nb, capsys):
    _load(nb)
    with pytest.raises(colab.NotebookStop):
        nb.run(7, schwelle_bestimmen=CALIBRATED)
    assert "Es sind keine Rankings geladen (Schritt 5 oder 6)." in capsys.readouterr().out
    nb.run(5, "rankings.csv")
    with pytest.raises(colab.NotebookStop):
        nb.run(7, schwelle_bestimmen=CALIBRATED)
    out = capsys.readouterr().out
    assert "❌ Die Kalibrierung aus Rankings ist nicht verfügbar" in out
    assert "Nur 4 Ranking-Paare bis Position 5, für eine Kalibrierung sind 20 nötig." in out
    assert "Wähle bei schwelle_bestimmen eine andere Option" in out
    assert nb.ns["decisions"] is None


@pytest.mark.parametrize("value", [0, -0.2, 1.5])
def test_smoke_invalid_own_threshold_stops(nb, capsys, value):
    _load(nb)
    with pytest.raises(colab.NotebookStop):
        nb.run(7, schwelle_bestimmen=OWN, eigene_schwelle=value)
    assert "❌ Die eigene Schwelle muss größer als 0 und höchstens 1 sein." in capsys.readouterr().out
    assert nb.ns.get("decisions") is None


def test_smoke_own_threshold_of_one_is_accepted(nb):
    _load(nb)
    nb.run(7, schwelle_bestimmen=OWN, eigene_schwelle=1.0)
    assert nb.ns["settings"]["Schwelle"] == 1.0


def test_install_line_pins_the_package_version():
    import tomllib

    import qum

    source = next(s for s in _code_cells() if s.startswith("#@title Schritt 2"))
    match = re.search(
        r'^!pip install -q "qum\[local\] @ git\+https://github\.com/DanielKremer91/query-url-matcher@v([^"]+)"$',
        source,
        flags=re.M,
    )
    assert match and match.group(1) == qum.__version__
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == qum.__version__
    committed = (ROOT / "query_url_matcher.ipynb").read_text(encoding="utf-8")
    assert f"query-url-matcher@v{qum.__version__}" in committed



@pytest.mark.parametrize("value", [-0.01, 0.1, 2])
def test_smoke_margin_outside_its_range_stops(nb, capsys, value):
    _load(nb)
    with pytest.raises(colab.NotebookStop):
        nb.run(7, schwelle_bestimmen=MEDIAN, abstand_fast_gleich=value)
    out = capsys.readouterr().out
    assert "Differenz von Cosinus-Scores (zum Beispiel 0.01), kein Prozentwert" in out
    assert KEPT not in out


def test_smoke_margin_zero_is_accepted(nb):
    _load(nb)
    nb.run(7, schwelle_bestimmen=MEDIAN, abstand_fast_gleich=0)
    assert nb.ns["settings"]["Abstand fast gleich"] == 0


def _closing_line(out):
    return next(line for line in out.splitlines() if line.startswith("ℹ️ Noch keine Urteile"))


def test_smoke_proposals_closing_line_lists_only_available_options(nb, capsys):
    _load(nb)
    nb.run(7)
    line = _closing_line(capsys.readouterr().out)
    assert "Aus Rankings kalibriert" not in line
    assert '"Mittlerer bester Score (nicht kalibriert)"' in line and '"Eigener Wert"' in line
    nb.uploads.append(_many_rankings())
    nb.run(5)
    nb.run(7)
    assert '"Aus Rankings kalibriert"' in _closing_line(capsys.readouterr().out)


def test_smoke_steps_8_and_9_after_proposals_only_say_so(nb, capsys):
    _load(nb)
    nb.run(7)
    capsys.readouterr()
    for step in (8, 9):
        with pytest.raises(colab.NotebookStop):
            nb.run(step)
        out = capsys.readouterr().out
        assert NO_VERDICTS_YET in out and "Bitte zuerst Schritt 7" not in out
    nb.run(4)  # ein früherer Schritt setzt den Zustand zurück
    with pytest.raises(colab.NotebookStop):
        nb.run(8)
    assert "❌ Bitte zuerst Schritt 7 (Schwelle und Urteile) ausführen." in capsys.readouterr().out
    nb.run(7, schwelle_bestimmen=MEDIAN)
    nb.run(8)
    assert "✅ Schritt 8 fertig" in capsys.readouterr().out


@pytest.mark.parametrize(
    "form",
    [
        {"schwelle_bestimmen": OWN, "eigene_schwelle": 0},
        {"schwelle_bestimmen": CALIBRATED},
        {"schwelle_bestimmen": MEDIAN, "abstand_fast_gleich": 0.5},
        {"schwelle_bestimmen": MEDIAN, "rankt_gut_bis_position": 0},
    ],
)
def test_smoke_rejected_step_7_says_the_last_verdicts_still_apply(nb, capsys, form):
    _load(nb)
    with pytest.raises(colab.NotebookStop):
        nb.run(7, **form)
    assert KEPT not in capsys.readouterr().out
    nb.run(7, schwelle_bestimmen=MEDIAN, eigene_schwelle=0, abstand_fast_gleich=0.01, rankt_gut_bis_position=10)
    before = nb.ns["decisions"]
    capsys.readouterr()
    with pytest.raises(colab.NotebookStop):
        nb.run(7, **form)
    out = capsys.readouterr().out
    assert out.startswith("❌ ") and out.rstrip().endswith(KEPT)
    assert nb.ns["decisions"] is before


def test_notebook_requests_a_gpu_runtime():
    meta = build()["metadata"]
    assert meta["accelerator"] == "GPU"
    assert meta["colab"]["gpuType"] == "T4"


def test_smoke_step_2_hides_hugging_face_token_notice(nb, monkeypatch, capsys):
    import logging

    http_logger = logging.getLogger("huggingface_hub.utils._http")
    monkeypatch.setattr(http_logger, "level", logging.NOTSET)
    _run_step_2(nb, monkeypatch, MODELS["e5-large"].label, gpu=True)
    http_logger.warning("Warning: You are sending unauthenticated requests to the HF Hub.")
    captured = capsys.readouterr()
    assert "unauthenticated" not in captured.out + captured.err
    assert http_logger.getEffectiveLevel() >= logging.ERROR


def test_upload_steps_explain_where_the_upload_button_appears():
    for source in _code_cells():
        if "colab.upload(" in source:
            first_hint = next(line for line in source.splitlines() if line.startswith("#@markdown"))
            assert "Play-Symbol" in first_hint and "Dateien auswählen" in first_hint, source.splitlines()[0]
