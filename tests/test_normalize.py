from qum.normalize import host_of, normalize_query, normalize_url


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  Hundefutter   Getreidefrei ") == "hundefutter getreidefrei"


def test_normalize_url_strips_fragment_trailing_slash_and_tracking():
    url = "HTTPS://Www.Example.de/Ratgeber/?utm_source=x&gclid=1&farbe=rot#abschnitt"
    assert normalize_url(url) == "https://example.de/Ratgeber?farbe=rot"


def test_normalize_url_root_with_and_without_slash_are_equal():
    assert normalize_url("https://example.de/") == normalize_url("https://example.de")


def test_normalize_url_keeps_path_case():
    assert normalize_url("https://example.de/A") != normalize_url("https://example.de/a")


def test_host_of_ignores_www_and_port():
    assert host_of("https://WWW.Example.de/x") == "example.de"
    assert host_of("https://example.de:443/x") == "example.de"
    assert host_of("www.example.de/x") == "example.de"


def test_same_page_with_and_without_www_slash_or_scheme_is_equal():
    # treppenverkleidung: Ranking mit www und ohne Schrägstrich, Frog-Export ohne www mit Schrägstrich
    frog = "https://toom.de/selbermachen/bauen-renovieren/treppen/treppen-verkleiden/"
    for ranking in (
        "https://www.toom.de/selbermachen/bauen-renovieren/treppen/treppen-verkleiden",
        "http://www.toom.de/selbermachen/bauen-renovieren/treppen/treppen-verkleiden/",
        "www.toom.de/selbermachen/bauen-renovieren/treppen/treppen-verkleiden",
        "toom.de/selbermachen/bauen-renovieren/treppen/treppen-verkleiden",
        "https://TOOM.de:443/selbermachen/bauen-renovieren/treppen/treppen-verkleiden",
    ):
        assert normalize_url(ranking) == normalize_url(frog), ranking


def test_other_subdomains_stay_different():
    assert normalize_url("https://shop.toom.de/a") != normalize_url("https://toom.de/a")
