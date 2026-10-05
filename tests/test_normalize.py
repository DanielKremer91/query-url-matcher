from qum.normalize import host_of, normalize_query, normalize_url


def test_normalize_query_lowercases_and_collapses_whitespace():
    assert normalize_query("  Hundefutter   Getreidefrei ") == "hundefutter getreidefrei"


def test_normalize_url_strips_fragment_trailing_slash_and_tracking():
    url = "HTTPS://Www.Example.de/Ratgeber/?utm_source=x&gclid=1&farbe=rot#abschnitt"
    assert normalize_url(url) == "https://www.example.de/Ratgeber?farbe=rot"


def test_normalize_url_root_with_and_without_slash_are_equal():
    assert normalize_url("https://example.de/") == normalize_url("https://example.de")


def test_normalize_url_keeps_path_case():
    assert normalize_url("https://example.de/A") != normalize_url("https://example.de/a")


def test_host_of():
    assert host_of("https://WWW.Example.de/x") == "www.example.de"
