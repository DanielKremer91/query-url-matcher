import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = {"gclid", "fbclid", "msclkid", "srsltid"}


def normalize_query(q) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def _split(u):
    """Zerlegt eine URL; ohne Protokoll ("www.toom.de/a") wird https angenommen, sonst landete alles im Pfad."""
    text = str(u).strip()
    if "://" not in text:
        text = "https://" + text.lstrip("/")
    return urlsplit(text)


def _host(parts) -> str:
    """Host ohne Groß-/Kleinschreibung, ohne "www." und ohne Standard-Port: www.toom.de und toom.de sind dieselbe Seite."""
    host = parts.netloc.lower()
    for port in (":80", ":443"):
        if host.endswith(port):
            host = host[: -len(port)]
    return host[4:] if host.startswith("www.") else host


def normalize_url(u) -> str:
    """Vergleichsschlüssel: http/https, www, Schrägstrich am Ende, Fragment und Tracking-Parameter zählen nicht."""
    parts = _split(u)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING)
    ]
    return urlunsplit(("https", _host(parts), parts.path.rstrip("/"), urlencode(query), ""))


def host_of(u) -> str:
    return _host(_split(u))
