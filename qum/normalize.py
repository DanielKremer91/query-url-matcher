import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = {"gclid", "fbclid", "msclkid", "srsltid"}


def normalize_query(q) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def normalize_url(u) -> str:
    parts = urlsplit(str(u).strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING)
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), "")
    )


def host_of(u) -> str:
    return urlsplit(str(u).strip()).netloc.lower()
