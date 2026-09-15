"""Credential sanitization.

Every camera source string passes through :func:`sanitize_source` before it
reaches a log record, an API response, an alert payload or an on-screen
overlay. RTSP URLs routinely embed ``user:password@`` and those must never be
written anywhere durable.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

#: Matches ``scheme://user:password@host`` in any URL-ish string.
_CREDENTIAL_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<userinfo>[^/@\s]+)@")

#: Keys whose values must never be logged verbatim.
SENSITIVE_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "apikey", "authorization", "bearer"}
)

REDACTED = "***"


def sanitize_source(source: object) -> str:
    """Return a log-safe representation of a camera source.

    * ``rtsp://bob:hunter2@cam/stream`` -> ``rtsp://bob:***@cam/stream``
    * ``0`` -> ``webcam:0``
    * a filesystem path is returned unchanged (no secrets there)

    The username is preserved because it is operationally useful when
    debugging which account a camera is being polled with; only the secret is
    removed.
    """
    if source is None:
        return ""
    if isinstance(source, int):
        return f"webcam:{source}"
    text = str(source)
    if not text:
        return ""
    if text.isdigit():
        return f"webcam:{text}"

    def _redact(match: re.Match[str]) -> str:
        userinfo = match.group("userinfo")
        user = userinfo.split(":", 1)[0]
        return f"{match.group('scheme')}{user}:{REDACTED}@"

    return _CREDENTIAL_RE.sub(_redact, text)


def sanitize_url(url: str) -> str:
    """Strip userinfo *and* query string from a URL for logging.

    Used for webhook endpoints, where a token is often carried in the query.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:  # pragma: no cover - malformed input
        return REDACTED
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:{REDACTED}@{host}"
    query = f"{REDACTED}" if parts.query else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def sanitize_mapping(data: dict[str, object]) -> dict[str, object]:
    """Copy a mapping with any sensitive-looking value redacted."""
    clean: dict[str, object] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            clean[key] = REDACTED
        elif isinstance(value, dict):
            clean[key] = sanitize_mapping(value)  # type: ignore[arg-type]
        elif isinstance(value, str) and "://" in value:
            clean[key] = sanitize_source(value)
        else:
            clean[key] = value
    return clean
