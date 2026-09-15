"""Structured logging.

Production modules use ``get_logger(__name__)`` and never ``print``. Two
formats are supported: human-readable text for a development terminal and
single-line JSON for log aggregation in a store deployment.

Structured fields are attached with the ``extra={"fields": {...}}`` convention
and are emitted as ``key=value`` pairs (text) or JSON object members (json).
All values pass through the credential sanitizer.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..utils.sanitize import sanitize_mapping

_CONFIGURED = False
_LOCK = threading.Lock()

#: Attributes present on every LogRecord; anything else was added by the caller.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "taskName"}
)


class StructuredFormatter(logging.Formatter):
    """Human-readable formatter that appends structured fields."""

    def __init__(self, use_color: bool = False) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-38s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = _extract_fields(record)
        if fields:
            rendered = " ".join(f"{k}={_render(v)}" for k, v in fields.items())
            base = f"{base} | {rendered}"
        if self.use_color:
            color = self._COLORS.get(record.levelname, "")
            if color:
                base = f"{color}{base}{self._RESET}"
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per line, suitable for shipping to an aggregator."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
        }
        payload.update(_extract_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def _extract_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Collect structured fields from a record, sanitized."""
    fields: dict[str, Any] = {}
    explicit = record.__dict__.get("fields")
    if isinstance(explicit, dict):
        fields.update(explicit)
    for key, value in record.__dict__.items():
        if key not in _RESERVED and key != "fields":
            fields[key] = value
    return sanitize_mapping(fields) if fields else {}


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_file: Path | str | None = None,
    *,
    force: bool = False,
) -> None:
    """Install handlers on the root logger. Idempotent unless ``force``."""
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED and not force:
            return
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        root.setLevel(getattr(logging, level.upper(), logging.INFO))

        stream = logging.StreamHandler(sys.stderr)
        if fmt == "json":
            stream.setFormatter(JsonFormatter())
        else:
            stream.setFormatter(StructuredFormatter(use_color=sys.stderr.isatty()))
        root.addHandler(stream)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate so a long-running store deployment cannot fill the disk.
            file_handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=32 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(JsonFormatter() if fmt == "json" else StructuredFormatter())
            root.addHandler(file_handler)

        # Third-party chatter that is never useful at INFO.
        for noisy in ("urllib3", "PIL", "matplotlib", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


class RateLimitedLogger:
    """Wrapper that suppresses repeated identical messages.

    Camera reconnect loops and dropped-frame warnings can fire many times per
    second; unbounded they would themselves become a performance problem.
    """

    def __init__(self, logger: logging.Logger, interval_seconds: float = 10.0) -> None:
        self._logger = logger
        self._interval = interval_seconds
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        self._lock = threading.Lock()

    def _should_emit(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is None or (now - last) >= self._interval:
                self._last[key] = now
                skipped = self._suppressed.pop(key, 0)
                return True, skipped
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False, 0

    def log(self, level: int, key: str, message: str, **kwargs: Any) -> None:
        emit, skipped = self._should_emit(key)
        if not emit:
            return
        if skipped:
            message = f"{message} (suppressed {skipped} similar messages)"
        self._logger.log(level, message, **kwargs)

    def warning(self, key: str, message: str, **kwargs: Any) -> None:
        self.log(logging.WARNING, key, message, **kwargs)

    def error(self, key: str, message: str, **kwargs: Any) -> None:
        self.log(logging.ERROR, key, message, **kwargs)

    def info(self, key: str, message: str, **kwargs: Any) -> None:
        self.log(logging.INFO, key, message, **kwargs)
