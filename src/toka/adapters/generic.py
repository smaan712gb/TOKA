"""Last-resort adapter for any JSON that carries token counts.

Bespoke adapters exist because each agent names its fields differently
and gets the semantics subtly wrong in its own way. But most logs — the
homegrown wrapper, the tool nobody has written an adapter for yet, an
export from something new — are still just objects with token counts
somewhere inside them.

This finds them by field-name pattern, at any nesting depth, and claims
the file weakly (0.5) so every purpose-built adapter outranks it. It is
the difference between "unsupported" and "works, with caveats".

Two things it deliberately does not do:

  It never guesses a provider. Field names say nothing about who served
  the request, so records come out provider-unknown and stay out of the
  dollar figures unless the model id itself identifies the vendor.

  It never assumes caching. If no cache-shaped field is found the record
  is marked cache_visible=False, so its prompt tokens are not reported as
  recoverable waste. A log that simply does not mention caching is not
  evidence that caching failed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, iter_objects, scoped_session

# Ordered: the first pattern to match a key wins, so more specific
# patterns must precede the looser ones.
_INPUT = re.compile(r"^(input|prompt)_?tokens?(_count)?$|^tokens_?(in|prompt)$", re.I)
_OUTPUT = re.compile(
    r"^(output|completion|generated|candidates)_?tokens?(_count)?$"
    r"|^tokens_?(out|generated)$",
    re.I,
)
_CACHE_READ = re.compile(
    r"^cache_?read(s)?(_input_tokens)?$|^cached_?(content_?)?tokens?(_count)?$", re.I
)
_CACHE_WRITE = re.compile(
    r"^cache_?(write|creation)(s)?(_input_tokens)?$", re.I
)
_TIMESTAMP = re.compile(r"^(timestamp|created_?at|ts|time|date)$", re.I)
_MODEL = re.compile(r"^(model|model_?id|model_?name|model_?version)$", re.I)
_SESSION = re.compile(r"^(session_?id|conversation_?id|thread_?id|trace_?id)$", re.I)

_MAX_DEPTH = 6


class GenericAdapter:
    name = "generic"
    provider = "unknown"

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            found = _scan(obj)
            if found.get("input") is not None and found.get("output") is not None:
                return 0.5  # always loses to a purpose-built adapter
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        fallback = scoped_session(path, path.stem)
        counters: dict[str, int] = {}
        for record in iter_objects(path):
            found = _scan(record)
            if found.get("input") is None or found.get("output") is None:
                continue

            model = found.get("model")
            session = str(found.get("session") or fallback)
            seq = counters.get(session, 0)
            counters[session] = seq + 1

            read = as_int(found.get("cache_read"))
            write = as_int(found.get("cache_write"))
            saw_cache = found.get("cache_read") is not None or (
                found.get("cache_write") is not None
            )

            yield Request(
                source=self.name,
                provider=_provider_of(model),
                session=session,
                seq=seq,
                timestamp=_stamp(found.get("timestamp")),
                model=model,
                fresh_input=as_int(found.get("input")),
                cache_write_5m=write,
                cache_write_1h=0,
                cache_read=read,
                output=as_int(found.get("output")),
                cache_visible=saw_cache,
            )


def _scan(obj, depth: int = 0) -> dict:
    """Collect the first match for each field kind, breadth-first-ish.

    Shallower keys win, so a top-level `model` beats one buried inside a
    nested request echo.
    """
    found: dict = {}
    if depth > _MAX_DEPTH or not isinstance(obj, dict):
        return found

    nested = []
    for key, value in obj.items():
        if isinstance(value, dict):
            nested.append(value)
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            for kind, pattern in (
                ("input", _INPUT),
                ("output", _OUTPUT),
                ("cache_read", _CACHE_READ),
                ("cache_write", _CACHE_WRITE),
            ):
                if kind not in found and pattern.match(key):
                    found[kind] = value
                    break
        if isinstance(value, str):
            if "model" not in found and _MODEL.match(key):
                found["model"] = value
            elif "session" not in found and _SESSION.match(key):
                found["session"] = value
            elif "timestamp" not in found and _TIMESTAMP.match(key):
                found["timestamp"] = value
        elif isinstance(value, (int, float)) and _TIMESTAMP.match(key):
            if "timestamp" not in found:
                found["timestamp"] = value

    for child in nested:
        for kind, value in _scan(child, depth + 1).items():
            found.setdefault(kind, value)
    return found


def _stamp(value) -> str | None:
    """Normalise epoch seconds/millis to ISO-8601; pass strings through."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    from datetime import datetime, timezone

    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n > 1e11:  # milliseconds
        n /= 1000.0
    try:
        return datetime.fromtimestamp(n, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _provider_of(model: str | None) -> str:
    if not model:
        return "unknown"
    m = model.lower()
    if "claude" in m or m.startswith("anthropic"):
        return "anthropic"
    if "gpt" in m or m.startswith(("openai", "o1", "o3")):
        return "openai"
    if "gemini" in m or m.startswith("google"):
        return "google"
    return "unknown"
