"""Google Gemini responses (`usageMetadata`).

Context caching exposes a cached-token count but no write premium, so
`cache_write_*` stay zero and the recoverable signal is cache miss, as
with the OpenAI-compatible shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, dig, iter_objects

SESSION_KEYS = ("session_id", "sessionId", "conversation_id", "trace_id")
TIMESTAMP_KEYS = ("timestamp", "createTime", "created_at", "time")


class GeminiAdapter:
    name = "gemini"
    provider = "google"

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            if _usage(obj):
                return 0.9
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        fallback_session = path.stem
        counters: dict[str, int] = {}
        for record in iter_objects(path):
            usage = _usage(record)
            if not usage:
                continue

            prompt = as_int(usage.get("promptTokenCount"))
            cached = as_int(usage.get("cachedContentTokenCount"))
            fresh = max(0, prompt - cached)

            session = _first(record, SESSION_KEYS) or fallback_session
            seq = counters.get(session, 0)
            counters[session] = seq + 1

            yield Request(
                source=self.name,
                provider=self.provider,
                session=str(session),
                seq=seq,
                timestamp=_first(record, TIMESTAMP_KEYS),
                model=record.get("modelVersion") or record.get("model"),
                fresh_input=fresh,
                cache_write_5m=0,
                cache_write_1h=0,
                cache_read=cached,
                output=as_int(usage.get("candidatesTokenCount")),
                thinking=as_int(usage.get("thoughtsTokenCount")),
            )


def _usage(record: dict) -> dict | None:
    for candidate in (
        record.get("usageMetadata"),
        dig(record, "response", "usageMetadata"),
    ):
        if isinstance(candidate, dict) and "promptTokenCount" in candidate:
            return candidate
    return None


def _first(record: dict, keys: tuple[str, ...]):
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None
