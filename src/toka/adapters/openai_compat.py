"""Any OpenAI-compatible usage record.

This one adapter covers a large share of the ecosystem: the OpenAI API
itself, plus every gateway and logger that mirrors its response shape —
LiteLLM, OpenRouter, Helicone, Langfuse exports, and most homegrown
wrappers.

Caching on this shape is automatic and has no write premium, so
`cache_write_*` stay zero and the recoverable signal is cache *miss*:
prompt tokens billed fresh that a warm prefix would have discounted.
Prefix churn is not observable here because the billing data does not
distinguish a rewrite from a first write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, dig, iter_objects

# Fields a logger might use for the conversation grouping key, in order.
SESSION_KEYS = (
    "session_id",
    "conversation_id",
    "thread_id",
    "trace_id",
    "run_id",
    "sessionId",
    "conversationId",
)
TIMESTAMP_KEYS = ("timestamp", "created_at", "createdAt", "time", "created")


class OpenAICompatAdapter:
    name = "openai-compatible"
    provider = "openai"

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            usage = _usage(obj)
            if not usage:
                continue
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                # Distinguish from Anthropic, which never uses these names.
                return 0.9
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        fallback_session = path.stem
        counters: dict[str, int] = {}
        for record in iter_objects(path):
            usage = _usage(record)
            if not usage or "prompt_tokens" not in usage:
                continue

            prompt = as_int(usage.get("prompt_tokens"))
            cached = as_int(dig(usage, "prompt_tokens_details", "cached_tokens"))

            # DeepSeek is OpenAI-compatible on the request side but names
            # its cache fields differently and puts them at the top level.
            # Without this branch its caching is invisible and every
            # prompt token reads as a miss.
            ds_hit = usage.get("prompt_cache_hit_tokens")
            ds_miss = usage.get("prompt_cache_miss_tokens")
            if ds_hit is not None or ds_miss is not None:
                cached = as_int(ds_hit)
                fresh = as_int(ds_miss) if ds_miss is not None else max(0, prompt - cached)
            else:
                # Guard against a logger that double-counts cached inside prompt.
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
                model=record.get("model") or dig(record, "response", "model"),
                fresh_input=fresh,
                cache_write_5m=0,
                cache_write_1h=0,
                cache_read=cached,
                output=as_int(usage.get("completion_tokens")),
                thinking=as_int(
                    dig(usage, "completion_tokens_details", "reasoning_tokens")
                ),
            )


def _usage(record: dict) -> dict | None:
    for candidate in (
        record.get("usage"),
        dig(record, "response", "usage"),
        dig(record, "body", "usage"),
    ):
        if isinstance(candidate, dict):
            return candidate
    return None


def _first(record: dict, keys: tuple[str, ...]):
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None
