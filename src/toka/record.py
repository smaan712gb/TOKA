"""The normalised record every adapter produces.

One `Request` is one billed model call, whatever produced it. Adapters map
their source format onto these fields; everything downstream — pricing,
analysis, reporting — only ever sees this shape.

Providers differ in how they expose caching, and the mapping matters:

  Anthropic  explicit write/read split, writes billed at a premium.
             cache_write_5m / cache_write_1h / cache_read all populated.
  OpenAI     automatic caching, no write concept and no write premium.
             cache_read populated, cache_write_* stay zero.
  Google     context caching with an explicit cached-token count.
             cache_read populated, cache_write_* stay zero.

Where a provider has no write premium, prefix churn is invisible in the
billing data — the waste shows up as cache *miss* instead. Both mechanisms
are measured, so a provider only needs one of them to be analysable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Request:
    source: str  # adapter that produced this record
    provider: str  # "anthropic" | "openai" | "google" | "unknown"
    session: str
    seq: int  # position within the session, 0-based
    timestamp: str | None
    model: str | None
    fresh_input: int  # prompt tokens billed at full rate
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    output: int
    thinking: int = 0

    @property
    def context_size(self) -> int:
        """Total prompt tokens the model saw for this request."""
        return (
            self.fresh_input
            + self.cache_write_5m
            + self.cache_write_1h
            + self.cache_read
        )

    @property
    def total_writes(self) -> int:
        return self.cache_write_5m + self.cache_write_1h
