"""Continue (VS Code / JetBrains) — `~/.continue/dev_data/*/tokensGenerated.jsonl`.

Continue logs one row per model call:

    {"timestamp": "...", "model": "gpt-4.1-2025-04-14", "provider": "openai",
     "promptTokens": 1234, "generatedTokens": 56}

Note what is absent: there is no cache accounting of any kind. Continue
records a prompt-token total and nothing about what was cached.

That absence is load-bearing. The provider caches regardless of whether
the client bothers to log it, so treating every prompt token as a miss
would report near-total waste on a setup that may be entirely healthy.
Records from this adapter are marked `cache_visible=False`, which keeps
them out of the recoverable figure while still counting their tokens and
cost.

`tokensGenerated.jsonl` carries no session id (its sibling
`chatInteraction.jsonl` does), so rows are grouped per file. Session
boundaries only matter for churn analysis, which is unavailable here
anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, iter_objects, scoped_session

MARKER = "tokensGenerated"


class ContinueAdapter:
    name = "continue"
    provider = "unknown"  # per-row; Continue routes to any provider

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            if obj.get("eventName") == MARKER and "promptTokens" in obj:
                return 1.0
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        session = scoped_session(path, path.stem)
        seq = 0
        for record in iter_objects(path):
            if record.get("eventName") != MARKER:
                continue
            yield Request(
                source=self.name,
                provider=_provider(record),
                session=session,
                seq=seq,
                timestamp=record.get("timestamp"),
                model=record.get("model"),
                fresh_input=as_int(record.get("promptTokens")),
                cache_write_5m=0,
                cache_write_1h=0,
                cache_read=0,
                output=as_int(record.get("generatedTokens")),
                cache_visible=False,
            )
            seq += 1


def _provider(record: dict) -> str:
    raw = (record.get("provider") or "").lower()
    if raw in ("anthropic", "openai", "google"):
        return raw
    model = (record.get("model") or "").lower()
    if "claude" in model:
        return "anthropic"
    if "gpt" in model or model.startswith(("o1", "o3")):
        return "openai"
    if "gemini" in model:
        return "google"
    return "unknown"
