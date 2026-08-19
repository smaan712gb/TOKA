"""Claude Code session transcripts (~/.claude/projects/**/*.jsonl).

Assistant messages carry a `usage` object with the full Anthropic cache
accounting, including the 5m/1h write split. This is the only common
format that exposes cache writes separately, which makes it the only one
where prefix churn is directly measurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, dig, iter_objects, scoped_session


class ClaudeCodeAdapter:
    name = "claude-code"
    provider = "anthropic"

    def detect(self, sample: list[dict]) -> float:
        if not sample:
            return 0.0
        # These three co-occurring are unique to Claude Code transcripts.
        markers = {"parentUuid", "sessionId", "isSidechain"}
        hits = sum(1 for obj in sample if markers <= obj.keys())
        if hits:
            return 1.0
        # Weaker signal: an Anthropic-shaped usage object.
        for obj in sample:
            if dig(obj, "message", "usage", "cache_read_input_tokens") is not None:
                return 0.7
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        session = scoped_session(path, path.stem)
        seq = 0
        for record in iter_objects(path):
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue

            five, hour = _split_writes(usage)
            yield Request(
                source=self.name,
                provider=self.provider,
                session=record.get("sessionId") or session,
                seq=seq,
                timestamp=record.get("timestamp"),
                model=message.get("model"),
                fresh_input=as_int(usage.get("input_tokens")),
                cache_write_5m=five,
                cache_write_1h=hour,
                cache_read=as_int(usage.get("cache_read_input_tokens")),
                output=as_int(usage.get("output_tokens")),
                thinking=as_int(
                    dig(usage, "output_tokens_details", "thinking_tokens")
                ),
            )
            seq += 1


def _split_writes(usage: dict) -> tuple[int, int]:
    """Return (5m tokens, 1h tokens).

    Prefer the explicit `cache_creation` breakdown. Older transcripts carry
    only the total; attribute those to the 5m tier, the cheaper of the two,
    so the estimate stays conservative rather than inflating measured cost.
    """
    total = as_int(usage.get("cache_creation_input_tokens"))
    breakdown = usage.get("cache_creation")
    if isinstance(breakdown, dict):
        five = as_int(breakdown.get("ephemeral_5m_input_tokens"))
        hour = as_int(breakdown.get("ephemeral_1h_input_tokens"))
        if five + hour > 0:
            return five, hour
    return total, 0
