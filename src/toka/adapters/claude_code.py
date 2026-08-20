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
        subagents = _subagent_sessions(path)
        seq: dict[str, int] = {}

        for record in iter_objects(path):
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue

            name = str(record.get("sessionId") or session)
            # A subagent runs its own conversation with its own cache, so
            # it is its own session. Pooling it with the parent sums
            # several contexts against a single peak and reads as churn:
            # on one real machine that inflated the churn share by 4.6
            # points and recoverable spend by 3.2.
            root = subagents.get(record.get("uuid"))
            if root is not None:
                name = f"{name}::agent::{root}"

            position = seq.get(name, 0)
            seq[name] = position + 1

            five, hour = _split_writes(usage)
            yield Request(
                source=self.name,
                provider=self.provider,
                session=name,
                seq=position,
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


def _subagent_sessions(path: Path) -> dict[str, str]:
    """Map each subagent message to the root of its own conversation.

    Claude Code runs subagents as `isSidechain` records that chain
    through `parentUuid` back to a root whose parent is either absent or
    not itself a sidechain. Every message in one chain shares a context
    and a cache entry, and different chains share neither.

    Returns {} for transcripts with no subagents, which is most of them,
    so the common case costs one cheap pass and nothing else.
    """
    parent: dict[str, str | None] = {}
    is_side: dict[str, bool] = {}
    for record in iter_objects(path):
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            parent[uuid] = record.get("parentUuid")
            is_side[uuid] = bool(record.get("isSidechain"))

    if not any(is_side.values()):
        return {}

    roots: dict[str, str] = {}
    for uuid, side in is_side.items():
        if not side:
            continue
        current = uuid
        seen = {current}
        while True:
            above = parent.get(current)
            # Stop at the top of the chain, at the parent conversation,
            # or on a cycle — a malformed transcript must not hang a scan.
            if above is None or not is_side.get(above) or above in seen:
                break
            seen.add(above)
            current = above
        roots[uuid] = current
    return roots


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
