"""Cline / Roo Code task history (VS Code global storage).

Each task is a directory containing `ui_messages.json`, whose
`api_req_started` entries carry the per-request token accounting:

    {"tokensIn": 12080, "tokensOut": 245,
     "cacheWrites": 0, "cacheReads": 0, "cost": 0.0489675}

Cline reports cache writes and reads **separately**, which most non-
Anthropic surfaces do not. That makes prefix churn directly measurable
here, the same as on Claude Code — Cline is a router, so when it is
pointed at an Anthropic model the write accounting comes through intact.

`tokensIn` is the uncached prompt count; cache reads and writes are
reported alongside it rather than folded into it, so they add rather
than subtract when reconstructing the full context size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, dig, iter_objects

# Only these entries represent a billed model call.
MARKER = "api_req_started"


class ClineAdapter:
    name = "cline"
    provider = "anthropic"

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            if obj.get("say") == MARKER and "ts" in obj:
                return 1.0
        # A ui_messages.json whose head is all user turns still has the
        # shape; claim it weakly so it beats no adapter at all.
        if sample and all(
            {"ts", "type"} <= obj.keys() for obj in sample[:5]
        ):
            return 0.6
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        # The task id is the containing directory; the filename is the
        # same for every task and would collapse them all into one.
        session = path.parent.name or path.stem
        model = _model_of(path)
        seq = 0

        for record in iter_objects(path):
            if record.get("say") != MARKER:
                continue
            payload = _payload(record)
            if payload is None:
                continue

            model_id = dig(record, "modelInfo", "modelId") or model
            yield Request(
                source=self.name,
                # Cline is a router — the task's model decides the rate
                # card, not the adapter. Hardcoding a provider here would
                # price a GPT task at Anthropic rates.
                provider=provider_of(model_id),
                session=session,
                seq=seq,
                timestamp=_iso(record.get("ts")),
                model=model_id,
                fresh_input=as_int(payload.get("tokensIn")),
                # Cline does not split TTL tiers; attribute to 5m, the
                # cheaper premium, so cost stays conservative.
                cache_write_5m=as_int(payload.get("cacheWrites")),
                cache_write_1h=0,
                cache_read=as_int(payload.get("cacheReads")),
                output=as_int(payload.get("tokensOut")),
            )
            seq += 1


def provider_of(model_id: str | None) -> str:
    """Infer the rate-card provider from a routed model id."""
    if not model_id:
        return "unknown"
    m = model_id.lower()
    if "claude" in m or m.startswith("anthropic"):
        return "anthropic"
    if "gpt" in m or m.startswith("openai") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if "gemini" in m or m.startswith("google"):
        return "google"
    return "unknown"


def _payload(record: dict) -> dict | None:
    """`text` holds the metrics as an embedded JSON string."""
    text = record.get("text")
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "tokensIn" not in payload:
        return None
    return payload


def _model_of(path: Path) -> str | None:
    """Fall back to the task's recorded model when a request omits it."""
    meta = path.parent / "task_metadata.json"
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None
    usage = data.get("model_usage")
    if isinstance(usage, list) and usage and isinstance(usage[0], dict):
        return usage[0].get("model_id")
    return None


def _iso(ts) -> str | None:
    """Cline stamps epoch milliseconds; the analyzer wants ISO-8601."""
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None
