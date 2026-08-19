"""Record usage for agents that keep no logs.

Every adapter in Toka reads something an agent already wrote down. That
leaves the largest gap in coverage: code calling a provider API directly
writes nothing at all, so DeepSeek, OpenRouter and every homegrown loop
are invisible to the analysis no matter how many adapters get added.

The fix is one line at the call site:

    response = client.messages.create(...)
    toka.log(response)

It appends a normalised record to `~/.toka` and returns. `toka` and
`toka --compare` pick it up from there with no further configuration.

Three rules this module holds to:

**It must never break the caller.** A metrics call that raises inside a
request handler is worse than no metrics. Failures return None instead of
propagating — but they are not swallowed either: an unrecognised response
shape warns once per process, so a shim quietly recording nothing for a
week is not a state you can end up in by accident. `strict=True` restores
raising, for tests.

**It records what it could not see.** A response whose shape has no cache
fields at all is written with `cache_visible: false`, which keeps its
prompt tokens out of the recoverable figure downstream. Counting them as
misses would report near-total waste on a healthy setup; counting them as
hits would report none. Neither is knowable, so neither is claimed.

**It writes nothing rather than something wrong.** A response with no
usage block at all is not logged as a request costing zero — it is
skipped and counted, because a free request is a claim, and one nobody
would think to check.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

# One record per model call, appended as JSON Lines. The marker makes the
# format unambiguous to detect, so its adapter never has to guess.
FORMAT_VERSION = 1

_LOCK = threading.Lock()
_SEQ: dict[str, int] = {}
_SESSION = uuid.uuid4().hex[:12]
_WARNED: set[str] = set()

#: How many calls were skipped because nothing usable could be read from
#: them. Exposed so a caller can assert on it rather than trust silence.
skipped = 0


def default_dir() -> Path:
    """Where records go. `TOKA_HOME` overrides it."""
    return Path(os.environ.get("TOKA_HOME") or (Path.home() / ".toka"))


def new_session(name: str | None = None) -> str:
    """Start a new conversation grouping and return its id.

    Sessions are the unit the analysis works in — write amplification is
    cumulative writes against peak context *within a session*, so lumping
    unrelated conversations together understates churn badly.
    """
    global _SESSION
    with _LOCK:
        _SESSION = name or uuid.uuid4().hex[:12]
        _SEQ.pop(_SESSION, None)
    return _SESSION


def log(
    response,
    *,
    session: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    path: Path | None = None,
    strict: bool = False,
) -> dict | None:
    """Append one usage record. Returns what was written, or None.

    `response` is whatever your provider SDK handed back — an Anthropic
    `Message`, an OpenAI `ChatCompletion`, a Google response, or the raw
    dict of any of them. Explicit `model` and `provider` override what is
    read from the response, for gateways that omit or rewrite them.
    """
    global skipped
    try:
        record = extract(response, model=model, provider=provider)
    except Exception:  # pragma: no cover - defensive; extract catches its own
        if strict:
            raise
        record = None

    if record is None:
        skipped += 1
        _warn_once(
            "unrecognised-shape",
            "toka.log could not find token usage on this response, so "
            "nothing was recorded. Pass the provider response object "
            "itself, or a dict of it.",
        )
        if strict:
            raise ValueError("no usage found on response")
        return None

    with _LOCK:
        name = session or _SESSION
        seq = _SEQ.get(name, 0)
        _SEQ[name] = seq + 1
        record["session"] = name
        record["seq"] = seq
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        target = (path or default_dir()) / f"{name}.jsonl"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            if strict:
                raise
            _warn_once("write-failed", f"toka.log could not write to {target}: {exc}")
            return None
    return record


def extract(response, *, model: str | None = None, provider: str | None = None):
    """Normalise a provider response into a record, or None.

    Kept separate from `log` so the mapping can be tested without
    touching the filesystem, and so a caller can route the output
    somewhere other than a file.
    """
    usage = _get(response, "usage", "usageMetadata", "usage_metadata")
    if usage is None:
        return None

    record = (
        _from_anthropic(usage)
        or _from_google(usage)
        or _from_openai(usage)
    )
    if record is None:
        return None

    record["toka"] = FORMAT_VERSION
    record["model"] = model or _get(response, "model", "modelVersion") or None
    record["provider"] = provider or record["provider"]
    return record


def _from_anthropic(usage) -> dict | None:
    """Anthropic splits writes from reads and bills writes at a premium.

    It is the only shape here that can carry a churn analysis, because it
    is the only one where a rewrite is distinguishable from a first
    write.
    """
    fresh = _get(usage, "input_tokens")
    if fresh is None:
        return None

    written = _int(_get(usage, "cache_creation_input_tokens"))
    detail = _get(usage, "cache_creation")
    write_5m = _int(_get(detail, "ephemeral_5m_input_tokens")) if detail else 0
    write_1h = _int(_get(detail, "ephemeral_1h_input_tokens")) if detail else 0
    if not (write_5m or write_1h):
        # No TTL breakdown. Assume the 5-minute tier: it is the default,
        # and it discounts more rewrites as expiry, which lowers the
        # churn we claim rather than raising it.
        write_5m = written

    return {
        "provider": "anthropic",
        "fresh_input": _int(fresh),
        "cache_write_5m": write_5m,
        "cache_write_1h": write_1h,
        "cache_read": _int(_get(usage, "cache_read_input_tokens")),
        "output": _int(_get(usage, "output_tokens")),
        "thinking": 0,
        # Anthropic always reports both cache fields, so their absence
        # would mean an unusual shape rather than an uncached call.
        "cache_visible": True,
    }


def _from_google(usage) -> dict | None:
    total = _get(usage, "promptTokenCount", "prompt_token_count")
    if total is None:
        return None

    cached = _get(usage, "cachedContentTokenCount", "cached_content_token_count")
    return {
        "provider": "google",
        # Google reports the prompt total inclusive of cached content.
        "fresh_input": max(0, _int(total) - _int(cached)),
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "cache_read": _int(cached),
        "output": _int(_get(usage, "candidatesTokenCount", "candidates_token_count")),
        "thinking": _int(_get(usage, "thoughtsTokenCount", "thoughts_token_count")),
        "cache_visible": cached is not None,
    }


def _from_openai(usage) -> dict | None:
    """The OpenAI shape, and everything mirroring it.

    Caching here is automatic and carries no write premium, so the write
    fields stay zero and the only recoverable signal is cache miss. That
    is a property of the billing model, not a gap in the logging.
    """
    prompt = _get(usage, "prompt_tokens")
    if prompt is None:
        return None

    details = _get(usage, "prompt_tokens_details")
    cached = _get(details, "cached_tokens") if details is not None else None

    # DeepSeek is OpenAI-compatible but names its cache fields differently
    # and puts them at the top level of usage.
    hit = _get(usage, "prompt_cache_hit_tokens")
    miss = _get(usage, "prompt_cache_miss_tokens")
    if hit is not None or miss is not None:
        cached = _int(hit)
        fresh = _int(miss) if miss is not None else max(0, _int(prompt) - cached)
        visible = True
    else:
        visible = cached is not None
        fresh = max(0, _int(prompt) - _int(cached))

    reasoning = _get(usage, "completion_tokens_details")
    return {
        "provider": "openai",
        "fresh_input": fresh,
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "cache_read": _int(cached),
        "output": _int(_get(usage, "completion_tokens")),
        "thinking": _int(
            _get(reasoning, "reasoning_tokens") if reasoning is not None else 0
        ),
        # A response with no cache fields at all is not evidence that
        # caching failed — it is evidence that this API does not say.
        "cache_visible": visible,
    }


def _get(obj, *keys):
    """Read a field from a dict or an SDK object, whichever it is."""
    if obj is None:
        return None
    for key in keys:
        if isinstance(obj, dict):
            if key in obj and obj[key] is not None:
                return obj[key]
        else:
            value = getattr(obj, key, None)
            if value is not None:
                return value
    return None


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _warn_once(key: str, message: str) -> None:
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)
