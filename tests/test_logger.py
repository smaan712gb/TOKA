"""The logging shim, and the round trip back out of it.

This is the only place Toka writes the data it later reads, so the test
that matters most is not "did it parse the response" but "does a call
logged here arrive intact at the other end of the analysis".
"""

import json

import pytest

from toka import logger
from toka.adapters import adapter_for
from toka.analyze import analyze
from toka.ingest import find_transcripts, read_all


class Detail:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Usage(Detail):
    pass


class Response(Detail):
    pass


ANTHROPIC = Response(
    model="claude-sonnet-4-5",
    usage=Usage(
        input_tokens=120,
        cache_creation_input_tokens=4000,
        cache_read_input_tokens=90_000,
        output_tokens=350,
        cache_creation=Detail(
            ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=3000
        ),
    ),
)

OPENAI = {
    "model": "gpt-5-codex",
    "usage": {
        "prompt_tokens": 10_000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 8_000},
        "completion_tokens_details": {"reasoning_tokens": 120},
    },
}

DEEPSEEK = {
    "model": "deepseek-chat",
    "usage": {
        "prompt_tokens": 10_000,
        "completion_tokens": 500,
        "prompt_cache_hit_tokens": 7_000,
        "prompt_cache_miss_tokens": 3_000,
    },
}

GOOGLE = {
    "modelVersion": "gemini-2.5-pro",
    "usageMetadata": {
        "promptTokenCount": 10_000,
        "cachedContentTokenCount": 6_000,
        "candidatesTokenCount": 400,
        "thoughtsTokenCount": 50,
    },
}


def test_an_anthropic_response_keeps_its_ttl_split():
    """The 5m and 1h tiers are billed at different multipliers, so
    collapsing them would misprice every cached write."""
    rec = logger.extract(ANTHROPIC)
    assert rec["provider"] == "anthropic"
    assert rec["fresh_input"] == 120
    assert rec["cache_write_5m"] == 1000
    assert rec["cache_write_1h"] == 3000
    assert rec["cache_read"] == 90_000
    assert rec["output"] == 350


def test_writes_without_a_ttl_breakdown_default_to_the_cheaper_claim():
    """Assuming 5m discounts more rewrites as expiry, which lowers the
    churn we report. The error, if any, is in the safe direction."""
    response = Response(
        model="claude-opus-5",
        usage=Usage(
            input_tokens=1,
            cache_creation_input_tokens=5_000,
            cache_read_input_tokens=0,
            output_tokens=1,
        ),
    )
    rec = logger.extract(response)
    assert rec["cache_write_5m"] == 5_000
    assert rec["cache_write_1h"] == 0


def test_openai_reports_no_write_premium():
    """Caching is automatic there and writes are not billed separately,
    so claiming writes would invent a cost the provider never charged."""
    rec = logger.extract(OPENAI)
    assert rec["provider"] == "openai"
    assert rec["cache_write_5m"] == 0 and rec["cache_write_1h"] == 0
    assert rec["cache_read"] == 8_000
    assert rec["fresh_input"] == 2_000  # prompt total is inclusive of cached
    assert rec["thinking"] == 120


def test_deepseek_cache_fields_are_read():
    """The whole reason this shim exists — DeepSeek writes no logs, and
    names its cache fields unlike anyone else."""
    rec = logger.extract(DEEPSEEK)
    assert rec["cache_read"] == 7_000
    assert rec["fresh_input"] == 3_000
    assert rec["cache_visible"]


def test_google_prompt_total_is_inclusive_of_cache():
    rec = logger.extract(GOOGLE)
    assert rec["provider"] == "google"
    assert rec["cache_read"] == 6_000
    assert rec["fresh_input"] == 4_000
    assert rec["model"] == "gemini-2.5-pro"


def test_a_response_with_no_cache_fields_is_marked_unmeasured():
    """Not a zero. An API that never mentions caching is not evidence
    that caching failed, and counting those tokens as misses would report
    near-total waste on a healthy setup."""
    rec = logger.extract({"model": "m", "usage": {"prompt_tokens": 500, "completion_tokens": 5}})
    assert rec["cache_visible"] is False
    assert rec["cache_read"] == 0


def test_a_response_with_no_usage_is_not_logged_as_a_free_request(tmp_path):
    """Writing zeros would be a claim — that the call cost nothing — and
    one nobody would think to check."""
    before = logger.skipped
    with pytest.warns(RuntimeWarning):
        assert logger.log({"id": "msg_1"}, path=tmp_path, session="s") is None
    assert logger.skipped == before + 1
    assert not list(tmp_path.glob("*.jsonl"))


def test_logging_never_raises_at_the_call_site():
    """A metrics call that throws inside a request handler is worse than
    no metrics at all."""
    assert logger.log(object(), path=None, session="s") is None
    assert logger.log(None, session="s") is None


def test_strict_mode_raises_for_tests():
    with pytest.raises(ValueError):
        logger.log({"no": "usage"}, strict=True, session="s")


def test_explicit_model_and_provider_win():
    """Gateways rewrite or drop both, and a router that reports the wrong
    provider prices the whole session wrong."""
    rec = logger.extract(OPENAI, model="claude-sonnet-4-5", provider="anthropic")
    assert rec["model"] == "claude-sonnet-4-5"
    assert rec["provider"] == "anthropic"


def test_a_logged_call_survives_the_round_trip(tmp_path):
    """End to end: log it, find it, parse it, price it. Every step in
    between is where a field quietly becomes a zero."""
    logger.log(ANTHROPIC, path=tmp_path, session="round-trip")
    logger.log(ANTHROPIC, path=tmp_path, session="round-trip")

    files = find_transcripts(tmp_path)
    assert len(files) == 1
    assert adapter_for(files[0]).name == "toka-log"

    requests = read_all(files)
    assert len(requests) == 2
    assert [r.seq for r in requests] == [0, 1]
    assert requests[0].provider == "anthropic"
    assert requests[0].cache_write_1h == 3000

    report = analyze(requests)
    assert report.total_cost > 0
    assert report.blind_sessions == 0


def test_sessions_are_separate_files_and_separate_sequences(tmp_path):
    """Write amplification is measured within a session, so merging two
    conversations would understate churn in both."""
    logger.log(ANTHROPIC, path=tmp_path, session="a")
    logger.log(ANTHROPIC, path=tmp_path, session="b")
    logger.log(ANTHROPIC, path=tmp_path, session="a")

    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert names == ["a.jsonl", "b.jsonl"]

    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in rows] == [0, 1]


def test_an_unmeasured_record_stays_unmeasured_through_the_adapter(tmp_path):
    """The suppression has to survive the file, not just the call."""
    logger.log(
        {"model": "m", "usage": {"prompt_tokens": 500, "completion_tokens": 5}},
        path=tmp_path,
        session="blind",
    )
    requests = read_all(find_transcripts(tmp_path))
    assert requests and requests[0].cache_visible is False

    report = analyze(requests)
    assert report.blind_sessions == 1
    assert report.recoverable == 0.0


def test_new_session_resets_the_sequence():
    first = logger.new_session("explicit-name")
    assert first == "explicit-name"
    assert logger.new_session() != "explicit-name"
