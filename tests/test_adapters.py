"""Adapter contract tests.

Adding a new agent format means adding an adapter; these lock the parts
everything downstream depends on so a new adapter cannot quietly break
detection or mis-map its token fields.
"""

from pathlib import Path

import pytest

from toka.adapters import ADAPTERS, adapter_for, parse
from toka.analyze import analyze
from toka.pricing import resolve

FIXTURES = Path(__file__).parent / "fixtures"


def test_every_adapter_declares_its_contract():
    for adapter in ADAPTERS:
        assert adapter.name and adapter.provider
        assert callable(adapter.detect) and callable(adapter.parse)


@pytest.mark.parametrize(
    "filename,expected",
    [("openai.jsonl", "openai-compatible"), ("gemini.jsonl", "gemini")],
)
def test_detection_routes_to_the_right_adapter(filename, expected):
    assert adapter_for(FIXTURES / filename).name == expected


def test_purpose_built_adapters_do_not_claim_each_others_formats():
    """Detection must be exclusive among adapters that know a format, or
    the registry picks by accident. `generic` is exempt — claiming broadly
    at low confidence is its entire job, and it always loses the tie."""
    from toka.adapters.base import sniff

    for filename in ("openai.jsonl", "gemini.jsonl"):
        sample = sniff(FIXTURES / filename)
        confident = [
            a.name for a in ADAPTERS if a.name != "generic" and a.detect(sample) >= 0.5
        ]
        assert len(confident) == 1, f"{filename} claimed by {confident}"


def test_openai_cached_tokens_are_not_double_counted():
    """`prompt_tokens` includes cached tokens, so fresh input is the
    difference — counting both would inflate context size."""
    reqs = list(parse(FIXTURES / "openai.jsonl"))
    assert len(reqs) == 3
    second = reqs[1]
    assert second.cache_read == 9000
    assert second.fresh_input == 3000  # 12000 - 9000
    assert second.context_size == 12000


def test_providers_without_write_premium_report_no_writes():
    for filename in ("openai.jsonl", "gemini.jsonl"):
        for req in parse(FIXTURES / filename):
            assert req.total_writes == 0


def test_sequence_numbers_are_per_session_and_ordered():
    reqs = list(parse(FIXTURES / "openai.jsonl"))
    assert [r.seq for r in reqs] == [0, 1, 2]
    assert len({r.session for r in reqs}) == 1


def test_unpriced_providers_are_excluded_from_cost_not_dropped():
    """A missing rate card must not fabricate dollars, and must not
    silently discard the tokens either."""
    reqs = list(parse(FIXTURES / "openai.jsonl"))
    report = analyze(reqs)
    assert report.total_cost == 0.0
    assert report.unpriced_requests == 3
    assert report.unpriced_tokens > 0


def test_non_anthropic_models_are_never_priced_by_analogy():
    price, exact = resolve("gpt-4o", "openai")
    assert price is None and not exact
    # An unknown *Anthropic* id may fall back within its own rate card.
    price, exact = resolve("claude-opus-9-unreleased", "anthropic")
    assert price is not None and not exact


def test_write_accounting_flagged_unreliable_when_reads_dwarf_writes():
    """Reads require a prior write. A source reporting many reads and no
    writes is under-reporting, not showing a stable prefix — claiming 0%
    churn off that data would tell the user they have no problem."""
    from toka.record import Request

    def req(seq, read, write):
        return Request(
            source="t", provider="anthropic", session="s", seq=seq,
            timestamp=None, model="claude-opus-5", fresh_input=100,
            cache_write_5m=write, cache_write_1h=0, cache_read=read, output=10,
        )

    unreliable = analyze([req(0, 1_000_000, 0), req(1, 1_000_000, 100)])
    assert not unreliable.write_accounting_reliable

    reliable = analyze([req(0, 10_000, 5_000), req(1, 10_000, 5_000)])
    assert reliable.write_accounting_reliable


def test_cline_router_infers_provider_from_model_not_adapter():
    """Cline routes to any model. Hardcoding a provider would price a
    GPT task at Anthropic rates."""
    from toka.adapters.cline import provider_of

    assert provider_of("anthropic/claude-sonnet-4.5") == "anthropic"
    assert provider_of("openai/gpt-4o") == "openai"
    assert provider_of("google/gemini-2.5-pro") == "google"
    assert provider_of(None) == "unknown"


def test_blind_sources_are_counted_even_when_also_unpriced():
    """Cache blindness is a property of the source, not of pricing. A
    source that is both blind and unpriced must still be reported as
    blind, or the user sees no warning at all."""
    from toka.record import Request

    reqs = [
        Request(
            source="continue", provider="openai", session="s", seq=i,
            timestamp=None, model="gpt-4.1", fresh_input=5000,
            cache_write_5m=0, cache_write_1h=0, cache_read=0, output=100,
            cache_visible=False,
        )
        for i in range(3)
    ]
    report = analyze(reqs)
    assert report.blind_sessions == 1
    assert report.blind_tokens == 15000
    assert report.recoverable_miss == 0.0  # never claimed on blind data


def test_blind_source_never_reports_recoverable_waste():
    """The trap this guards: a source logging only a prompt-token total
    would otherwise read as ~100% cache miss on a healthy setup."""
    from toka.record import Request

    reqs = [
        Request(
            source="x", provider="anthropic", session="s", seq=i,
            timestamp=None, model="claude-opus-5", fresh_input=100_000,
            cache_write_5m=0, cache_write_1h=0, cache_read=0, output=100,
            cache_visible=False,
        )
        for i in range(5)
    ]
    report = analyze(reqs)
    assert report.total_cost > 0      # cost is still real
    assert report.recoverable == 0.0  # but no waste is claimed


def test_generic_never_outranks_a_purpose_built_adapter():
    """The generic adapter would parse most files. It must always lose to
    an adapter that knows the format's semantics."""
    from toka.adapters.base import sniff
    from toka.adapters.generic import GenericAdapter

    for filename in ("openai.jsonl", "gemini.jsonl"):
        sample = sniff(FIXTURES / filename)
        assert GenericAdapter().detect(sample) < 0.9
        assert adapter_for(FIXTURES / filename).name != "generic"


def test_generic_finds_nested_token_counts():
    reqs = list(parse(FIXTURES / "homegrown.jsonl"))
    assert len(reqs) == 2
    r = reqs[0]
    assert r.fresh_input == 500 and r.output == 80
    assert r.cache_read == 9000 and r.total_writes == 1200
    assert r.provider == "anthropic"      # inferred from model id
    assert r.cache_visible                # cache fields were present


def test_generic_marks_cache_invisible_when_no_cache_fields_found():
    """A log that never mentions caching is not evidence caching failed."""
    import json, tempfile, os
    from toka.adapters.generic import GenericAdapter

    fd, p = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"model": "claude-opus-5",
                             "prompt_tokens": 100, "completion_tokens": 20}) + "\n")
    try:
        reqs = list(GenericAdapter().parse(Path(p)))
        assert len(reqs) == 1 and not reqs[0].cache_visible
    finally:
        os.unlink(p)


def test_aider_parses_human_formatted_counts():
    reqs = list(parse(FIXTURES / ".aider.chat.history.md"))
    assert len(reqs) == 2
    assert reqs[0].fresh_input == 3100 and reqs[0].output == 226
    assert reqs[1].fresh_input == 12000 and reqs[1].output == 1400
    assert reqs[0].model == "gpt-4o"
    assert all(not r.cache_visible for r in reqs)  # aider logs no caching


def test_unverified_adapters_declare_themselves():
    """Adapters not tested against real traffic must say so, so their
    numbers are not mistaken for verified ones."""
    from toka.adapters.aider import AiderAdapter

    assert AiderAdapter().verified is False


def test_deepseek_cache_fields_are_read():
    """DeepSeek is OpenAI-compatible on requests but names its cache
    fields differently and puts them top-level. Missing that branch makes
    its caching invisible and reads every prompt token as a miss."""
    reqs = list(parse(FIXTURES / "deepseek.jsonl"))
    assert len(reqs) == 2
    assert reqs[0].cache_read == 9200 and reqs[0].fresh_input == 800
    assert reqs[1].cache_read == 11500 and reqs[1].fresh_input == 500
    assert all(r.cache_visible for r in reqs)


def test_cache_multipliers_are_per_model_not_global():
    """Cache economics differ by provider — Anthropic reads are 0.1x,
    OpenAI ~0.5x, DeepSeek ~0.26x. A global constant misprices everyone
    who is not Anthropic."""
    from toka.pricing import ModelPrice, cost

    anthropic = ModelPrice(5.0, 25.0, 512)
    cheap_reads = ModelPrice(5.0, 25.0, 512, cache_read_mult=0.5)
    kw = dict(fresh_input=0, cache_write_5m=0, cache_write_1h=0,
              cache_read=1_000_000, output=0)
    assert cost(anthropic, **kw) == 0.5      # 1M * $5 * 0.1
    assert cost(cheap_reads, **kw) == 2.5    # 1M * $5 * 0.5
