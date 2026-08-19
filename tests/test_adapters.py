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


def test_adapters_do_not_claim_each_others_formats():
    """Detection must be exclusive, or the registry picks by accident."""
    for filename in ("openai.jsonl", "gemini.jsonl"):
        from toka.adapters.base import sniff

        sample = sniff(FIXTURES / filename)
        confident = [a.name for a in ADAPTERS if a.detect(sample) >= 0.5]
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
