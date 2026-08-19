"""Cross-agent comparison.

The comparison exists to be read as a ranking, which makes its failure
mode specific: an agent that logs nothing about caching must never look
like an agent that caches badly. Most of these tests defend that line.
"""

from pathlib import Path

from toka.analyze import analyze
from toka.compare import Row, render
from toka.discovery import candidates
from toka.record import Request


def _req(seq, *, fresh=0, write=0, read=0, visible=True, session="s"):
    return Request(
        source="test",
        provider="anthropic",
        session=session,
        seq=seq,
        timestamp=None,
        model="claude-sonnet-4-5",
        fresh_input=fresh,
        cache_write_5m=write,
        cache_write_1h=0,
        cache_read=read,
        output=100,
        cache_visible=visible,
    )


def _row(agent, requests):
    report = analyze(requests)
    return Row(agent=agent, report=report, files=1)


def test_a_blind_source_reports_no_hit_rate_rather_than_zero():
    """The load-bearing distinction. A source with no cache fields
    computes to 0% because cache_read is always zero — which reads as the
    worst agent on the table when it is in fact the unmeasured one."""
    row = _row("Continue", [_req(0, fresh=1000, visible=False)])
    assert row.cache_blind
    assert row.hit_rate is None
    assert row.confidence == "no cache data"


def test_blind_rows_are_excluded_from_the_ranking():
    """Only one agent here was actually measured, so there is no spread
    to report. Claiming one would rank a measurement against a blank."""
    rows = [
        _row("Claude Code", [_req(0, write=1000, read=9000)]),
        _row("Continue", [_req(0, fresh=1000, visible=False)]),
    ]
    text = render(rows)
    assert "Spread:" not in text
    assert "not logged" in text


def test_a_partially_blind_source_says_so():
    """Some sessions visible, some not — neither 'full' nor 'no cache
    data' is honest, and the row is still rankable on what was seen."""
    row = _row(
        "Mixed",
        [
            _req(0, write=1000, read=9000, session="a"),
            _req(0, fresh=1000, visible=False, session="b"),
        ],
    )
    assert not row.cache_blind
    assert row.confidence == "partial — some sources blind"
    assert row.hit_rate is not None


def test_reads_far_exceeding_writes_are_reported_as_unreliable():
    """A read requires a prior write. Reads at 100x writes means the
    source under-reports writes, not that the prefix never broke."""
    row = _row("Cline", [_req(0, write=10, read=100_000)])
    assert row.confidence == "miss only — writes unreported"


def test_a_complete_source_reports_full_confidence():
    row = _row("Claude Code", [_req(0, write=1000, read=5000)])
    assert row.confidence == "full"


def test_a_real_spread_is_reported():
    rows = [
        _row("Good", [_req(0, write=1000, read=99_000)]),
        _row("Bad", [_req(0, fresh=5000, write=5000, read=1000)]),
    ]
    text = render(rows)
    assert "Spread:" in text
    assert "Good" in text and "Bad" in text


def test_the_empty_state_names_a_command_that_exists(monkeypatch, capsys):
    """The no-results message is the only instruction a new user gets, so
    it is run against the real parser rather than eyeballed. It previously
    named `toka compare --path`, which argparse rejects outright."""
    from toka import cli, compare

    text = render([])
    hints = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("toka ")]
    assert hints, "the empty state must suggest something"

    monkeypatch.setattr(compare, "collect", lambda extra=None: [])
    for hint in hints:
        assert cli.main(hint.split()[1:]) == 0, f"{hint!r} is not a valid invocation"


def test_discovery_returns_only_paths_that_exist():
    """Every candidate is probed before it is offered, so a machine
    without Cline installed does not produce an empty Cline row."""
    for agent, paths in candidates().items():
        for p in paths:
            assert p.exists(), f"{agent} offered a non-existent path: {p}"
