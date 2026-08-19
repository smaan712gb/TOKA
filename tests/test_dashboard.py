"""The dashboard, held to the same rules as the text report.

A page for non-technical readers is exactly where a suppressed number is
most likely to quietly come back as a zero — nobody reading it will know
to ask. These tests are that guard.
"""

from toka.analyze import analyze
from toka.compare import Row
from toka.dashboard import render
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
    return Row(agent=agent, report=analyze(requests), files=1)


def test_a_cache_blind_agent_is_named_as_excluded_not_folded_in():
    """The reader cannot see what was left out unless the page says so."""
    html = render(
        [
            _row("Claude Code", [_req(0, write=1000, read=9000)]),
            _row("Continue", [_req(0, fresh=5000, visible=False)]),
        ]
    )
    assert "Excludes Continue" in html
    assert "no cache information" in html


def test_churn_is_withheld_when_writes_are_under_reported():
    """The chart that would read as a reassuring 0%."""
    html = render([_row("Cline", [_req(0, write=10, read=100_000)])])
    assert "Not available" in html
    assert "incomplete" in html
    # and no cause bar is drawn — the method note still explains the term,
    # which is the point: the concept is described, the number withheld.
    assert 'class="cause' not in html
    assert "the cache was still live" not in html


def test_churn_names_the_agents_it_could_not_include():
    """Mixing an under-reporting source into the average would drag churn
    toward zero without the reader ever seeing why."""
    html = render(
        [
            _row("Claude Code", [_req(0, write=9000, read=1000)]),
            _row("Cline", [_req(0, write=10, read=100_000)]),
        ]
    )
    assert "Prefix churn" in html
    assert "Measured on Claude Code only" in html
    assert "Cline under-reports" in html


def test_a_blind_row_reads_not_logged_rather_than_zero():
    html = render([_row("Continue", [_req(0, fresh=5000, visible=False)])])
    assert "not logged" in html
    assert "is not a zero" in html


def test_every_charted_value_is_also_written_out_as_text():
    """Light mode fails 3:1 contrast on two of the four hues, so a value
    reachable only by looking at a colour would be unreadable for some
    readers. Each cost category carries its own printed number."""
    html = render([_row("Claude Code", [_req(0, fresh=100, write=1000, read=9000)])])
    for label in (
        "Paid in full",
        "Written to cache",
        "Read from cache",
        "Output",
    ):
        assert label in html
    assert html.count("<td class=\"num\">$") >= 4


def test_agent_names_are_escaped():
    """Agent names come from filesystem discovery, so they are input."""
    html = render([_row("<script>x</script>", [_req(0, write=1, read=1)])])
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_the_empty_state_names_a_command_that_parses():
    from toka.cli import build_parser

    html = render([])
    assert "No logs found" in html
    assert "toka --compare /your/logs --html report.html" in html
    build_parser().parse_args(
        "--compare /your/logs --html report.html".split()
    )  # raises SystemExit if the suggestion is wrong


def test_the_page_is_self_contained():
    """It is written to disk and opened from a file:// URL, so anything
    fetched from a network would simply be missing."""
    html = render([_row("Claude Code", [_req(0, write=1000, read=9000)])])
    for external in ("http://", "https://", "<script", "@import"):
        assert external not in html
