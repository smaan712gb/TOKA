"""The promises, enforced.

Two claims on the front page are not features but invariants: Toka talks
to no network, and it does not modify your logs. A feature that breaks
is a bug; an invariant that breaks is a betrayal, because the user
believed it when deciding to point the tool at private material.

So neither is asserted in prose alone. Both fail the build.
"""

import ast
import socket
from pathlib import Path

import pytest

from toka.adapters import parse_all
from toka.analyze import analyze
from toka.dashboard import render as render_html
from toka.ingest import find_transcripts
from toka.report import render as render_text

SRC = Path(__file__).resolve().parent.parent / "src" / "toka"

# Anything that could open a connection, start a process, or execute
# text. None of it has a place in a tool that reads local files.
FORBIDDEN = {
    "socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests",
    "httpx", "aiohttp", "ftplib", "telnetlib", "smtplib", "poplib",
    "imaplib", "xmlrpc", "webbrowser", "subprocess", "pickle", "marshal",
    "shelve", "ctypes", "multiprocessing",
}


def _modules_imported_by(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_module_imports_anything_that_could_reach_the_network():
    """Checked by reading the source, not by watching it run — an import
    that only fires on an unusual branch is still a network dependency,
    and a test that exercises the happy path would never see it."""
    offenders = {}
    for path in sorted(SRC.rglob("*.py")):
        bad = _modules_imported_by(path) & FORBIDDEN
        if bad:
            offenders[path.name] = sorted(bad)
    assert not offenders, f"forbidden imports: {offenders}"


def test_no_source_file_calls_eval_or_exec():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "compile"}, path.name


def test_a_full_analysis_opens_no_socket(tmp_path, monkeypatch):
    """The belt to the braces above: run the whole pipeline with the
    socket layer booby-trapped."""
    log = tmp_path / "s.jsonl"
    log.write_text(
        '{"model":"gpt-5-codex","usage":{"prompt_tokens":100,'
        '"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":50}}}\n',
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Toka attempted to open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    requests, _used, _skipped = parse_all(find_transcripts(tmp_path))
    report = analyze(requests)
    render_text(report)
    from toka.compare import Row

    render_html([Row(agent="t", report=report, files=1)])


def test_the_source_logs_are_never_modified(tmp_path):
    """Toka is pointed at directories people did not back up."""
    log = tmp_path / "s.jsonl"
    body = (
        '{"model":"claude-sonnet-4-5","message":{"usage":'
        '{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":5}}}\n'
    )
    log.write_text(body, encoding="utf-8")
    before = (log.read_bytes(), log.stat().st_mtime_ns)

    requests, _used, _skipped = parse_all(find_transcripts(tmp_path))
    analyze(requests)

    assert (log.read_bytes(), log.stat().st_mtime_ns) == before


def test_analysis_writes_nothing_anywhere_unless_asked(tmp_path):
    """No cache directory, no index, no sidecar files. The only writes
    Toka makes are the ones named on the command line."""
    log = tmp_path / "s.jsonl"
    log.write_text('{"usage":{"prompt_tokens":10,"completion_tokens":1}}\n', encoding="utf-8")
    before = {p.name for p in tmp_path.iterdir()}

    requests, _used, _skipped = parse_all(find_transcripts(tmp_path))
    analyze(requests)

    assert {p.name for p in tmp_path.iterdir()} == before


# Bodies are built by name rather than listed inline: a two-megabyte
# fixture in a parametrize list becomes a two-megabyte test id, which
# pytest then tries to store in an environment variable.
# Special bytes are built with bytes([...]) rather than written as escape
# sequences, because these literals have already been mangled once in
# transit and a raw NUL in a source file is a syntax error, not a test
# failure — it takes the whole module out.
NUL = bytes([0])
NL = bytes([10])

# Bodies are keyed by name rather than listed in the parametrize call: a
# two-megabyte fixture there becomes a two-megabyte test id, which pytest
# then tries to store in an environment variable.
HOSTILE: dict[str, bytes] = {
    "empty": b"",
    "truncated": b'{"usage": {"prompt_tokens": 10',
    "binary": bytes(range(256)) * 8,
    "embedded-nulls": b'{"a": 1}' + NUL + NUL + NL + b'{"b": 2}' + NL,
    "invalid-utf8": bytes([0xFF, 0xFE]) + b'{"model": "bad"}' + NL,
    "very-long-line": b'{"x": "' + b"a" * 2_000_000 + b'"}' + NL,
    "not-json": b"this is not json at all" + NL + b"nor is this" + NL,
    "json-array": b'[{"usage": {"prompt_tokens": 5, "completion_tokens": 1}}]',
    "null-fields": b'{"usage": {"prompt_tokens": null}}' + NL,
    "wrong-types": b'{"usage": {"prompt_tokens": "lots", "completion_tokens": []}}' + NL,
    "deeply-nested": b'{"a":' * 200 + b"1" + b"}" * 200 + NL,
}


@pytest.mark.parametrize("case", sorted(HOSTILE))
def test_hostile_input_never_takes_down_the_analysis(tmp_path, case):
    """One bad record must cost that record, not the run. These are the
    files a real log directory actually contains."""
    suffix = ".json" if case == "json-array" else ".jsonl"
    (tmp_path / f"{case}{suffix}").write_bytes(HOSTILE[case])
    (tmp_path / "good.jsonl").write_text(
        '{"usage":{"prompt_tokens":100,"completion_tokens":10}}\n', encoding="utf-8"
    )

    requests, _used, _skipped = parse_all(find_transcripts(tmp_path))
    report = analyze(requests)
    render_text(report)  # must not raise either


def test_nonsense_numbers_do_not_produce_nonsense_money(tmp_path):
    """Negative and absurd token counts appear in logs from buggy
    wrappers. They must not turn into a negative bill or a NaN."""
    (tmp_path / "s.jsonl").write_text(
        '{"model":"gpt-5-codex","usage":{"prompt_tokens":-500,"completion_tokens":1e400}}\n'
        '{"model":"gpt-5-codex","usage":{"prompt_tokens":10,"completion_tokens":5}}\n',
        encoding="utf-8",
    )
    requests, _used, _skipped = parse_all(find_transcripts(tmp_path))
    report = analyze(requests)

    text = render_text(report)
    assert "nan" not in text.lower()
    assert "inf" not in text.lower()
    assert report.recoverable >= 0.0


def test_the_same_input_produces_the_same_output(tmp_path):
    """Reports get committed and diffed. Two runs over identical input
    must not differ, or every diff is noise."""
    (tmp_path / "a.jsonl").write_text(
        '{"model":"claude-sonnet-4-5","message":{"usage":'
        '{"input_tokens":10,"cache_creation_input_tokens":50,'
        '"cache_read_input_tokens":900,"output_tokens":5}}}\n',
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        '{"model":"claude-sonnet-4-5","message":{"usage":'
        '{"input_tokens":20,"cache_read_input_tokens":100,"output_tokens":5}}}\n',
        encoding="utf-8",
    )

    def once() -> str:
        requests, _u, _s = parse_all(find_transcripts(tmp_path))
        return render_text(analyze(requests))

    assert once() == once()


# Built with chr() so no escape sequence in this file can be mangled in
# transit and quietly stop testing what it says it tests.
ESC = chr(27)
BEL = chr(7)


def test_log_content_cannot_inject_terminal_escapes(tmp_path):
    """Session ids and model names are printed, and they come out of
    files Toka did not write. An ANSI sequence in one can clear the
    screen or overwrite lines already printed — which is a way to
    fabricate output the reader has every reason to trust."""
    from toka.record import Request

    hostile = Request(
        source="t",
        provider="anthropic",
        session=ESC + "[2J" + ESC + "[31mwiped",
        seq=0,
        timestamp=None,
        model=ESC + "]0;retitled" + BEL + "evil",
        fresh_input=10,
        cache_write_5m=100,
        cache_write_1h=0,
        cache_read=10,
        output=5,
    )
    text = render_text(analyze([hostile]))
    assert ESC not in text
    assert BEL not in text


def test_log_content_cannot_inject_escapes_into_the_dashboard(tmp_path):
    from toka.compare import Row
    from toka.record import Request

    hostile = Request(
        source="t", provider="anthropic", session=ESC + "[31m", seq=0,
        timestamp=None, model="m", fresh_input=1, cache_write_5m=1,
        cache_write_1h=0, cache_read=1, output=1,
    )
    page = render_html(
        [Row(agent=ESC + "[31m<script>alert(1)</script>", report=analyze([hostile]), files=1)]
    )
    assert ESC not in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_stripping_escapes_keeps_the_name_readable():
    """The point is to disarm the control codes, not to redact the name —
    a session you cannot identify is not much use in a report."""
    from toka.report import _plain

    assert _plain(ESC + "[31msession-42") == "[31msession-42"
    assert _plain("ordinary-name") == "ordinary-name"
