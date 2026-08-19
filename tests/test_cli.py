"""The command line as a stranger meets it.

Two failures here are invisible from inside the project: the report
crashing on a console encoding the developer does not have, and the
first-run message for someone who does not use the one agent Toka reads
by default. Both were real.
"""

import subprocess
import sys
from pathlib import Path

from toka import cli

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "openai.jsonl"


def test_the_report_survives_a_legacy_console_encoding():
    """The report contains em dashes. A redirected stream uses the locale
    encoding, so on a cp437/cp850 Windows box `toka > report.txt` raised
    UnicodeEncodeError and produced nothing at all.

    Run in a subprocess because the encoding is fixed when the stream is
    created — there is no way to reproduce it in-process.
    """
    result = subprocess.run(
        [sys.executable, "-m", "toka.cli", str(FIXTURE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **_clean_env(),
            "PYTHONIOENCODING": "cp437",
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    assert "UnicodeEncodeError" not in result.stderr, result.stderr
    assert result.returncode == 0
    assert "WHERE THE MONEY WENT" in result.stdout


def test_first_run_points_at_the_other_agents_it_can_see(monkeypatch, capsys):
    """Running `toka` with nothing installed used to print one line about
    a path the reader has never heard of."""
    monkeypatch.setattr(cli, "default_transcript_root", lambda: Path("nope-does-not-exist"))
    monkeypatch.setattr(
        "toka.discovery.candidates", lambda: {"Cline": [Path("x")], "Continue": [Path("y")]}
    )

    assert cli.main([]) == 1
    err = capsys.readouterr().err
    assert "Cline" in err and "Continue" in err
    assert "--compare" in err


def test_first_run_with_nothing_at_all_explains_the_way_forward(monkeypatch, capsys):
    """The genuinely empty case — no agent on the machine writes logs.
    Toka can still measure them, but only if the user knows about the
    shim, and this is the only place they would find out."""
    monkeypatch.setattr(cli, "default_transcript_root", lambda: Path("nope-does-not-exist"))
    monkeypatch.setattr("toka.discovery.candidates", dict)

    assert cli.main([]) == 1
    err = capsys.readouterr().err
    assert "toka.log(response)" in err
    assert "toka /path/to/your/logs" in err


def test_an_explicit_bad_path_still_says_so_plainly(monkeypatch, capsys):
    """The guidance is for the no-argument case. Someone who named a path
    wants to know that path is wrong, not a tour of the alternatives."""
    assert cli.main(["definitely-not-here"]) == 1
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "--compare" not in err


def test_the_no_files_message_names_the_formats_it_reads(tmp_path, capsys):
    """It claimed to look only for .jsonl while accepting four suffixes."""
    (tmp_path / "notes.rst").write_text("not a transcript", encoding="utf-8")
    assert cli.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert ".jsonl" in err and ".md" in err


def _clean_env() -> dict:
    import os

    return {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE")
    }
