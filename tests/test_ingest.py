"""Walking a real filesystem.

Both failures here were found by pointing Toka at a live machine rather
than at fixtures, and neither is reachable from a tidy test directory
unless you build the mess deliberately.
"""

from pathlib import Path

import pytest

from toka.adapters.base import scoped_session
from toka.ingest import find_transcripts


def test_an_unreadable_entry_does_not_abandon_the_scan(tmp_path, monkeypatch):
    """A broken symlink, a Windows reparse point, a directory with no
    permission — any one of them raised mid-walk and the user got a
    traceback instead of a report. Found on a real machine, where
    `Claude/.../debug/latest` is a reparse point Windows will not stat.
    """
    good = tmp_path / "good.jsonl"
    good.write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}", encoding="utf-8")

    real_stat = Path.stat

    def exploding_stat(self, *args, **kwargs):
        if self.name == "bad.jsonl":
            raise OSError(1920, "The file cannot be accessed by the system")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", exploding_stat)

    found = find_transcripts(tmp_path)
    assert [p.name for p in found] == ["good.jsonl"]


def test_a_missing_root_yields_nothing_rather_than_raising(tmp_path):
    assert find_transcripts(tmp_path / "not-here") == []


def test_expensive_directories_are_never_descended(tmp_path):
    """Pruning before descending, not filtering afterwards — the point is
    to not walk a node_modules tree at all."""
    junk = tmp_path / "node_modules" / "pkg"
    junk.mkdir(parents=True)
    (junk / "data.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "real.jsonl").write_text("{}", encoding="utf-8")

    assert [p.name for p in find_transcripts(tmp_path)] == ["real.jsonl"]


def test_only_readable_formats_are_offered(tmp_path):
    for name in ("a.jsonl", "b.ndjson", "c.json", "d.md", "e.png", "f.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    names = {p.name for p in find_transcripts(tmp_path)}
    assert names == {"a.jsonl", "b.ndjson", "c.json", "d.md"}


def test_same_named_files_in_different_folders_are_different_sessions(tmp_path):
    """The bug this exists to prevent: sixty transcripts all called
    `audit.jsonl` merged into one session, which summed sixty sessions'
    cache writes against one session's peak context and overstated churn
    by three and a half points."""
    a = tmp_path / "one" / "audit.jsonl"
    b = tmp_path / "two" / "audit.jsonl"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("{}", encoding="utf-8")

    assert scoped_session(a, a.stem) != scoped_session(b, b.stem)
    assert scoped_session(a, a.stem).startswith("audit-")


def test_a_session_id_is_stable_across_runs(tmp_path):
    """It ends up in reports and diffs, so it cannot be random."""
    path = tmp_path / "s.jsonl"
    path.write_text("{}", encoding="utf-8")
    assert scoped_session(path, "s") == scoped_session(path, "s")


def test_directory_scoped_sessions_group_their_files(tmp_path):
    """Cline splits one task across several files in one directory, so
    those must stay a single session."""
    task = tmp_path / "task-123"
    task.mkdir()
    a, b = task / "api_conversation_history.json", task / "ui_messages.json"
    for p in (a, b):
        p.write_text("{}", encoding="utf-8")

    assert scoped_session(a.parent, a.parent.name) == scoped_session(
        b.parent, b.parent.name
    )
