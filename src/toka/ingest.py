"""Transcript discovery.

Parsing lives in `toka.adapters`; this module only finds candidate files
and re-exports the pieces callers expect.
"""

from __future__ import annotations

import os
from pathlib import Path

from .adapters import parse_all
from .record import Request

# Formats worth sniffing. Everything else is skipped without reading.
SUFFIXES = (".jsonl", ".ndjson", ".json", ".md")

# Directories that never contain transcripts but are expensive to walk.
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}


def find_transcripts(root: Path) -> list[Path]:
    """Candidate transcript files under `root`, newest first.

    Walking a real machine means walking things that cannot be read: a
    broken symlink, a Windows reparse point, a directory the user has no
    permission on, a network mount that has gone away. Any one of them
    used to raise mid-scan and take the whole run with it — a single
    unreadable entry anywhere under the root and there was no report at
    all, only a traceback. Every filesystem call here is allowed to fail
    and be skipped.

    Symlinks are not followed, so a loop cannot hang the scan, and
    uninteresting directories are pruned before descending rather than
    filtered afterwards — which also means never walking into
    `node_modules` in the first place.
    """
    found: list[tuple[float, Path]] = []
    for dirpath, dirnames, filenames in os.walk(
        root, onerror=lambda _err: None, followlinks=False
    ):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in SUFFIXES:
                continue
            path = Path(dirpath) / name
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue  # unreadable: skip it, do not abandon the scan
            found.append((mtime, path))

    found.sort(key=lambda pair: pair[0], reverse=True)
    return [path for _mtime, path in found]


def read_all(paths: list[Path]) -> list[Request]:
    records, _, _ = parse_all(paths)
    return records


__all__ = ["Request", "find_transcripts", "read_all", "parse_all"]
