"""Adapter contract.

An adapter turns one source format into `Request` records. Adding support
for a new agent means one file implementing this protocol and one line in
the registry — no changes anywhere downstream.

`detect` is given the first few parsed JSON objects from a file and returns
a confidence in 0.0-1.0. The registry picks the highest scorer, so an
adapter should return 0.0 when it sees a format it does not own rather
than guessing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ..record import Request

SNIFF_LINES = 200


@runtime_checkable
class Adapter(Protocol):
    name: str
    provider: str

    def detect(self, sample: list[dict]) -> float: ...

    def parse(self, path: Path) -> Iterator[Request]: ...


def sniff(path: Path, limit: int = SNIFF_LINES) -> list[dict]:
    """First `limit` JSON objects in a file, for format detection.

    Handles both JSON Lines and a single top-level JSON array, which is
    what most trace exporters emit.
    """
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(64_000)
    except OSError:
        return out

    stripped = head.lstrip()
    if stripped.startswith("["):
        # Whole-file JSON array — needs a complete read to parse.
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return out
        if isinstance(data, list):
            return [d for d in data[:limit] if isinstance(d, dict)]
        return out

    for line in head.splitlines()[:limit]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def iter_objects(path: Path) -> Iterator[dict]:
    """Every JSON object in a file, JSON Lines or top-level array."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            first = fh.read(1)
            fh.seek(0)
            if first == "[":
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    return
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                return
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def scoped_session(scope: Path, label: str) -> str:
    """A session id that cannot collide with an unrelated file's.

    Most formats do not record a session id, so adapters fall back to
    something from the path. A bare filename is not enough: one machine
    held sixty transcripts all named `audit.jsonl` in different
    directories, and merging them summed sixty sessions' cache writes
    against a single session's peak context. Write amplification is
    measured per session, so that inflates churn — the one number that is
    only worth reporting as a lower bound.

    `scope` is whatever genuinely identifies the conversation: the file
    for formats where one file is one session, the containing directory
    for formats like Cline that split a task across several files. The
    label stays readable; the digest makes it unique.
    """
    try:
        key = str(scope.resolve())
    except OSError:  # unresolvable path — its literal form is still stable
        key = str(scope)
    digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{label}-{digest}" if label else digest


def dig(obj: dict, *path: str):
    """Nested lookup that returns None instead of raising."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def as_int(value) -> int:
    """Coerce a logged token count to a number we can do arithmetic on.

    Logs are written by other people's code, and buggy wrappers emit
    things a token count cannot be. `1e400` parses as float infinity and
    used to raise OverflowError out of the adapter, killing the analysis
    of every other file in the run. A negative count is equally
    impossible.

    Anything unusable becomes zero, which can only ever lower a total.
    That keeps a corrupt record from inventing spend — the failure
    direction the whole tool is built to avoid.
    """
    try:
        number = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0  # None, a string, NaN, or infinity
    return number if number > 0 else 0
