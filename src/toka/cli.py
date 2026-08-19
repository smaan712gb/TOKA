"""toka analyze — read agent transcripts, report where the money went."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import ADAPTERS, parse_all
from .analyze import analyze
from .ingest import SUFFIXES, find_transcripts
from .report import render


def default_transcript_root() -> Path:
    return Path.home() / ".claude" / "projects"


def make_output_safe() -> None:
    """Make sure the report can be printed anywhere it might be printed.

    The report contains em dashes. On a UTF-8 locale that is a non-issue,
    but Windows consoles still default to cp437/cp850 in places, and a
    redirected stream uses the locale encoding rather than the console's
    — so `toka > report.txt` died with UnicodeEncodeError and produced no
    report at all. A dash degraded to '?' is a worse report; a traceback
    is no report.

    Redirected output is switched to UTF-8, since a file has no reason to
    inherit a legacy codepage. A live console keeps its own encoding and
    only gains the replacing error handler, because reconfiguring the
    encoding underneath a terminal is how you get mojibake.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # not a reconfigurable text stream; nothing to do


def build_parser() -> argparse.ArgumentParser:
    """The command line, as one inspectable object.

    Separated from `main` so the docs test can ask the real parser what
    flags exist instead of trusting the README to have kept up.
    """
    parser = argparse.ArgumentParser(
        prog="toka",
        description="Measure where an agent's token spend actually goes.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="analyse every agent found on this machine, side by side",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="transcript file or directory (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "--project",
        help="only sessions whose transcript directory contains this substring",
    )
    parser.add_argument("--top", type=int, default=10, help="worst-N sessions to list")
    parser.add_argument("--out", type=Path, help="also write the report here")
    parser.add_argument(
        "--scan",
        action="append",
        metavar="PATH",
        default=None,
        help="also look here for logs (repeatable; see also TOKA_SCAN)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="also write a dashboard here, for people who don't read terminals",
    )
    return parser


def _no_default_logs(root: Path) -> int:
    """Explain what to do instead of dead-ending on a missing directory.

    Running `toka` with no arguments is the first thing anyone does, and
    for anyone who does not use Claude Code it used to print one line
    about a path they have never heard of. The machine is already able to
    find their agents — so look, and say what turned up.
    """
    from .discovery import candidates

    print(f"No Claude Code logs at {root}.", file=sys.stderr)
    found = sorted(candidates())
    if found:
        print(
            "\nOther agents were found on this machine:\n  "
            + "\n  ".join(found)
            + "\n\nRun `toka --compare` to analyse them.",
            file=sys.stderr,
        )
        return 1

    print(
        "\nNo supported agent logs were found anywhere either. Toka reads logs"
        "\nagents have already written, so there are two ways forward:"
        "\n"
        "\n  toka /path/to/your/logs     if you know where they are"
        "\n"
        "\nOr, if your code calls a model API directly, it writes no logs at"
        "\nall — record them yourself with one line at the call site:"
        "\n"
        "\n  import toka"
        "\n  toka.log(response)"
        "\n"
        "\nThen run `toka` again.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    make_output_safe()
    args = build_parser().parse_args(argv)

    if args.compare:
        from .compare import collect, render as render_compare

        extra = {"(supplied)": args.path} if args.path else {}
        for i, location in enumerate(args.scan or []):
            extra[f"Scanned {i + 1}" if len(args.scan) > 1 else "Scanned"] = Path(location)
        print("discovering agents...", file=sys.stderr)
        rows = collect(extra)
        print(render_compare(rows))
        if args.html:
            _write_html(args.html, rows)
        return 0

    root = args.path or default_transcript_root()
    if not root.exists():
        if args.path is None:
            return _no_default_logs(root)
        print(f"error: {root} does not exist", file=sys.stderr)
        return 1

    if root.is_file():
        paths = [root]
    else:
        paths = find_transcripts(root)
        if args.project:
            needle = args.project.lower()
            paths = [p for p in paths if needle in str(p).lower()]

    if not paths:
        kinds = ", ".join(SUFFIXES)
        print(
            f"error: no readable files ({kinds}) found under {root}",
            file=sys.stderr,
        )
        return 1

    print(f"scanning {len(paths)} file(s)...", file=sys.stderr)
    requests, used, skipped = parse_all(paths)
    if not requests:
        print(
            "error: no billed model requests found. Supported formats: "
            + ", ".join(a.name for a in ADAPTERS),
            file=sys.stderr,
        )
        return 1

    for name, count in used.most_common():
        print(f"  {name}: {count} file(s)", file=sys.stderr)
    if skipped:
        print(
            f"  skipped: {len(skipped)} file(s) that are not transcripts",
            file=sys.stderr,
        )

    report = analyze(requests)
    text = render(report, top=args.top)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}", file=sys.stderr)

    if args.html:
        # One agent is still a comparison of one — the dashboard reads a
        # list of rows either way, so there is no second rendering path to
        # keep in step with this one.
        from .compare import Row

        agent = used.most_common(1)[0][0] if used else "this agent"
        _write_html(args.html, [Row(agent=agent, report=report, files=len(paths))])
    return 0


def _write_html(path: Path, rows) -> None:
    from .dashboard import render as render_html

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(rows), encoding="utf-8")
    print(f"dashboard written to {path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
