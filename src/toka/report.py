"""Render an analysis as plain text."""

from __future__ import annotations

import re

from .analyze import Report, worst_sessions

# C0 and C1 control characters, minus tab and newline.
_CONTROL_CHARS = "".join(
    chr(c)
    for c in list(range(0x00, 0x09)) + list(range(0x0B, 0x20)) + list(range(0x7F, 0xA0))
)
_CONTROL = re.compile("[" + re.escape(_CONTROL_CHARS) + "]")


def _plain(text: str) -> str:
    """Strip control characters out of anything read from a log.

    Session ids, model names and agent names come from files Toka did not
    write. ANSI escapes in one of them can clear the screen, retitle the
    terminal, or overwrite lines already printed — which means
    fabricating output the reader has every reason to trust. Names are
    for identifying things, so nothing is lost by removing them.
    """
    return _CONTROL.sub("", str(text))


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def render(report: Report, top: int = 10) -> str:
    lines: list[str] = []
    add = lines.append

    total = report.total_cost
    add("=" * 66)
    add("TOKA — agent spend analysis")
    add("=" * 66)
    add("")
    add(f"  sessions analysed     {len(report.sessions):>12,}")
    add(f"  model requests        {report.total_requests:>12,}")
    add(f"  total cost            {_money(total):>12}")
    add("")

    add("WHERE THE MONEY WENT")
    add("-" * 66)
    rows = [
        ("fresh input (1.0x)", report.cost_fresh_input),
        ("cache writes (1.25-2.0x)", report.cost_cache_write),
        ("cache reads (0.1x)", report.cost_cache_read),
        ("output", report.cost_output),
    ]
    for label, value in rows:
        pct = 100.0 * value / total if total else 0.0
        add(f"  {label:<28} {_money(value):>12}   {pct:5.1f}%")
    add("")

    prompt_tokens = sum(s.prompt_tokens for s in report.sessions.values())
    reads = sum(s.cache_read for s in report.sessions.values())
    writes = sum(s.total_writes for s in report.sessions.values())
    fresh = sum(s.fresh_input for s in report.sessions.values())

    add("CACHE BEHAVIOUR")
    add("-" * 66)
    add(f"  prompt tokens billed  {_tokens(prompt_tokens):>12}")
    add(f"    served from cache   {_tokens(reads):>12}   "
        f"{100.0 * reads / prompt_tokens if prompt_tokens else 0:5.1f}%")
    add(f"    written to cache    {_tokens(writes):>12}   "
        f"{100.0 * writes / prompt_tokens if prompt_tokens else 0:5.1f}%")
    add(f"    missed cache        {_tokens(fresh):>12}   "
        f"{100.0 * fresh / prompt_tokens if prompt_tokens else 0:5.1f}%")
    add("")

    amps = [s.write_amplification for s in report.sessions.values() if s.peak_context]
    if amps:
        amps.sort()
        median = amps[len(amps) // 2]
        add(f"  write amplification   median {median:>5.2f}x   "
            f"worst {max(amps):.2f}x   (1.00x is ideal)")
    add("")
    if not report.write_accounting_reliable:
        add("  rewrites by cause    UNAVAILABLE")
        add(f"    This source reports {_tokens(reads)} cache reads against only")
        add(f"    {_tokens(writes)} writes. Reads require a prior write, so the")
        add("    write counts are incomplete — not evidence of a stable prefix.")
        add("    Churn analysis is suppressed; the cache-miss figure below is")
        add("    unaffected and still holds.")
    else:
        expiry = sum(s.expiry_writes for s in report.sessions.values())
        churn = sum(s.excess_writes for s in report.sessions.values())
        baseline = max(0, writes - expiry - churn)
        add("  rewrites by cause")
        for label, value in (
            ("first pass over context", baseline),
            ("TTL expiry (idle gap)", expiry),
            ("prefix churn", churn),
        ):
            pct = 100.0 * value / writes if writes else 0.0
            add(f"    {label:<24} {_tokens(value):>10}   {pct:5.1f}%")
    add("")

    add("RECOVERABLE")
    add("-" * 66)
    add(f"  cache misses          {_money(report.recoverable_miss):>12}")
    add(f"  redundant writes      {_money(report.recoverable_excess_writes):>12}")
    add(f"  {'total':<21} {_money(report.recoverable):>12}   "
        f"{report.recoverable_pct:5.1f}% of spend")
    add("")

    ranked = worst_sessions(report, top)
    if ranked and ranked[0].cost > 0:
        add(f"WORST {len(ranked)} SESSIONS")
        add("-" * 66)
        add(f"  {'session':<14} {'reqs':>6} {'peak ctx':>10} "
            f"{'amp':>7} {'hit':>6} {'recoverable':>13}")
        for s in ranked:
            add(
                f"  {_plain(s.session)[:12]:<14} {s.requests:>6,} "
                f"{_tokens(s.peak_context):>10} "
                f"{s.write_amplification:>6.2f}x "
                f"{100 * s.cache_hit_rate:>5.1f}% "
                f"{_money(s.cost):>13}"
            )
        add("")

    if report.approximated_models:
        add("NOTE — Anthropic ids priced by prefix match:")
        for model in sorted(report.approximated_models):
            add(f"  {_plain(model)}")
        add("")

    if report.blind_sessions:
        add("NOTE — sources that log no cache accounting:")
        add(f"  {report.blind_sessions} session(s), {_tokens(report.blind_tokens)} prompt tokens")
        add("  Excluded from 'recoverable' — with no cache fields logged, a miss")
        add("  is indistinguishable from an unrecorded hit, and counting every")
        add("  prompt token as waste would report ~100% on a healthy setup.")
        add("")

    if report.unpriced_requests:
        add("NOTE — no rate card, excluded from all dollar figures:")
        add(f"  {report.unpriced_requests:,} requests, "
            f"{_tokens(report.unpriced_tokens)} tokens")
        for model in sorted(report.unpriced_models):
            add(f"  {_plain(model)}")
        add("")

    add("Method: 'cache misses' prices fresh input after a session's first")
    add("request against the 0.1x it would have cost warm. 'redundant writes'")
    add("prices cache writes beyond one full pass over peak context. Both are")
    add("lower bounds — real waste is at least this, likely more.")
    return "\n".join(lines)
