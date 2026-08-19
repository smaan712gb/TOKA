"""The comparison as a page someone outside engineering can read.

The text report answers a question you already know to ask. This answers
the one a person asks when they open a bill: how much of this did we not
need to spend, and what do we do about it?

Design constraints, in the order they were decided:

  Form before colour.  The lead number is a hero figure, not a chart — a
  one-bar chart of "recoverable" would be a chart of a single value. The
  cost split is part-to-whole, so it is one horizontal stacked bar. The
  rewrite causes have one member that matters — churn is the only
  recoverable one — so that chart is emphasis: churn in the accent hue,
  the rest in de-emphasis grey, rather than four competing colours.

  Colour assigned by slot, in fixed order, never by rank. Orange means
  cache writes in the cost split and stays orange for churn in the causes
  chart, because churn *is* redundant writes. A reader who learns the hue
  once keeps it.

  Every chart has a table twin. The palette clears colour-blind
  separation in both modes, but two light-mode hues land under 3:1
  against the surface, so every value is written out as text as well — no
  number here is reachable only by looking at a colour.

The honesty rules from the text report carry over unchanged. A source
that cannot see its own cache writes has its churn suppressed here too,
and the page names which agents were left out of which chart rather than
quietly averaging them in.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from .compare import Row
from .report import _money, _plain, _tokens

# Validated with the dataviz palette checker against each mode's surface.
# Light passes every check with a contrast WARN on aqua (2.74:1) and
# yellow (2.11:1) — which is why the in-table labels below are mandatory
# rather than decorative. Dark passes clean.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

# Fixed slot order for the cost split. Position is meaning: slot 0 is
# fresh input, 1 writes, 2 reads, 3 output. Dropping an empty category
# must never repaint the others.
COST_SLOTS = [
    ("Paid in full", "fresh context — nothing cached could serve it"),
    ("Written to cache", "the premium paid to store context"),
    ("Read from cache", "context served at a tenth of the price"),
    ("Output", "what the model wrote back"),
]


@dataclass
class Segment:
    label: str
    note: str
    value: float
    slot: int

    def pct(self, total: float) -> float:
        return 100.0 * self.value / total if total else 0.0


def render(rows: list[Row], *, title: str = "Toka") -> str:
    """A self-contained HTML page. No network, no assets, no scripts."""
    if not rows:
        return _page(title, _empty_state())

    body = "".join(
        [
            _headline(rows),
            _kpis(rows),
            _cost_split(rows),
            _causes(rows),
            _leaderboard(rows),
            _method(rows),
        ]
    )
    return _page(title, body)


def _measured(rows: list[Row]) -> list[Row]:
    """Agents whose caching was actually observed."""
    return [r for r in rows if not r.cache_blind]


def _churnable(rows: list[Row]) -> list[Row]:
    """Agents whose cache-write counts can carry a churn analysis.

    A read requires a prior write, so a source reporting reads without
    writes is under-reporting, not running a stable prefix. Averaging it
    in would drag the churn share toward a reassuring number.
    """
    return [r for r in _measured(rows) if r.report.write_accounting_reliable]


def _headline(rows: list[Row]) -> str:
    total = sum(r.report.total_cost for r in rows)
    recoverable = sum(r.report.recoverable for r in rows)
    pct = 100.0 * recoverable / total if total else 0.0

    blind = [r.agent for r in rows if r.cache_blind]
    note = ""
    if blind:
        note = (
            '<p class="caveat">Excludes '
            + html.escape(_plain(", ".join(blind)))
            + " — those logs record no cache information at all, so nothing "
            "about their spend can be called avoidable either way.</p>"
        )

    return f"""
<section class="hero">
  <p class="hero-label">Avoidable spend</p>
  <p class="hero-figure">{html.escape(_money(recoverable))}</p>
  <p class="hero-sub">
    of {html.escape(_money(total))} spent — <strong>{pct:.1f}%</strong> of the
    bill went on context that had already been paid for once.
  </p>
  {note}
</section>
"""


def _kpis(rows: list[Row]) -> str:
    measured = _measured(rows)
    measured_prompt = sum(r.prompt_tokens for r in measured)
    reads = sum(s.cache_read for r in measured for s in r.report.sessions.values())
    read_pct = 100.0 * reads / measured_prompt if measured_prompt else 0.0
    requests = sum(r.requests for r in rows)
    sessions = sum(r.sessions for r in rows)

    tiles = [
        (
            "Total spend",
            _money(sum(r.report.total_cost for r in rows)),
            "across every agent found",
        ),
        (
            "Context processed",
            _tokens(sum(r.prompt_tokens for r in rows)),
            f"over {requests:,} model requests",
        ),
        (
            "Served from cache",
            "not logged" if not measured else f"{read_pct:.1f}%",
            "of context, where it could be measured",
        ),
        (
            "Sessions analysed",
            f"{sessions:,}",
            f"from {len(rows)} agent{'' if len(rows) == 1 else 's'}",
        ),
    ]
    cells = "".join(
        f"""
    <div class="tile">
      <p class="tile-label">{html.escape(label)}</p>
      <p class="tile-value">{html.escape(value)}</p>
      <p class="tile-note">{html.escape(note)}</p>
    </div>"""
        for label, value, note in tiles
    )
    return f'<section class="kpis">{cells}</section>'


def _cost_split(rows: list[Row]) -> str:
    reports = [r.report for r in rows]
    totals = [
        sum(r.cost_fresh_input for r in reports),
        sum(r.cost_cache_write for r in reports),
        sum(r.cost_cache_read for r in reports),
        sum(r.cost_output for r in reports),
    ]
    segs = [
        Segment(COST_SLOTS[i][0], COST_SLOTS[i][1], value, i)
        for i, value in enumerate(totals)
    ]
    total = sum(totals)
    if total <= 0:
        return ""

    # flex-grow carries the proportion exactly; the 2px gap between
    # segments is the surface doing the separating, so no segment needs a
    # border drawn around it.
    bars = "".join(
        f'<div class="seg slot{s.slot}" style="flex-grow:{s.value:.6f}"'
        f' title="{html.escape(s.label)}: {html.escape(_money(s.value))}'
        f' ({s.pct(total):.1f}%)"></div>'
        for s in segs
        if s.value > 0
    )

    body = "".join(
        f"""
      <tr>
        <td><span class="key slot{s.slot}"></span>{html.escape(s.label)}</td>
        <td class="num">{html.escape(_money(s.value))}</td>
        <td class="num">{s.pct(total):.1f}%</td>
        <td class="muted">{html.escape(s.note)}</td>
      </tr>"""
        for s in segs
    )

    return f"""
<section class="card">
  <h2>Where the money went</h2>
  <p class="lede">
    Reading context back out of the cache costs a tenth of putting it in. The
    orange band is what it cost to put it in — pay that once and it is cheap,
    pay it over and over and it is the bill.
  </p>
  <div class="stack">{bars}</div>
  <table class="data">
    <thead>
      <tr>
        <th>Category</th><th class="num">Cost</th>
        <th class="num">Share</th><th>What it is</th>
      </tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</section>
"""


def _causes(rows: list[Row]) -> str:
    usable = _churnable(rows)
    withheld = [r.agent for r in _measured(rows) if r not in usable]

    if not usable:
        who = ", ".join(withheld) or "The agents found"
        return f"""
<section class="card">
  <h2>Why the cache was rewritten</h2>
  <p class="lede">Not available for the agents found.</p>
  <p class="caveat">
    {html.escape(_plain(who))} reports far more cache reads than writes. A read
    requires a prior write, so those write counts are incomplete — and an
    incomplete count would show a reassuring 0% churn rather than an unknown
    one. The chart is withheld rather than guessed at. The avoidable-spend
    figure above does not depend on it and still holds.
  </p>
</section>
"""

    stats = [s for r in usable for s in r.report.sessions.values()]
    writes = sum(s.total_writes for s in stats)
    expiry = sum(s.expiry_writes for s in stats)
    churn = sum(s.excess_writes for s in stats)
    baseline = max(0, writes - expiry - churn)
    if writes <= 0:
        return ""

    causes = [
        (
            "First pass over context",
            baseline,
            False,
            "Unavoidable — every token has to be cached once.",
        ),
        (
            "Cache expired while idle",
            expiry,
            False,
            "Unavoidable — the entry timed out between turns.",
        ),
        (
            "Prefix churn",
            churn,
            True,
            "Avoidable — the cache was still live and was invalidated anyway.",
        ),
    ]
    top = max(value for _, value, _, _ in causes) or 1

    bars = "".join(
        f"""
      <div class="cause{' accent' if hot else ''}">
        <p class="cause-label">{html.escape(label)}</p>
        <div class="cause-track">
          <div class="cause-bar" style="width:{100.0 * value / top:.2f}%"></div>
          <span class="cause-value">{html.escape(_tokens(value))} &middot; {100.0 * value / writes:.1f}%</span>
        </div>
        <p class="cause-note">{html.escape(note)}</p>
      </div>"""
        for label, value, hot, note in causes
    )

    note = ""
    if withheld:
        note = (
            '<p class="caveat">Measured on '
            + html.escape(_plain(", ".join(r.agent for r in usable)))
            + " only. "
            + html.escape(_plain(", ".join(withheld)))
            + " under-reports its cache writes, so including it would "
            "understate churn rather than describe it.</p>"
        )

    return f"""
<section class="card">
  <h2>Why the cache was rewritten</h2>
  <p class="lede">
    Two of these three are the cost of doing business. The third is the one
    worth fixing: context still sitting in the cache, thrown away and paid
    for all over again.
  </p>
  <div class="causes">{bars}</div>
  {note}
</section>
"""


def _leaderboard(rows: list[Row]) -> str:
    body = "".join(
        f"""
      <tr>
        <td>{html.escape(_plain(r.agent))}</td>
        <td class="num">{r.sessions:,}</td>
        <td class="num">{r.requests:,}</td>
        <td class="num">{html.escape(_tokens(r.prompt_tokens))}</td>
        <td class="num">{'not logged' if r.hit_rate is None else f'{r.hit_rate:.1f}%'}</td>
        <td class="num">{'&mdash;' if r.cache_blind else f'{r.recoverable_pct:.1f}%'}</td>
        <td class="muted">{html.escape(r.confidence)}</td>
      </tr>"""
        for r in rows
    )
    return f"""
<section class="card">
  <h2>Agent by agent</h2>
  <p class="lede">
    <strong>Cache hit rate is the comparable number.</strong> It reflects how
    an agent builds its prompts, not what it was asked to do. Token counts and
    cost are <em>not</em> comparable — they say how much each agent was used,
    not how well it was built.
  </p>
  <table class="data">
    <thead>
      <tr>
        <th>Agent</th><th class="num">Sessions</th><th class="num">Requests</th>
        <th class="num">Context</th><th class="num">Cache hit rate</th>
        <th class="num">Avoidable</th><th>What we could see</th>
      </tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
  <p class="caveat">
    <strong>&ldquo;not logged&rdquo; is not a zero.</strong> An agent that
    records no cache fields scores 0% on arithmetic alone, which would rank a
    well-behaved tool last on no evidence. Those rows are excluded from every
    comparison on this page.
  </p>
</section>
"""


def _method(rows: list[Row]) -> str:
    unpriced = sorted({m for r in rows for m in r.report.unpriced_models})
    extra = ""
    if unpriced:
        extra = (
            '<p class="caveat">No published rate card for '
            + html.escape(_plain(", ".join(unpriced)))
            + ". Those requests are counted in the token totals and left out "
            "of every dollar figure — pricing them by analogy with a provider "
            "we do have rates for would be a guess wearing a dollar sign.</p>"
        )
    return f"""
<section class="card method">
  <h2>How to read this</h2>
  <p>
    Both avoidable-spend figures are <strong>lower bounds</strong>.
    &ldquo;Paid in full&rdquo; counts context billed at the full rate after a
    session's first request, priced against the tenth-of-the-price it would
    have cost warm. &ldquo;Prefix churn&rdquo; counts cache writes beyond one
    complete pass over the largest context the session ever held, after
    discounting rewrites that followed an idle gap longer than the cache
    lifetime — those expired, and no amount of good engineering brings them
    back.
  </p>
  <p>
    Real waste is at least this much and probably more. Nothing here is
    estimated to fill a gap: where the logs cannot support a number, the page
    says so instead of printing one.
  </p>
  {extra}
</section>
"""


def _empty_state() -> str:
    return """
<section class="hero">
  <p class="hero-label">Nothing to report</p>
  <p class="hero-figure">No logs found</p>
  <p class="hero-sub">
    Toka looked in the default locations for Claude Code, Cline, Roo Code,
    Continue and Aider, and found nothing to read. Point it at a directory:
    <code>toka --compare /your/logs --html report.html</code>
  </p>
</section>
"""


def _slot_css(colors: list[str]) -> str:
    return "".join(f"  --slot{i}: {c};\n" for i, c in enumerate(colors))


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — where the agent spend went</title>
<style>
:root {{
  color-scheme: light dark;
{_slot_css(LIGHT)}
  --surface: #fcfcfb;
  --card: #ffffff;
  --ink: #1a1a19;
  --ink-2: #55554f;
  --muted: #85857d;
  --rule: #e6e6e1;
  --dim: #c9c9c3;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
{_slot_css(DARK)}
    --surface: #171716;
    --card: #201f1e;
    --ink: #f2f2ef;
    --ink-2: #b0b0a8;
    --muted: #85857d;
    --rule: #302f2d;
    --dim: #46453f;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 40px 24px 72px;
  background: var(--surface);
  color: var(--ink);
  font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
main {{ max-width: 880px; margin: 0 auto; }}
h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 16px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); margin: 0 0 32px; font-size: 14px; }}
.hero {{ margin: 0 0 28px; }}
.hero-label {{
  margin: 0; font-size: 13px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}}
/* Proportional figures: tabular-nums makes a large number look loose. */
.hero-figure {{
  margin: 4px 0 8px; font-size: clamp(48px, 9vw, 68px); font-weight: 600;
  line-height: 1.05; letter-spacing: -0.03em;
}}
.hero-sub {{ margin: 0; font-size: 17px; color: var(--ink-2); max-width: 62ch; }}
.kpis {{
  display: grid; gap: 12px; margin: 0 0 28px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}}
.tile {{
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}}
.tile-label {{ margin: 0; font-size: 12px; color: var(--muted); font-weight: 500; }}
.tile-value {{
  margin: 4px 0 2px; font-size: 26px; font-weight: 600; letter-spacing: -0.02em;
}}
.tile-note {{ margin: 0; font-size: 12px; color: var(--muted); }}
.card {{
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 22px 24px; margin: 0 0 20px;
}}
.lede {{ margin: 0 0 18px; color: var(--ink-2); max-width: 68ch; }}
.caveat {{
  margin: 18px 0 0; padding-top: 14px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--muted); max-width: 72ch;
}}
/* Stacked bar: 24px cap, 4px rounded outer ends, square inside, and a
   2px gap of surface between segments rather than a stroke. */
.stack {{ display: flex; gap: 2px; height: 24px; margin: 0 0 20px; }}
.seg {{ min-width: 2px; border-radius: 1px; }}
.seg:first-child {{ border-radius: 4px 1px 1px 4px; }}
.seg:last-child {{ border-radius: 1px 4px 4px 1px; }}
.slot0 {{ background: var(--slot0); }}
.slot1 {{ background: var(--slot1); }}
.slot2 {{ background: var(--slot2); }}
.slot3 {{ background: var(--slot3); }}
.key {{
  display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 8px;
}}
table.data {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
table.data th {{
  text-align: left; font-weight: 500; font-size: 12px; color: var(--muted);
  padding: 0 12px 8px 0; border-bottom: 1px solid var(--rule);
  text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;
}}
table.data td {{ padding: 9px 12px 9px 0; border-bottom: 1px solid var(--rule); }}
table.data tr:last-child td {{ border-bottom: none; }}
/* tabular-nums only in columns, where digits have to line up. */
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.muted {{ color: var(--muted); font-size: 13px; }}
/* Emphasis: one cause is the story, the other two are context. */
.causes {{ display: grid; gap: 18px; }}
.cause-label {{ margin: 0 0 6px; font-size: 14px; font-weight: 500; }}
.cause-track {{ display: flex; align-items: center; gap: 10px; }}
.cause-bar {{
  height: 14px; border-radius: 0 4px 4px 0; background: var(--dim); min-width: 2px;
}}
.cause.accent .cause-bar {{ background: var(--slot1); }}
.cause.accent .cause-label {{ font-weight: 600; }}
.cause-value {{
  font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
.cause-note {{ margin: 5px 0 0; font-size: 13px; color: var(--muted); }}
.method p {{ color: var(--ink-2); max-width: 72ch; }}
code {{
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 5px; padding: 1px 5px;
}}
footer {{ color: var(--muted); font-size: 12px; margin-top: 28px; }}
@media (max-width: 620px) {{
  table.data {{ display: block; overflow-x: auto; }}
}}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="sub">Where your agents' token spend actually went.</p>
{body}
<footer>
  Generated locally by Toka. It reads local log files and sends nothing
  anywhere.
</footer>
</main>
</body>
</html>
"""
