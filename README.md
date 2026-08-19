# Toka

**Measure where an AI agent's token spend actually goes.**

Agents don't get expensive because models are expensive. They get expensive
because the same context is paid for over and over. Toka reads the logs you
already have and tells you how much of your bill was avoidable — and why.

It runs locally, reads local files, and talks to no network. Nothing leaves
your machine.

```bash
pip install toka
toka
```

That's it. With no arguments it reads `~/.claude/projects` and prints a report.

---

## What it found

Run against 11.1 billion prompt tokens of real agent traffic — 37 sessions,
36,311 model requests, $11,039 of spend:

```
WHERE THE MONEY WENT
  fresh input (1.0x)                  $9.57     0.1%
  cache writes (1.25-2.0x)        $2,642.61    23.8%
  cache reads (0.1x)              $7,334.93    66.0%
  output                          $1,118.04    10.1%

  rewrites by cause
    first pass over context       16.9M     7.3%
    TTL expiry (idle gap)         44.6M    19.3%
    prefix churn                 169.5M    73.4%

RECOVERABLE                      $1,752.39    15.9% of spend
```

Two findings worth stating plainly:

**Cache misses are solved.** A 97.9% hit rate; fresh uncached input was $9.57
out of $11,039. If you were planning to optimize cache misses, don't.

**Prefix churn is not.** Nearly three quarters of all cache *rewrites*
happened while the cache was still live — the prefix was invalidated and
rewritten from scratch, paying 1.25–2× where a read costs 0.1×. Median write
amplification was 5×; the worst session rewrote its context 86 times over.

---

## Finding out *why*

The report tells you churn is costing you. The guard tells you which bytes
did it — the thing you can't get from a dashboard.

```python
from toka import PrefixGuard

guard = PrefixGuard()          # one per conversation

report = guard.check(system=system, tools=tools, messages=messages)
if not report.stable:
    print(report.explain())
```

```
prefix broke at turn 2 — 100% of the cached prefix invalidated (390 chars)
  segment: system[0]
  cause:   looks like a timestamp
  before:  ...Current date: 2026-08-19T10:33:01Z\nYou are a coding assistant...
  after:   ...Current date: 2026-08-19T10:34:15Z\nYou are a coding assistant...
  note:    system renders before messages, so all history was lost too

  Move volatile content out of the system prompt and into a later message,
  after the last cache breakpoint.
```

It runs offline against the request you're about to send — no API call, no
key, nothing leaves the process. Growth by appending is never reported as a
break, so it stays quiet until something actually goes wrong.

It also knows that a change in `tools` costs more than the same change in
`messages`, because tools render first and take the whole prompt with them.

---

## Usage

```bash
toka                              # ~/.claude/projects
toka path/to/logs/                # any directory of transcripts
toka session.jsonl                # a single file
toka --project my-repo            # only matching transcript paths
toka --top 20                     # list the 20 worst sessions
toka --out report.txt             # also write to a file
```

Formats are detected automatically — you never name one.

---

## Supported agents

| Adapter | Covers |
|---|---|
| `claude-code` | Claude Code session transcripts |
| `openai-compatible` | The OpenAI API and every gateway that mirrors its response shape — LiteLLM, OpenRouter, Helicone, Langfuse exports, most homegrown wrappers |
| `gemini` | Google `usageMetadata` |

**A caveat that matters:** only Anthropic bills cache writes separately. On
OpenAI and Google, cached tokens are simply discounted with no write premium,
so prefix churn is *invisible in their billing data*. On those providers Toka
reports the cache-miss signal instead. The token accounting is correct
everywhere; the churn analysis is Anthropic-only until other providers expose
write accounting.

---

## How the numbers are computed

Two mechanisms, both deliberately **lower bounds**. Toka would rather
under-report than sell you a number that doesn't survive scrutiny.

**Cache miss.** Prompt tokens billed at full rate that a warm prefix would
have served at 0.1×. A session's first request is excluded — it has no cache
to hit yet, so its fresh input is unavoidable.

**Prefix churn.** In a well-built session the context grows monotonically and
each token is written to cache once, so cumulative writes should land near the
session's peak context. Writing several times that means the cached prefix
kept breaking. Rewrites that followed an idle gap longer than the cache TTL
are discounted first — those are expiry, not churn, and they aren't
recoverable. What remains is prefix that was still live and got invalidated
anyway.

Without the TTL discount the headline reads ~4 points higher. It's in there
because a number that counts unavoidable re-warming as waste is a number that
falls apart the first time someone checks it.

Models with no rate card are excluded from every dollar figure and reported
separately, rather than priced by analogy against a provider we do have rates
for.

---

## Adding an agent

One file and one registry line. Implement `detect` and `parse`:

```python
class MyAgentAdapter:
    name = "my-agent"
    provider = "openai"

    def detect(self, sample: list[dict]) -> float:
        # Confidence in 0.0–1.0. Return 0.0 for formats you don't own —
        # the registry picks the highest scorer, so guessing hurts.
        return 1.0 if "my_marker" in sample[0] else 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        ...
```

Register it in `src/toka/adapters/__init__.py`. `tests/test_adapters.py`
covers the contract — detection must be exclusive, cached tokens must not be
double-counted, and providers without a write premium must report zero
writes.

```bash
pip install -e ".[dev]"
pytest
```

---

## Status

Early. The measurement layer is real and the numbers hold up; the reporting
is plain text and the adapter set is small. Issues and adapters welcome.

Apache 2.0.
