"""Toka — measure and cut what agents actually spend on context.

Three parts:

    toka.analyze   read your logs, price them, find the recoverable waste
    toka.guard     tell you which bytes broke your prompt cache
    toka.repair    fix the breaks that carry no risk; propose the rest

"How much", then "why", then "fix it".

`toka.log` sits in front of all three, for agents that keep no logs of
their own: call it on a provider response and the analysis has something
to read. Two more sit on top and are reached through the CLI rather than
imported — `toka.compare` runs the analysis across every agent on the
machine, and `toka.dashboard` renders the result as a page for people who
do not read terminals.
"""

from .guard import CheckReport, PrefixGuard, render, sorted_keys_warning
from .logger import log, new_session
from .repair import RepairResult, repair, repair_safely, verify
from .record import Request

__version__ = "0.14.0"

__all__ = [
    "PrefixGuard",
    "CheckReport",
    "Request",
    "render",
    "sorted_keys_warning",
    "repair",
    "repair_safely",
    "verify",
    "RepairResult",
    "log",
    "new_session",
    "__version__",
]
