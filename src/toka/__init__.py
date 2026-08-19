"""Toka — measure and cut what agents actually spend on context.

Three parts:

    toka.analyze   read your logs, price them, find the recoverable waste
    toka.guard     tell you which bytes broke your prompt cache
    toka.repair    fix the breaks that carry no risk; propose the rest

"How much", then "why", then "fix it".

Two more sit on top of those and are reached through the CLI rather than
imported: `toka.compare` runs the analysis across every agent on the
machine, and `toka.dashboard` renders the result as a page for people who
do not read terminals.
"""

from .guard import CheckReport, PrefixGuard, render, sorted_keys_warning
from .repair import RepairResult, repair, repair_safely, verify
from .record import Request

__version__ = "0.9.0"

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
    "__version__",
]
