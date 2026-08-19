"""Toka — measure and cut what agents actually spend on context.

Two halves:

    toka.analyze   read your logs, price them, find the recoverable waste
    toka.guard     tell you which bytes broke your prompt cache

The first answers "how much"; the second answers "why".
"""

from .guard import CheckReport, PrefixGuard, render, sorted_keys_warning
from .record import Request

__version__ = "0.2.0"

__all__ = [
    "PrefixGuard",
    "CheckReport",
    "Request",
    "render",
    "sorted_keys_warning",
    "__version__",
]
