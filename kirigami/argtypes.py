"""argparse value types shared by the command-line tools in scripts/: a bad value stops the
tool at once with a readable message, before anything is computed.

    p.add_argument("--dt", type=positive, default=1e-4)
    p.add_argument("--steps", type=count, default=40)
"""
from __future__ import annotations

import argparse


def _number(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if v != v or v in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return v


def positive(text: str) -> float:
    """A number > 0 (a length, a time step, a tolerance)."""
    v = _number(text)
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be positive, not {v:g}")
    return v


def nonneg(text: str) -> float:
    """A number >= 0 (an amplitude, a kerf that may be zero)."""
    v = _number(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, not {v:g}")
    return v


def count(text: str) -> int:
    """A whole number >= 1 (steps, frames, iterations)."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {n}")
    return n


def count0(text: str) -> int:
    """A whole number >= 0, where 0 means "all" or "none"."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, not {n}")
    return n
