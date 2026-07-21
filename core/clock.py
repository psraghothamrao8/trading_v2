"""Time. Every datetime in this system is timezone-aware Asia/Kolkata.

Implements the timezone half of §0.6 ("timestamps in Asia/Kolkata") and the
session-structure arithmetic of §8.3.

Never call ``datetime.now()`` anywhere else in the codebase -- call
:func:`now_ist`. Tests freeze time with :func:`set_clock`, which is why the
rest of the system must route through this module.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Injectable clock. Production leaves this None (-> real wall clock); tests set
# it to a lambda returning a fixed aware datetime.
_clock_override: Optional[Callable[[], _dt.datetime]] = None


def set_clock(fn: Optional[Callable[[], _dt.datetime]]) -> None:
    """Install (or clear, with ``None``) a clock override. Test-only hook."""
    global _clock_override
    _clock_override = fn


def now_ist() -> _dt.datetime:
    """Current time as a tz-aware datetime in Asia/Kolkata."""
    if _clock_override is not None:
        ts = _clock_override()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=IST)
        return ts.astimezone(IST)
    return _dt.datetime.now(IST)


def today_ist() -> _dt.date:
    """Current date in Asia/Kolkata."""
    return now_ist().date()


def to_ist(ts: _dt.datetime) -> _dt.datetime:
    """Coerce any datetime to tz-aware IST. Naive input is *assumed* IST."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=IST)
    return ts.astimezone(IST)


def parse_hhmm(value: str) -> _dt.time:
    """Parse ``"HH:MM"`` or ``"HH:MM:SS"`` from config into a ``time``.

    Config uses both forms (e.g. ``"14:45"`` and ``"09:06:30"``), so both parse.
    """
    parts = value.split(":")
    if len(parts) == 2:
        return _dt.time(int(parts[0]), int(parts[1]))
    if len(parts) == 3:
        return _dt.time(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Cannot parse time-of-day {value!r}; expected HH:MM[:SS]")


def at(day: _dt.date, hhmm: str) -> _dt.datetime:
    """Combine a date with a config ``"HH:MM[:SS]"`` string into IST."""
    return _dt.datetime.combine(day, parse_hhmm(hhmm), tzinfo=IST)


def is_before(ts: _dt.datetime, hhmm: str) -> bool:
    """True if ``ts`` (IST) falls strictly before the given time-of-day."""
    return to_ist(ts).timetz().replace(tzinfo=None) < parse_hhmm(hhmm)


def is_after(ts: _dt.datetime, hhmm: str) -> bool:
    """True if ``ts`` (IST) falls strictly after the given time-of-day."""
    return to_ist(ts).timetz().replace(tzinfo=None) > parse_hhmm(hhmm)


def within(ts: _dt.datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """True if ``ts`` (IST) falls inside ``[start, end]`` inclusive."""
    t = to_ist(ts).timetz().replace(tzinfo=None)
    return parse_hhmm(start_hhmm) <= t <= parse_hhmm(end_hhmm)


def isoformat(ts: _dt.datetime) -> str:
    """Canonical journal timestamp: ISO-8601 with the IST offset."""
    return to_ist(ts).isoformat(timespec="seconds")
