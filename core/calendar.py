"""NSE trading calendar: holidays, sessions, expiries, Muhurat, event blocks.

Implements §8.3 (session structure, expiry schedule) and the calendar half of
the §3 event-day veto.

Everything here reads from config -- ``config/holidays.yaml`` for holidays and
``config/settings.yaml`` for session times and expiry rules. **No weekday is
hardcoded**: SEBI and the exchanges have changed expiry days more than once,
so the expiry weekday is a config value with an ``as_of`` date (§8.3).
"""

from __future__ import annotations

import datetime as _dt
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml

from core import clock
from core.config import CONFIG_DIR, ConfigError, get_events, get_settings

log = logging.getLogger(__name__)

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_warned_unverified = False


# ---------------------------------------------------------------------------
# Holiday loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_holiday_file(config_dir: str | None = None) -> dict:
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    path = directory / "holidays.yaml"
    if not path.exists():
        raise ConfigError(f"holidays.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def holidays(config_dir: str | None = None) -> frozenset[_dt.date]:
    """All configured NSE trading holidays as dates."""
    global _warned_unverified
    data = _load_holiday_file(config_dir)

    if not data.get("meta", {}).get("verified_against_nse", False) and not _warned_unverified:
        log.warning(
            "holidays.yaml is not yet verified against NSE (meta.verified_against_nse=false). "
            "Run scripts/refresh_holidays.py --check before trusting any backtest date range."
        )
        _warned_unverified = True

    out: set[_dt.date] = set()
    for _year, entries in (data.get("holidays") or {}).items():
        for entry in entries or []:
            out.add(_dt.date.fromisoformat(entry["date"]))
    return frozenset(out)


@lru_cache(maxsize=None)
def muhurat_sessions(config_dir: str | None = None) -> dict[_dt.date, tuple[str, str]]:
    """Diwali Muhurat sessions: ``date -> (start_hhmm, end_hhmm)``."""
    data = _load_holiday_file(config_dir)
    out: dict[_dt.date, tuple[str, str]] = {}
    for entry in (data.get("muhurat") or {}).get("sessions", []) or []:
        out[_dt.date.fromisoformat(entry["date"])] = (entry["start"], entry["end"])
    return out


def muhurat_engines_enabled(config_dir: str | None = None) -> bool:
    """Whether engines may trade the Muhurat session. Default: no."""
    data = _load_holiday_file(config_dir)
    return bool((data.get("muhurat") or {}).get("trade_engines", False))


def reset_calendar_cache() -> None:
    """Drop cached calendar data. Tests call this after writing temp config."""
    global _warned_unverified
    _load_holiday_file.cache_clear()
    holidays.cache_clear()
    muhurat_sessions.cache_clear()
    blocked_event_dates.cache_clear()
    _warned_unverified = False


# ---------------------------------------------------------------------------
# Trading days
# ---------------------------------------------------------------------------


def is_weekend(day: _dt.date) -> bool:
    return day.weekday() >= 5


def is_holiday(day: _dt.date, config_dir: str | None = None) -> bool:
    return day in holidays(config_dir)


def is_trading_day(day: _dt.date, config_dir: str | None = None) -> bool:
    """A normal continuous session. Muhurat days are holidays, not trading days.

    A Muhurat day is a listed holiday with a special evening window; treating
    it as a trading day would let engines schedule 09:15 entries into a closed
    market. Ask :func:`is_muhurat` explicitly if you want that session.
    """
    return not is_weekend(day) and not is_holiday(day, config_dir)


def is_muhurat(day: _dt.date, config_dir: str | None = None) -> bool:
    return day in muhurat_sessions(config_dir)


def next_trading_day(day: _dt.date, config_dir: str | None = None) -> _dt.date:
    """The next session strictly after ``day``."""
    candidate = day + _dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(candidate, config_dir):
            return candidate
        candidate += _dt.timedelta(days=1)
    raise ConfigError(f"No trading day found within 30 days after {day} -- holidays.yaml looks wrong")


def previous_trading_day(day: _dt.date, config_dir: str | None = None) -> _dt.date:
    candidate = day - _dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(candidate, config_dir):
            return candidate
        candidate -= _dt.timedelta(days=1)
    raise ConfigError(f"No trading day found within 30 days before {day}")


def trading_days_between(start: _dt.date, end: _dt.date, config_dir: str | None = None) -> list[_dt.date]:
    """Inclusive list of sessions in ``[start, end]``."""
    out: list[_dt.date] = []
    day = start
    while day <= end:
        if is_trading_day(day, config_dir):
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


def sessions_until(target: _dt.date, frm: _dt.date, config_dir: str | None = None) -> int:
    """Number of sessions from ``frm`` (exclusive) to ``target`` (inclusive).

    Used by the §3 physical-settlement guard: "close by expiry minus 2
    sessions" is ``sessions_until(expiry, today) <= 2``.
    """
    if target < frm:
        return -len(trading_days_between(target, frm, config_dir))
    return len(trading_days_between(frm + _dt.timedelta(days=1), target, config_dir))


# ---------------------------------------------------------------------------
# Session windows (§8.3)
# ---------------------------------------------------------------------------


def session_window(day: _dt.date, name: str = "continuous") -> tuple[_dt.datetime, _dt.datetime]:
    """Start/end datetimes for a named session window on ``day``.

    ``name`` is one of ``preopen``, ``continuous``, ``closing_auction``.
    """
    settings = get_settings()
    mapping = {
        "preopen": ("market.session.preopen_start", "market.session.preopen_matching_end"),
        "continuous": ("market.session.continuous_start", "market.session.continuous_end"),
        "closing_auction": (
            "market.session.closing_auction_start",
            "market.session.closing_auction_end",
        ),
    }
    if name not in mapping:
        raise ValueError(f"Unknown session window {name!r}; expected one of {sorted(mapping)}")
    start_key, end_key = mapping[name]
    return clock.at(day, settings.require(start_key)), clock.at(day, settings.require(end_key))


def is_market_open(ts: Optional[_dt.datetime] = None, config_dir: str | None = None) -> bool:
    """True during the continuous session on a trading day."""
    ts = clock.to_ist(ts) if ts else clock.now_ist()
    if not is_trading_day(ts.date(), config_dir):
        return False
    start, end = session_window(ts.date(), "continuous")
    return start <= ts <= end


def is_preopen(ts: Optional[_dt.datetime] = None, config_dir: str | None = None) -> bool:
    ts = clock.to_ist(ts) if ts else clock.now_ist()
    if not is_trading_day(ts.date(), config_dir):
        return False
    start, end = session_window(ts.date(), "preopen")
    return start <= ts <= end


# ---------------------------------------------------------------------------
# Expiries (§8.3) — weekday comes from config, never from code
# ---------------------------------------------------------------------------


def _configured_weekday(path: str) -> int:
    settings = get_settings()
    name = str(settings.require(path)).lower()
    if name not in _WEEKDAYS:
        raise ConfigError(f"Config {path} = {name!r} is not a weekday name")
    return _WEEKDAYS[name]


def expiry_config_verified() -> bool:
    """§8.3: expiry rules must be verified against NSE/Kite before F&O go-live."""
    return bool(get_settings().get("market.expiry.verified", False))


def weekly_expiry(day: _dt.date, index: str = "NIFTY", config_dir: str | None = None) -> _dt.date:
    """The weekly expiry session on or after ``day`` for ``index``.

    If the configured weekday is a holiday the expiry moves to the previous
    trading day -- the exchange convention.
    """
    weekday = _configured_weekday(f"market.expiry.weekly.{index.upper()}.weekday")
    delta = (weekday - day.weekday()) % 7
    candidate = day + _dt.timedelta(days=delta)
    while not is_trading_day(candidate, config_dir):
        candidate -= _dt.timedelta(days=1)
    if candidate < day:
        # Rolled back past `day` -- take next week instead.
        return weekly_expiry(day + _dt.timedelta(days=7 - delta if delta else 7), index, config_dir)
    return candidate


def monthly_expiry(
    year: int, month: int, kind: str = "index", config_dir: str | None = None
) -> _dt.date:
    """Last configured weekday of the month, rolled back over holidays.

    ``kind`` is ``index`` or ``stock_fno``.
    """
    weekday = _configured_weekday(f"market.expiry.monthly.{kind}.weekday")
    which = str(get_settings().get(f"market.expiry.monthly.{kind}.which", "last")).lower()
    if which != "last":
        raise ConfigError(f"Only 'last' is supported for monthly expiry `which`, got {which!r}")

    if month == 12:
        last_day = _dt.date(year, 12, 31)
    else:
        last_day = _dt.date(year, month + 1, 1) - _dt.timedelta(days=1)

    candidate = last_day - _dt.timedelta(days=(last_day.weekday() - weekday) % 7)
    while not is_trading_day(candidate, config_dir):
        candidate -= _dt.timedelta(days=1)
    return candidate


def is_expiry_day(day: _dt.date, config_dir: str | None = None) -> bool:
    """True if ``day`` is any weekly or monthly expiry.

    §7: expiry days halve all new sizes. §3: the STT-trap guard fires here.
    """
    settings = get_settings()
    for index in (settings.get("market.expiry.weekly", {}) or {}):
        if weekly_expiry(day, index, config_dir) == day:
            return True
    for kind in (settings.get("market.expiry.monthly", {}) or {}):
        if monthly_expiry(day.year, day.month, kind, config_dir) == day:
            return True
    return False


def next_expiry(
    day: _dt.date, index: str = "NIFTY", monthly: bool = False, config_dir: str | None = None
) -> _dt.date:
    """The next expiry on or after ``day``."""
    if monthly:
        candidate = monthly_expiry(day.year, day.month, "index", config_dir)
        if candidate < day:
            year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
            candidate = monthly_expiry(year, month, "index", config_dir)
        return candidate
    return weekly_expiry(day, index, config_dir)


def lot_size(symbol: str) -> int:
    """Configured lot size. Raises if unknown -- guessing a lot size is a bug."""
    settings = get_settings()
    lots = settings.get("market.lot_sizes", {}) or {}
    key = symbol.upper()
    if key not in lots:
        raise ConfigError(
            f"No lot size configured for {symbol!r}. Add it to "
            f"config/settings.yaml `market.lot_sizes` with an as_of date (§8.3) "
            f"-- lot sizes change and must never be guessed."
        )
    return int(lots[key])


# ---------------------------------------------------------------------------
# Event blocks (§3) — events.yaml
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def blocked_event_dates(config_dir: str | None = None) -> dict[_dt.date, str]:
    """``date -> note`` for every blocked event day in ``events.yaml``."""
    events = get_events(config_dir)
    categories: Iterable[str] = events.get("blocking_categories", []) or []
    out: dict[_dt.date, str] = {}
    for category in categories:
        for entry in events.get(category, []) or []:
            try:
                day = _dt.date.fromisoformat(entry["date"])
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"Bad entry in events.yaml `{category}`: {entry!r} ({exc})") from exc
            out[day] = f"{category}: {entry.get('note', '')}".strip().rstrip(":")
    return out


def is_blocked_event_day(day: _dt.date, config_dir: str | None = None) -> bool:
    """§3: no new entries on these days."""
    return day in blocked_event_dates(config_dir)


def event_note(day: _dt.date, config_dir: str | None = None) -> str:
    return blocked_event_dates(config_dir).get(day, "")


def next_session_is_blocked(day: _dt.date, config_dir: str | None = None) -> bool:
    """§6.4: the overnight engine may not hold into a blocked session."""
    return is_blocked_event_day(next_trading_day(day, config_dir), config_dir)


# ---------------------------------------------------------------------------
# Summary used by ``--status``
# ---------------------------------------------------------------------------


def describe(day: Optional[_dt.date] = None, config_dir: str | None = None) -> dict[str, object]:
    """Human-readable calendar state for the CLI status output."""
    day = day or clock.today_ist()
    info: dict[str, object] = {
        "date": day.isoformat(),
        "weekday": day.strftime("%A"),
        "trading_day": is_trading_day(day, config_dir),
        "holiday": is_holiday(day, config_dir),
        "muhurat": is_muhurat(day, config_dir),
        "blocked_event": is_blocked_event_day(day, config_dir),
        "event_note": event_note(day, config_dir),
        "expiry_config_verified": expiry_config_verified(),
    }
    try:
        info["is_expiry_day"] = is_expiry_day(day, config_dir)
        info["next_weekly_expiry"] = next_expiry(day, "NIFTY").isoformat()
        info["next_monthly_expiry"] = next_expiry(day, monthly=True).isoformat()
    except ConfigError as exc:
        info["expiry_error"] = str(exc)
    try:
        info["next_trading_day"] = next_trading_day(day, config_dir).isoformat()
    except ConfigError as exc:  # pragma: no cover - only on a broken holiday file
        info["next_trading_day"] = f"ERROR: {exc}"
    return info
