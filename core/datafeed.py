"""Market data: historical candles (chunked for Kite's limits) and live ticks.

Implements the data half of §5.2, plus §8.4 (price-band status per symbol) and
§9.5 (websocket auto-reconnect with backoff, resubscribe, journalled gap).

Kite caps the date range of a single historical request by interval, and rate
limits the API. :meth:`DataFeed.historical` chunks and sleeps accordingly; the
limits live in ``settings.yaml`` under ``datafeed.kite`` with an ``as_of`` date
because they change.

Storage layout::

    data/parquet/<interval>/<SYMBOL>.parquet     # OHLCV, IST-indexed
    data/instruments.parquet                     # the Kite instrument dump
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.config import ConfigError, data_path, get_settings, get_universe
from core.journal import Journal, get_journal
from core.risk import BandInfo

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Historical or live data could not be obtained."""


@dataclass
class Chunk:
    """One historical request window."""

    start: _dt.date
    end: _dt.date

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Chunk({self.start}..{self.end})"


def plan_chunks(start: _dt.date, end: _dt.date, max_days: int) -> list[Chunk]:
    """Split ``[start, end]`` into windows Kite will accept.

    Kite rejects a request whose range exceeds the per-interval cap, so this
    is not an optimisation -- it is the difference between data and an error.
    """
    if max_days <= 0:
        raise ValueError(f"max_days must be positive, got {max_days}")
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    chunks: list[Chunk] = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + _dt.timedelta(days=max_days - 1), end)
        chunks.append(Chunk(cursor, stop))
        cursor = stop + _dt.timedelta(days=1)
    return chunks


class DataFeed:
    """Historical candles, the instruments cache, and price-band status.

    ``kite`` is injectable so tests never touch the network. In production the
    orchestrator passes ``KiteBroker.kite``.
    """

    def __init__(
        self,
        kite: Any = None,
        journal: Journal | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.kite = kite
        self.journal = journal or get_journal()
        self._sleep = sleeper or time.sleep
        self._instruments: Optional[pd.DataFrame] = None
        self._token_cache: dict[str, int] = {}

    # -- instruments ------------------------------------------------------

    def instruments(self, exchange: str = "NSE", refresh: bool = False) -> pd.DataFrame:
        """The Kite instrument dump, cached to Parquet.

        Downloading this on every run is wasteful (it is a multi-MB CSV) and
        pointless -- it changes at most daily.
        """
        cache = data_path(f"instruments_{exchange.lower()}.parquet")
        if not refresh and self._instruments is not None:
            return self._instruments
        if not refresh and cache.exists():
            age = clock.now_ist() - clock.to_ist(
                _dt.datetime.fromtimestamp(cache.stat().st_mtime)
            )
            if age < _dt.timedelta(days=1):
                self._instruments = pd.read_parquet(cache)
                return self._instruments

        if self.kite is None:
            if cache.exists():
                log.warning("No Kite session; using a stale instruments cache at %s", cache)
                self._instruments = pd.read_parquet(cache)
                return self._instruments
            raise DataError(
                "No Kite session and no instruments cache. Run "
                "`python scripts/morning_auth.py` then "
                "`python scripts/download_history.py --instruments`."
            )

        frame = pd.DataFrame(self.kite.instruments(exchange))
        frame.to_parquet(cache, index=False)
        self._instruments = frame
        log.info("Instruments cache refreshed for %s: %d rows", exchange, len(frame))
        return frame

    def instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Resolve a trading symbol to its Kite instrument token."""
        key = f"{exchange}:{symbol}"
        if key in self._token_cache:
            return self._token_cache[key]
        frame = self.instruments(exchange)
        match = frame[frame["tradingsymbol"] == symbol.upper()]
        if match.empty:
            raise DataError(f"{symbol} not found in the {exchange} instrument list")
        token = int(match.iloc[0]["instrument_token"])
        self._token_cache[key] = token
        return token

    def resolve_nifty500(self) -> list[str]:
        """Resolve the NIFTY-500 universe (§6.2, §6.6) from real instruments.

        ``universe.yaml`` deliberately does not hand-maintain 500 tickers.
        When an explicit list is configured it wins; otherwise this filters the
        NSE equity instrument dump to EQ-series names.
        """
        universe = get_universe()
        explicit = list(universe.get("nifty500_explicit", []) or [])
        if explicit:
            return explicit
        if not universe.get("nifty500_from_instruments", False):
            raise ConfigError(
                "NIFTY500 is not configured: set `nifty500_from_instruments: true` "
                "or populate `nifty500_explicit` in config/universe.yaml"
            )
        frame = self.instruments("NSE")
        equities = frame[
            (frame.get("segment") == "NSE") & (frame.get("instrument_type") == "EQ")
        ]
        return sorted(equities["tradingsymbol"].astype(str).str.upper().unique().tolist())

    # -- historical -------------------------------------------------------

    def max_days_for(self, interval: str) -> int:
        """Kite's per-request range cap for an interval, from config (§8.3)."""
        limits = self.settings.get("datafeed.kite.max_days_per_request", {}) or {}
        if interval not in limits:
            raise ConfigError(
                f"No max_days_per_request configured for interval {interval!r}. "
                f"Add it to config/settings.yaml `datafeed.kite.max_days_per_request` "
                f"with an as_of date -- guessing Kite's limit produces silent gaps."
            )
        return int(limits[interval])

    def historical(
        self,
        symbol: str,
        interval: str,
        start: _dt.date,
        end: _dt.date,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """Download candles, chunked and rate-limited, as an IST-indexed frame.

        A chunk that fails is logged and skipped rather than aborting the whole
        symbol -- a partial history with a journalled gap beats no history and
        no record of why.
        """
        if self.kite is None:
            raise DataError(
                f"No Kite session; cannot download {symbol}. Run scripts/morning_auth.py first."
            )
        token = self.instrument_token(symbol, exchange)
        max_days = self.max_days_for(interval)
        pause = float(self.settings.get("datafeed.kite.sleep_between_requests_sec", 0.4))

        frames: list[pd.DataFrame] = []
        chunks = plan_chunks(start, end, max_days)
        for index, chunk in enumerate(chunks):
            try:
                rows = self.kite.historical_data(
                    instrument_token=token,
                    from_date=chunk.start,
                    to_date=chunk.end,
                    interval=interval,
                )
            except Exception as exc:
                log.error("historical %s %s %s failed: %s", symbol, interval, chunk, exc)
                self.journal.record_error(
                    "datafeed",
                    f"{symbol} {interval} {chunk.start}..{chunk.end}: {exc}",
                    severity="WARNING",
                )
                continue
            if rows:
                frames.append(pd.DataFrame(rows))
            if index < len(chunks) - 1:
                self._sleep(pause)

        if not frames:
            return _empty_ohlcv()
        return _normalise_ohlcv(pd.concat(frames, ignore_index=True))

    # -- parquet store ----------------------------------------------------

    def parquet_path(self, symbol: str, interval: str) -> Path:
        return data_path("parquet", interval, f"{symbol.upper()}.parquet")

    def save(self, symbol: str, interval: str, frame: pd.DataFrame) -> Path:
        """Persist candles, merging with anything already stored."""
        path = self.parquet_path(symbol, interval)
        if path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame.to_parquet(path)
        return path

    def load(
        self,
        symbol: str,
        interval: str,
        start: _dt.date | None = None,
        end: _dt.date | None = None,
    ) -> pd.DataFrame:
        """Read stored candles. Missing data raises rather than returning empty.

        A silently empty frame becomes a backtest with zero trades, which reads
        as "the strategy found nothing" instead of "the data was never
        downloaded" (§0.8).
        """
        path = self.parquet_path(symbol, interval)
        if not path.exists():
            raise DataError(
                f"No {interval} data for {symbol} at {path}. Run "
                f"`python scripts/download_history.py --symbols {symbol} --interval {interval}`."
            )
        frame = pd.read_parquet(path)
        if start is not None:
            frame = frame[frame.index.date >= start]
        if end is not None:
            frame = frame[frame.index.date <= end]
        return frame

    def has(self, symbol: str, interval: str) -> bool:
        return self.parquet_path(symbol, interval).exists()

    def update(
        self, symbol: str, interval: str, until: _dt.date | None = None, exchange: str = "NSE"
    ) -> int:
        """Fetch only what is missing since the last stored candle. Returns rows added."""
        until = until or clock.today_ist()
        path = self.parquet_path(symbol, interval)
        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty:
                start = existing.index.max().date() + _dt.timedelta(days=1)
            else:
                start = _default_start(interval)
        else:
            start = _default_start(interval)
        if start > until:
            return 0
        frame = self.historical(symbol, interval, start, until, exchange)
        if frame.empty:
            return 0
        self.save(symbol, interval, frame)
        return len(frame)

    # -- quotes and bands (§8.4) -----------------------------------------

    def ltp(self, symbols: Sequence[str]) -> dict[str, float]:
        if not symbols or self.kite is None:
            return {}
        keys = [s if ":" in s else f"NSE:{s}" for s in symbols]
        data = self.kite.ltp(keys)
        return {k.split(":", 1)[1]: float(v["last_price"]) for k, v in data.items()}

    def bands(self, symbols: Sequence[str]) -> dict[str, BandInfo]:
        """Per-symbol circuit-band status, for the §3 kernel veto.

        Kite's quote payload carries ``lower_circuit_limit`` /
        ``upper_circuit_limit``. Symbols without bands (F&O stocks) come back
        with ``upper``/``lower`` as None, which the kernel reads as "no band
        veto applies".
        """
        if not symbols or self.kite is None:
            return {}
        keys = [s if ":" in s else f"NSE:{s}" for s in symbols]
        try:
            quotes = self.kite.quote(keys)
        except Exception as exc:
            log.error("band fetch failed: %s", exc)
            self.journal.record_error("datafeed", f"band fetch failed: {exc}", severity="WARNING")
            return {}

        out: dict[str, BandInfo] = {}
        for key, payload in quotes.items():
            symbol = key.split(":", 1)[1]
            last = payload.get("last_price")
            if last is None:
                continue
            upper = payload.get("upper_circuit_limit") or None
            lower = payload.get("lower_circuit_limit") or None
            out[symbol] = BandInfo(
                last_price=float(last),
                upper=float(upper) if upper else None,
                lower=float(lower) if lower else None,
            )
        return out


# ---------------------------------------------------------------------------
# Live ticks (§9.5)
# ---------------------------------------------------------------------------


class TickStream:
    """Kite websocket wrapper: auto-reconnect, resubscribe, journalled gaps.

    §9.5 is explicit that a dropped websocket must reconnect with backoff,
    resubscribe to everything it had, and *journal the gap* -- because a silent
    gap looks exactly like a quiet market, and engines would happily trade on
    stale state.
    """

    def __init__(
        self,
        kite_ticker: Any = None,
        journal: Journal | None = None,
        alerts: Any = None,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self.ticker = kite_ticker
        self.journal = journal or get_journal()
        self._alerts = alerts
        self.max_backoff = max_backoff_seconds

        self._subscribed: set[int] = set()
        self._token_to_symbol: dict[int, str] = {}
        self._handlers: list[Callable[[Any], None]] = []
        self._connected = False
        self._disconnected_at: Optional[_dt.datetime] = None
        self._reconnect_attempts = 0
        self._lock = threading.RLock()

    @property
    def alerts(self) -> Any:
        if self._alerts is None:
            from live.alerts import get_alerts

            self._alerts = get_alerts()
        return self._alerts

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def subscribed_tokens(self) -> set[int]:
        return set(self._subscribed)

    def on_tick(self, handler: Callable[[Any], None]) -> None:
        self._handlers.append(handler)

    def subscribe(self, tokens: Iterable[int], symbols: dict[int, str] | None = None) -> None:
        """Subscribe, remembering the set so a reconnect can restore it."""
        with self._lock:
            new = set(tokens) - self._subscribed
            self._subscribed |= set(tokens)
            if symbols:
                self._token_to_symbol.update(symbols)
        if new and self.ticker is not None and self._connected:
            self.ticker.subscribe(list(new))
            self.ticker.set_mode(self.ticker.MODE_FULL, list(new))

    def unsubscribe(self, tokens: Iterable[int]) -> None:
        with self._lock:
            self._subscribed -= set(tokens)
        if self.ticker is not None and self._connected:
            self.ticker.unsubscribe(list(tokens))

    # -- lifecycle callbacks ---------------------------------------------

    def handle_connect(self) -> None:
        """Called on (re)connect: resubscribe and close out any journalled gap."""
        with self._lock:
            self._connected = True
            self._reconnect_attempts = 0
            tokens = list(self._subscribed)
            gap_start = self._disconnected_at
            self._disconnected_at = None

        if tokens and self.ticker is not None:
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
            log.info("Websocket connected; resubscribed to %d tokens", len(tokens))

        if gap_start is not None:
            gap = (clock.now_ist() - gap_start).total_seconds()
            self.journal.record_error(
                "websocket",
                f"Reconnected after a {gap:.0f}s data gap "
                f"({clock.isoformat(gap_start)} .. {clock.isoformat(clock.now_ist())})",
                severity="WARNING",
                gap_seconds=gap,
                tokens=len(tokens),
            )
            self.alerts.error(
                "websocket",
                f"Tick feed reconnected after a {gap:.0f}s gap. "
                f"Positions opened during the gap may have stale stops.",
                severity="WARNING",
            )

    def handle_close(self, code: Any = None, reason: Any = None) -> None:
        """Called on disconnect: start the gap clock."""
        with self._lock:
            if self._connected:
                self._disconnected_at = clock.now_ist()
            self._connected = False
        log.warning("Websocket closed (code=%s reason=%s)", code, reason)

    def handle_ticks(self, ticks: Sequence[dict[str, Any]]) -> None:
        from core.types import Tick

        for raw in ticks:
            token = raw.get("instrument_token")
            tick = Tick(
                symbol=self._token_to_symbol.get(token, str(token)),
                last_price=float(raw.get("last_price", 0.0)),
                timestamp=clock.to_ist(raw.get("timestamp") or clock.now_ist()),
                volume=int(raw.get("volume_traded") or 0),
                oi=int(raw.get("oi") or 0),
                instrument_token=token,
            )
            for handler in self._handlers:
                try:
                    handler(tick)
                except Exception as exc:
                    log.error("tick handler failed for %s: %s", tick.symbol, exc, exc_info=True)

    def backoff_delay(self) -> float:
        """Exponential backoff, capped. Used by the reconnect loop."""
        with self._lock:
            self._reconnect_attempts += 1
            attempts = self._reconnect_attempts
        return min(2.0 ** attempts, self.max_backoff)

    def start(self) -> None:
        """Wire the callbacks onto a real KiteTicker and connect."""
        if self.ticker is None:
            raise DataError("TickStream.start() needs a KiteTicker instance")
        self.ticker.on_connect = lambda ws, response: self.handle_connect()
        self.ticker.on_close = lambda ws, code, reason: self.handle_close(code, reason)
        self.ticker.on_error = lambda ws, code, reason: self.handle_close(code, reason)
        self.ticker.on_ticks = lambda ws, ticks: self.handle_ticks(ticks)
        self.ticker.connect(threaded=True, disable_ssl_verification=False)

    def stop(self) -> None:
        if self.ticker is not None:
            try:
                self.ticker.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._connected = False


# ---------------------------------------------------------------------------
# frame helpers
# ---------------------------------------------------------------------------


def _empty_ohlcv() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="date", tz=clock.IST)
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=index)


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Kite returns naive-ish dicts; normalise to an IST-indexed OHLCV frame."""
    if frame.empty:
        return _empty_ohlcv()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(clock.IST)
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in OHLCV_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[OHLCV_COLUMNS]


def _default_start(interval: str) -> _dt.date:
    settings = get_settings()
    key = "daily_history_start" if interval == "day" else "intraday_history_start"
    return _dt.date.fromisoformat(str(settings.require(f"datafeed.{key}")))


# ---------------------------------------------------------------------------
# indicators used by more than one engine
# ---------------------------------------------------------------------------


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range. Used by §6.1, §6.5, §6.6."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(frame: pd.DataFrame) -> pd.Series:
    """Session VWAP. §6.1 trails by this and tightens stops to it.

    Resets each session -- a VWAP carried across days is meaningless.
    """
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].fillna(0)
    day = frame.index.date
    cumulative_pv = (typical * volume).groupby(day).cumsum()
    cumulative_volume = volume.groupby(day).cumsum()
    return cumulative_pv / cumulative_volume.replace(0, pd.NA)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. §6.6 uses 20-EMA daily, §6.7 10-EMA 15m."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average. §6.4 gates on the 200-DMA."""
    return series.rolling(window=period, min_periods=period).mean()


def percentile_rank(series: pd.Series, value: float) -> float:
    """Percentile of ``value`` within ``series`` (0-100).

    §6.8 gates on the India VIX 1-year percentile; §6.9 on a 3-year FII
    positioning percentile.
    """
    clean = series.dropna()
    if clean.empty:
        raise ValueError("Cannot compute a percentile of an empty series")
    return float((clean <= value).sum()) / len(clean) * 100.0


def sessions_from(frame: pd.DataFrame) -> list[_dt.date]:
    """Distinct trading dates present in an intraday frame."""
    return sorted({ts.date() for ts in frame.index})


def first_n_minutes(frame: pd.DataFrame, day: _dt.date, minutes: int) -> pd.DataFrame:
    """The first ``minutes`` of a session -- §6.7's 'first-15-min high'."""
    start, _ = trading_calendar.session_window(day, "continuous")
    end = start + _dt.timedelta(minutes=minutes)
    return frame[(frame.index >= start) & (frame.index < end)]
