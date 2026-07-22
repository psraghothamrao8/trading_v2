"""Market-data sources. Kite is one; free public data is another.

:class:`YFinanceSource` implements the same handful of methods
``core.datafeed.DataFeed`` calls on a Kite client -- ``instruments``,
``historical_data``, ``ltp``, ``quote`` -- so the datafeed, the backtester and
the paper runtime work against either without knowing which they have.

What free data can and cannot do
--------------------------------
Verified empirically against live data, not read off a docs page:

* **Daily history: complete.** ``RELIANCE.NS`` returns ~2,850 rows back to
  2015. Good enough for every daily-bar backtest.
* **5-minute history: 60 days, hard stop.** Yahoo refuses anything older. No
  free source has multi-year intraday NSE history.
* **Live quotes: delayed**, typically 15 minutes. Fine for alert-only paper
  running; not fine for a 5-minute-scale entry rule.
* **No circuit bands, no order routing.** The §3 band veto degrades to "no
  band data" and the paper broker is the only broker.

The consequence, stated plainly because it changes what you can trust: §6.1,
§6.2, §6.3, §6.5 and §6.7 all need 5-minute bars across the 2019-2024
walk-forward windows, so **those five cannot be backtested on free data**.
``overnight``, ``pead`` and ``flows`` can. Live paper trading of all eleven
still works, because 5-minute bars are built forward from polling rather than
pulled from history.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from core import clock
from core.config import get_universe, resolve_universe

log = logging.getLogger(__name__)

# Yahoo's cap on intraday history. Not a config value -- it is their limit, and
# pretending it is tunable would invite someone to "fix" it by raising a number.
YF_INTRADAY_MAX_DAYS = 60

# What we actually request. Yahoo rejects a window starting at exactly the
# 60-day boundary ("must be within the last 60 days"), so asking for 60 returns
# nothing at all. One day of margin turns an empty result into a full one.
YF_INTRADAY_SAFE_DAYS = 59

# Index and volatility symbols. NIFTYBEES.NS returns NaN closes on Yahoo, so
# the index proxy is the index itself.
YF_SPECIAL_SYMBOLS = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTYBEES": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "INDIA VIX": "^INDIAVIX",
    "SENSEX": "^BSESN",
}

# Kite interval -> yfinance interval.
INTERVAL_MAP = {
    "minute": "1m",
    "3minute": "5m",       # Yahoo has no 3m; 5m is the nearest honest match
    "5minute": "5m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "1h",
    "day": "1d",
}


class SourceError(RuntimeError):
    """The data source could not satisfy the request."""


def to_yahoo_symbol(symbol: str) -> str:
    """Map an NSE trading symbol to its Yahoo ticker."""
    key = symbol.upper().strip()
    if key in YF_SPECIAL_SYMBOLS:
        return YF_SPECIAL_SYMBOLS[key]
    if key.startswith("^") or key.endswith(".NS"):
        return key
    return f"{key}.NS"


def from_yahoo_symbol(ticker: str) -> str:
    """Inverse of :func:`to_yahoo_symbol`, for labelling results."""
    for nse, yahoo in YF_SPECIAL_SYMBOLS.items():
        if yahoo == ticker:
            return nse
    return ticker[:-3] if ticker.endswith(".NS") else ticker


class YFinanceSource:
    """Free NSE data via Yahoo Finance, shaped like a Kite client.

    Deliberately quacks like ``kiteconnect.KiteConnect`` rather than sitting
    behind a new abstraction: ``DataFeed`` already talks to exactly four
    methods, so matching them means zero changes anywhere downstream.
    """

    name = "free"

    def __init__(self, session: Any = None) -> None:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError(
                "yfinance is not installed. pip install -r requirements.txt"
            ) from exc
        self._yf = yfinance
        self._session = session
        self._instruments_cache: Optional[list[dict[str, Any]]] = None
        self._token_map: dict[int, str] = {}

    # -- instruments ------------------------------------------------------

    def instruments(self, exchange: str = "NSE") -> list[dict[str, Any]]:
        """A synthetic instrument list built from ``universe.yaml``.

        Yahoo has no instrument master. Rather than invent one, this exposes
        exactly the symbols the owner has declared -- which is also the only
        set the engines are allowed to trade.

        F&O (``NFO``) returns empty: free data has no option chain, so §6.8
        cannot propose a contract. That is a real limitation, surfaced as an
        empty list the wheel engine already handles, not a fake chain.
        """
        if exchange.upper() != "NSE":
            log.warning(
                "Free data has no %s instruments (no option chain). §6.8 wheel "
                "cannot propose contracts without Kite.", exchange,
            )
            return []

        if self._instruments_cache is not None:
            return self._instruments_cache

        symbols: list[str] = []
        for name in ("NIFTY50", "NIFTY100", "NIFTY200"):
            try:
                symbols.extend(resolve_universe(name))
            except Exception:
                continue

        # Index proxies, by every name the system might ask for: the config key
        # itself (NIFTY, INDIAVIX), the spot symbol ("NIFTY 50") and the ETF
        # proxy (NIFTYBEES). All of them map to the same Yahoo ticker, and an
        # engine asking by any of those names must resolve.
        universe = get_universe()
        for key, proxy in (universe.get("index_proxies", {}) or {}).items():
            symbols.append(str(key))
            for field in ("spot_symbol", "etf_proxy"):
                value = proxy.get(field)
                if value:
                    symbols.append(str(value))
        symbols.extend(YF_SPECIAL_SYMBOLS)

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for index, symbol in enumerate(symbols, start=1):
            key = symbol.upper()
            if key in seen:
                continue
            seen.add(key)
            token = 900_000 + index
            self._token_map[token] = key
            rows.append({
                "instrument_token": token,
                "tradingsymbol": key,
                "name": key,
                "exchange": "NSE",
                "segment": "NSE",
                "instrument_type": "EQ",
                "lot_size": 1,
                "tick_size": 0.05,
            })
        self._instruments_cache = rows
        return rows

    # -- history ----------------------------------------------------------

    def max_history_days(self, interval: str) -> Optional[int]:
        """How far back this source can serve ``interval``, or None for no limit.

        ``DataFeed`` asks before planning chunks, so a 2019-2026 intraday
        request becomes one clamped call instead of nineteen that each go to
        Yahoo and come back empty.
        """
        if INTERVAL_MAP.get(interval) == "1d":
            return None
        return YF_INTRADAY_SAFE_DAYS

    def historical_data(
        self,
        instrument_token: int,
        from_date: _dt.date,
        to_date: _dt.date,
        interval: str,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Kite-shaped candles: ``[{date, open, high, low, close, volume}]``."""
        symbol = self._token_map.get(instrument_token)
        if symbol is None:
            self.instruments("NSE")
            symbol = self._token_map.get(instrument_token)
        if symbol is None:
            raise SourceError(f"Unknown instrument token {instrument_token}")

        yf_interval = INTERVAL_MAP.get(interval)
        if yf_interval is None:
            raise SourceError(
                f"Free data has no {interval!r} interval. Available: "
                f"{', '.join(sorted(INTERVAL_MAP))}"
            )

        if yf_interval != "1d":
            oldest = clock.today_ist() - _dt.timedelta(days=YF_INTRADAY_SAFE_DAYS)
            if from_date < oldest:
                # Clamp rather than fail: a partial window with a loud warning
                # beats an exception that stops a 200-symbol download, and the
                # caller can see from the row count what it actually got.
                log.warning(
                    "%s %s: free intraday history is capped at %d days. Requested "
                    "from %s, clamping to %s. Backtests over 2019-2024 are NOT "
                    "possible on free data for intraday engines.",
                    symbol, interval, YF_INTRADAY_MAX_DAYS, from_date, oldest,
                )
                from_date = oldest
            if to_date < oldest:
                log.warning(
                    "%s %s: the whole requested window is older than the %d-day "
                    "free-data limit; returning nothing.",
                    symbol, interval, YF_INTRADAY_MAX_DAYS,
                )
                return []

        ticker = to_yahoo_symbol(symbol)
        try:
            frame = self._yf.download(
                ticker,
                start=from_date.isoformat(),
                end=(to_date + _dt.timedelta(days=1)).isoformat(),
                interval=yf_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise SourceError(f"yfinance failed for {ticker}: {exc}") from exc

        return self._to_kite_rows(frame, symbol)

    def _to_kite_rows(self, frame: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []

        # yfinance returns a MultiIndex column frame for single tickers in
        # recent versions; flatten to the plain OHLCV names.
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.droplevel(1, axis=1)
        frame = frame.rename(columns=str.lower)

        required = {"open", "high", "low", "close"}
        if not required <= set(frame.columns):
            raise SourceError(
                f"{symbol}: unexpected yfinance columns {list(frame.columns)}"
            )

        index = frame.index
        if getattr(index, "tz", None) is None:
            index = index.tz_localize("Asia/Kolkata")
        else:
            index = index.tz_convert("Asia/Kolkata")

        rows: list[dict[str, Any]] = []
        for timestamp, row in zip(index, frame.itertuples(index=False)):
            values = row._asdict()
            close = values.get("close")
            if close is None or pd.isna(close):
                # NIFTYBEES.NS and some thin tickers return NaN rows. Dropping
                # them is right: a NaN close silently becomes a zero return.
                continue
            rows.append({
                "date": timestamp.to_pydatetime(),
                "open": float(values.get("open", close)),
                "high": float(values.get("high", close)),
                "low": float(values.get("low", close)),
                "close": float(close),
                "volume": int(values.get("volume") or 0),
            })
        return rows

    # -- quotes -----------------------------------------------------------

    def ltp(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Kite-shaped LTP: ``{"NSE:INFY": {"last_price": 1500.0}}``.

        Free quotes are typically delayed ~15 minutes. Every caller that
        matters logs the source, so a delayed price is never mistaken for a
        live one.
        """
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            symbol = key.split(":", 1)[1] if ":" in key else key
            price = self._last_price(symbol)
            if price is not None:
                out[key] = {"last_price": price}
        return out

    def quote(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Kite-shaped quote. Circuit-band fields are absent, not guessed.

        ``core.datafeed.bands()`` reads ``upper_circuit_limit`` /
        ``lower_circuit_limit``; omitting them makes the §3 band veto
        correctly conclude "no band data for this symbol" rather than
        inventing a band and vetoing real trades.
        """
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            symbol = key.split(":", 1)[1] if ":" in key else key
            price = self._last_price(symbol)
            if price is None:
                continue
            out[key] = {"last_price": price, "source": "free", "delayed": True}
        return out

    def _last_price(self, symbol: str) -> Optional[float]:
        ticker = to_yahoo_symbol(symbol)
        try:
            handle = self._yf.Ticker(ticker)
            fast = getattr(handle, "fast_info", None)
            if fast is not None:
                price = fast.get("lastPrice") if hasattr(fast, "get") else getattr(fast, "last_price", None)
                if price and not pd.isna(price):
                    return float(price)
            history = handle.history(period="1d", interval="1m", auto_adjust=False)
            if history is not None and not history.empty:
                close = history["Close"].dropna()
                if not close.empty:
                    return float(close.iloc[-1])
        except Exception as exc:
            log.warning("free quote failed for %s: %s", ticker, exc)
        return None

    # -- capability reporting ---------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """What this source can actually do. Printed by ``--status``."""
        return {
            "source": "free (yfinance)",
            "daily_history": "full",
            "intraday_history_days": YF_INTRADAY_MAX_DAYS,
            "quotes": "delayed ~15 min",
            "circuit_bands": False,
            "option_chain": False,
            "order_routing": False,
            "backtestable_engines": ["overnight", "pead", "flows"],
            "not_backtestable": ["filings", "sympathy", "pairs", "preopen", "panic_reversion"],
        }


def get_source(name: str = "kite", kite: Any = None) -> Any:
    """Build a data source by name.

    ``kite`` returns the live Kite client (full capability, needs a paid
    subscription and a daily token); ``free`` returns :class:`YFinanceSource`.
    """
    key = (name or "kite").lower()
    if key == "free":
        return YFinanceSource()
    if key == "kite":
        if kite is not None:
            return kite
        from core.broker import KiteBroker

        return KiteBroker().kite
    raise SourceError(f"Unknown data source {name!r}; expected 'kite' or 'free'")
