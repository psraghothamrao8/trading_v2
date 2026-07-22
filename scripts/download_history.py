"""Bulk historical download. Implements §5.2.

    python scripts/download_history.py --instruments
    python scripts/download_history.py --universe NIFTY500 --interval day
    python scripts/download_history.py --universe NIFTY200 --interval 5minute
    python scripts/download_history.py --symbols RELIANCE INFY --interval day

NIFTY-500 daily since 2015, and 5-minute as far back as Kite's historical API
allows. Per-request range caps and rate limits come from
``settings.yaml -> datafeed.kite`` (with an ``as_of`` date, §8.3) and are
honoured by :meth:`core.datafeed.DataFeed.historical`.

The download is resumable: ``--update`` fetches only what is missing since each
symbol's last stored candle, so an interrupted run costs minutes, not hours.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` as well as `python -m scripts.<name>`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import datetime as _dt
import logging
import sys
import time
from typing import Sequence

from core import clock
from core.broker import KiteBroker
from core.config import ConfigError, get_settings, resolve_universe
from core.datafeed import DataFeed, DataError
from core.journal import get_journal
from core.logging_config import setup_logging

log = logging.getLogger(__name__)


def resolve_symbols(feed: DataFeed, universe: str | None, symbols: Sequence[str] | None) -> list[str]:
    """Turn CLI arguments into a concrete symbol list."""
    if symbols:
        return [s.upper() for s in symbols]
    if not universe:
        raise ConfigError("Pass either --universe or --symbols")
    if universe.upper() == "NIFTY500":
        return feed.resolve_nifty500()
    return resolve_universe(universe)


def download(
    feed: DataFeed,
    symbols: Sequence[str],
    interval: str,
    start: _dt.date,
    end: _dt.date,
    update_only: bool,
) -> dict[str, int]:
    """Download (or update) each symbol. Returns ``{symbol: rows}``.

    One symbol failing never aborts the run -- 499 of 500 symbols is a usable
    dataset, and the failures are journalled and printed at the end.
    """
    results: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    total = len(symbols)

    for index, symbol in enumerate(symbols, 1):
        try:
            if update_only:
                rows = feed.update(symbol, interval, until=end)
            else:
                frame = feed.historical(symbol, interval, start, end)
                rows = len(frame)
                if rows:
                    feed.save(symbol, interval, frame)
            results[symbol] = rows
            log.info("[%d/%d] %s %s: %d rows", index, total, symbol, interval, rows)
        except (DataError, Exception) as exc:  # noqa: B902 - one bad symbol must not stop the run
            failures.append((symbol, str(exc)))
            log.error("[%d/%d] %s FAILED: %s", index, total, symbol, exc)
            get_journal().record_error(
                "download_history", f"{symbol} {interval}: {exc}", severity="WARNING"
            )

    if failures:
        print(f"\n{len(failures)} symbol(s) failed:", file=sys.stderr)
        for symbol, error in failures[:20]:
            print(f"  {symbol}: {error[:120]}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more (see the journal)", file=sys.stderr)
    return results


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download historical candles (§5.2)")
    parser.add_argument("--universe", help="NIFTY50 | NIFTY100 | NIFTY200 | NIFTY500")
    parser.add_argument("--symbols", nargs="+", help="explicit symbol list")
    parser.add_argument("--interval", default="day",
                        help="day | 5minute | 15minute | 60minute | minute")
    parser.add_argument("--start", help="YYYY-MM-DD (default from settings.yaml)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--update", action="store_true",
                        help="fetch only what is missing since the last stored candle")
    parser.add_argument("--instruments", action="store_true",
                        help="refresh the instruments cache and exit")
    parser.add_argument("--source", choices=["kite", "free"],
                        help="data source; default from settings.yaml datafeed.source")
    args = parser.parse_args(argv)

    setup_logging()

    source_name = args.source or str(settings.get("datafeed.source", "kite"))

    if source_name == "kite":
        broker = KiteBroker()
        if not broker.is_authenticated():
            print("ERROR: not authenticated with Kite (§8.1: tokens expire daily ~07:30 IST).",
                  file=sys.stderr)
            print("       Run `python scripts/morning_auth.py` first,", file=sys.stderr)
            print("       or use --source free for daily data without Kite.", file=sys.stderr)
            return 2
        source = broker.kite
    else:
        from core.sources import YFinanceSource

        source = YFinanceSource()
        caps = source.capabilities()
        print(f"Source: {caps['source']} — daily history {caps['daily_history']}, "
              f"intraday capped at {caps['intraday_history_days']} days")
        if args.interval != "day":
            print()
            print("  WARNING: free intraday history stops at 60 days, so the 2019-2024")
            print("  walk-forward windows cannot be filled. These engines cannot be")
            print(f"  backtested on free data: {', '.join(caps['not_backtestable'])}.")
            print("  Live paper trading of all engines still works.")
            print()

    feed = DataFeed(kite=source)

    if args.instruments:
        for exchange in ("NSE", "NFO"):
            frame = feed.instruments(exchange, refresh=True)
            print(f"{exchange}: {len(frame)} instruments cached")
        return 0

    default_start_key = (
        "datafeed.daily_history_start" if args.interval == "day"
        else "datafeed.intraday_history_start"
    )
    start = _dt.date.fromisoformat(args.start or str(settings.require(default_start_key)))
    end = _dt.date.fromisoformat(args.end) if args.end else clock.today_ist()

    try:
        symbols = resolve_symbols(feed, args.universe, args.symbols)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Downloading {args.interval} candles for {len(symbols)} symbols, {start} -> {end}")
    if not settings.get("datafeed.kite.verified", False):
        print("WARNING: datafeed.kite limits are UNVERIFIED (§8.3). If requests are rejected,")
        print("         check Kite's current per-interval range caps and update settings.yaml.")

    began = time.monotonic()
    results = download(feed, symbols, args.interval, start, end, args.update)
    elapsed = time.monotonic() - began

    rows = sum(results.values())
    empty = [s for s, n in results.items() if n == 0]
    print(f"\nDone in {elapsed / 60:.1f} min: {rows:,} rows across {len(results)} symbols")
    if empty:
        print(f"{len(empty)} symbol(s) returned no rows: {', '.join(empty[:10])}"
              + (" ..." if len(empty) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
