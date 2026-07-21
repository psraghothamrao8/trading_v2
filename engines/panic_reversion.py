"""panic_reversion.py — Implements §6.7.

WHY (from the spec): India's market is structurally dip-bought (SIP flows
arrive monthly regardless of mood). Sentiment crashes without a fundamental
filing snap back with high frequency -- but only sentiment crashes, hence the
mandatory no-negative-filing cross-check.

That cross-check is not optional and is not a heuristic. A stock down 6% with
a fraud disclosure is not oversold; it is repriced. Removing that query turns
this engine into a knife-catcher.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.datafeed import ema, first_n_minutes
from core.types import EntryType, Regime, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class PanicCandidate:
    """A symbol that crashed yesterday and is eligible for a reclaim entry."""

    symbol: str
    trigger: str                # "index" | "stock"
    crash_date: _dt.date
    move_pct: float
    session_low: float


class PanicReversionEngine(Engine):
    """§6.7. Enabled only in the PANIC regime (§7)."""

    name = "panic_reversion"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.watchlist: dict[str, PanicCandidate] = {}

    @property
    def allowed_regimes(self) -> set[Regime]:
        return {Regime(r) for r in self.config.get("regimes", ["PANIC"])}

    # -- triggers ---------------------------------------------------------

    def detect(self, ctx: Context) -> list[PanicCandidate]:
        """§6.7 triggers A and B, with the mandatory filing cross-check."""
        candidates: list[PanicCandidate] = []

        # Trigger A: NIFTY day <= -3%.
        index_symbol = self._index_symbol()
        index_move = self._day_move_pct(ctx, index_symbol)
        index_threshold = float(self.config.require("trigger_a_index_drop_pct"))
        index_panicked = index_move is not None and index_move <= index_threshold
        if index_panicked:
            log.info("panic trigger A: index %s at %.2f%%", index_symbol, index_move)

        # Trigger B: a NIFTY-100 stock <= -6% with NO negative filing that day.
        stock_threshold = float(self.config.require("trigger_b_stock_drop_pct"))
        journal = ctx.the_journal()
        require_clean = bool(self.config.get("require_no_negative_filing", True))

        for symbol in self._trigger_b_universe():
            move = self._day_move_pct(ctx, symbol)
            if move is None or move > stock_threshold:
                continue

            # §6.7 MANDATORY cross-check. Do not remove.
            if require_clean and journal.has_negative_filing(symbol, ctx.today.isoformat()):
                log.info(
                    "panic: skipping %s (%.2f%%) -- it has a MATERIAL_NEGATIVE filing "
                    "today, so this is a repricing, not a sentiment crash (§6.7)",
                    symbol, move,
                )
                continue

            bars = ctx.bars_for(symbol, "day")
            if bars is None or bars.empty:
                continue
            candidates.append(
                PanicCandidate(
                    symbol=symbol, trigger="stock", crash_date=ctx.today,
                    move_pct=move, session_low=float(bars["low"].iloc[-1]),
                )
            )

        if index_panicked:
            bars = ctx.bars_for(index_symbol, "day")
            if bars is not None and not bars.empty:
                candidates.append(
                    PanicCandidate(
                        symbol=index_symbol, trigger="index", crash_date=ctx.today,
                        move_pct=index_move, session_low=float(bars["low"].iloc[-1]),
                    )
                )
        return candidates

    def _trigger_b_universe(self) -> list[str]:
        from core.config import resolve_universe

        return resolve_universe(str(self.config.require("trigger_b_universe")))

    def _index_symbol(self) -> str:
        from core.config import get_universe

        return str(get_universe().get("index_proxies.NIFTY.etf_proxy", "NIFTYBEES"))

    def _day_move_pct(self, ctx: Context, symbol: str) -> Optional[float]:
        bars = ctx.bars_for(symbol, "day")
        if bars is None or len(bars) < 2:
            return None
        previous = float(bars["close"].iloc[-2])
        if previous <= 0:
            return None
        current = float(bars["close"].iloc[-1])
        return (current - previous) / previous * 100.0

    # -- entries ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.7: entry next session 09:30-10:30 on reclaim of the first-15-min high."""
        if not self.auto_trade:
            return []
        if ctx.regime not in self.allowed_regimes and ctx.regime is not Regime.NA:
            return []

        # Yesterday's crashes become today's watchlist.
        for candidate in self.detect(ctx):
            self.watchlist[candidate.symbol] = candidate

        if not self.within_window(ctx, "entry_window"):
            return []

        max_concurrent = int(self.config.require("max_concurrent"))
        signals: list[Signal] = []
        consumed: list[str] = []

        for symbol, candidate in self.watchlist.items():
            # Only trade the session AFTER the crash.
            if candidate.crash_date >= ctx.today:
                continue
            if ctx.position_for(symbol, self.name) is not None:
                continue
            if self.concurrent_positions(ctx) + len(signals) >= max_concurrent:
                break

            intraday = ctx.bars_for(symbol, "5minute")
            if intraday is None or intraday.empty:
                continue

            opening_range = first_n_minutes(intraday, ctx.today, 15)
            if opening_range.empty:
                continue
            opening_high = float(opening_range["high"].max())

            price = ctx.price(symbol)
            if price is None or price <= opening_high:
                continue

            # §6.7 stop: below the session low.
            today_bars = intraday[intraday.index.date == ctx.today]
            session_low = float(today_bars["low"].min()) if not today_bars.empty else candidate.session_low
            stop = round(session_low * 0.999, 2)
            if stop >= price:
                continue

            risk = price - stop
            signals.append(
                self.signal(
                    symbol, Side.BUY,
                    stop=stop,
                    reference_price=price,
                    targets=(price + risk,),
                    ttl=TTL.INTRADAY,
                    entry_type=EntryType.MARKET,
                    reason=(
                        f"panic reversion: {candidate.trigger} crash {candidate.move_pct:.1f}% "
                        f"on {candidate.crash_date}, reclaimed the first-15-min high "
                        f"{opening_high:,.2f}, no negative filing"
                    ),
                    segment=Segment.EQUITY_INTRADAY.value,
                    crash_date=candidate.crash_date.isoformat(),
                    crash_move_pct=candidate.move_pct,
                    opening_high=opening_high,
                    book_fraction_at_1r=float(self.config.get("book_fraction_at_1r", 0.5)),
                )
            )
            consumed.append(symbol)

        for symbol in consumed:
            self.watchlist.pop(symbol, None)
        self._expire_watchlist(ctx)
        return signals

    def _expire_watchlist(self, ctx: Context) -> None:
        """A crash more than one session old is no longer the setup."""
        stale = [
            symbol for symbol, candidate in self.watchlist.items()
            if len(trading_calendar.trading_days_between(candidate.crash_date, ctx.today)) > 2
        ]
        for symbol in stale:
            self.watchlist.pop(symbol, None)

    # -- management -------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.7: book 50% at +1R, trail the rest by the 15-minute 10-EMA."""
        trail_config = self.config.section("trail")
        period = int(trail_config.require("period"))
        interval = str(trail_config.require("timeframe"))

        signals: list[Signal] = []
        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            stop = position.stop
            if stop is not None and not position.meta.get("booked_1r"):
                risk = abs(position.average_price - stop)
                target = position.average_price + risk
                if risk > 0 and price >= target:
                    fraction = float(self.config.get("book_fraction_at_1r", 0.5))
                    position.meta["booked_1r"] = True
                    signals.append(self._exit(
                        position, price,
                        max(int(abs(position.quantity) * fraction), 1),
                        f"+1R scale-out ({fraction:.0%})",
                    ))

            bars = ctx.bars_for(position.symbol, interval)
            if bars is None or bars.empty:
                continue
            ema_series = ema(bars["close"], period)
            if pd.isna(ema_series.iloc[-1]):
                continue
            if price < float(ema_series.iloc[-1]):
                signals.append(self._exit(
                    position, price, abs(position.quantity),
                    f"{period}-EMA({interval}) trail broken (§6.7)",
                ))
        return signals

    def _exit(self, position: Any, price: float, quantity: int, reason: str) -> Signal:
        return self.signal(
            position.symbol, Side.SELL,
            stop=None,
            reference_price=price,
            ttl=TTL.INTRADAY,
            reason=reason,
            segment=Segment.EQUITY_INTRADAY.value,
            exit=True,
            quantity=quantity,
        )
