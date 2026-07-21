"""overnight.py — Overnight drift. Implements §6.4.

WHY (from the spec): a large share of long-run index return accrues overnight
-- news lands while the market is shut and gaps carry it. Harvest the drift
with filters that skip the toxic nights.

The three filters are the whole strategy. Holding the index overnight
unconditionally is a coin flip with fat tails; holding it only above the
200-DMA, only when today was not already ugly, and never into a known event,
is the version with an edge.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.datafeed import sma
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


class OvernightEngine(Engine):
    """§6.4. One index position, entered 15:20, exited 09:16 next session."""

    name = "overnight"

    def universe(self) -> list[str]:
        """The index proxy, from universe.yaml."""
        from core.config import get_universe

        proxy = get_universe().get("index_proxies.NIFTY.etf_proxy", "NIFTYBEES")
        return [str(self.config.get("symbol", proxy))]

    @property
    def symbol(self) -> str:
        return self.universe()[0]

    # -- entry ------------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.4 entry at 15:20, subject to the three filters."""
        if not self.auto_trade:
            return []
        if not self._at_entry_time(ctx):
            return []
        if ctx.position_for(self.symbol, self.name) is not None:
            return []

        reasons = self.blocking_reasons(ctx)
        if reasons:
            log.info("overnight: no entry tonight -- %s", "; ".join(reasons))
            return []

        price = ctx.price(self.symbol)
        if price is None:
            log.warning("overnight: no price for %s", self.symbol)
            return None or []

        stop = self._stop_for(price)
        return [
            self.signal(
                self.symbol, Side.BUY,
                stop=stop,
                reference_price=price,
                ttl=TTL.OVERNIGHT,
                entry_type=EntryType.MARKET,
                reason="overnight drift: above 200-DMA, no event tomorrow, today not weak (§6.4)",
                segment=Segment.EQUITY_DELIVERY.value,
                exit_at=str(self.config.require("exit_at")),
                gift_nifty_exit_threshold_pct=float(
                    self.config.require("gift_nifty_exit_threshold_pct")
                ),
            )
        ]

    def blocking_reasons(self, ctx: Context) -> list[str]:
        """Every §6.4 filter that currently forbids an entry, as text.

        Returned as a list rather than a bool so the digest can say *why* the
        engine sat out -- "no signal" and "blocked by the event calendar" are
        very different pieces of information.
        """
        reasons: list[str] = []

        # Filter 1: index close > 200-DMA.
        above, detail = self._above_dma(ctx)
        if not above:
            reasons.append(detail)

        # Filter 2: the next session is not a blocked event day.
        if trading_calendar.next_session_is_blocked(ctx.today):
            nxt = trading_calendar.next_trading_day(ctx.today)
            reasons.append(f"next session {nxt} is blocked ({trading_calendar.event_note(nxt)})")

        # Filter 3: today >= -1.5%.
        today_move = self._today_move_pct(ctx)
        floor = float(self.config.require("max_today_move_pct"))
        if today_move is not None and today_move < floor:
            reasons.append(f"today is {today_move:.2f}% (< {floor}%)")

        # §6.9 feeds a veto in: FII positioning at an extreme high blocks new
        # overnight longs.
        if ctx.extras.get("flows_veto_overnight_longs"):
            reasons.append("§6.9 FII positioning percentile >= 90 vetoes new overnight longs")

        return reasons

    def _above_dma(self, ctx: Context) -> tuple[bool, str]:
        period = int(self.config.require("require_close_above_dma"))
        bars = ctx.bars_for(self.symbol, "day")
        if bars is None or len(bars) < period:
            return False, f"not enough history for the {period}-DMA filter"
        dma = sma(bars["close"], period)
        if pd.isna(dma.iloc[-1]):
            return False, f"{period}-DMA is not yet defined"
        close = float(bars["close"].iloc[-1])
        value = float(dma.iloc[-1])
        if close > value:
            return True, ""
        return False, f"close {close:,.2f} is not above the {period}-DMA {value:,.2f}"

    def _today_move_pct(self, ctx: Context) -> Optional[float]:
        bars = ctx.bars_for(self.symbol, "day")
        if bars is None or len(bars) < 2:
            return None
        previous = float(bars["close"].iloc[-2])
        current = ctx.price(self.symbol) or float(bars["close"].iloc[-1])
        if previous <= 0:
            return None
        return (current - previous) / previous * 100.0

    def _stop_for(self, price: float) -> float:
        """§6.4: size so a 2% adverse gap is about the daily loss limit.

        The §3 sizing formula is ``qty = risk_budget / |entry - stop|``. Setting
        the stop exactly one adverse-gap away therefore makes that gap cost
        exactly the per-trade risk budget -- which is how the spec's sizing
        instruction is expressed in this system's arithmetic.
        """
        gap_pct = float(self.config.require("adverse_gap_sizing_pct"))
        return round(price * (1 - gap_pct / 100.0), 2)

    def _at_entry_time(self, ctx: Context) -> bool:
        entry_at = str(self.config.require("entry_at"))
        return clock.within(ctx.now, entry_at, entry_at) or (
            clock.is_after(ctx.now, entry_at)
            and clock.is_before(ctx.now, str(self.settings.require("market.session.continuous_end")))
        )

    # -- exit -------------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.4 exit at 09:16, or earlier in pre-open if GIFT Nifty is <= -1%."""
        position = ctx.position_for(self.symbol, self.name)
        if position is None:
            return []

        price = ctx.price(self.symbol)
        if price is None:
            return []

        # §6.4: if GIFT Nifty shows <= -1% before open, exit in pre-open
        # instead of waiting for 09:16.
        gap = ctx.extras.get("gift_nifty_gap_pct")
        threshold = float(self.config.require("gift_nifty_exit_threshold_pct"))
        if gap is not None and gap <= threshold and trading_calendar.is_preopen(ctx.now):
            return [self._exit(position, price, f"GIFT Nifty {gap:.2f}% <= {threshold}% (§6.4)")]

        exit_at = str(self.config.require("exit_at"))
        if clock.is_after(ctx.now, exit_at) or clock.within(ctx.now, exit_at, exit_at):
            return [self._exit(position, price, f"{exit_at} scheduled exit (§6.4)")]
        return []

    def _exit(self, position: Any, price: float, reason: str) -> Signal:
        return self.signal(
            position.symbol, Side.SELL,
            stop=None,
            reference_price=price,
            ttl=TTL.OVERNIGHT,
            entry_type=EntryType.MARKET,
            reason=reason,
            segment=Segment.EQUITY_DELIVERY.value,
            exit=True,
            quantity=abs(position.quantity),
        )
