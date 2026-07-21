"""flows.py — FII positioning tilt. Implements §6.9.

WHY (from the spec): NSE publishes FII/DII cash flows and FII index-futures
positioning daily. Extremes are contrarian gold: when FII index-futures long
share drops to historic lows, forward index returns have skewed sharply
positive -- the sellers are exhausted.

Note the asymmetry, which is deliberate and must not be "fixed": a LOW
percentile is a long signal, but a HIGH percentile is **not** a short signal.
It is a veto on other engines' risk. Shorting a market with SIP flows arriving
monthly is a different and much worse trade.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

from core import calendar as trading_calendar
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class FlowReading:
    """One night's FII/DII reading with its 3-year percentile."""

    trade_date: _dt.date
    fii_cash_cr: Optional[float]
    dii_cash_cr: Optional[float]
    long_ratio: Optional[float]
    percentile: Optional[float]

    @property
    def usable(self) -> bool:
        """A percentile from too little history is not a percentile."""
        return self.percentile is not None


class FlowsEngine(Engine):
    """§6.9. Max one position; also feeds vetoes to §6.4 and §6.8."""

    name = "flows"

    def universe(self) -> list[str]:
        from core.config import get_universe

        return [str(get_universe().get("index_proxies.NIFTY.etf_proxy", "NIFTYBEES"))]

    @property
    def symbol(self) -> str:
        return self.universe()[0]

    # -- the reading ------------------------------------------------------

    def latest_reading(self, ctx: Context) -> Optional[FlowReading]:
        """The most recent nightly flows row (written by the 20:30 job)."""
        rows = ctx.the_journal().query(
            "SELECT * FROM flows ORDER BY trade_date DESC LIMIT 1"
        )
        if not rows:
            log.info("flows: no nightly data yet; run scripts/nightly_downloads.py")
            return None
        row = rows[0]
        try:
            trade_date = _dt.date.fromisoformat(str(row["trade_date"])[:10])
        except ValueError:
            trade_date = ctx.today
        return FlowReading(
            trade_date=trade_date,
            fii_cash_cr=row["fii_cash_cr"],
            dii_cash_cr=row["dii_cash_cr"],
            long_ratio=row["long_ratio"],
            percentile=row["ratio_percentile_3y"],
        )

    def regime_context(self, ctx: Context) -> dict[str, Any]:
        """§6.9: 'feed the daily reading to the regime router as context'.

        Also emits the two vetoes the spec attaches to a high percentile, which
        the orchestrator places into ``ctx.extras`` for §6.4 and §6.8.
        """
        reading = self.latest_reading(ctx)
        if reading is None or not reading.usable:
            return {"flows_available": False}

        veto_at = float(self.config.require("veto_signal_percentile_min"))
        extreme_high = reading.percentile >= veto_at

        return {
            "flows_available": True,
            "fii_cash_cr": reading.fii_cash_cr,
            "dii_cash_cr": reading.dii_cash_cr,
            "fii_long_ratio": reading.long_ratio,
            "fii_ratio_percentile_3y": reading.percentile,
            "flows_as_of": reading.trade_date.isoformat(),
            # §6.9: percentile >= 90 -> no shorting; instead veto new overnight
            # longs (§6.4) and halve premium-selling size (§6.8).
            "flows_veto_overnight_longs": extreme_high,
            "flows_halve_premium_selling": extreme_high
            and bool(self.config.get("halve_premium_selling_at_high_percentile", True)),
        }

    # -- entries ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.9: percentile <= 10 -> swing long the index at the next open."""
        if not self.auto_trade:
            return []
        if ctx.position_for(self.symbol, self.name) is not None:
            return []
        if self.concurrent_positions(ctx) >= int(self.config.require("max_positions")):
            return []

        reading = self.latest_reading(ctx)
        if reading is None or not reading.usable:
            return []

        threshold = float(self.config.require("long_signal_percentile_max"))
        if reading.percentile > threshold:
            return []

        price = ctx.price(self.symbol)
        if price is None:
            return []

        stop_pct = float(self.config.require("stop_pct"))
        stop = round(price * (1 + stop_pct / 100.0), 2)

        return [
            self.signal(
                self.symbol, Side.BUY,
                stop=stop,
                reference_price=price,
                ttl=TTL.SWING,
                entry_type=EntryType.MARKET,
                reason=(
                    f"FII index-futures long share at the {reading.percentile:.1f}th "
                    f"3-year percentile (<= {threshold:.0f}); sellers are exhausted (§6.9)"
                ),
                segment=Segment.EQUITY_DELIVERY.value,
                entry_percentile=reading.percentile,
                exit_percentile_above=float(self.config.require("exit.percentile_above")),
                exit_profit_pct=float(self.config.require("exit.profit_pct")),
                exit_max_sessions=int(self.config.require("exit.max_sessions")),
            )
        ]

    # -- exits ------------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.9 exit: percentile > 40, or +4%, or 20 sessions -- whichever first."""
        exit_config = self.config.section("exit")
        percentile_exit = float(exit_config.require("percentile_above"))
        profit_target = float(exit_config.require("profit_pct"))
        max_sessions = int(exit_config.require("max_sessions"))

        reading = self.latest_reading(ctx)
        signals: list[Signal] = []

        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            gain_pct = (price - position.average_price) / position.average_price * 100.0
            if gain_pct >= profit_target:
                signals.append(self._exit(position, price, f"+{gain_pct:.1f}% target (§6.9)"))
                continue

            if reading is not None and reading.usable and reading.percentile > percentile_exit:
                signals.append(self._exit(
                    position, price,
                    f"FII percentile back to {reading.percentile:.1f} (> {percentile_exit:.0f})",
                ))
                continue

            if position.opened_at is not None:
                held = len(trading_calendar.trading_days_between(
                    position.opened_at.date(), ctx.today
                )) - 1
                if held >= max_sessions:
                    signals.append(self._exit(
                        position, price, f"{max_sessions}-session time exit (§6.9)"
                    ))
        return signals

    def _exit(self, position: Any, price: float, reason: str) -> Signal:
        return self.signal(
            position.symbol, Side.SELL,
            stop=None,
            reference_price=price,
            ttl=position.ttl,
            reason=reason,
            segment=Segment.EQUITY_DELIVERY.value,
            exit=True,
            quantity=abs(position.quantity),
        )
