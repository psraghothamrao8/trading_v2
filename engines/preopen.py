"""preopen.py — Pre-open auction imbalance. Implements §6.5.

WHY (from the spec): NSE's 09:00-09:08 call auction publishes indicative price
and matched/unmatched order quantities. Persistent one-sided imbalance in the
auction routinely continues into the first minutes of trade -- public data
almost no retail trader reads programmatically.

"Persistent" is why there are two snapshots. A single reading at 09:06:30 can
be an artefact of one large order that gets pulled before 09:07:59; a reading
that still holds at 09:07:45 is a real imbalance going into the match.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core import clock
from core.datafeed import atr
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class PreopenCandidate:
    """A symbol that passed the §6.5 imbalance screen at snapshot time."""

    symbol: str
    indicative_price: float
    prev_close: float
    gap_pct: float
    imbalance_ratio: float
    direction: Side
    snapshot_at: _dt.datetime

    @property
    def is_long(self) -> bool:
        return self.direction is Side.BUY


def imbalance_ratio(buy_qty: Optional[int], sell_qty: Optional[int]) -> Optional[float]:
    """``unmatched buy / unmatched sell``.

    A zero sell quantity means infinite imbalance, which is real but not a
    number the thresholds can compare -- treated as a very large finite value
    so the long threshold fires and the short threshold cannot.
    """
    if buy_qty is None or sell_qty is None:
        return None
    if buy_qty <= 0 and sell_qty <= 0:
        return None
    if sell_qty <= 0:
        return float("inf")
    return buy_qty / sell_qty


class PreopenEngine(Engine):
    """§6.5. Two snapshots, then continuation entries in 09:15-09:20."""

    name = "preopen"

    def __init__(self, nse: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._nse = nse
        self.snapshots: dict[str, list[PreopenCandidate]] = {}
        self._entered_today: set[str] = set()
        self._snapshot_date: Optional[_dt.date] = None

    @property
    def nse(self) -> Any:
        if self._nse is None:
            from core.nse import NSEClient

            self._nse = NSEClient(journal=self.journal)
        return self._nse

    # -- snapshots --------------------------------------------------------

    def take_snapshot(self, ctx: Context, label: str) -> list[PreopenCandidate]:
        """Snapshot the NSE pre-open feed and screen it (§6.5)."""
        self._reset_if_new_day(ctx)
        try:
            rows = self.nse.preopen_snapshot()
        except Exception as exc:
            log.error("pre-open snapshot failed: %s", exc)
            self.journal.record_error("preopen", f"snapshot {label}: {exc}", severity="WARNING")
            return []

        universe = set(self.universe())
        min_gap = float(self.config.require("min_abs_gap_pct"))
        long_ratio = float(self.config.require("imbalance_ratio_long_min"))
        short_ratio = float(self.config.require("imbalance_ratio_short_max"))

        candidates: list[PreopenCandidate] = []
        for row in rows:
            symbol = row["symbol"]
            if universe and symbol not in universe:
                continue

            indicative = row.get("indicative_price")
            prev_close = row.get("prev_close")
            if not indicative or not prev_close:
                continue

            gap_pct = (indicative - prev_close) / prev_close * 100.0
            if abs(gap_pct) < min_gap:
                continue

            ratio = imbalance_ratio(row.get("total_buy_quantity"), row.get("total_sell_quantity"))
            if ratio is None:
                continue

            if gap_pct > 0 and ratio >= long_ratio:
                direction = Side.BUY
            elif gap_pct < 0 and ratio <= short_ratio:
                direction = Side.SELL
            else:
                continue

            candidates.append(
                PreopenCandidate(
                    symbol=symbol,
                    indicative_price=float(indicative),
                    prev_close=float(prev_close),
                    gap_pct=gap_pct,
                    imbalance_ratio=ratio,
                    direction=direction,
                    snapshot_at=ctx.now,
                )
            )

        self.snapshots[label] = candidates
        log.info("pre-open snapshot %s: %d candidates", label, len(candidates))
        return candidates

    def persistent_candidates(self, ctx: Context) -> list[PreopenCandidate]:
        """Candidates present in BOTH snapshots, with the same direction.

        This is the "persistent one-sided imbalance" the spec is describing.
        With only one snapshot taken, that one is used -- but the engine says
        so, because a single reading is weaker evidence.
        """
        labels = list(self.snapshots)
        if not labels:
            return []
        if len(labels) == 1:
            log.info("only one pre-open snapshot taken; imbalance persistence unverified")
            return self.snapshots[labels[0]]

        first, second = self.snapshots[labels[0]], self.snapshots[labels[-1]]
        first_by_symbol = {c.symbol: c for c in first}
        return [
            candidate for candidate in second
            if candidate.symbol in first_by_symbol
            and first_by_symbol[candidate.symbol].direction is candidate.direction
        ]

    # -- entries ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.5: at 09:15-09:20, enter on CONTINUATION only."""
        self._reset_if_new_day(ctx)
        if not self.auto_trade:
            return []
        if not self.within_window(ctx, "entry_window"):
            return []

        max_per_day = int(self.config.require("max_trades_per_day"))
        if len(self._entered_today) >= max_per_day:
            return []

        # §6.5: direction must agree with any overnight §6.1 filing; a
        # disagreeing filing is a veto.
        filings = self._overnight_filing_directions(ctx)

        signals: list[Signal] = []
        for candidate in self.persistent_candidates(ctx):
            if candidate.symbol in self._entered_today:
                continue
            if len(self._entered_today) + len(signals) >= max_per_day:
                break

            filing_direction = filings.get(candidate.symbol)
            if filing_direction is not None and filing_direction is not candidate.direction:
                log.info(
                    "preopen: vetoing %s -- an overnight filing disagrees with the "
                    "auction imbalance (§6.5)", candidate.symbol,
                )
                continue

            price = ctx.price(candidate.symbol)
            if price is None:
                continue

            # §6.5: continuation only -- price must trade BEYOND the indicative
            # price in the gap direction. Fading the auction is a different
            # (and much worse) strategy.
            if candidate.is_long and price <= candidate.indicative_price:
                continue
            if not candidate.is_long and price >= candidate.indicative_price:
                continue

            stop = self._stop_for(candidate, ctx)
            if stop is None:
                continue

            risk = abs(price - stop)
            signals.append(
                self.signal(
                    candidate.symbol, candidate.direction,
                    stop=stop,
                    reference_price=price,
                    targets=(price + risk * candidate.direction.sign,),
                    ttl=TTL.INTRADAY,
                    entry_type=EntryType.MARKET,
                    reason=(
                        f"pre-open imbalance gap {candidate.gap_pct:+.2f}% "
                        f"ratio {candidate.imbalance_ratio:.2f}, continuation confirmed"
                    ),
                    segment=Segment.EQUITY_INTRADAY.value,
                    indicative_price=candidate.indicative_price,
                    imbalance_ratio=candidate.imbalance_ratio,
                    gap_pct=candidate.gap_pct,
                    book_fraction_at_1r=float(self.config.get("book_fraction_at_1r", 0.5)),
                    flat_by=str(self.config.require("flat_by")),
                )
            )
            self._entered_today.add(candidate.symbol)
        return signals

    def _stop_for(self, candidate: PreopenCandidate, ctx: Context) -> Optional[float]:
        """§6.5: stop = indicative price -/+ 0.5 x ATR(5m)."""
        interval = str(self.config.get("atr_timeframe", "5minute"))
        bars = ctx.bars_for(candidate.symbol, interval)
        if bars is None or bars.empty:
            return None
        atr_series = atr(bars, 14)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        offset = float(self.config.require("stop_atr_mult")) * float(atr_series.iloc[-1])
        if candidate.is_long:
            return round(candidate.indicative_price - offset, 2)
        return round(candidate.indicative_price + offset, 2)

    def _overnight_filing_directions(self, ctx: Context) -> dict[str, Side]:
        """Material filings since yesterday's close, as a direction per symbol."""
        rows = ctx.the_journal().query(
            "SELECT symbol, label FROM announcements "
            "WHERE trade_date >= ? AND label IN ('MATERIAL_POSITIVE','MATERIAL_NEGATIVE')",
            ((ctx.today - _dt.timedelta(days=1)).isoformat(),),
        )
        return {
            row["symbol"]: (Side.BUY if row["label"] == "MATERIAL_POSITIVE" else Side.SELL)
            for row in rows
        }

    # -- management -------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.5: book 50% at +1R, flat by 10:30 -- this edge is minutes-scale."""
        flat_by = str(self.config.require("flat_by"))
        signals: list[Signal] = []

        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            if clock.is_after(ctx.now, flat_by):
                signals.append(self._exit(
                    position, price, abs(position.quantity),
                    f"{flat_by} time exit -- pre-open edge is minutes-scale (§6.5)",
                ))
                continue

            stop = position.stop
            if stop is not None and not position.meta.get("booked_1r"):
                risk = abs(position.average_price - stop)
                direction = 1 if position.is_long else -1
                target = position.average_price + risk * direction
                reached = price >= target if position.is_long else price <= target
                if reached and risk > 0:
                    fraction = float(self.config.get("book_fraction_at_1r", 0.5))
                    position.meta["booked_1r"] = True
                    signals.append(self._exit(
                        position, price,
                        max(int(abs(position.quantity) * fraction), 1),
                        f"+1R scale-out ({fraction:.0%})",
                    ))
        return signals

    def _exit(self, position: Any, price: float, quantity: int, reason: str) -> Signal:
        return self.signal(
            position.symbol,
            Side.SELL if position.is_long else Side.BUY,
            stop=None,
            reference_price=price,
            ttl=TTL.INTRADAY,
            reason=reason,
            segment=Segment.EQUITY_INTRADAY.value,
            exit=True,
            quantity=quantity,
        )

    def _reset_if_new_day(self, ctx: Context) -> None:
        if self._snapshot_date != ctx.today:
            self.snapshots.clear()
            self._entered_today.clear()
            self._snapshot_date = ctx.today
