"""sympathy.py — Second-order reaction. Implements §6.2.

WHY (from the spec): a material event moves the subject stock in seconds, but
its listed suppliers/customers/peers reprice over hours. The LLM knows business
relationships no price feed contains -- this is the rare edge where the AI
stack beats faster money.

The "< 1/3 of the primary's move" filter is what keeps that true. If the
related name has already moved with the primary, the crowd has already made
the connection and there is nothing left to capture.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core.datafeed import atr, vwap
from core.llm import LLMError, load_schema
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine
from engines.filings import ClassifiedFiling

log = logging.getLogger(__name__)


@dataclass
class RelatedName:
    """One LLM-identified related company."""

    symbol: str
    relation: str
    direction: str              # POSITIVE | NEGATIVE
    confidence: float
    reason: str

    @property
    def side(self) -> Side:
        return Side.BUY if self.direction == "POSITIVE" else Side.SELL


class SympathyEngine(Engine):
    """§6.2. Triggered only by a §6.1 MATERIAL event."""

    name = "sympathy"

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._traded_today: set[str] = set()
        self._filings_handled: set[str] = set()
        self._day: Optional[_dt.date] = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from core.llm import get_llm

            self._llm = get_llm()
        return self._llm

    def universe(self) -> list[str]:
        """§6.2 screens against NIFTY-500."""
        try:
            return super().universe()
        except Exception:
            from core.datafeed import DataFeed

            return DataFeed(journal=self.journal).resolve_nifty500()

    # -- the trigger ------------------------------------------------------

    def on_filing(self, filing: ClassifiedFiling, ctx: Context) -> list[Signal]:
        """§6.2: trigger ONLY from a §6.1 MATERIAL event."""
        self._reset_if_new_day(ctx)

        if not filing.is_material:
            return []
        if filing.announcement_id in self._filings_handled:
            return []
        self._filings_handled.add(filing.announcement_id)

        max_per_day = int(self.config.require("max_trades_per_day"))
        if len(self._traded_today) >= max_per_day:
            log.info("sympathy: already at %d trades today (§6.2)", max_per_day)
            return []

        related = self.find_related(filing)
        if not related:
            return []

        primary_move = self._move_since_filing(filing.symbol, filing.timestamp, ctx)
        if primary_move is None or primary_move == 0:
            log.debug("sympathy: no measurable primary move for %s yet", filing.symbol)
            return []

        # §6.2: max 1 sympathy trade per filing.
        max_per_filing = int(self.config.require("max_trades_per_filing"))
        signals: list[Signal] = []

        for candidate in related:
            if len(signals) >= max_per_filing:
                break
            signal = self._signal_for(candidate, filing, primary_move, ctx)
            if signal is not None:
                signals.append(signal)
                self._traded_today.add(candidate.symbol)
        return signals

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """Drain any filings the orchestrator queued for us this cycle."""
        self._reset_if_new_day(ctx)
        signals: list[Signal] = []
        for filing in ctx.extras.get("material_filings", []) or []:
            signals.extend(self.on_filing(filing, ctx))
        return signals

    # -- the LLM step -----------------------------------------------------

    def find_related(self, filing: ClassifiedFiling) -> list[RelatedName]:
        """§6.2: LLM returns <= 3 listed NSE names with relation and direction."""
        payload = {
            "subject_symbol": filing.symbol,
            "label": filing.label,
            "headline": filing.headline,
            "body": filing.body[:6000],
        }
        try:
            result = self.llm.classify("sympathy", payload, load_schema("sympathy"))
        except LLMError as exc:
            log.error("sympathy mapping failed for %s: %s", filing.symbol, exc)
            self.journal.record_error("sympathy", f"{filing.symbol}: {exc}", severity="WARNING")
            return []

        max_names = int(self.config.require("max_related_names"))
        out: list[RelatedName] = []
        for row in (result.get("related") or [])[:max_names]:
            symbol = str(row["symbol"]).upper()
            if symbol == filing.symbol:
                continue            # the prompt forbids it; enforce it anyway
            out.append(
                RelatedName(
                    symbol=symbol,
                    relation=str(row["relation"]),
                    direction=str(row["direction"]),
                    confidence=float(row["confidence"]),
                    reason=str(row.get("reason", "")),
                )
            )
        return out

    # -- the trade --------------------------------------------------------

    def _signal_for(
        self,
        candidate: RelatedName,
        filing: ClassifiedFiling,
        primary_move: float,
        ctx: Context,
    ) -> Optional[Signal]:
        """§6.2 gate: liquid, confidence >= 0.7, move < 1/3 of the primary's."""
        if not self.auto_trade:
            return None
        if candidate.symbol in self._traded_today:
            return None

        min_confidence = float(self.config.require("min_confidence"))
        if candidate.confidence < min_confidence:
            log.info("sympathy: %s confidence %.2f < %.2f",
                     candidate.symbol, candidate.confidence, min_confidence)
            return None

        if candidate.symbol not in set(self.universe()):
            log.info("sympathy: %s is not in the liquid universe", candidate.symbol)
            return None

        turnover_cr = self._avg_turnover_cr(candidate.symbol, ctx)
        min_turnover = float(self.config.require("min_avg_daily_turnover_cr"))
        if turnover_cr is not None and turnover_cr < min_turnover:
            log.info("sympathy: %s turnover Rs %.1f cr < Rs %.0f cr",
                     candidate.symbol, turnover_cr, min_turnover)
            return None

        # §6.2: its move since the filing must be < 1/3 of the primary's move.
        candidate_move = self._move_since_filing(candidate.symbol, filing.timestamp, ctx)
        if candidate_move is None:
            return None
        fraction = float(self.config.require("max_move_fraction_of_primary"))
        if abs(candidate_move) >= abs(primary_move) * fraction:
            log.info(
                "sympathy: %s already moved %.2f%% vs primary %.2f%% "
                "(>= %.0f%% of it); the crowd got there first",
                candidate.symbol, candidate_move, primary_move, fraction * 100,
            )
            return None

        price = ctx.price(candidate.symbol)
        if price is None:
            return None

        stop = self._stop_for(candidate, ctx, price)
        if stop is None:
            return None
        risk = abs(price - stop)

        return self.signal(
            candidate.symbol, candidate.side,
            stop=stop,
            reference_price=price,
            targets=(price + risk * candidate.side.sign,),
            ttl=TTL.INTRADAY,
            entry_type=EntryType.MARKET,
            reason=(
                f"sympathy to {filing.symbol} {filing.label} via {candidate.relation} "
                f"({candidate.confidence:.2f}): {candidate.reason[:100]}"
            ),
            segment=Segment.EQUITY_INTRADAY.value,
            primary_symbol=filing.symbol,
            primary_move_pct=round(primary_move, 2),
            candidate_move_pct=round(candidate_move, 2),
            relation=candidate.relation,
            confidence=candidate.confidence,
            announcement_id=filing.announcement_id,
            book_fraction_at_1r=0.5,
        )

    def _stop_for(self, candidate: RelatedName, ctx: Context, price: float) -> Optional[float]:
        """§6.2: 'mechanics mirror 6.1' -- 1.5xATR, tightened to VWAP."""
        filings_config = self.settings.section("engines.filings")
        interval = str(filings_config.get("atr_timeframe", "5minute"))
        bars = ctx.bars_for(candidate.symbol, interval)
        if bars is None or bars.empty:
            return None

        atr_series = atr(bars, int(filings_config.get("atr_period", 14)))
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        offset = float(filings_config.require("stop_atr_mult")) * float(atr_series.iloc[-1])

        is_long = candidate.side is Side.BUY
        stop = price - offset if is_long else price + offset

        vwap_series = vwap(bars)
        if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]):
            vwap_value = float(vwap_series.iloc[-1])
            if is_long and stop < vwap_value < price:
                stop = vwap_value
            elif not is_long and price < vwap_value < stop:
                stop = vwap_value
        return round(stop, 2)

    def _move_since_filing(
        self, symbol: str, since: _dt.datetime, ctx: Context
    ) -> Optional[float]:
        interval = str(self.settings.get("engines.filings.atr_timeframe", "5minute"))
        bars = ctx.bars_for(symbol, interval)
        if bars is None or bars.empty:
            return None
        before = bars[bars.index <= since]
        if before.empty:
            return None
        reference = float(before["close"].iloc[-1])
        current = ctx.price(symbol) or float(bars["close"].iloc[-1])
        if reference <= 0:
            return None
        return (current - reference) / reference * 100.0

    def _avg_turnover_cr(self, symbol: str, ctx: Context) -> Optional[float]:
        """20-day average daily turnover in INR crore (§6.2 liquidity gate)."""
        bars = ctx.bars_for(symbol, "day")
        if bars is None or len(bars) < 20:
            return None
        recent = bars.iloc[-20:]
        turnover = (recent["close"] * recent["volume"]).mean()
        return float(turnover) / 1e7        # 1 crore = 10^7

    # -- management -------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.2 mechanics mirror §6.1: book 50% at +1R, trail by VWAP."""
        interval = str(self.settings.get("engines.filings.atr_timeframe", "5minute"))
        signals: list[Signal] = []

        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            stop = position.stop
            direction = 1 if position.is_long else -1
            if stop is not None and not position.meta.get("booked_1r"):
                risk = abs(position.average_price - stop)
                target = position.average_price + risk * direction
                reached = price >= target if position.is_long else price <= target
                if reached and risk > 0:
                    position.meta["booked_1r"] = True
                    signals.append(self._exit(
                        position, price, max(int(abs(position.quantity) * 0.5), 1),
                        "+1R scale-out (50%)",
                    ))

            bars = ctx.bars_for(position.symbol, interval)
            if bars is None or bars.empty:
                continue
            vwap_series = vwap(bars)
            if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
                continue
            vwap_value = float(vwap_series.iloc[-1])
            broke = price < vwap_value if position.is_long else price > vwap_value
            if broke:
                signals.append(self._exit(
                    position, price, abs(position.quantity), "VWAP trail broken"
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
        if self._day != ctx.today:
            self._traded_today.clear()
            self._filings_handled.clear()
            self._day = ctx.today
