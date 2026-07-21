"""pead.py — Post-earnings drift + concall tone. Implements §6.6.

WHY (from the spec): institutions can't buy a surprise in one day; big beats
drift for weeks (a documented anomaly, persistent in India). The LLM reads what
the price gap can't: whether management's guidance backs the number.

The tone gate is the entire reason this engine is not just "buy gaps". A 40%
profit beat with management refusing to guide is exactly the setup that fades.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core import calendar as trading_calendar
from core.datafeed import atr, ema
from core.llm import LLMError, load_schema
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class EarningsCandidate:
    """A result-day gap that passed the volume screen and is awaiting entry."""

    symbol: str
    result_date: _dt.date
    gap_pct: float
    volume_multiple: float
    pre_earnings_close: float
    tone: Optional[int] = None
    tone_reason: str = ""
    guidance: str = ""

    def sessions_since(self, today: _dt.date) -> int:
        return len(trading_calendar.trading_days_between(self.result_date, today)) - 1


class PeadEngine(Engine):
    """§6.6."""

    name = "pead"

    def __init__(self, llm: Any = None, nse: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._nse = nse
        # Candidates awaiting the 20-EMA pullback, keyed by symbol.
        self.watchlist: dict[str, EarningsCandidate] = {}

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from core.llm import get_llm

            self._llm = get_llm()
        return self._llm

    # -- detection --------------------------------------------------------

    def detect(self, ctx: Context) -> list[EarningsCandidate]:
        """§6.6: gap >= +3% with volume >= 2x the 20-day average, NIFTY-500."""
        min_gap = float(self.config.require("min_gap_pct"))
        min_volume = float(self.config.require("min_volume_multiple"))
        volume_days = int(self.config.get("volume_avg_days", 20))

        found: list[EarningsCandidate] = []
        for symbol in self.universe():
            if symbol in self.watchlist:
                continue
            bars = ctx.bars_for(symbol, "day")
            if bars is None or len(bars) < volume_days + 2:
                continue

            today = bars.iloc[-1]
            previous_close = float(bars["close"].iloc[-2])
            if previous_close <= 0:
                continue

            gap_pct = (float(today["open"]) - previous_close) / previous_close * 100.0
            if gap_pct < min_gap:
                continue

            average_volume = float(bars["volume"].iloc[-(volume_days + 1):-1].mean())
            if average_volume <= 0:
                continue
            volume_multiple = float(today["volume"]) / average_volume
            if volume_multiple < min_volume:
                continue

            found.append(
                EarningsCandidate(
                    symbol=symbol,
                    result_date=bars.index[-1].date(),
                    gap_pct=gap_pct,
                    volume_multiple=volume_multiple,
                    pre_earnings_close=previous_close,
                )
            )
        return found

    def score_tone(self, candidate: EarningsCandidate, ctx: Context) -> Optional[int]:
        """§6.6: LLM scores management tone 0-10; require >= 7.

        With no transcript available the honest answer is None, not a guess.
        The engine then does not trade -- refusing to score is a valid outcome.
        """
        transcript = self._fetch_transcript(candidate, ctx)
        if not transcript:
            log.info(
                "pead: no transcript/results text for %s; cannot score tone, so no trade",
                candidate.symbol,
            )
            return None

        payload = {
            "symbol": candidate.symbol,
            "quarter": self._quarter_label(candidate.result_date),
            "gap_pct": round(candidate.gap_pct, 2),
            "transcript_excerpt": transcript[:12_000],
        }
        try:
            result = self.llm.classify("pead_tone", payload, load_schema("pead_tone"))
        except LLMError as exc:
            log.error("pead tone scoring failed for %s: %s", candidate.symbol, exc)
            self.journal.record_error("pead", f"{candidate.symbol}: {exc}", severity="WARNING")
            return None

        candidate.tone = int(result["tone"])
        candidate.tone_reason = str(result.get("reason", ""))
        candidate.guidance = str(result.get("guidance", ""))
        return candidate.tone

    def _fetch_transcript(self, candidate: EarningsCandidate, ctx: Context) -> str:
        """Results PDF / concall text, via the §6.1 announcements store."""
        rows = ctx.the_journal().query(
            "SELECT body, headline FROM announcements WHERE symbol=? AND trade_date>=? "
            "ORDER BY id DESC LIMIT 5",
            (candidate.symbol, candidate.result_date.isoformat()),
        )
        parts = [f"{r['headline']}\n{r['body']}" for r in rows if r["body"]]
        return "\n\n".join(parts)

    @staticmethod
    def _quarter_label(day: _dt.date) -> str:
        """Indian fiscal quarter label, e.g. Q3FY26. FY starts in April."""
        quarter = ((day.month - 4) % 12) // 3 + 1
        fy = day.year + 1 if day.month >= 4 else day.year
        return f"Q{quarter}FY{fy % 100:02d}"

    # -- entries ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """Detect new gaps, score tone, then look for the 20-EMA pullback."""
        if not self.auto_trade:
            return []

        min_tone = int(self.config.require("min_tone_score"))
        for candidate in self.detect(ctx):
            tone = self.score_tone(candidate, ctx)
            if tone is None:
                continue
            if tone < min_tone:
                log.info(
                    "pead: %s gapped %.1f%% but tone is %d (< %d): %s",
                    candidate.symbol, candidate.gap_pct, tone, min_tone,
                    candidate.tone_reason[:120],
                )
                continue
            self.watchlist[candidate.symbol] = candidate
            log.info("pead: watching %s, tone %d, gap %.1f%%",
                     candidate.symbol, tone, candidate.gap_pct)

        return self._pullback_entries(ctx)

    def _pullback_entries(self, ctx: Context) -> list[Signal]:
        """§6.6 entry: first pullback to the daily 20-EMA within 5 sessions."""
        entry_config = self.config.section("entry")
        ema_period = int(entry_config.require("ema_period"))
        within = int(entry_config.require("within_sessions"))
        max_concurrent = int(self.config.require("max_concurrent"))

        signals: list[Signal] = []
        expired: list[str] = []

        for symbol, candidate in self.watchlist.items():
            age = candidate.sessions_since(ctx.today)
            if age > within:
                expired.append(symbol)
                continue
            if ctx.position_for(symbol, self.name) is not None:
                continue
            if self.concurrent_positions(ctx) + len(signals) >= max_concurrent:
                break

            bars = ctx.bars_for(symbol, "day")
            if bars is None or len(bars) < ema_period + 1:
                continue
            ema_series = ema(bars["close"], ema_period)
            if pd.isna(ema_series.iloc[-1]):
                continue

            ema_value = float(ema_series.iloc[-1])
            low = float(bars["low"].iloc[-1])
            price = ctx.price(symbol) or float(bars["close"].iloc[-1])

            # The pullback has happened when the session's low touched the EMA.
            if low > ema_value:
                continue

            # §6.6 stop: below the pre-earnings close. If price is already
            # below it the setup is dead, not a bigger opportunity.
            stop = round(candidate.pre_earnings_close * 0.999, 2)
            if price <= stop:
                expired.append(symbol)
                continue

            signals.append(
                self.signal(
                    symbol, Side.BUY,
                    stop=stop,
                    reference_price=price,
                    ttl=TTL.SWING,
                    entry_type=EntryType.MARKET,
                    reason=(
                        f"PEAD: gap {candidate.gap_pct:+.1f}%, volume "
                        f"{candidate.volume_multiple:.1f}x, tone {candidate.tone}/10 "
                        f"({candidate.guidance})"
                    ),
                    segment=Segment.EQUITY_DELIVERY.value,
                    tone=candidate.tone,
                    gap_pct=candidate.gap_pct,
                    result_date=candidate.result_date.isoformat(),
                    pre_earnings_close=candidate.pre_earnings_close,
                    atr_trail_mult=float(self.config.require("exit.atr_trail_mult")),
                    time_exit_days=int(self.config.require("exit.time_exit_days")),
                )
            )

        for symbol in expired:
            self.watchlist.pop(symbol, None)
        return signals

    # -- management -------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.6 exit: 3xATR(14,d) trail, or a 30-day time exit."""
        exit_config = self.config.section("exit")
        multiple = float(exit_config.require("atr_trail_mult"))
        period = int(exit_config.get("atr_period", 14))
        max_days = int(exit_config.require("time_exit_days"))

        signals: list[Signal] = []
        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            if position.opened_at is not None:
                held = len(trading_calendar.trading_days_between(
                    position.opened_at.date(), ctx.today
                )) - 1
                if held >= max_days:
                    signals.append(self._exit(
                        position, price, f"{max_days}-session time exit (§6.6)"
                    ))
                    continue

            bars = ctx.bars_for(position.symbol, "day")
            if bars is None or bars.empty:
                continue
            atr_series = atr(bars, period)
            if pd.isna(atr_series.iloc[-1]):
                continue

            # Chandelier-style trail from the highest close since entry.
            since_entry = bars[bars.index >= pd.Timestamp(position.opened_at)] \
                if position.opened_at is not None else bars
            if since_entry.empty:
                continue
            peak = float(since_entry["close"].max())
            trail = peak - multiple * float(atr_series.iloc[-1])
            if price <= trail:
                signals.append(self._exit(
                    position, price,
                    f"{multiple}xATR trail from {peak:,.2f} (§6.6)",
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

    def universe(self) -> list[str]:
        """§6.6 operates on NIFTY-500, resolved from the instruments cache."""
        try:
            return super().universe()
        except Exception:
            from core.datafeed import DataFeed

            return DataFeed(journal=self.journal).resolve_nifty500()
