"""filings.py — Disclosure-latency engine. Implements §6.1.

WHY (from the spec): every price-moving fact must legally hit the exchange
announcements feed first. Most humans read it hours later; HFT trades order
flow, not text. Reacting to *clearly material* filings within a minute captures
the drift before the crowd.

This engine is also the shared news sensor for §6.2 (sympathy) and §6.6 (PEAD),
and its database is what §6.3 and §6.7 query before they trade.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core import clock
from core.datafeed import atr, vwap
from core.llm import LLMError, load_schema
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)

MATERIAL_LABELS = {"MATERIAL_POSITIVE", "MATERIAL_NEGATIVE"}


@dataclass
class ClassifiedFiling:
    """An announcement plus its verdict. Passed to §6.2 unchanged."""

    symbol: str
    announcement_id: str
    content_hash: str
    headline: str
    body: str
    timestamp: _dt.datetime
    label: str
    confidence: float
    reason: str
    est_revenue_impact_pct: Optional[float]
    latency_sec: float
    attachment_url: Optional[str] = None

    @property
    def is_material(self) -> bool:
        return self.label in MATERIAL_LABELS

    @property
    def is_positive(self) -> bool:
        return self.label == "MATERIAL_POSITIVE"

    def age_minutes(self, now: _dt.datetime) -> float:
        return (now - self.timestamp).total_seconds() / 60.0


def content_hash(symbol: str, headline: str, body: str) -> str:
    """§6.1 dedupe key half: a stable hash of the announcement's content."""
    blob = f"{symbol}|{headline.strip()}|{body.strip()}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


class FilingsEngine(Engine):
    """§6.1. Polls announcements, classifies, alerts, and (once promoted) trades."""

    name = "filings"

    def __init__(self, nse: Any = None, llm: Any = None, alerts: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._nse = nse
        self._llm = llm
        self._alerts = alerts
        self._last_poll: Optional[_dt.datetime] = None
        # Filings classified this session, newest last. §6.2 reads this.
        self.recent: list[ClassifiedFiling] = []

    # -- lazy collaborators ----------------------------------------------

    @property
    def nse(self) -> Any:
        if self._nse is None:
            from core.nse import NSEClient

            self._nse = NSEClient(journal=self.journal)
        return self._nse

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from core.llm import get_llm

            self._llm = get_llm()
        return self._llm

    @property
    def alerts(self) -> Any:
        if self._alerts is None:
            from live.alerts import get_alerts

            self._alerts = get_alerts()
        return self._alerts

    # -- polling and classification ---------------------------------------

    def poll(self, ctx: Context) -> list[ClassifiedFiling]:
        """§6.1: poll every 30s, 08:00-15:35 IST, dedupe, classify, alert.

        Returns only the newly-classified MATERIAL items, because that is what
        §6.2 consumes and what the trading path acts on.
        """
        if not self.within_window(ctx, "poll_window"):
            return []

        try:
            rows = self.nse.announcements(since=self._last_poll)
        except Exception as exc:
            log.error("announcement poll failed: %s", exc)
            self.journal.record_error("filings", f"poll failed: {exc}", severity="WARNING")
            return []
        self._last_poll = ctx.now

        material: list[ClassifiedFiling] = []
        for row in rows:
            filing = self._classify_row(row, ctx)
            if filing is None:
                continue
            self.recent.append(filing)
            if filing.is_material:
                material.append(filing)
        return material

    def _classify_row(self, row: dict[str, Any], ctx: Context) -> Optional[ClassifiedFiling]:
        """Dedupe, fetch the PDF, classify, journal and alert one announcement."""
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            return None

        headline = str(row.get("headline") or "")
        body = str(row.get("body") or "")
        attachment = row.get("attachment_url")

        # §6.1: download the attached PDF and extract text (first 4 pages max).
        if attachment and not body:
            body = self._extract_pdf_text(attachment)

        digest = content_hash(symbol, headline, body)
        announcement_id = str(row.get("announcement_id") or digest)

        payload = {
            "symbol": symbol,
            "headline": headline,
            "body": body[:8000],
            "timestamp": clock.isoformat(row.get("timestamp") or ctx.now),
        }

        try:
            result = self.llm.classify("filings", payload, load_schema("filings"))
        except LLMError as exc:
            log.error("classification failed for %s: %s", symbol, exc)
            self.journal.record_error("filings", f"{symbol}: {exc}", severity="WARNING")
            return None

        filing = ClassifiedFiling(
            symbol=symbol,
            announcement_id=announcement_id,
            content_hash=digest,
            headline=headline,
            body=body,
            timestamp=clock.to_ist(row.get("timestamp") or ctx.now),
            label=str(result["label"]),
            confidence=float(result["confidence"]),
            reason=str(result.get("reason", "")),
            est_revenue_impact_pct=result.get("est_revenue_impact_pct"),
            latency_sec=getattr(result, "latency_sec", 0.0),
            attachment_url=attachment,
        )

        # The UNIQUE index is the dedupe: a False here means we have seen this
        # exact announcement before and must not alert or trade on it again.
        is_new = self.journal.record_announcement(
            announcement_id=filing.announcement_id,
            content_hash=filing.content_hash,
            ts=clock.isoformat(filing.timestamp),
            trade_date=ctx.today.isoformat(),
            symbol=filing.symbol,
            headline=filing.headline,
            body=filing.body[:4000],
            attachment_url=filing.attachment_url,
            label=filing.label,
            confidence=filing.confidence,
            llm_reason=filing.reason,
            est_revenue_impact_pct=filing.est_revenue_impact_pct,
            classify_latency_sec=filing.latency_sec,
        )
        if not is_new:
            return None

        # §6.1: EVERY material item gets an instant alert, whether or not it is
        # tradable. The alert is the product; the trade is the bonus.
        if filing.is_material:
            self.alerts.material_filing(
                filing.symbol, filing.label, filing.confidence, filing.reason,
                headline=filing.headline, latency_sec=filing.latency_sec,
            )
        return filing

    def _extract_pdf_text(self, url: str) -> str:
        """§6.1: extract text from the attached PDF, first 4 pages max."""
        max_pages = int(self.config.get("pdf_max_pages", 4))
        try:
            import httpx
            from pypdf import PdfReader
        except ImportError as exc:
            log.warning("PDF extraction unavailable (%s); classifying on the headline", exc)
            return ""

        try:
            response = httpx.get(url, timeout=20.0, follow_redirects=True)
            if response.status_code >= 400:
                log.warning("PDF fetch %s returned %s", url, response.status_code)
                return ""
            import io

            reader = PdfReader(io.BytesIO(response.content))
            pages = reader.pages[:max_pages]
            return "\n".join((page.extract_text() or "") for page in pages)
        except Exception as exc:
            log.warning("PDF extraction failed for %s: %s", url, exc)
            return ""

    # -- trading ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """Poll, then convert eligible material filings into Signals."""
        signals: list[Signal] = []
        for filing in self.poll(ctx):
            signal = self.build_signal(filing, ctx)
            if signal is not None:
                signals.append(signal)
        return signals

    def build_signal(self, filing: ClassifiedFiling, ctx: Context) -> Optional[Signal]:
        """§6.1 auto-trade gate, then the entry/stop construction.

        Gate (all must hold):
          * stock is in NIFTY-200
          * confidence >= 0.8
          * time is 09:20-14:30
          * the stock has not already gapped > 5% since the filing
          * classification latency <= the configured budget
          * the filing is younger than 10 minutes

        Always evaluated regardless of ``auto_trade``: a backtest is how the
        promotion decision gets made, and gating signal generation on the flag
        the backtest informs would make an unpromoted engine permanently
        unbacktestable. The one authoritative ``auto_trade`` check lives in
        ``live.session.Session.run_cycle``.
        """
        if not filing.is_material:
            return None

        if filing.symbol not in set(self.universe()):
            log.debug("%s not in %s; alert only", filing.symbol, self.config.get("universe"))
            return None

        if filing.confidence < float(self.config.require("min_confidence")):
            return None

        if not self.within_window(ctx, "trade_window"):
            return None

        # §6.1 NOTE: latency > budget -> alert anyway (already done), skip the trade.
        budget = float(self.settings.get("llm.classify_latency_budget_seconds", 20))
        if filing.latency_sec > budget:
            log.info(
                "%s classified in %.1fs (> %.0fs budget); alerted but not trading (§6.1)",
                filing.symbol, filing.latency_sec, budget,
            )
            return None

        # §6.1 NOTE: never trade a filing older than 10 minutes -- the edge is gone.
        max_age = float(self.config.require("max_filing_age_minutes"))
        age = filing.age_minutes(ctx.now)
        if age > max_age:
            log.info("%s filing is %.1f min old (> %.0f); edge is gone (§6.1)",
                     filing.symbol, age, max_age)
            return None

        price = ctx.price(filing.symbol)
        if price is None:
            log.warning("no price for %s; cannot size the trade", filing.symbol)
            return None

        move_pct = self._move_since_filing(filing, ctx, price)
        max_gap = float(self.config.require("max_gap_since_filing_pct"))
        if move_pct is not None and abs(move_pct) > max_gap:
            log.info("%s already moved %.2f%% since the filing (> %.1f%%); skipping",
                     filing.symbol, move_pct, max_gap)
            return None

        bars = ctx.bars_for(filing.symbol, self.config.get("atr_timeframe", "5minute"))
        stop = self._stop_for(filing, ctx, price, bars)
        if stop is None:
            log.warning("no ATR/VWAP available for %s; cannot place a stop", filing.symbol)
            return None

        side = Side.BUY if filing.is_positive else Side.SELL
        risk = abs(price - stop)

        # §6.1: POSITIVE trades may carry `swing_hold` -> hold <= 5 sessions,
        # stop at entry. NEGATIVE is always MIS intraday (§6.1 says "MIS short").
        swing = bool(self.config.get("swing_hold", False)) and filing.is_positive
        ttl = TTL.SWING if swing else TTL.INTRADAY
        if swing:
            stop = price          # §6.1: swing holds move the stop to entry

        return self.signal(
            filing.symbol,
            side,
            stop=stop,
            reference_price=price,
            targets=(price + risk * side.sign,),      # §6.1: book 50% at +1R
            ttl=ttl,
            entry_type=EntryType.MARKET,
            reason=f"{filing.label} ({filing.confidence:.2f}): {filing.reason[:120]}",
            segment=(Segment.EQUITY_DELIVERY if swing else Segment.EQUITY_INTRADAY).value,
            announcement_id=filing.announcement_id,
            label=filing.label,
            confidence=filing.confidence,
            book_fraction_at_1r=float(self.config.get("book_fraction_at_1r", 0.5)),
            trail=str(self.config.get("trail", "vwap")),
            swing_hold=swing,
            max_sessions=int(self.config.get("swing_hold_max_sessions", 5)) if swing else None,
        )

    def _move_since_filing(
        self, filing: ClassifiedFiling, ctx: Context, price: float
    ) -> Optional[float]:
        """Percent move from the bar at the filing time to now."""
        bars = ctx.bars_for(filing.symbol, self.config.get("atr_timeframe", "5minute"))
        if bars is None or bars.empty:
            return None
        at_filing = bars[bars.index <= filing.timestamp]
        if at_filing.empty:
            return None
        reference = float(at_filing["close"].iloc[-1])
        if reference <= 0:
            return None
        return (price - reference) / reference * 100.0

    def _stop_for(
        self,
        filing: ClassifiedFiling,
        ctx: Context,
        price: float,
        bars: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """§6.1 stop: entry -/+ 1.5xATR(5m,14), tightened to VWAP if VWAP is nearer.

        "Tightened" is the operative word: VWAP only replaces the ATR stop when
        it is *closer* to price, never when it is further away. A looser stop
        would quietly increase position size through the §3 sizing formula.
        """
        if bars is None or bars.empty:
            return None

        period = int(self.config.get("atr_period", 14))
        multiple = float(self.config.require("stop_atr_mult"))
        atr_series = atr(bars, period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        atr_value = float(atr_series.iloc[-1])

        if filing.is_positive:
            stop = price - multiple * atr_value
        else:
            stop = price + multiple * atr_value

        vwap_series = vwap(bars)
        if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]):
            vwap_value = float(vwap_series.iloc[-1])
            if filing.is_positive and vwap_value > stop and vwap_value < price:
                stop = vwap_value          # tighter long stop
            elif not filing.is_positive and vwap_value < stop and vwap_value > price:
                stop = vwap_value          # tighter short stop

        return round(stop, 2)

    # -- management -------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.1: book 50% at +1R, trail the rest by VWAP, force-flat 15:10.

        The 15:10 force-flat is enforced by the §3 kernel for the whole book;
        what this method owns is the +1R scale-out and the VWAP trail.
        """
        signals: list[Signal] = []
        for position in ctx.positions_for_engine(self.name):
            price = ctx.price(position.symbol)
            if price is None:
                continue

            if position.meta.get("swing_hold"):
                signal = self._manage_swing(position, ctx, price)
                if signal is not None:
                    signals.append(signal)
                continue

            entry = position.average_price
            stop = position.stop
            direction = 1 if position.is_long else -1

            # +1R scale-out, once.
            if stop is not None and not position.meta.get("booked_1r"):
                risk = abs(entry - stop)
                target = entry + risk * direction
                reached = price >= target if position.is_long else price <= target
                if reached and risk > 0:
                    fraction = float(self.config.get("book_fraction_at_1r", 0.5))
                    quantity = max(int(abs(position.quantity) * fraction), 1)
                    position.meta["booked_1r"] = True
                    signals.append(self._exit_signal(
                        position, price, quantity, f"+1R scale-out ({fraction:.0%})"
                    ))

            # VWAP trail on the remainder.
            bars = ctx.bars_for(position.symbol, self.config.get("atr_timeframe", "5minute"))
            if bars is not None and not bars.empty:
                vwap_series = vwap(bars)
                if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]):
                    vwap_value = float(vwap_series.iloc[-1])
                    broke = price < vwap_value if position.is_long else price > vwap_value
                    if broke:
                        signals.append(self._exit_signal(
                            position, price, abs(position.quantity), "VWAP trail broken"
                        ))
        return signals

    def _manage_swing(self, position: Any, ctx: Context, price: float) -> Optional[Signal]:
        """§6.1 swing holds: <= 5 sessions, stop at entry."""
        max_sessions = position.meta.get("max_sessions") or int(
            self.config.get("swing_hold_max_sessions", 5)
        )
        opened = position.opened_at
        if opened is not None:
            from core import calendar as trading_calendar

            held = len(trading_calendar.trading_days_between(opened.date(), ctx.today)) - 1
            if held >= max_sessions:
                return self._exit_signal(
                    position, price, abs(position.quantity),
                    f"swing time exit after {held} sessions (§6.1)",
                )
        return None

    def _exit_signal(self, position: Any, price: float, quantity: int, reason: str) -> Signal:
        return self.signal(
            position.symbol,
            Side.SELL if position.is_long else Side.BUY,
            stop=None,
            reference_price=price,
            ttl=position.ttl,
            reason=reason,
            segment=position.segment.value,
            exit=True,
            quantity=quantity,
        )
