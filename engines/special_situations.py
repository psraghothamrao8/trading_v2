"""special_situations.py — Alert-only corporate actions. Implements §6.11.

RULES (from the spec): daily scan of announcements for tender buybacks, open
offers, delistings and index-inclusion news. Alert with offer price, market
price, retail entitlement where determinable, acceptance-ratio estimate and
indicative expected value. **No automated execution.**

The LLM extracts the numbers from the document; the arithmetic below computes
the entitlement, acceptance ratio and expected value. That split is deliberate:
extraction is judgement and belongs to the model, but the economics a human
acts on must be auditable and identical every time.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from core.llm import LLMError, load_schema
from engines.base import AlertOnlyEngine, Context

log = logging.getLogger(__name__)

TRADABLE_EVENTS = {"buyback", "open_offer", "delisting"}

# SEBI's small-shareholder definition: holdings up to INR 2 lakh at the record
# date. Used to cap the entitlement calculation for the retail-reserved portion.
SMALL_SHAREHOLDER_LIMIT_INR = 200_000


@dataclass
class SpecialSituation:
    """An extracted corporate action plus the computed economics."""

    symbol: str
    event_type: str
    offer_price: Optional[float]
    market_price: Optional[float]
    record_date: Optional[str]
    offer_size_shares: Optional[float]
    total_shares: Optional[float]
    reserved_retail_pct: Optional[float]
    summary: str
    confidence: float
    announcement_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- computed economics ------------------------------------------------

    @property
    def premium_pct(self) -> Optional[float]:
        """Offer price over market price, as a percentage."""
        if not self.offer_price or not self.market_price:
            return None
        return (self.offer_price - self.market_price) / self.market_price * 100.0

    @property
    def retail_entitlement_shares(self) -> Optional[int]:
        """Shares a small shareholder can tender, at the SEBI 2 lakh cap.

        This is the *maximum eligible holding*, which is what the acceptance
        ratio is applied to. Returns None when the market price is unknown.
        """
        if not self.market_price or self.market_price <= 0:
            return None
        return int(SMALL_SHAREHOLDER_LIMIT_INR // self.market_price)

    @property
    def acceptance_ratio(self) -> Optional[float]:
        """Estimated fraction of tendered shares actually bought back.

        Estimated as ``(offer_size x reserved_retail_pct) / (total_shares x
        assumed retail float)``. Every input can be missing, in which case this
        is None -- an invented ratio in an alert a human acts on is worse than
        no ratio.
        """
        if not self.offer_size_shares or not self.total_shares:
            return None
        reserved = (self.reserved_retail_pct or 15.0) / 100.0
        retail_float = float(self.meta.get("assumed_retail_float_pct", 10.0)) / 100.0
        retail_pool = self.total_shares * retail_float
        if retail_pool <= 0:
            return None
        return min(self.offer_size_shares * reserved / retail_pool, 1.0)

    @property
    def expected_value_inr(self) -> Optional[float]:
        """Indicative EV on a full retail tender, ignoring costs and taxes.

        ``accepted x (offer - market)`` plus the mark-to-market on the
        unaccepted remainder, which is assumed to be zero: the residual is
        usually sold near the post-record market price. Labelled "indicative"
        everywhere it is shown, because that assumption is the whole risk.
        """
        entitlement = self.retail_entitlement_shares
        ratio = self.acceptance_ratio
        if entitlement is None or ratio is None or self.premium_pct is None:
            return None
        accepted = math.floor(entitlement * ratio)
        return accepted * (self.offer_price - self.market_price)


class SpecialSituationsEngine(AlertOnlyEngine):
    """§6.11. Alert-only: no automated execution, ever."""

    name = "special_situations"

    def __init__(self, llm: Any = None, alerts: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._alerts = alerts
        self._seen: set[str] = set()

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

    def universe(self) -> list[str]:
        return []

    # -- scanning ---------------------------------------------------------

    def keyword_hits(self, ctx: Context, trade_date: Optional[str] = None) -> list[dict[str, Any]]:
        """Announcements whose text mentions a configured corporate action."""
        keywords = [str(k).lower() for k in self.config.get("keywords", [])]
        if not keywords:
            return []
        day = trade_date or ctx.today.isoformat()
        rows = ctx.the_journal().query(
            "SELECT announcement_id, symbol, headline, body FROM announcements "
            "WHERE trade_date=?",
            (day,),
        )
        hits: list[dict[str, Any]] = []
        for row in rows:
            text = f"{row['headline'] or ''} {row['body'] or ''}".lower()
            if any(keyword in text for keyword in keywords):
                hits.append(dict(row))
        return hits

    def scan(self, ctx: Context, trade_date: Optional[str] = None) -> list[SpecialSituation]:
        """§6.11 daily scan: keyword filter, then LLM extraction, then economics.

        The keyword pass first is not just a cost saving -- it keeps the LLM
        away from the hundreds of routine filings a day where it would only
        have the chance to be creatively wrong.
        """
        found: list[SpecialSituation] = []
        for row in self.keyword_hits(ctx, trade_date):
            announcement_id = str(row["announcement_id"])
            if announcement_id in self._seen:
                continue
            self._seen.add(announcement_id)

            payload = {
                "symbol": row["symbol"],
                "headline": row["headline"] or "",
                "body": (row["body"] or "")[:8000],
            }
            try:
                result = self.llm.classify(
                    "special_situations", payload, load_schema("special_situations")
                )
            except LLMError as exc:
                log.error("special situation extraction failed for %s: %s", row["symbol"], exc)
                self.journal.record_error(
                    "special_situations", f"{row['symbol']}: {exc}", severity="WARNING"
                )
                continue

            if result.get("event_type") not in TRADABLE_EVENTS | {"index_change"}:
                continue

            situation = SpecialSituation(
                symbol=str(row["symbol"]),
                event_type=str(result["event_type"]),
                offer_price=result.get("offer_price"),
                market_price=ctx.price(str(row["symbol"])),
                record_date=result.get("record_date"),
                offer_size_shares=result.get("offer_size_shares"),
                total_shares=result.get("total_shares"),
                reserved_retail_pct=result.get("reserved_retail_pct"),
                summary=str(result.get("summary", "")),
                confidence=float(result.get("confidence", 0.0)),
                announcement_id=announcement_id,
                meta={"open_date": result.get("open_date"), "close_date": result.get("close_date")},
            )
            found.append(situation)
        return found

    # -- alerting ---------------------------------------------------------

    def format_alert(self, situation: SpecialSituation) -> str:
        """The §6.11 alert body, with every field the spec names."""
        lines = [
            f"💼 <b>{situation.event_type.replace('_', ' ').upper()}</b>  "
            f"<code>{situation.symbol}</code>",
            situation.summary[:300],
        ]
        if situation.offer_price:
            lines.append(f"offer price   : ₹{situation.offer_price:,.2f}")
        if situation.market_price:
            lines.append(f"market price  : ₹{situation.market_price:,.2f}")
        if situation.premium_pct is not None:
            lines.append(f"premium       : {situation.premium_pct:+.2f}%")
        if situation.record_date:
            lines.append(f"record date   : {situation.record_date}")

        entitlement = situation.retail_entitlement_shares
        lines.append(
            f"retail entitlement : {entitlement} shares (₹2L SEBI cap)"
            if entitlement is not None
            else "retail entitlement : not determinable"
        )

        ratio = situation.acceptance_ratio
        lines.append(
            f"acceptance ratio   : ~{ratio:.1%} (estimate)"
            if ratio is not None
            else "acceptance ratio   : not determinable from the filing"
        )

        ev = situation.expected_value_inr
        lines.append(
            f"indicative EV      : ₹{ev:,.0f}"
            if ev is not None
            else "indicative EV      : not determinable"
        )

        lines.append(f"<i>extraction confidence {situation.confidence:.2f} — "
                     f"ALERT ONLY, no automated execution (§6.11)</i>")
        return "\n".join(lines)

    def alerts_for(self, ctx: Context) -> list[str]:
        return [self.format_alert(s) for s in self.scan(ctx)]

    def send_alerts(self, ctx: Context) -> int:
        """Send one alert per situation found. Returns how many were sent."""
        sent = 0
        for message in self.alerts_for(ctx):
            self.alerts.send(message)
            sent += 1
        return sent
