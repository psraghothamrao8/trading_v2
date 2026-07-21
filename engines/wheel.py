"""wheel.py — Cash-secured puts / covered calls. Implements §6.8.

WHY (from the spec): selling puts on stocks I'd happily own converts waiting
into premium. The IV gate ensures I only sell insurance when it's expensive.

Two rules make this engine unusual and both are deliberate:

1. **Every order is proposed via Telegram and requires confirmation, even in
   paper.** Stock options settle physically (§8.5); an assignment is a real
   delivery obligation, so no wheel order is ever placed unattended.
2. **Assignment is never accepted automatically.** The §3 physical-settlement
   guard will force a close/roll unless the owner explicitly allows delivery.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from core import calendar as trading_calendar
from core.datafeed import percentile_rank
from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class WheelProposal:
    """A wheel order awaiting the owner's Telegram confirmation (§6.8)."""

    request_id: str
    symbol: str
    action: str                 # sell_put | buy_back | roll | covered_call | accept_assignment
    strike: float
    expiry: _dt.date
    option_type: str            # CE | PE
    premium: float
    lots: int
    description: str
    created_at: _dt.datetime
    confirmed: Optional[bool] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pending(self) -> bool:
        return self.confirmed is None


class WheelEngine(Engine):
    """§6.8. Alert-first by construction: nothing trades without confirmation."""

    name = "wheel"

    def __init__(self, broker_quotes: Any = None, alerts: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._quotes = broker_quotes
        self._alerts = alerts
        self.proposals: dict[str, WheelProposal] = {}

    @property
    def alerts(self) -> Any:
        if self._alerts is None:
            from live.alerts import get_alerts

            self._alerts = get_alerts()
        return self._alerts

    def universe(self) -> list[str]:
        """§6.8: the owner-approved list only. Never anything else."""
        from core.config import get_universe

        return [str(w["symbol"]) for w in get_universe().get("wheel_approved", [])]

    # -- the IV gate ------------------------------------------------------

    def vix_percentile(self, ctx: Context) -> Optional[float]:
        """India VIX 1-year percentile (§6.8 gate).

        Returns None when there is not enough VIX history stored -- and the
        engine then does nothing, because a percentile computed from 12
        observations is not a percentile.
        """
        current = ctx.india_vix
        if current is None:
            return None

        lookback = int(self.config.require("vix_percentile_lookback_days"))
        rows = ctx.the_journal().query(
            "SELECT meta FROM surveillance_snapshots WHERE list_name='india_vix' "
            "ORDER BY trade_date DESC LIMIT ?", (lookback,)
        )
        history = ctx.extras.get("india_vix_history")
        if history is None and rows:
            import json

            history = []
            for row in rows:
                try:
                    history.append(float(json.loads(row["meta"] or "{}").get("value")))
                except (TypeError, ValueError):
                    continue

        if not history or len(history) < 30:
            log.info(
                "wheel: only %d India VIX observations available; the 1-year "
                "percentile gate (§6.8) is not meaningful yet, so no orders.",
                len(history or []),
            )
            return None
        return percentile_rank(pd.Series(history), current)

    def iv_gate_open(self, ctx: Context) -> bool:
        """§6.8: run only when India VIX 1-year percentile >= 50."""
        percentile = self.vix_percentile(ctx)
        if percentile is None:
            return False
        minimum = float(self.config.require("min_vix_percentile"))
        if percentile < minimum:
            log.info("wheel: VIX percentile %.1f < %.0f; premium is not expensive enough",
                     percentile, minimum)
            return False
        return True

    # -- proposals --------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.8 proposes; it never returns a tradable Signal directly.

        Every wheel order goes out as a Telegram confirmation request. The
        orchestrator turns a *confirmed* proposal into a Signal via
        :meth:`signal_for_confirmed`. Returning `[]` here is the mechanism, not
        an omission.
        """
        if not self.enabled:
            return []

        self.propose_new_puts(ctx)
        self.propose_management(ctx)
        return []

    def propose_new_puts(self, ctx: Context) -> list[WheelProposal]:
        """§6.8: sell 1 lot cash-secured put ~0.25 delta, 30-45 DTE."""
        if not self.iv_gate_open(ctx):
            return []

        # §6.9: at an extreme-high FII positioning percentile, halve premium
        # selling size. With `lots: 1` configured that means: do not sell.
        if ctx.extras.get("flows_halve_premium_selling"):
            log.info("wheel: §6.9 says halve premium selling; 1 lot cannot halve, so skipping")
            return []
        if ctx.extras.get("premium_selling_disabled"):
            log.info("wheel: premium selling disabled in this regime (§7)")
            return []

        out: list[WheelProposal] = []
        for symbol in self.universe():
            if self._has_open_option(ctx, symbol):
                continue
            if any(p.pending and p.symbol == symbol for p in self.proposals.values()):
                continue

            contract = self._select_contract(ctx, symbol, "PE")
            if contract is None:
                continue

            proposal = self._make_proposal(
                symbol=symbol, action="sell_put", contract=contract, ctx=ctx,
                description=(
                    f"SELL {contract['lots']} lot cash-secured PUT {symbol} "
                    f"{contract['strike']:.0f} PE exp {contract['expiry']} "
                    f"@ ~{contract['premium']:.2f} (delta ~{contract.get('delta', '?')})\n"
                    f"Assignment means taking delivery of {symbol} — physical settlement (§8.5)."
                ),
            )
            out.append(proposal)
        return out

    def propose_management(self, ctx: Context) -> list[WheelProposal]:
        """§6.8: buy back at 50% of credit; roll down-and-out if breached."""
        out: list[WheelProposal] = []
        buy_back_at = float(self.config.require("buy_back_at_credit_fraction"))
        roll_dte = int(self.config.require("roll_if_breached_and_dte_gt"))

        for position in ctx.positions_for_engine(self.name):
            if position.segment is not Segment.EQUITY_OPTIONS or not position.is_short:
                continue

            credit = float(position.meta.get("entry_premium") or position.average_price)
            current = ctx.price(position.symbol)
            if current is None or credit <= 0:
                continue

            expiry = _as_date(position.meta.get("expiry"))
            dte = (
                len(trading_calendar.trading_days_between(ctx.today, expiry)) - 1
                if expiry else None
            )

            if current <= credit * buy_back_at:
                out.append(self._make_proposal(
                    symbol=position.symbol, action="buy_back",
                    contract={
                        "strike": position.meta.get("strike"), "expiry": expiry,
                        "option_type": position.meta.get("option_type"),
                        "premium": current, "lots": abs(position.quantity),
                        "tradingsymbol": position.symbol,
                    },
                    ctx=ctx,
                    description=(
                        f"BUY BACK {position.symbol} at {current:.2f}, which is "
                        f"{current / credit:.0%} of the {credit:.2f} credit "
                        f"(target {buy_back_at:.0%}). Locks in the win."
                    ),
                ))
                continue

            strike = position.meta.get("strike")
            underlying = position.meta.get("underlying") or position.symbol
            spot = ctx.price(str(underlying))
            breached = (
                strike is not None and spot is not None
                and str(position.meta.get("option_type", "")).upper() == "PE"
                and spot < float(strike)
            )
            if breached and dte is not None and dte > roll_dte:
                out.append(self._make_proposal(
                    symbol=position.symbol, action="roll",
                    contract={
                        "strike": strike, "expiry": expiry,
                        "option_type": position.meta.get("option_type"),
                        "premium": current, "lots": abs(position.quantity),
                        "tradingsymbol": position.symbol,
                    },
                    ctx=ctx,
                    description=(
                        f"ROLL down-and-out: {underlying} spot {spot:,.2f} has breached "
                        f"the {strike} strike with {dte} sessions to expiry."
                    ),
                ))
        return out

    def _make_proposal(
        self, symbol: str, action: str, contract: dict[str, Any], ctx: Context, description: str
    ) -> WheelProposal:
        request_id = uuid.uuid4().hex[:8]
        proposal = WheelProposal(
            request_id=request_id,
            symbol=str(contract.get("tradingsymbol") or symbol),
            action=action,
            strike=float(contract.get("strike") or 0.0),
            expiry=contract.get("expiry") or ctx.today,
            option_type=str(contract.get("option_type") or "PE"),
            premium=float(contract.get("premium") or 0.0),
            lots=int(contract.get("lots") or 1),
            description=description,
            created_at=ctx.now,
            meta={"underlying": symbol, **contract},
        )
        self.proposals[request_id] = proposal
        # §6.8: EVERY wheel order is proposed via Telegram and requires
        # confirmation, even in paper.
        self.alerts.confirmation_request(request_id, description)
        log.info("wheel proposal %s: %s", request_id, description.splitlines()[0])
        return proposal

    def confirm(self, request_id: str, approved: bool) -> Optional[WheelProposal]:
        """Record the owner's Telegram answer. Called by the /confirm handler."""
        proposal = self.proposals.get(request_id)
        if proposal is None:
            return None
        proposal.confirmed = approved
        log.info("wheel proposal %s %s", request_id, "CONFIRMED" if approved else "REJECTED")
        return proposal

    def signal_for_confirmed(self, request_id: str, ctx: Context) -> Optional[Signal]:
        """Turn a confirmed proposal into a Signal. The only path to an order."""
        proposal = self.proposals.get(request_id)
        if proposal is None or proposal.confirmed is not True:
            return None

        side = Side.SELL if proposal.action in ("sell_put", "covered_call") else Side.BUY
        # Options have no meaningful stop; the §3 kernel sizes from meta.quantity.
        lot = trading_calendar.lot_size(str(proposal.meta.get("underlying", proposal.symbol))) \
            if proposal.meta.get("use_lot_size") else 1

        return self.signal(
            proposal.symbol, side,
            stop=None,
            reference_price=proposal.premium,
            ttl=TTL.SWING,
            entry_type=EntryType.LIMIT,
            reason=f"wheel {proposal.action} (confirmed {request_id})",
            segment=Segment.EQUITY_OPTIONS.value,
            option_type=proposal.option_type,
            strike=proposal.strike,
            expiry=proposal.expiry.isoformat() if hasattr(proposal.expiry, "isoformat") else proposal.expiry,
            underlying=proposal.meta.get("underlying"),
            underlying_type="stock",
            instrument_type=proposal.option_type,
            entry_premium=proposal.premium,
            quantity=proposal.lots * lot,
            request_id=request_id,
            # §3 physical-settlement guard: never bypassed without an explicit
            # per-trade opt-in from the owner.
            allow_delivery=bool(proposal.meta.get("allow_delivery", False)),
        )

    # -- contract selection -----------------------------------------------

    def _select_contract(self, ctx: Context, symbol: str, option_type: str) -> Optional[dict[str, Any]]:
        """Pick a ~0.25 delta contract 30-45 DTE from the injected chain.

        The chain comes from ``ctx.extras['option_chains']``, populated by the
        orchestrator from the broker. Engines do not call the broker (§0.3).
        """
        chains = ctx.extras.get("option_chains") or {}
        chain = chains.get(symbol)
        if not chain:
            log.debug("wheel: no option chain available for %s", symbol)
            return None

        low_dte, high_dte = self.config.require("dte_range")
        target_delta = float(self.config.require("target_delta"))

        eligible = []
        for row in chain:
            if str(row.get("instrument_type", "")).upper() != option_type:
                continue
            expiry = _as_date(row.get("expiry"))
            if expiry is None:
                continue
            dte = (expiry - ctx.today).days
            if not (low_dte <= dte <= high_dte):
                continue
            delta = row.get("delta")
            if delta is None:
                continue
            eligible.append((abs(abs(float(delta)) - target_delta), row, expiry))

        if not eligible:
            return None
        _, best, expiry = min(eligible, key=lambda item: item[0])
        return {
            "tradingsymbol": best.get("tradingsymbol"),
            "strike": float(best.get("strike", 0.0)),
            "expiry": expiry,
            "option_type": option_type,
            "premium": float(best.get("last_price") or best.get("premium") or 0.0),
            "delta": best.get("delta"),
            "lots": int(self.config.get("lots", 1)),
            "use_lot_size": True,
        }

    def _has_open_option(self, ctx: Context, underlying: str) -> bool:
        return any(
            position.meta.get("underlying") == underlying
            for position in ctx.positions_for_engine(self.name)
        )


def _as_date(value: Any) -> Optional[_dt.date]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
