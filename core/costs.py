"""Indian cost model — §4. Gross backtests are lies.

Every backtest fill and every paper fill runs through this module. It
implements Zerodha's full charge structure as a **config-driven table**:
brokerage, STT/CTT, exchange transaction charges, SEBI fees, stamp duty, GST,
DP charges, plus the §4 slippage model.

Nothing here is a literal. Every rate lives in ``settings.yaml -> costs`` with
an ``as_of`` date, because the government and the exchanges change them --
often mid-year, and always without warning to a backtest written last year.

Charge structure (the arithmetic, once, so the code below reads as arithmetic
and not as folklore)::

    brokerage      per executed order, per the segment's rule
    stt            per the segment's rate, on the side(s) that pay
    exchange       turnover (or premium for options) x rate
    sebi           turnover x rate
    stamp duty     buy side only, turnover x rate
    gst            18% of (brokerage + exchange + sebi)
    dp             equity delivery SELL only, flat per scrip per day

Options charge on **premium**, not on notional. Getting that wrong inflates
option costs by orders of magnitude and makes every option strategy look
unprofitable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from core.config import ConfigError, Settings, get_settings
from core.types import Segment, Side

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Charges:
    """Itemised charges for one executed order. All values in INR."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    dp: float = 0.0
    turnover: float = 0.0
    segment: str = ""
    side: str = ""

    @property
    def total(self) -> float:
        return round(
            self.brokerage + self.stt + self.exchange + self.sebi
            + self.stamp_duty + self.gst + self.dp,
            2,
        )

    def breakdown(self) -> dict[str, float]:
        """Itemised view, for the digest and for calculator comparison."""
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange": round(self.exchange, 2),
            "sebi": round(self.sebi, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "gst": round(self.gst, 2),
            "dp": round(self.dp, 2),
            "total": self.total,
        }

    def __add__(self, other: "Charges") -> "Charges":
        return Charges(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange=self.exchange + other.exchange,
            sebi=self.sebi + other.sebi,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            gst=self.gst + other.gst,
            dp=self.dp + other.dp,
            turnover=self.turnover + other.turnover,
            segment=self.segment or other.segment,
            side="both",
        )


@dataclass
class RoundTrip:
    """Entry + exit costs plus net P&L for a completed trade."""

    entry: Charges
    exit: Charges
    gross_pnl: float
    slippage: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def costs(self) -> float:
        return round(self.entry.total + self.exit.total, 2)

    @property
    def net_pnl(self) -> float:
        """§4: net of costs. This is the only P&L number that means anything."""
        return round(self.gross_pnl - self.costs, 2)

    def breakdown(self) -> dict[str, Any]:
        return {
            "gross_pnl": round(self.gross_pnl, 2),
            "entry": self.entry.breakdown(),
            "exit": self.exit.breakdown(),
            "costs": self.costs,
            "net_pnl": self.net_pnl,
        }


class CostModel:
    """Config-driven Zerodha charge calculator (§4)."""

    def __init__(self, settings: Settings | None = None) -> None:
        root = settings or get_settings()
        self.settings = root.section("costs")
        self.as_of = str(self.settings.get("as_of", "unknown"))
        if not self.settings.get("verified_against_calculator", False):
            log.info(
                "Cost model rates as_of %s are not yet verified against Zerodha's "
                "brokerage calculator (costs.verified_against_calculator=false). "
                "See tests/test_costs.py for the three sample trades to check.",
                self.as_of,
            )

    # -- individual components -------------------------------------------

    def brokerage(self, segment: Segment, turnover: float) -> float:
        """Per executed order. ₹0 on equity delivery; min(0.03%, ₹20) elsewhere."""
        rule = self.settings.get(f"brokerage.{segment.value}", None)
        if rule is None:
            raise ConfigError(
                f"No brokerage rule configured for segment {segment.value!r}. "
                f"Add it to settings.yaml `costs.brokerage`."
            )
        kind = str(rule.get("type"))
        if kind == "flat":
            return float(rule["value"])
        if kind == "min_of":
            return min(turnover * float(rule["pct"]) / 100.0, float(rule["flat"]))
        raise ConfigError(f"Unknown brokerage rule type {kind!r} for {segment.value}")

    def stt(
        self,
        segment: Segment,
        side: Side,
        turnover: float,
        premium: float | None = None,
        intrinsic: float | None = None,
        exercised: bool = False,
    ) -> float:
        """STT/CTT. Options charge on premium; exercised options on intrinsic.

        The exercised branch is the §3 "STT trap" made arithmetic: a long ITM
        option carried into expiry pays on intrinsic value, which for a deep
        ITM contract dwarfs the premium-based charge on a normal sell.
        """
        if exercised:
            rule = self.settings.get("stt.options_exercise", None)
            if rule is None:
                raise ConfigError("costs.stt.options_exercise is not configured")
            if side is not Side.BUY:
                return 0.0
            basis = intrinsic if intrinsic is not None else turnover
            return float(basis) * float(rule["rate"])

        rule = self.settings.get(f"stt.{segment.value}", None)
        if rule is None:
            raise ConfigError(f"No STT rule configured for segment {segment.value!r}")

        applies_to = str(rule.get("side", "both")).lower()
        if applies_to == "sell" and side is not Side.SELL:
            return 0.0
        if applies_to == "buy" and side is not Side.BUY:
            return 0.0

        basis_name = str(rule.get("basis", "turnover"))
        if basis_name == "premium":
            basis = premium if premium is not None else turnover
        elif basis_name == "intrinsic":
            basis = intrinsic if intrinsic is not None else 0.0
        else:
            basis = turnover
        return float(basis) * float(rule["rate"])

    def exchange_charges(self, segment: Segment, turnover: float, premium: float | None = None) -> float:
        """NSE transaction charges. Options are charged on premium."""
        rate = self.settings.get(f"exchange_transaction_charges.{segment.value}", None)
        if rate is None:
            raise ConfigError(
                f"No exchange transaction charge configured for {segment.value!r}"
            )
        basis = premium if (segment is Segment.EQUITY_OPTIONS and premium is not None) else turnover
        return float(basis) * float(rate)

    def sebi_charges(self, turnover: float) -> float:
        """₹10 per crore, all segments."""
        return turnover * float(self.settings.require("sebi_charges.rate"))

    def stamp_duty(self, segment: Segment, side: Side, turnover: float) -> float:
        """Buy side only, per segment."""
        if side is not Side.BUY:
            return 0.0
        rate = self.settings.get(f"stamp_duty.{segment.value}", None)
        if rate is None:
            raise ConfigError(f"No stamp duty configured for {segment.value!r}")
        return turnover * float(rate)

    def gst(self, brokerage: float, exchange: float, sebi: float) -> float:
        """18% on (brokerage + exchange transaction charges + SEBI fees)."""
        return (brokerage + exchange + sebi) * float(self.settings.require("gst.rate"))

    def dp_charges(self, segment: Segment, side: Side) -> float:
        """Flat per scrip per day, equity delivery SELL only."""
        if segment is not Segment.EQUITY_DELIVERY or side is not Side.SELL:
            return 0.0
        return float(self.settings.require("dp_charges.per_scrip_per_day_sell"))

    # -- the entry point --------------------------------------------------

    def charges(
        self,
        segment: Segment,
        side: Side,
        price: float,
        quantity: int,
        *,
        exercised: bool = False,
        intrinsic_value: float | None = None,
    ) -> Charges:
        """All charges for one executed order.

        For options, ``price`` is the **premium per unit**, so ``turnover`` is
        the premium value -- which is what STT and exchange charges apply to.
        """
        turnover = abs(price * quantity)
        premium = turnover if segment is Segment.EQUITY_OPTIONS else None

        brokerage = self.brokerage(segment, turnover)
        stt = self.stt(
            segment, side, turnover,
            premium=premium,
            intrinsic=intrinsic_value,
            exercised=exercised,
        )
        exchange = self.exchange_charges(segment, turnover, premium=premium)
        sebi = self.sebi_charges(turnover)
        stamp = self.stamp_duty(segment, side, turnover)
        gst = self.gst(brokerage, exchange, sebi)
        dp = self.dp_charges(segment, side)

        return Charges(
            brokerage=brokerage, stt=stt, exchange=exchange, sebi=sebi,
            stamp_duty=stamp, gst=gst, dp=dp, turnover=turnover,
            segment=segment.value, side=side.value,
        )

    def round_trip(
        self,
        segment: Segment,
        side: Side,
        entry_price: float,
        exit_price: float,
        quantity: int,
        *,
        exercised: bool = False,
        intrinsic_value: float | None = None,
    ) -> RoundTrip:
        """Costs and net P&L for a complete trade. ``side`` is the ENTRY side."""
        entry = self.charges(segment, side, entry_price, quantity)
        exit_charges = self.charges(
            segment, side.opposite, exit_price, quantity,
            exercised=exercised, intrinsic_value=intrinsic_value,
        )
        gross = (exit_price - entry_price) * quantity * side.sign
        return RoundTrip(entry=entry, exit=exit_charges, gross_pnl=gross)

    # -- slippage (§4) ----------------------------------------------------

    def slippage_per_unit(self, segment: Segment, price: float) -> float:
        """§4: 0.03%/side liquid equity, 0.05%/side options, 1 tick minimum."""
        slip = self.settings.section("slippage")
        is_option = segment is Segment.EQUITY_OPTIONS
        pct = float(slip.require("options_pct" if is_option else "liquid_equity_pct")) / 100.0
        tick = float(
            slip.require("tick_size_options" if is_option else "tick_size_equity")
        )
        min_ticks = int(slip.require("minimum_ticks"))
        return max(price * pct, tick * min_ticks)

    def apply_slippage(self, segment: Segment, side: Side, price: float) -> float:
        """Adjust a reference price by slippage. Slippage always costs money."""
        slip = self.slippage_per_unit(segment, price)
        adjusted = price + slip if side is Side.BUY else price - slip
        tick = float(
            self.settings.require(
                "slippage.tick_size_options" if segment is Segment.EQUITY_OPTIONS
                else "slippage.tick_size_equity"
            )
        )
        return round(max(adjusted, tick), 2)

    def round_to_tick(self, segment: Segment, price: float) -> float:
        """Snap a price to the instrument's tick size."""
        tick = float(
            self.settings.require(
                "slippage.tick_size_options" if segment is Segment.EQUITY_OPTIONS
                else "slippage.tick_size_equity"
            )
        )
        return round(round(price / tick) * tick, 2)


_default: Optional[CostModel] = None


def get_cost_model() -> CostModel:
    """Process-wide cost model singleton."""
    global _default
    if _default is None:
        _default = CostModel()
    return _default


def set_cost_model(model: Optional[CostModel]) -> None:
    global _default
    _default = model
