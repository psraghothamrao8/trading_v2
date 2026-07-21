"""Domain types shared by engines, the risk kernel, the broker and the journal.

``Signal`` is specified in §6.0. Everything else exists so that the one-way
flow -- engine -> Signal -> risk.check() -> broker -- can be expressed in
types rather than in convention.
"""

from __future__ import annotations

import datetime as _dt
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class Side(str, enum.Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for a long, -1 for a short. Used in P&L arithmetic."""
        return 1 if self is Side.BUY else -1


class EntryType(str, enum.Enum):
    """How the order reaches the book."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"           # stop-loss limit
    SL_M = "SL-M"       # stop-loss market


class Product(str, enum.Enum):
    """§8.6 product types.

    MIS  -- intraday equity/short (auto square-off ~15:20; our 15:10 fires first)
    CNC  -- overnight equity (delivery)
    NRML -- overnight F&O
    """

    MIS = "MIS"
    CNC = "CNC"
    NRML = "NRML"


class Segment(str, enum.Enum):
    """Cost-model segment (§4). Drives every charge lookup in ``core.costs``."""

    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"
    EQUITY_FUTURES = "equity_futures"
    EQUITY_OPTIONS = "equity_options"


class TTL(str, enum.Enum):
    """How long a position is meant to live. The broker wrapper derives the
    product type from this (§8.6), so it is part of the Signal contract."""

    INTRADAY = "INTRADAY"     # flat by 15:10  -> MIS
    OVERNIGHT = "OVERNIGHT"   # held to next session -> CNC / NRML
    SWING = "SWING"           # multi-session  -> CNC / NRML
    GTT = "GTT"               # resting order


class Regime(str, enum.Enum):
    """§7 regime classification."""

    NA = "NA"
    TREND = "TREND"
    CHOP = "CHOP"
    PANIC = "PANIC"


class Verdict(str, enum.Enum):
    """Result of ``core.risk.check()``."""

    ALLOW = "ALLOW"
    REJECT = "REJECT"


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class Signal:
    """§6.0 -- the only thing an engine may emit.

    Engines never place orders. The orchestrator converts a Signal into an
    :class:`Order` and submits it to ``core.risk.check()``.

    ``targets`` is an ordered list of price levels; the manage() loop books
    ``book_fraction`` at each. ``meta`` carries engine-specific payload the
    kernel may inspect (e.g. ``allow_delivery`` for the §3 physical-settlement
    guard, or ``option_type``/``strike`` for option signals).
    """

    symbol: str
    side: Side
    entry_type: EntryType
    stop: Optional[float]
    targets: tuple[float, ...]
    ttl: TTL
    reason: str
    engine: str
    meta: dict[str, Any] = field(default_factory=dict)

    signal_id: str = field(default_factory=_new_id)
    created_at: Optional[_dt.datetime] = None
    limit_price: Optional[float] = None
    reference_price: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Signal.symbol is required")
        if self.entry_type is EntryType.LIMIT and self.limit_price is None:
            raise ValueError(f"{self.engine}: LIMIT signal for {self.symbol} needs limit_price")

    @property
    def is_option(self) -> bool:
        """True when the signal targets an option contract."""
        return bool(self.meta.get("option_type")) or self.meta.get("instrument_type") in {"CE", "PE"}

    @property
    def is_derivative(self) -> bool:
        """True for futures or options -- what the §3 ban-list veto applies to."""
        return self.is_option or self.meta.get("instrument_type") in {"FUT", "FUTIDX", "FUTSTK"}

    @property
    def allow_delivery(self) -> bool:
        """§3 physical-settlement guard opt-out, set per trade by the owner."""
        return bool(self.meta.get("allow_delivery", False))

    @property
    def risk_per_unit(self) -> Optional[float]:
        """``|entry - stop|`` -- the denominator of the §3 sizing formula."""
        if self.stop is None or self.reference_price is None:
            return None
        return abs(self.reference_price - self.stop)


@dataclass
class Order:
    """A sized, routable instruction. Only the orchestrator constructs these."""

    symbol: str
    side: Side
    quantity: int
    entry_type: EntryType
    product: Product
    engine: str
    signal_id: str
    segment: Segment = Segment.EQUITY_INTRADAY
    price: Optional[float] = None           # limit price
    trigger_price: Optional[float] = None   # SL / SL-M
    stop: Optional[float] = None
    ttl: TTL = TTL.INTRADAY
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    order_id: str = field(default_factory=_new_id)
    broker_order_id: Optional[str] = None
    created_at: Optional[_dt.datetime] = None

    # Set to False by the orchestrator when the order closes an existing
    # position. Most §3 vetoes apply only to NEW entries -- a veto that blocks
    # an exit would trap us in exactly the stock we are trying to leave.
    is_entry: bool = True

    @property
    def notional(self) -> float:
        """Best-effort exposure, for the per-engine capital caps."""
        ref = self.price if self.price is not None else self.meta.get("reference_price")
        return float(ref or 0.0) * self.quantity

    @property
    def is_derivative(self) -> bool:
        return self.segment in {Segment.EQUITY_FUTURES, Segment.EQUITY_OPTIONS}


@dataclass(frozen=True)
class Fill:
    """An executed (or simulated) fill, after costs."""

    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float
    timestamp: _dt.datetime
    engine: str
    costs: float = 0.0
    is_paper: bool = True
    broker_order_id: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    fill_id: str = field(default_factory=_new_id)

    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass(frozen=True)
class Tick:
    """A single market data update."""

    symbol: str
    last_price: float
    timestamp: _dt.datetime
    volume: int = 0
    oi: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None
    instrument_token: Optional[int] = None


@dataclass
class Position:
    """A live position, as the risk kernel and engines see it."""

    symbol: str
    quantity: int                    # signed: negative is short
    average_price: float
    engine: str
    product: Product
    ttl: TTL = TTL.INTRADAY
    stop: Optional[float] = None
    opened_at: Optional[_dt.datetime] = None
    last_price: Optional[float] = None
    segment: Segment = Segment.EQUITY_INTRADAY
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def notional(self) -> float:
        price = self.last_price if self.last_price is not None else self.average_price
        return abs(self.quantity) * price

    def unrealised_pnl(self, mark: Optional[float] = None) -> float:
        price = mark if mark is not None else self.last_price
        if price is None:
            return 0.0
        return (price - self.average_price) * self.quantity


@dataclass(frozen=True)
class RiskDecision:
    """The return value of ``core.risk.check()``.

    A REJECT always carries a machine-readable ``reason_code`` (so tests assert
    on the veto that fired, not on prose) and a human ``reason`` for Telegram.
    """

    verdict: Verdict
    reason_code: str = ""
    reason: str = ""
    quantity: Optional[int] = None      # sized quantity when ALLOW
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def __bool__(self) -> bool:
        return self.allowed

    @classmethod
    def allow(cls, quantity: int | None = None, **meta: Any) -> "RiskDecision":
        return cls(Verdict.ALLOW, quantity=quantity, meta=meta)

    @classmethod
    def reject(cls, code: str, reason: str, **meta: Any) -> "RiskDecision":
        return cls(Verdict.REJECT, reason_code=code, reason=reason, meta=meta)
