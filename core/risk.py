"""THE RISK KERNEL — §3.

Every order in this system, paper or live, passes through :meth:`RiskKernel.check`
before it can reach the broker. No engine may bypass it (§0.3), and
``tests/test_architecture.py`` fails the build if one tries.

The kernel does two jobs:

1. **Size** the order -- ``qty = floor((capital x risk_per_trade_pct/100) / |entry - stop|)``
2. **Veto** it, running the §3 checks in a deliberate order: cheap universal
   blocks first (kill switch, loss limits), then India-specific instrument
   vetoes, then time and exposure limits.

Design rules encoded here
-------------------------
* **Vetoes apply to NEW ENTRIES, not exits.** An order with ``is_entry=False``
  skips the instrument vetoes. Blocking an exit because the stock is near its
  circuit band would trap us in exactly the position we are trying to leave.
  The loss-limit and kill-switch blocks are the exception -- but they *force*
  exits rather than blocking them.
* **Every number comes from ``settings.yaml``.** Nothing is hardcoded.
* **Every rejection is journalled** with a machine-readable ``reason_code``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence

from core import calendar as trading_calendar
from core import clock
from core.config import Settings, get_settings
from core.journal import Journal, get_journal
from core.types import (
    EntryType,
    Order,
    Position,
    RiskDecision,
    Segment,
    Side,
    Signal,
    TTL,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reason codes -- tests assert on these, not on prose.
# ---------------------------------------------------------------------------


class Reason:
    """Machine-readable rejection codes."""

    KILL_SWITCH = "KILL_SWITCH_ACTIVE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    WEEKLY_LOSS_LIMIT = "WEEKLY_LOSS_LIMIT"
    MAX_TRADES_PER_ENGINE = "MAX_NEW_TRADES_PER_DAY_PER_ENGINE"
    MAX_CONCURRENT_POSITIONS = "MAX_CONCURRENT_POSITIONS_TOTAL"
    ENGINE_CAPITAL_CAP = "ENGINE_CAPITAL_CAP"
    ALERT_ONLY_ENGINE = "ALERT_ONLY_ENGINE"
    FNO_BAN_LIST = "FNO_BAN_LIST"
    SURVEILLANCE = "ASM_GSM_SURVEILLANCE"
    CIRCUIT_BAND = "CIRCUIT_BAND_PROXIMITY"
    BLOCKED_EVENT_DAY = "BLOCKED_EVENT_DAY"
    OVERNIGHT_INTO_EVENT = "OVERNIGHT_INTO_BLOCKED_EVENT"
    ENTRY_CUTOFF = "INTRADAY_ENTRY_CUTOFF"
    PHYSICAL_SETTLEMENT = "PHYSICAL_SETTLEMENT_GUARD"
    STT_TRAP = "STT_TRAP_GUARD"
    EQUITY_SHORT_OVERNIGHT = "EQUITY_SHORT_OVERNIGHT_IMPOSSIBLE"
    ZERO_QUANTITY = "ZERO_QUANTITY"
    NO_STOP = "NO_STOP_DEFINED"
    MARKET_CLOSED = "MARKET_CLOSED"
    BLACKLISTED = "BLACKLISTED_SYMBOL"


# ---------------------------------------------------------------------------
# Market state the kernel needs in order to veto
# ---------------------------------------------------------------------------


@dataclass
class BandInfo:
    """§8.4 price band state for one symbol."""

    last_price: float
    upper: Optional[float] = None
    lower: Optional[float] = None

    def distance_to_band_pct(self) -> Optional[float]:
        """Percent distance to the *nearer* band, or None when unbanded."""
        distances = []
        if self.upper:
            distances.append((self.upper - self.last_price) / self.last_price * 100.0)
        if self.lower:
            distances.append((self.last_price - self.lower) / self.last_price * 100.0)
        return min(distances) if distances else None


@dataclass
class MarketState:
    """Everything the kernel needs to know about the market right now.

    Populated by the orchestrator from the datafeed and the nightly NSE
    downloads. Empty collections mean "no data", and the kernel treats missing
    surveillance data as *not* a veto -- but the nightly job alerts loudly when
    the lists are stale (§8.2 "never silently degrade"), which is where that
    risk is managed.
    """

    fno_ban_list: set[str] = field(default_factory=set)
    surveillance: set[str] = field(default_factory=set)   # ASM + GSM, any stage
    bands: dict[str, BandInfo] = field(default_factory=dict)
    spot: dict[str, float] = field(default_factory=dict)
    lists_as_of: Optional[_dt.date] = None

    def band_for(self, symbol: str) -> Optional[BandInfo]:
        return self.bands.get(symbol)


class PositionSource(Protocol):
    """Anything that can report open positions -- broker or paper book."""

    def positions(self) -> list[Position]: ...


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def size_quantity(
    entry: float,
    stop: float,
    capital: float | None = None,
    risk_pct: float | None = None,
    lot_size: int = 1,
) -> int:
    """§3 sizing: ``floor((capital x risk_pct/100) / |entry - stop|)``.

    ``lot_size`` rounds down to whole lots for F&O -- a 1.7-lot order is not a
    thing, and rounding *up* would silently exceed the risk budget.
    """
    settings = get_settings()
    capital = float(capital if capital is not None else settings.require("risk.capital"))
    risk_pct = float(risk_pct if risk_pct is not None else settings.require("risk.risk_per_trade_pct"))

    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        raise ValueError(
            f"Cannot size a trade with entry {entry} == stop {stop}: risk per unit is zero."
        )
    budget = capital * (risk_pct / 100.0)
    raw = math.floor(budget / risk_per_unit)
    if lot_size > 1:
        raw = (raw // lot_size) * lot_size
    return max(raw, 0)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


class RiskKernel:
    """The single gate every order passes through (§0.3, §3)."""

    def __init__(
        self,
        journal: Journal | None = None,
        settings: Settings | None = None,
        market: MarketState | None = None,
        positions: PositionSource | None = None,
    ) -> None:
        self.journal = journal or get_journal()
        self.settings = settings or get_settings()
        self.market = market or MarketState()
        self.positions_source = positions

        self._killed = False
        self._kill_reason = ""
        self._daily_blocked_date: Optional[_dt.date] = None
        self._weekly_blocked_until: Optional[_dt.date] = None

    # -- state ------------------------------------------------------------

    @property
    def is_killed(self) -> bool:
        return self._killed

    def arm_kill(self, reason: str) -> None:
        """Latch the kill switch: no new order passes until it is cleared."""
        self._killed = True
        self._kill_reason = reason
        log.critical("Risk kernel KILLED: %s", reason)

    def clear_kill(self) -> None:
        """Clear the latch. Only a human (or a new session) should do this."""
        self._killed = False
        self._kill_reason = ""

    def block_for_day(self, day: _dt.date | None = None) -> None:
        """§3 daily loss limit breach: no new orders until the next session."""
        self._daily_blocked_date = day or clock.today_ist()

    def block_for_week(self, until_monday: _dt.date | None = None) -> None:
        """§3 weekly loss limit breach: no new orders until Monday."""
        today = clock.today_ist()
        if until_monday is None:
            until_monday = today + _dt.timedelta(days=(7 - today.weekday()) % 7 or 7)
        self._weekly_blocked_until = until_monday

    def current_positions(self) -> list[Position]:
        if self.positions_source is None:
            return []
        return self.positions_source.positions()

    # -- loss limits ------------------------------------------------------

    def daily_pnl(self, day: _dt.date | None = None) -> float:
        day = day or clock.today_ist()
        return self.journal.realised_pnl_between(day.isoformat(), day.isoformat())

    def weekly_pnl(self, day: _dt.date | None = None) -> float:
        day = day or clock.today_ist()
        monday = day - _dt.timedelta(days=day.weekday())
        return self.journal.realised_pnl_between(monday.isoformat(), day.isoformat())

    def daily_limit_breached(self, day: _dt.date | None = None) -> bool:
        capital = float(self.settings.require("risk.capital"))
        limit = capital * float(self.settings.require("risk.daily_loss_limit_pct")) / 100.0
        return self.daily_pnl(day) <= -limit

    def weekly_limit_breached(self, day: _dt.date | None = None) -> bool:
        capital = float(self.settings.require("risk.capital"))
        limit = capital * float(self.settings.require("risk.weekly_loss_limit_pct")) / 100.0
        return self.weekly_pnl(day) <= -limit

    # -- the gate ---------------------------------------------------------

    def check(self, order: Order, now: _dt.datetime | None = None) -> RiskDecision:
        """§3: ``ALLOW`` or ``REJECT(reason)``. Runs before EVERY order.

        Order of evaluation is deliberate:
          1. kill switch and loss limits  (block everything, including entries
             we might otherwise want)
          2. structural checks            (alert-only engine, blacklist, qty)
          3. India instrument vetoes      (ban list, ASM/GSM, circuit band)
          4. day and time vetoes          (event days, 14:45 cutoff)
          5. settlement guards            (physical settlement, STT trap)
          6. exposure limits              (trade counts, concurrency, caps)
        """
        now = clock.to_ist(now) if now else clock.now_ist()
        today = now.date()

        # NOTE: these are chained with explicit `is not None`, never with `or`.
        # RiskDecision.__bool__ returns `allowed`, so a REJECT is falsy and an
        # `or` chain would silently skip every veto.
        stages = (
            lambda: self._check_halts(order, today),
            lambda: self._check_structural(order),
            lambda: self._check_instrument_vetoes(order, today),
            lambda: self._check_time_and_day(order, now, today),
            lambda: self._check_settlement_guards(order, now, today),
            lambda: self._check_exposure(order, today),
        )
        for stage in stages:
            decision = stage()
            if decision is not None:
                self._journal_rejection(order, decision)
                return decision

        return RiskDecision.allow(quantity=order.quantity)

    # -- 1. halts ---------------------------------------------------------

    def _check_halts(self, order: Order, today: _dt.date) -> Optional[RiskDecision]:
        if self._killed:
            return RiskDecision.reject(
                Reason.KILL_SWITCH, f"Kill switch is armed: {self._kill_reason}"
            )

        # Exits are always allowed past the loss limits -- the limits exist to
        # get us OUT, so blocking a closing order would be self-defeating.
        if not order.is_entry:
            return None

        if self._daily_blocked_date == today or self.daily_limit_breached(today):
            self._daily_blocked_date = today
            pct = self.settings.require("risk.daily_loss_limit_pct")
            return RiskDecision.reject(
                Reason.DAILY_LOSS_LIMIT,
                f"Daily loss limit {pct}% breached (realised {self.daily_pnl(today):,.0f}); "
                f"no new orders until the next session.",
                daily_pnl=self.daily_pnl(today),
            )

        if (self._weekly_blocked_until and today < self._weekly_blocked_until) or \
                self.weekly_limit_breached(today):
            if not self._weekly_blocked_until:
                self.block_for_week()
            pct = self.settings.require("risk.weekly_loss_limit_pct")
            return RiskDecision.reject(
                Reason.WEEKLY_LOSS_LIMIT,
                f"Weekly loss limit {pct}% breached (realised {self.weekly_pnl(today):,.0f}); "
                f"no new orders until Monday.",
                weekly_pnl=self.weekly_pnl(today),
            )
        return None

    # -- 2. structural ----------------------------------------------------

    def _check_structural(self, order: Order) -> Optional[RiskDecision]:
        alert_only = set(self.settings.get("risk.alert_only_engines", []) or [])
        if order.engine in alert_only:
            return RiskDecision.reject(
                Reason.ALERT_ONLY_ENGINE,
                f"Engine {order.engine!r} is alert-only (§3) and may not place orders.",
            )

        from core.config import get_universe

        blacklist = set(get_universe().get("blacklist.symbols", []) or [])
        if order.symbol in blacklist:
            return RiskDecision.reject(
                Reason.BLACKLISTED, f"{order.symbol} is on the permanent blacklist."
            )

        if order.quantity <= 0:
            return RiskDecision.reject(
                Reason.ZERO_QUANTITY,
                f"Sized quantity is {order.quantity}; the stop is too far for the "
                f"risk budget, or no reference price was available.",
            )

        # §8.6: shorting equity overnight is impossible in the cash segment.
        if (
            order.is_entry
            and order.side is Side.SELL
            and order.ttl in (TTL.OVERNIGHT, TTL.SWING)
            and order.segment in (Segment.EQUITY_DELIVERY, Segment.EQUITY_INTRADAY)
        ):
            return RiskDecision.reject(
                Reason.EQUITY_SHORT_OVERNIGHT,
                "§8.6: equity cannot be shorted overnight in the cash segment.",
            )

        if order.is_entry and order.stop is None and order.entry_type is not EntryType.SL_M:
            return RiskDecision.reject(
                Reason.NO_STOP,
                f"{order.engine} sent an entry for {order.symbol} with no stop; "
                f"sizing (§3) is undefined without one.",
            )
        return None

    # -- 3. India instrument vetoes ---------------------------------------

    def _check_instrument_vetoes(self, order: Order, today: _dt.date) -> Optional[RiskDecision]:
        # Exits are never blocked by instrument state.
        if not order.is_entry:
            return None

        vetoes = self.settings.section("risk.vetoes")
        underlying = str(order.meta.get("underlying") or order.symbol)

        # §3: F&O ban list (MWPL >= 95%) -> reject DERIVATIVES orders.
        if vetoes.get("fno_ban_list", True) and order.is_derivative:
            if underlying in self.market.fno_ban_list or order.symbol in self.market.fno_ban_list:
                return RiskDecision.reject(
                    Reason.FNO_BAN_LIST,
                    f"{underlying} is in the F&O ban list (MWPL >= 95%); "
                    f"derivatives entries are rejected.",
                )

        # §3: ASM/GSM surveillance -> reject ALL new entries, any engine.
        if vetoes.get("surveillance_asm_gsm", True):
            if order.symbol in self.market.surveillance or underlying in self.market.surveillance:
                return RiskDecision.reject(
                    Reason.SURVEILLANCE,
                    f"{order.symbol} is under ASM/GSM surveillance (100% margin, "
                    f"position caps); no engine may open a new position.",
                )

        # §3: within 1% of the circuit band -> reject new entries.
        threshold = float(vetoes.get("circuit_band_proximity_pct", 1.0))
        band = self.market.band_for(order.symbol)
        if band is not None:
            distance = band.distance_to_band_pct()
            if distance is not None and distance <= threshold:
                return RiskDecision.reject(
                    Reason.CIRCUIT_BAND,
                    f"{order.symbol} is {distance:.2f}% from its circuit band "
                    f"(threshold {threshold}%); a locked stock cannot be exited.",
                    distance_pct=distance,
                )
        return None

    # -- 4. day and time --------------------------------------------------

    def _check_time_and_day(
        self, order: Order, now: _dt.datetime, today: _dt.date
    ) -> Optional[RiskDecision]:
        if not order.is_entry:
            return None

        vetoes = self.settings.section("risk.vetoes")

        # §3: no new entries on blocked event days.
        if vetoes.get("blocked_event_days", True):
            if trading_calendar.is_blocked_event_day(today):
                return RiskDecision.reject(
                    Reason.BLOCKED_EVENT_DAY,
                    f"{today} is a blocked event day "
                    f"({trading_calendar.event_note(today)}); no new entries.",
                )
            # §3: no overnight holds INTO a blocked day.
            if order.ttl in (TTL.OVERNIGHT, TTL.SWING) and \
                    trading_calendar.next_session_is_blocked(today):
                nxt = trading_calendar.next_trading_day(today)
                return RiskDecision.reject(
                    Reason.OVERNIGHT_INTO_EVENT,
                    f"Next session {nxt} is a blocked event day "
                    f"({trading_calendar.event_note(nxt)}); no overnight holds into it.",
                )

        # §3: no new INTRADAY entries after 14:45.
        cutoff = str(vetoes.require("no_new_intraday_entries_after"))
        if order.ttl is TTL.INTRADAY and clock.is_after(now, cutoff):
            return RiskDecision.reject(
                Reason.ENTRY_CUTOFF,
                f"No new intraday entries after {cutoff} IST (now {now:%H:%M}); "
                f"force-flat is at {vetoes.require('mis_force_flat_at')}.",
            )
        return None

    # -- 5. settlement guards ---------------------------------------------

    def _check_settlement_guards(
        self, order: Order, now: _dt.datetime, today: _dt.date
    ) -> Optional[RiskDecision]:
        """§3 physical-settlement guard and STT-trap guard.

        These apply to *stock* options. Index options are cash-settled, so the
        delivery risk does not exist there -- the guard keys off
        ``meta['underlying_type']`` which the option engines set.
        """
        guard = self.settings.section("risk.vetoes.physical_settlement_guard")
        stt_guard = self.settings.section("risk.vetoes.stt_trap_guard")

        is_option = order.segment is Segment.EQUITY_OPTIONS
        if not is_option:
            return None

        underlying_type = str(order.meta.get("underlying_type", "stock")).lower()
        expiry = order.meta.get("expiry")
        expiry_date = _as_date(expiry)

        # --- physical settlement: SHORT stock options near expiry ---------
        if (
            guard.get("enabled", True)
            and underlying_type == "stock"
            and order.side is Side.SELL
            and order.is_entry
            and expiry_date is not None
        ):
            sessions_left = trading_calendar.sessions_until(expiry_date, today)
            close_by = int(guard.get("close_by_sessions_before_expiry", 2))
            if sessions_left <= close_by:
                strike = order.meta.get("strike")
                spot = self.market.spot.get(str(order.meta.get("underlying") or order.symbol))
                near_money = _is_itm_or_near(
                    strike, spot, order.meta.get("option_type"),
                    float(guard.get("itm_buffer_pct", 2.0)),
                )
                if near_money and not order.meta.get("allow_delivery", False):
                    return RiskDecision.reject(
                        Reason.PHYSICAL_SETTLEMENT,
                        f"Short stock option {order.symbol} expires in {sessions_left} "
                        f"session(s) and is ITM or within "
                        f"{guard.get('itm_buffer_pct')}% of spot. Stock F&O settle "
                        f"physically; close/roll it or set allow_delivery=true (§3).",
                        sessions_to_expiry=sessions_left,
                    )

        # --- STT trap: LONG ITM options into the expiry close --------------
        if (
            stt_guard.get("enabled", True)
            and order.side is Side.BUY
            and order.is_entry
            and expiry_date == today
        ):
            deadline = str(stt_guard.require("exit_long_itm_options_by"))
            if clock.is_after(now, deadline):
                return RiskDecision.reject(
                    Reason.STT_TRAP,
                    f"Expiry day: no new long option entries after {deadline} IST. "
                    f"Long ITM options held into the close pay delivery STT on "
                    f"intrinsic value (§3 STT-trap guard).",
                )
        return None

    # -- 6. exposure ------------------------------------------------------

    def _check_exposure(self, order: Order, today: _dt.date) -> Optional[RiskDecision]:
        if not order.is_entry:
            return None

        max_per_engine = int(self.settings.require("risk.max_new_trades_per_day_per_engine"))
        used = self.journal.new_entry_count_today(today.isoformat(), order.engine)
        if used >= max_per_engine:
            return RiskDecision.reject(
                Reason.MAX_TRADES_PER_ENGINE,
                f"{order.engine} already opened {used} trades today "
                f"(limit {max_per_engine}).",
                used=used,
            )

        positions = self.current_positions()
        max_total = int(self.settings.require("risk.max_concurrent_positions_total"))
        open_symbols = {p.symbol for p in positions if not p.is_flat}
        if order.symbol not in open_symbols and len(open_symbols) >= max_total:
            return RiskDecision.reject(
                Reason.MAX_CONCURRENT_POSITIONS,
                f"{len(open_symbols)} concurrent positions open (limit {max_total}).",
                open_positions=len(open_symbols),
            )

        cap_pct = self.settings.get(f"risk.per_engine_capital_cap_pct.{order.engine}", None)
        if cap_pct is not None and order.notional > 0:
            capital = float(self.settings.require("risk.capital"))
            cap_value = capital * float(cap_pct) / 100.0
            engine_exposure = sum(p.notional for p in positions if p.engine == order.engine)
            if engine_exposure + order.notional > cap_value:
                return RiskDecision.reject(
                    Reason.ENGINE_CAPITAL_CAP,
                    f"{order.engine} exposure {engine_exposure + order.notional:,.0f} would "
                    f"exceed its {cap_pct}% cap ({cap_value:,.0f}).",
                    cap_value=cap_value,
                    projected=engine_exposure + order.notional,
                )
        return None

    # -- expiry-day sweeps -------------------------------------------------

    def stt_trap_exits_due(self, now: _dt.datetime | None = None) -> list[Position]:
        """§3 STT-trap guard, exit side: long ITM options to close by 15:00.

        Blocking new entries is only half the rule. An option bought days ago
        must still be forced out on expiry day, because a long ITM option held
        into the close pays delivery STT on intrinsic value -- which can exceed
        the option's whole remaining premium.

        Returns the positions the orchestrator must flatten now.
        """
        guard = self.settings.section("risk.vetoes.stt_trap_guard")
        if not guard.get("enabled", True):
            return []

        now = clock.to_ist(now) if now else clock.now_ist()
        today = now.date()
        if not clock.is_after(now, str(guard.require("exit_long_itm_options_by"))):
            return []

        due: list[Position] = []
        for position in self.current_positions():
            if position.segment is not Segment.EQUITY_OPTIONS or not position.is_long:
                continue
            if _as_date(position.meta.get("expiry")) != today:
                continue
            spot = self.market.spot.get(str(position.meta.get("underlying") or position.symbol))
            if _is_itm(position.meta.get("strike"), spot, position.meta.get("option_type")):
                due.append(position)
        return due

    def physical_settlement_rolls_due(self, now: _dt.datetime | None = None) -> list[Position]:
        """§3 physical-settlement guard, exit side.

        Short stock options that are ITM or within the configured buffer of
        spot, with ``close_by_sessions_before_expiry`` sessions or fewer left,
        and without ``allow_delivery``. Stock F&O settle physically, and a
        surprise delivery obligation can exceed the account.
        """
        guard = self.settings.section("risk.vetoes.physical_settlement_guard")
        if not guard.get("enabled", True):
            return []

        now = clock.to_ist(now) if now else clock.now_ist()
        today = now.date()
        close_by = int(guard.get("close_by_sessions_before_expiry", 2))
        buffer_pct = float(guard.get("itm_buffer_pct", 2.0))

        due: list[Position] = []
        for position in self.current_positions():
            if position.segment is not Segment.EQUITY_OPTIONS or not position.is_short:
                continue
            if str(position.meta.get("underlying_type", "stock")).lower() != "stock":
                continue
            if position.meta.get("allow_delivery", False):
                continue
            expiry = _as_date(position.meta.get("expiry"))
            if expiry is None or trading_calendar.sessions_until(expiry, today) > close_by:
                continue
            spot = self.market.spot.get(str(position.meta.get("underlying") or position.symbol))
            if _is_itm_or_near(
                position.meta.get("strike"), spot, position.meta.get("option_type"), buffer_pct
            ):
                due.append(position)
        return due

    # -- journalling ------------------------------------------------------

    def _journal_rejection(self, order: Order, decision: RiskDecision) -> None:
        """§3: every rejection is journalled with its reason."""
        self.journal.record_rejection(
            decision,
            engine=order.engine,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            signal_id=order.signal_id,
            order_id=order.order_id,
        )
        log.warning(
            "REJECT [%s] %s %s x%d -> %s: %s",
            order.engine, order.side.value, order.symbol, order.quantity,
            decision.reason_code, decision.reason,
        )


# ---------------------------------------------------------------------------
# Signal -> Order, with sizing
# ---------------------------------------------------------------------------


def build_order(
    signal: Signal,
    reference_price: float | None = None,
    *,
    kernel: RiskKernel | None = None,
    lot_size: int = 1,
    size_multiplier: float = 1.0,
    is_entry: bool = True,
) -> Order:
    """Convert a Signal into a sized Order (§6.0: engines never do this).

    ``size_multiplier`` implements the §7 expiry-day rule ("halve all new
    sizes") without any engine knowing about expiry days.
    """
    from core.broker import product_for

    settings = get_settings()
    price = reference_price if reference_price is not None else signal.reference_price
    if price is None:
        raise ValueError(f"No reference price for {signal.symbol}; cannot size (§3).")

    segment = Segment(signal.meta.get("segment", Segment.EQUITY_INTRADAY.value))
    if signal.is_option:
        segment = Segment.EQUITY_OPTIONS
    elif signal.meta.get("instrument_type") in {"FUT", "FUTIDX", "FUTSTK"}:
        segment = Segment.EQUITY_FUTURES
    elif signal.ttl in (TTL.OVERNIGHT, TTL.SWING) and segment is Segment.EQUITY_INTRADAY:
        segment = Segment.EQUITY_DELIVERY

    if signal.stop is None:
        quantity = int(signal.meta.get("quantity", 0))
    else:
        quantity = size_quantity(
            price,
            signal.stop,
            capital=float(settings.require("risk.capital")),
            risk_pct=float(settings.require("risk.risk_per_trade_pct")),
            lot_size=lot_size,
        )
    if size_multiplier != 1.0:
        quantity = int(quantity * size_multiplier)
        if lot_size > 1:
            quantity = (quantity // lot_size) * lot_size

    product = product_for(signal.ttl, segment, signal.side)

    return Order(
        symbol=signal.symbol,
        side=signal.side,
        quantity=quantity,
        entry_type=signal.entry_type,
        product=product,
        engine=signal.engine,
        signal_id=signal.signal_id,
        segment=segment,
        price=signal.limit_price,
        stop=signal.stop,
        ttl=signal.ttl,
        reason=signal.reason,
        meta={**signal.meta, "reference_price": price, "targets": list(signal.targets)},
        created_at=clock.now_ist(),
        is_entry=is_entry,
    )


# ---------------------------------------------------------------------------
# Flatten / kill helpers (§3 kill())
# ---------------------------------------------------------------------------


def flatten_all(
    broker: Any,
    journal: Journal | None = None,
    *,
    reason: str = "flatten",
    engines: Optional[Iterable[str]] = None,
    intraday_only: bool = False,
) -> int:
    """Close open positions. Returns how many were flattened.

    Exit orders bypass the entry vetoes by construction (``is_entry=False``) --
    see the module docstring. Individual failures are logged and counted, never
    allowed to abort the sweep.
    """
    from core.types import Order as _Order

    journal = journal or get_journal()
    engine_filter = set(engines) if engines else None
    flattened = 0

    for position in broker.positions():
        if position.is_flat:
            continue
        if engine_filter and position.engine not in engine_filter:
            continue
        if intraday_only and position.ttl is not TTL.INTRADAY:
            continue

        exit_side = Side.SELL if position.is_long else Side.BUY
        order = _Order(
            symbol=position.symbol,
            side=exit_side,
            quantity=abs(position.quantity),
            entry_type=EntryType.MARKET,
            product=position.product,
            engine=position.engine,
            signal_id="",
            segment=position.segment,
            ttl=position.ttl,
            reason=reason,
            meta={"reference_price": position.last_price or position.average_price,
                  "exit": True, **position.meta},
            created_at=clock.now_ist(),
            is_entry=False,
        )
        try:
            broker_order_id = broker.place_order(order)
            order.broker_order_id = broker_order_id
            journal.record_order(order, mode=getattr(broker, "mode", "paper"), status="FILLED")
            flattened += 1
        except Exception as exc:
            log.error("flatten failed for %s: %s", position.symbol, exc)
            journal.record_error("flatten_all", f"{position.symbol}: {exc}", severity="ERROR")
    return flattened


def force_flat_intraday(broker: Any, journal: Journal | None = None) -> int:
    """§3: MIS positions are force-flat by 15:10, never left to the broker's
    ~15:20 auto square-off (which carries penalties)."""
    return flatten_all(broker, journal, reason="15:10 force-flat (§3)", intraday_only=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def _is_itm(strike: Any, spot: Optional[float], option_type: Any) -> bool:
    """Strictly in-the-money. Unknown spot fails safe (assume ITM)."""
    if strike is None or spot is None:
        return True
    strike = float(strike)
    opt = str(option_type or "").upper()
    if opt == "CE":
        return spot > strike
    if opt == "PE":
        return spot < strike
    return False


def _is_itm_or_near(
    strike: Any, spot: Optional[float], option_type: Any, buffer_pct: float
) -> bool:
    """True when a short option is ITM or within ``buffer_pct`` of spot.

    With no spot available the answer is **True** -- the §3 guard fails safe.
    Being wrong about a physical-delivery obligation can exceed the account.
    """
    if strike is None or spot is None:
        return True
    strike = float(strike)
    buffer_value = spot * buffer_pct / 100.0
    opt = str(option_type or "").upper()
    if opt == "CE":
        return spot >= strike - buffer_value
    if opt == "PE":
        return spot <= strike + buffer_value
    return abs(spot - strike) <= buffer_value


_default_kernel: Optional[RiskKernel] = None


def get_kernel() -> RiskKernel:
    """Process-wide kernel singleton."""
    global _default_kernel
    if _default_kernel is None:
        _default_kernel = RiskKernel()
    return _default_kernel


def set_kernel(kernel: Optional[RiskKernel]) -> None:
    global _default_kernel
    _default_kernel = kernel


def check(order: Order, now: _dt.datetime | None = None) -> RiskDecision:
    """§0.3 module-level gate: ``core.risk.check(order)``.

    This is the function the architectural law names. Everything routes here.
    """
    return get_kernel().check(order, now=now)
