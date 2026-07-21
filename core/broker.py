"""Broker layer: Kite Connect wrapper plus a local paper simulator.

Implements §0.1 (paper default, live needs config **and** an interactive
confirmation), §8.1 (daily token reality) and §8.6 (product-type selection).

**Architectural law (§0.3):** no engine imports this module. Only
``live/orchestrator.py`` may, and only after ``core.risk.check()`` returned
ALLOW. ``tests/test_architecture.py`` enforces that.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from core import clock
from core.config import Secrets, get_secrets, get_settings
from core.types import EntryType, Fill, Order, Position, Product, Segment, Side, TTL

log = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Any broker-side failure: auth, rejection, connectivity."""


class NotAuthenticated(BrokerError):
    """The Kite access token is missing or expired (§8.1 -- happens daily)."""


# ---------------------------------------------------------------------------
# §8.6 product-type selection
# ---------------------------------------------------------------------------


def product_for(ttl: TTL, segment: Segment, side: Side) -> Product:
    """Pick the broker product type from a Signal's ttl (§8.6).

    Intraday equity/short -> MIS. Overnight equity -> CNC. Overnight F&O ->
    NRML. Shorting equity overnight is impossible in the cash segment, so that
    combination raises rather than silently downgrading to MIS.
    """
    is_derivative = segment in (Segment.EQUITY_FUTURES, Segment.EQUITY_OPTIONS)

    if ttl is TTL.INTRADAY:
        return Product.MIS
    if ttl in (TTL.OVERNIGHT, TTL.SWING, TTL.GTT):
        if is_derivative:
            return Product.NRML
        if side is Side.SELL:
            raise BrokerError(
                "§8.6: cannot short equity overnight in the cash segment "
                f"(ttl={ttl.value}, segment={segment.value}). Reject the Signal."
            )
        return Product.CNC
    raise BrokerError(f"Unhandled ttl {ttl!r}")


# ---------------------------------------------------------------------------
# Broker interface
# ---------------------------------------------------------------------------


@dataclass
class AccountSnapshot:
    """What ``--status`` prints and the digest reports."""

    mode: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    authenticated: bool = False
    equity_available: Optional[float] = None
    equity_used: Optional[float] = None
    net: Optional[float] = None
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


class Broker(ABC):
    """Everything the orchestrator is allowed to ask a broker to do."""

    mode: str = "paper"

    @abstractmethod
    def is_authenticated(self) -> bool:
        """§8.1: verified by the 08:30 scheduler job every morning."""

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Route an order; return a broker order id. Only ever called post-ALLOW."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None: ...

    @abstractmethod
    def open_orders(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def ltp(self, symbols: list[str]) -> dict[str, float]: ...

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Full quote incl. circuit bands (§8.4). Default: derive from ltp."""
        return {s: {"last_price": p} for s, p in self.ltp(symbols).items()}

    def margins(self) -> dict[str, Any]:
        return {}

    def option_chain(self, underlying: str, expiry: str) -> list[dict[str, Any]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement option_chain; §6.8 (wheel) needs it."
        )


# ---------------------------------------------------------------------------
# Paper broker (§0.1 default)
# ---------------------------------------------------------------------------


class PaperBroker(Broker):
    """Simulates fills locally. **Never** sends a broker order.

    Prices come from an injected ``price_source`` -- in the paper runtime that
    is the live datafeed, in tests a dict. Fills are marked at the reference
    price adjusted by the §4 slippage model, so paper P&L and backtest P&L use
    identical arithmetic.
    """

    mode = "paper"

    def __init__(
        self,
        price_source: Any = None,
        starting_capital: float | None = None,
        cost_model: Any = None,
    ) -> None:
        settings = get_settings()
        self.price_source = price_source
        self.starting_capital = float(
            starting_capital
            if starting_capital is not None
            else settings.get("execution.paper_starting_capital", settings.require("risk.capital"))
        )
        self.cash = self.starting_capital
        self.cost_model = cost_model
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._fills: list[Fill] = []
        self._seq = 0

    # -- interface --------------------------------------------------------

    def is_authenticated(self) -> bool:
        """Paper mode needs no broker session."""
        return True

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(
            mode=self.mode,
            user_id="PAPER",
            user_name="Paper Trading",
            authenticated=True,
            equity_available=self.cash,
            equity_used=sum(p.notional for p in self._positions.values()),
            net=self.equity(),
            meta={"positions": len(self._positions), "fills": len(self._fills)},
        )

    def place_order(self, order: Order) -> str:
        """Simulate immediate execution at the reference price plus slippage."""
        self._seq += 1
        broker_order_id = f"PAPER{self._seq:06d}"
        price = self._reference_price(order)
        if price is None:
            raise BrokerError(
                f"Paper fill for {order.symbol} needs a reference price; none available. "
                "Attach `reference_price` to the order meta or wire a price_source."
            )
        fill_price = self._apply_slippage(price, order)
        costs = self._compute_costs(order, fill_price)

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=clock.now_ist(),
            engine=order.engine,
            costs=costs,
            is_paper=True,
            broker_order_id=broker_order_id,
            meta={"reference_price": price, "segment": order.segment.value},
        )
        self._fills.append(fill)
        self._apply_fill(fill, order)
        self._orders[broker_order_id] = {
            "order_id": broker_order_id,
            "status": "COMPLETE",
            "symbol": order.symbol,
            "quantity": order.quantity,
        }
        log.info(
            "PAPER FILL %s %s x%d @ %.2f (costs %.2f) [%s]",
            order.side.value, order.symbol, order.quantity, fill_price, costs, order.engine,
        )
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        record = self._orders.get(broker_order_id)
        if record and record["status"] not in ("COMPLETE", "CANCELLED"):
            record["status"] = "CANCELLED"

    def open_orders(self) -> list[dict[str, Any]]:
        return [o for o in self._orders.values() if o["status"] not in ("COMPLETE", "CANCELLED")]

    def positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_flat]

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        if self.price_source is None:
            return {}
        if isinstance(self.price_source, dict):
            return {s: float(self.price_source[s]) for s in symbols if s in self.price_source}
        return self.price_source.ltp(symbols)

    # -- paper-only helpers ----------------------------------------------

    def equity(self) -> float:
        """Cash plus mark-to-market on open positions."""
        marks = self.ltp([p.symbol for p in self._positions.values()]) or {}
        unrealised = sum(
            p.unrealised_pnl(marks.get(p.symbol)) for p in self._positions.values()
        )
        return self.cash + unrealised

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    def seed_position(self, position: Position) -> None:
        """Inject a starting position. Used by tests and by a mid-day restart."""
        self._positions[self._key(position.symbol, position.engine)] = position

    def _key(self, symbol: str, engine: str) -> str:
        return f"{engine}:{symbol}"

    def _reference_price(self, order: Order) -> Optional[float]:
        if order.price is not None:
            return float(order.price)
        ref = order.meta.get("reference_price")
        if ref is not None:
            return float(ref)
        prices = self.ltp([order.symbol])
        return prices.get(order.symbol)

    def _apply_slippage(self, price: float, order: Order) -> float:
        """§4 slippage: 0.03%/side liquid equity, 0.05%/side options, 1 tick min."""
        settings = get_settings()
        is_option = order.segment is Segment.EQUITY_OPTIONS
        pct = float(
            settings.require("costs.slippage.options_pct")
            if is_option
            else settings.require("costs.slippage.liquid_equity_pct")
        ) / 100.0
        tick = float(
            settings.require("costs.slippage.tick_size_options")
            if is_option
            else settings.require("costs.slippage.tick_size_equity")
        )
        min_ticks = int(settings.require("costs.slippage.minimum_ticks"))
        slip = max(price * pct, tick * min_ticks)
        # Slippage always hurts: pay up to buy, receive less to sell.
        adjusted = price + slip if order.side is Side.BUY else price - slip
        return round(max(adjusted, tick), 2)

    def _compute_costs(self, order: Order, fill_price: float) -> float:
        if self.cost_model is None:
            return 0.0
        return float(
            self.cost_model.charges(
                segment=order.segment,
                side=order.side,
                price=fill_price,
                quantity=order.quantity,
            ).total
        )

    def _apply_fill(self, fill: Fill, order: Order) -> None:
        key = self._key(fill.symbol, fill.engine)
        signed = fill.quantity * fill.side.sign
        existing = self._positions.get(key)

        self.cash -= fill.costs
        if existing is None:
            self._positions[key] = Position(
                symbol=fill.symbol,
                quantity=signed,
                average_price=fill.price,
                engine=fill.engine,
                product=order.product,
                ttl=order.ttl,
                stop=order.stop,
                opened_at=fill.timestamp,
                last_price=fill.price,
                segment=order.segment,
            )
            self.cash -= fill.price * signed
            return

        prior_qty = existing.quantity
        new_qty = prior_qty + signed

        if prior_qty != 0 and (prior_qty > 0) != (signed > 0):
            # Reducing or flipping: realise P&L on the closed portion.
            closed = min(abs(prior_qty), abs(signed))
            realised = (fill.price - existing.average_price) * closed * (1 if prior_qty > 0 else -1)
            self.cash += realised
            self.cash += existing.average_price * closed * (1 if prior_qty > 0 else -1)
            remaining = abs(signed) - closed
            if remaining:
                existing.average_price = fill.price
                self.cash -= fill.price * remaining * (1 if signed > 0 else -1)
        else:
            # Adding: weighted-average the entry.
            total = abs(prior_qty) + abs(signed)
            existing.average_price = (
                existing.average_price * abs(prior_qty) + fill.price * abs(signed)
            ) / total
            self.cash -= fill.price * signed

        existing.quantity = new_qty
        existing.last_price = fill.price
        if new_qty == 0:
            self._positions.pop(key, None)


# ---------------------------------------------------------------------------
# Kite broker
# ---------------------------------------------------------------------------


class KiteBroker(Broker):
    """Thin wrapper over ``kiteconnect.KiteConnect``.

    §8.1: the access token expires daily around 07:30 IST. This class does not
    try to refresh it -- ``scripts/morning_auth.py`` is a human-in-the-loop
    flow by design, and the 08:30 job alerts when the token is dead.
    """

    mode = "live"

    def __init__(self, secrets: Secrets | None = None, kite: Any = None) -> None:
        self.secrets = secrets or get_secrets()
        self._kite = kite
        self._instruments_cache: Optional[list[dict[str, Any]]] = None

    @property
    def kite(self) -> Any:
        if self._kite is None:
            if not self.secrets.kite_api_key:
                raise NotAuthenticated("KITE_API_KEY is not set in .env")
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise BrokerError(
                    "kiteconnect is not installed. `pip install -r requirements.txt`"
                ) from exc
            self._kite = KiteConnect(api_key=self.secrets.kite_api_key)
            if self.secrets.kite_access_token:
                self._kite.set_access_token(self.secrets.kite_access_token)
        return self._kite

    def is_authenticated(self) -> bool:
        if not self.secrets.kite_access_token:
            return False
        try:
            self.kite.profile()
            return True
        except Exception as exc:  # kiteconnect raises a family of exceptions
            log.warning("Kite token check failed (§8.1 daily expiry?): %s", exc)
            return False

    def account(self) -> AccountSnapshot:
        try:
            profile = self.kite.profile()
            funds = self.kite.margins("equity")
            available = funds.get("available", {}).get("live_balance")
            used = funds.get("utilised", {}).get("debits")
            return AccountSnapshot(
                mode=self.mode,
                user_id=profile.get("user_id"),
                user_name=profile.get("user_name"),
                authenticated=True,
                equity_available=available,
                equity_used=used,
                net=funds.get("net"),
            )
        except Exception as exc:
            return AccountSnapshot(mode=self.mode, authenticated=False, error=str(exc))

    def place_order(self, order: Order) -> str:
        params: dict[str, Any] = {
            "variety": self.kite.VARIETY_REGULAR,
            "exchange": order.meta.get("exchange", "NSE"),
            "tradingsymbol": order.meta.get("tradingsymbol", order.symbol),
            "transaction_type": order.side.value,
            "quantity": order.quantity,
            "product": order.product.value,
            "order_type": _kite_order_type(order.entry_type),
        }
        if order.entry_type in (EntryType.LIMIT, EntryType.SL):
            params["price"] = order.price
        if order.entry_type in (EntryType.SL, EntryType.SL_M):
            params["trigger_price"] = order.trigger_price
        try:
            return str(self.kite.place_order(**params))
        except Exception as exc:
            raise BrokerError(f"Kite rejected {order.symbol} {order.side.value}: {exc}") from exc

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=broker_order_id)
        except Exception as exc:
            raise BrokerError(f"Cancel failed for {broker_order_id}: {exc}") from exc

    def open_orders(self) -> list[dict[str, Any]]:
        try:
            return [
                o for o in self.kite.orders()
                if o.get("status") in ("OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED")
            ]
        except Exception as exc:
            raise BrokerError(f"Could not fetch orders: {exc}") from exc

    def positions(self) -> list[Position]:
        try:
            raw = self.kite.positions().get("net", [])
        except Exception as exc:
            raise BrokerError(f"Could not fetch positions: {exc}") from exc
        out: list[Position] = []
        for row in raw:
            if not row.get("quantity"):
                continue
            out.append(
                Position(
                    symbol=row["tradingsymbol"],
                    quantity=int(row["quantity"]),
                    average_price=float(row["average_price"]),
                    engine=row.get("tag") or "unknown",
                    product=Product(row.get("product", "MIS")),
                    last_price=float(row.get("last_price") or 0) or None,
                    meta={"exchange": row.get("exchange")},
                )
            )
        return out

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        keys = [s if ":" in s else f"NSE:{s}" for s in symbols]
        try:
            data = self.kite.ltp(keys)
        except Exception as exc:
            raise BrokerError(f"LTP fetch failed: {exc}") from exc
        return {k.split(":", 1)[1]: float(v["last_price"]) for k, v in data.items()}

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Full quote including the §8.4 circuit bands the kernel vetoes on."""
        if not symbols:
            return {}
        keys = [s if ":" in s else f"NSE:{s}" for s in symbols]
        try:
            data = self.kite.quote(keys)
        except Exception as exc:
            raise BrokerError(f"Quote fetch failed: {exc}") from exc
        return {k.split(":", 1)[1]: v for k, v in data.items()}

    def margins(self) -> dict[str, Any]:
        try:
            return self.kite.margins()
        except Exception as exc:
            raise BrokerError(f"Margins fetch failed: {exc}") from exc

    def instruments(self, exchange: str = "NFO") -> list[dict[str, Any]]:
        if self._instruments_cache is None:
            self._instruments_cache = self.kite.instruments(exchange)
        return self._instruments_cache

    def option_chain(self, underlying: str, expiry: str) -> list[dict[str, Any]]:
        """Option contracts for an underlying and expiry (§6.8 wheel)."""
        return [
            row for row in self.instruments("NFO")
            if row.get("name") == underlying.upper()
            and str(row.get("expiry")) == expiry
            and row.get("instrument_type") in ("CE", "PE")
        ]


def _kite_order_type(entry: EntryType) -> str:
    return {
        EntryType.MARKET: "MARKET",
        EntryType.LIMIT: "LIMIT",
        EntryType.SL: "SL",
        EntryType.SL_M: "SL-M",
    }[entry]


# ---------------------------------------------------------------------------
# §0.1 mode resolution — live needs config AND an interactive confirmation
# ---------------------------------------------------------------------------


def resolve_mode(override: str | None = None) -> str:
    """Configured execution mode, before the interactive gate."""
    if override:
        return override.lower()
    secrets = get_secrets()
    if secrets.execution_mode_override:
        return secrets.execution_mode_override.lower()
    return str(get_settings().get("execution.mode", "paper")).lower()


def confirm_live(interactive: bool = True, stream: Any = None) -> bool:
    """§0.1: live trading requires an explicit ``y`` typed at startup.

    Returns True only on an exact ``y``/``yes``. A non-interactive process (no
    TTY, a cron job, a test) can never answer yes -- which is the point.
    """
    out = stream or sys.stdout
    if not interactive:
        print(
            "LIVE mode requested but the process is non-interactive. "
            "Refusing to trade live without a typed confirmation (§0.1).",
            file=out,
        )
        return False
    if not sys.stdin.isatty():
        print(
            "LIVE mode requested but stdin is not a TTY. Refusing (§0.1).",
            file=out,
        )
        return False
    settings = get_settings()
    capital = settings.require("risk.capital")
    print("", file=out)
    print("=" * 70, file=out)
    print("  LIVE EXECUTION MODE REQUESTED", file=out)
    print("=" * 70, file=out)
    print(f"  Capital at risk       : INR {capital:,.0f}", file=out)
    print(f"  Daily loss limit      : {settings.require('risk.daily_loss_limit_pct')}%", file=out)
    print(f"  Weekly loss limit     : {settings.require('risk.weekly_loss_limit_pct')}%", file=out)
    print("  Real orders will be sent to Zerodha. Paper mode is the default.", file=out)
    print("=" * 70, file=out)
    answer = input("  Type 'y' to confirm LIVE trading, anything else to abort: ").strip().lower()
    confirmed = answer in ("y", "yes")
    print(f"  -> {'LIVE CONFIRMED' if confirmed else 'ABORTED, staying in paper'}\n", file=out)
    return confirmed


def get_broker(
    mode: str | None = None,
    *,
    interactive: bool = True,
    price_source: Any = None,
    cost_model: Any = None,
    kite: Any = None,
) -> Broker:
    """Build the broker for the resolved mode.

    §0.1: ``live`` requires BOTH ``execution.mode: live`` (or ``EXECUTION_MODE``)
    AND a typed ``y``. Anything else falls back to :class:`PaperBroker`.
    """
    resolved = resolve_mode(mode)
    if resolved == "live":
        if confirm_live(interactive=interactive):
            log.warning("LIVE MODE CONFIRMED -- real orders will be sent")
            return KiteBroker(kite=kite)
        log.warning("Live mode not confirmed; falling back to paper (§0.1)")
    elif resolved != "paper":
        raise BrokerError(f"Unknown execution mode {resolved!r}; expected 'paper' or 'live'")
    return PaperBroker(price_source=price_source, cost_model=cost_model)
