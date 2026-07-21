"""Event-driven backtester — §4.

Walks a timeline of bars, hands each engine a :class:`~engines.base.Context`
truncated to "now", routes every Signal through the **real** §3 risk kernel,
and fills through the **real** §4 cost model. Backtest and paper trading share
that path deliberately: a backtest that skips the kernel measures a strategy
this system would never actually run.

Lookahead-bias defence
----------------------
The Context's bar frames are sliced with ``index.searchsorted(now)`` before the
engine sees them. An engine cannot peek at a future bar because the future does
not exist in the frame it holds. Fills happen at the *next* bar's open where the
signal is a market order, never at the close that produced the signal.

Walk-forward (§4)
-----------------
``tune`` 2019-2022, ``validate`` 2023-2024, ``test`` 2025-present. The test
window is consumed **exactly once**: every run is recorded in the journal, and
a second run prints a loud warning rather than silently letting you fit to it.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.config import ConfigError, Settings, get_settings
from core.costs import CostModel, get_cost_model
from core.datafeed import DataFeed
from core.journal import Journal, get_journal
from core.risk import MarketState, RiskKernel, build_order
from core.types import (
    EntryType,
    Fill,
    Position,
    Product,
    Regime,
    Segment,
    Side,
    Signal,
    TTL,
)
from backtest.metrics import (
    BacktestTrade,
    Metrics,
    PromotionVerdict,
    compute_metrics,
    engine_capital_for,
    evaluate_gates,
)
from engines.base import Context, Engine

log = logging.getLogger(__name__)


class BacktestError(RuntimeError):
    """The backtest could not run -- missing data, bad window, unknown engine."""


@dataclass
class Window:
    """A named walk-forward window (§4)."""

    name: str
    start: _dt.date
    end: Optional[_dt.date]

    def contains(self, day: _dt.date) -> bool:
        if day < self.start:
            return False
        return self.end is None or day <= self.end

    def __str__(self) -> str:
        return f"{self.name} {self.start}..{self.end or 'present'}"


def load_windows(settings: Settings | None = None) -> dict[str, Window]:
    """Read the §4 walk-forward windows from config."""
    settings = settings or get_settings()
    out: dict[str, Window] = {}
    for name, spec in (settings.require("backtest.windows") or {}).items():
        end = spec.get("end")
        out[name] = Window(
            name=name,
            start=_dt.date.fromisoformat(str(spec["start"])),
            end=_dt.date.fromisoformat(str(end)) if end else None,
        )
    return out


# ---------------------------------------------------------------------------
# Simulated book
# ---------------------------------------------------------------------------


@dataclass
class OpenPosition:
    """A position the backtest book is carrying."""

    position: Position
    entry_ts: _dt.datetime
    entry_costs: float
    stop: Optional[float]
    targets: list[float] = field(default_factory=list)
    signal_reason: str = ""


class BacktestBook:
    """The simulated broker: fills, positions, and completed trades.

    Implements the ``positions()`` contract the risk kernel expects, so the
    kernel's concurrency and capital-cap vetoes work unchanged in a backtest.
    """

    mode = "backtest"

    def __init__(self, cost_model: CostModel | None = None, starting_equity: float = 0.0) -> None:
        self.costs = cost_model or get_cost_model()
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self._open: dict[str, OpenPosition] = {}
        self.trades: list[BacktestTrade] = []
        self.fills: list[Fill] = []

    # -- kernel interface --------------------------------------------------

    def positions(self) -> list[Position]:
        return [op.position for op in self._open.values() if not op.position.is_flat]

    # -- execution ---------------------------------------------------------

    def key(self, engine: str, symbol: str) -> str:
        return f"{engine}:{symbol}"

    def open_position(
        self,
        order: Any,
        fill_price: float,
        timestamp: _dt.datetime,
        targets: Sequence[float] = (),
    ) -> Fill:
        """Open (or add to) a position at ``fill_price``, after slippage."""
        executed = self.costs.apply_slippage(order.segment, order.side, fill_price)
        charges = self.costs.charges(order.segment, order.side, executed, order.quantity)

        fill = Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            quantity=order.quantity, price=executed, timestamp=timestamp,
            engine=order.engine, costs=charges.total, is_paper=True,
            meta={"segment": order.segment.value, "reference_price": fill_price},
        )
        self.fills.append(fill)

        key = self.key(order.engine, order.symbol)
        existing = self._open.get(key)
        signed = order.quantity * order.side.sign

        if existing is None:
            self._open[key] = OpenPosition(
                position=Position(
                    symbol=order.symbol, quantity=signed, average_price=executed,
                    engine=order.engine, product=order.product, ttl=order.ttl,
                    stop=order.stop, opened_at=timestamp, last_price=executed,
                    segment=order.segment, meta=dict(order.meta),
                ),
                entry_ts=timestamp,
                entry_costs=charges.total,
                stop=order.stop,
                targets=list(targets),
                signal_reason=order.reason,
            )
        else:
            position = existing.position
            total = abs(position.quantity) + abs(signed)
            position.average_price = (
                position.average_price * abs(position.quantity) + executed * abs(signed)
            ) / total
            position.quantity += signed
            existing.entry_costs += charges.total
        self.cash -= charges.total
        return fill

    def close_position(
        self,
        engine: str,
        symbol: str,
        exit_price: float,
        timestamp: _dt.datetime,
        reason: str,
        quantity: int | None = None,
    ) -> Optional[BacktestTrade]:
        """Close all or part of a position, recording a :class:`BacktestTrade`."""
        key = self.key(engine, symbol)
        open_position = self._open.get(key)
        if open_position is None:
            return None

        position = open_position.position
        closing = abs(position.quantity) if quantity is None else min(quantity, abs(position.quantity))
        if closing <= 0:
            return None

        exit_side = Side.SELL if position.is_long else Side.BUY
        executed = self.costs.apply_slippage(position.segment, exit_side, exit_price)
        charges = self.costs.charges(position.segment, exit_side, executed, closing)

        direction = 1 if position.is_long else -1
        gross = (executed - position.average_price) * closing * direction
        entry_share = open_position.entry_costs * (closing / abs(position.quantity))

        self.fills.append(
            Fill(
                order_id="", symbol=symbol, side=exit_side, quantity=closing,
                price=executed, timestamp=timestamp, engine=engine,
                costs=charges.total, is_paper=True,
                meta={"exit": True, "reason": reason},
            )
        )

        trade = BacktestTrade(
            engine=engine, symbol=symbol,
            side="BUY" if direction > 0 else "SELL",
            quantity=closing,
            entry_ts=open_position.entry_ts, entry_price=position.average_price,
            exit_ts=timestamp, exit_price=executed,
            gross_pnl=round(gross, 2),
            costs=round(entry_share + charges.total, 2),
            exit_reason=reason,
            stop=open_position.stop,
            meta={"signal_reason": open_position.signal_reason},
        )
        self.trades.append(trade)
        self.cash += trade.net_pnl

        position.quantity -= closing * direction
        open_position.entry_costs -= entry_share
        if position.quantity == 0:
            self._open.pop(key, None)
        return trade

    def mark(self, prices: dict[str, float]) -> None:
        for open_position in self._open.values():
            price = prices.get(open_position.position.symbol)
            if price is not None:
                open_position.position.last_price = price

    def open_positions(self) -> list[OpenPosition]:
        return list(self._open.values())

    @property
    def equity(self) -> float:
        return self.cash


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Everything one engine-window run produced."""

    engine: str
    window: Window
    metrics: Metrics
    verdict: PromotionVerdict
    trades: list[BacktestTrade]
    bars_processed: int = 0
    signals_emitted: int = 0
    signals_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


class Backtester:
    """Drives one engine over one window."""

    def __init__(
        self,
        feed: DataFeed | None = None,
        cost_model: CostModel | None = None,
        journal: Journal | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.feed = feed or DataFeed(journal=journal)
        self.cost_model = cost_model or get_cost_model()
        self.journal = journal or get_journal()

    # -- data ---------------------------------------------------------------

    def load_bars(
        self, symbols: Sequence[str], intervals: Sequence[str], window: Window
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Load stored bars for the window, padded for indicator warm-up.

        The pad matters: a 200-DMA gate (§6.4) evaluated on the first day of the
        window needs 200 prior sessions, or the engine sits out the first year
        of every backtest and the result is meaningless.
        """
        pad_days = int(self.settings.get("backtest.warmup_pad_days", 400))
        start = window.start - _dt.timedelta(days=pad_days)
        end = window.end or clock.today_ist()

        bars: dict[tuple[str, str], pd.DataFrame] = {}
        missing: list[str] = []
        for symbol in symbols:
            for interval in intervals:
                try:
                    frame = self.feed.load(symbol, interval, start=start, end=end)
                except Exception as exc:
                    missing.append(f"{symbol}/{interval}")
                    log.debug("no %s bars for %s: %s", interval, symbol, exc)
                    continue
                if not frame.empty:
                    bars[(symbol.upper(), interval)] = frame
        if missing:
            log.warning(
                "%d symbol/interval pairs had no stored data and were skipped: %s%s",
                len(missing), ", ".join(missing[:10]), " ..." if len(missing) > 10 else "",
            )
        if not bars:
            raise BacktestError(
                f"No data at all for {len(symbols)} symbols over {window}. "
                f"Run scripts/download_history.py first."
            )
        return bars

    def build_timeline(
        self, bars: dict[tuple[str, str], pd.DataFrame], window: Window
    ) -> list[pd.Timestamp]:
        """Every distinct bar timestamp inside the window, ascending."""
        stamps: set[pd.Timestamp] = set()
        for frame in bars.values():
            inside = frame.index[
                (frame.index.date >= window.start)
                & ((window.end is None) | (frame.index.date <= (window.end or window.start)))
            ] if window.end else frame.index[frame.index.date >= window.start]
            stamps.update(inside)
        return sorted(stamps)

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        engine: Engine,
        window: Window,
        symbols: Sequence[str] | None = None,
        intervals: Sequence[str] = ("day",),
        market: MarketState | None = None,
    ) -> BacktestResult:
        """Run ``engine`` over ``window``. Returns metrics and a §4 verdict."""
        self._guard_test_window(engine.name, window)

        symbols = list(symbols) if symbols else engine.universe()
        if not symbols:
            raise BacktestError(
                f"Engine {engine.name} has an empty universe; nothing to backtest."
            )

        bars = self.load_bars(symbols, intervals, window)
        timeline = self.build_timeline(bars, window)
        if not timeline:
            raise BacktestError(f"No bars inside {window} for {engine.name}")

        engine_capital = engine_capital_for(engine.name, self.settings)
        book = BacktestBook(self.cost_model, starting_equity=engine_capital)
        kernel = RiskKernel(
            journal=self.journal,
            settings=self.settings,
            market=market or MarketState(),
            positions=book,
        )

        # Pre-index each frame so slicing to "now" is O(log n), not O(n).
        indices = {key: frame.index for key, frame in bars.items()}

        signals_emitted = 0
        signals_rejected = 0
        rejection_reasons: dict[str, int] = {}
        pending: list[tuple[Signal, Any]] = []

        for stamp in timeline:
            now = clock.to_ist(stamp.to_pydatetime())

            visible = self._slice(bars, indices, stamp)
            prices = self._prices_at(visible)
            book.mark(prices)

            # Market orders raised on the previous bar execute at THIS bar's
            # open. Filling at the signal bar's close is the classic lookahead
            # bug -- the close is not tradable once you have seen it.
            for signal, order in pending:
                open_price = self._open_price(visible, order.symbol, intervals[0])
                if open_price is None:
                    continue
                book.open_position(order, open_price, now, targets=signal.targets)
            pending = []

            ctx = Context(
                now=now,
                regime=Regime.NA,
                bars=visible,
                prices=prices,
                positions=book.positions(),
                journal=self.journal,
                is_backtest=True,
            )

            self._apply_exits(book, ctx, engine, prices, now)

            for signal in list(engine.manage(ctx)) + list(engine.on_schedule(ctx)):
                signals_emitted += 1
                reference = signal.reference_price or prices.get(signal.symbol)
                if reference is None:
                    continue
                try:
                    order = build_order(signal, reference, kernel=kernel)
                except (ValueError, Exception) as exc:  # noqa: B902
                    log.debug("could not build order for %s: %s", signal.symbol, exc)
                    continue

                decision = kernel.check(order, now=now)
                if not decision.allowed:
                    signals_rejected += 1
                    rejection_reasons[decision.reason_code] = (
                        rejection_reasons.get(decision.reason_code, 0) + 1
                    )
                    continue
                self.journal.record_order(order, mode="backtest")
                pending.append((signal, order))

        # Close anything still open at the end of the window, at the last price.
        final_prices = self._prices_at(self._slice(bars, indices, timeline[-1]))
        final_now = clock.to_ist(timeline[-1].to_pydatetime())
        for open_position in list(book.open_positions()):
            price = final_prices.get(open_position.position.symbol)
            if price is not None:
                book.close_position(
                    open_position.position.engine, open_position.position.symbol,
                    price, final_now, "end of backtest window",
                )

        metrics = compute_metrics(book.trades, engine.name, window.name, engine_capital)
        verdict = evaluate_gates(metrics, engine_capital, self.settings)

        self.journal.record_backtest_run(
            engine.name, window.name, window.start.isoformat(),
            window.end.isoformat() if window.end else None,
            verdict.verdict, metrics.as_dict(),
        )

        return BacktestResult(
            engine=engine.name, window=window, metrics=metrics, verdict=verdict,
            trades=book.trades, bars_processed=len(timeline),
            signals_emitted=signals_emitted, signals_rejected=signals_rejected,
            rejection_reasons=rejection_reasons,
        )

    # -- helpers -----------------------------------------------------------

    def _slice(
        self,
        bars: dict[tuple[str, str], pd.DataFrame],
        indices: dict[tuple[str, str], pd.Index],
        stamp: pd.Timestamp,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Truncate every frame to ``<= stamp``. The lookahead defence."""
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for key, frame in bars.items():
            position = indices[key].searchsorted(stamp, side="right")
            if position:
                out[key] = frame.iloc[:position]
        return out

    def _prices_at(self, visible: dict[tuple[str, str], pd.DataFrame]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for (symbol, _interval), frame in visible.items():
            if not frame.empty:
                prices[symbol] = float(frame["close"].iloc[-1])
        return prices

    def _open_price(
        self, visible: dict[tuple[str, str], pd.DataFrame], symbol: str, interval: str
    ) -> Optional[float]:
        frame = visible.get((symbol.upper(), interval))
        if frame is None or frame.empty:
            return None
        return float(frame["open"].iloc[-1])

    def _apply_exits(
        self,
        book: BacktestBook,
        ctx: Context,
        engine: Engine,
        prices: dict[str, float],
        now: _dt.datetime,
    ) -> None:
        """Stops, targets and the §3 15:10 force-flat, applied before signals.

        Stops are checked against the bar's LOW/HIGH, not its close: a stop
        that only triggers on closes systematically understates losses.
        """
        force_flat_at = str(self.settings.require("risk.vetoes.mis_force_flat_at"))
        for open_position in list(book.open_positions()):
            position = open_position.position
            symbol = position.symbol
            frame = ctx.bars_for(symbol, "day")
            if frame is None:
                frame = ctx.bars_for(symbol, "5minute")
            price = prices.get(symbol)
            if price is None:
                continue

            if open_position.stop is not None and frame is not None and not frame.empty:
                low = float(frame["low"].iloc[-1])
                high = float(frame["high"].iloc[-1])
                hit = (
                    low <= open_position.stop if position.is_long
                    else high >= open_position.stop
                )
                if hit:
                    book.close_position(
                        position.engine, symbol, open_position.stop, now, "stop"
                    )
                    continue

            if position.ttl is TTL.INTRADAY and clock.is_after(now, force_flat_at):
                book.close_position(
                    position.engine, symbol, price, now, f"{force_flat_at} force-flat (§3)"
                )

    def _guard_test_window(self, engine: str, window: Window) -> None:
        """§4: the final test window is consumed exactly once."""
        if window.name != "test":
            return
        if not self.settings.get("backtest.test_window_single_use", True):
            return
        runs = self.journal.test_window_runs(engine)
        if runs:
            log.warning(
                "TEST WINDOW RE-RUN: %s has already consumed the untouched test window "
                "%d time(s). Every additional run fits the strategy to data that was "
                "supposed to be seen once. Treat the result as tainted (§4).",
                engine, runs,
            )
            print(
                f"\n{'!' * 72}\n"
                f"  WARNING: {engine} has already used the `test` window {runs} time(s).\n"
                f"  §4 says the final test window is consumed EXACTLY ONCE.\n"
                f"  This result is no longer an out-of-sample measurement.\n"
                f"{'!' * 72}\n"
            )


# ---------------------------------------------------------------------------
# Buy-and-hold sanity check (§5.3 acceptance)
# ---------------------------------------------------------------------------


class BuyAndHold(Engine):
    """Buy on the first bar, hold. The §5.3 sanity backtest.

    Its job is to prove the plumbing: if buy-and-hold NIFTY does not produce a
    believable net number, no engine result can be trusted.
    """

    name = "buy_and_hold"

    def __init__(self, symbol: str = "NIFTYBEES", **kwargs: Any) -> None:
        settings = kwargs.pop("settings", None) or get_settings()
        # This engine is not in settings.yaml (it is a test harness, not a
        # strategy), so give it a minimal config rather than failing to load.
        data = settings.as_dict()
        data.setdefault("engines", {})["buy_and_hold"] = {
            "enabled": True, "auto_trade": True, "universe": None,
        }
        data.setdefault("risk", {}).setdefault("per_engine_capital_cap_pct", {})
        data["risk"]["per_engine_capital_cap_pct"]["buy_and_hold"] = 100
        from core.config import Settings as _Settings

        super().__init__(settings=_Settings(data, source="<buy_and_hold>"), **kwargs)
        self.symbol = symbol.upper()
        self._bought = False

    def universe(self) -> list[str]:
        return [self.symbol]

    def on_schedule(self, ctx: Context) -> list[Signal]:
        if self._bought or ctx.position_for(self.symbol, self.name):
            return []
        price = ctx.price(self.symbol)
        if price is None:
            return []
        self._bought = True
        return [
            self.signal(
                self.symbol, Side.BUY,
                stop=price * 0.5,          # a nominal stop so §3 sizing works
                reference_price=price,
                ttl=TTL.SWING,
                reason="buy and hold sanity check (§5.3)",
                segment=Segment.EQUITY_DELIVERY.value,
            )
        ]
