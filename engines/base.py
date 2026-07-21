"""Engine interface — §6.0.

An engine's only output is a list of :class:`~core.types.Signal`. It never
sizes an order, never checks risk, and never touches the broker (§0.3). The
orchestrator converts Signals to Orders via ``core.risk.build_order`` and
submits them to ``core.risk.check``.

:class:`Context` is the read-only world the engine sees. Everything an engine
needs -- prices, the journal, the clock, the regime, its own config block --
arrives through it, which is what makes engines testable against mocked data
and reusable inside the backtester without change.
"""

from __future__ import annotations

import datetime as _dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from core import clock
from core.config import Settings, get_settings, get_universe, resolve_universe
from core.journal import Journal, get_journal
from core.types import Fill, Position, Regime, Signal, Tick

log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """An engine could not run. Raised loudly, never swallowed (§0.8)."""


@dataclass
class Context:
    """Everything an engine may read. Engines must not mutate it.

    ``bars`` maps ``(symbol, interval) -> DataFrame`` of OHLCV, IST-indexed and
    **already truncated to `now`** by the caller. That truncation is the whole
    lookahead-bias defence: an engine physically cannot see a future bar
    because the frame does not contain one.
    """

    now: _dt.datetime
    regime: Regime = Regime.NA
    bars: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    positions: list[Position] = field(default_factory=list)
    journal: Optional[Journal] = None
    india_vix: Optional[float] = None
    index_level: Optional[float] = None
    extras: dict[str, Any] = field(default_factory=dict)
    is_backtest: bool = False

    @property
    def today(self) -> _dt.date:
        return self.now.date()

    def bars_for(self, symbol: str, interval: str = "day") -> Optional[pd.DataFrame]:
        """Bars for a symbol, or None when the data was never loaded."""
        return self.bars.get((symbol.upper(), interval))

    def require_bars(self, symbol: str, interval: str = "day") -> pd.DataFrame:
        """Bars, or a loud failure. Silent empties become 'strategy found nothing'."""
        frame = self.bars_for(symbol, interval)
        if frame is None or frame.empty:
            raise EngineError(
                f"No {interval} bars for {symbol} at {self.now:%Y-%m-%d %H:%M}. "
                f"Download them with scripts/download_history.py."
            )
        return frame

    def price(self, symbol: str) -> Optional[float]:
        """Last known price: the live tick if present, else the latest close."""
        if symbol in self.prices:
            return self.prices[symbol]
        frame = self.bars_for(symbol, "day")
        if frame is not None and not frame.empty:
            return float(frame["close"].iloc[-1])
        return None

    def position_for(self, symbol: str, engine: str | None = None) -> Optional[Position]:
        for position in self.positions:
            if position.symbol == symbol and (engine is None or position.engine == engine):
                return position
        return None

    def positions_for_engine(self, engine: str) -> list[Position]:
        return [p for p in self.positions if p.engine == engine and not p.is_flat]

    def the_journal(self) -> Journal:
        return self.journal or get_journal()


class Engine(ABC):
    """§6.0 common interface.

    Subclasses implement whichever hooks apply:

    * ``on_schedule`` -- timed engines (most of them)
    * ``on_tick``     -- streaming engines
    * ``manage``      -- stops, trails, exits for open positions
    * ``on_fill``     -- react to an execution

    The defaults return nothing rather than raising, so an engine that is
    purely scheduled does not need an empty ``on_tick``. That is a genuine
    "not applicable", not a stub (§0.8).
    """

    name: str = "base"

    def __init__(self, settings: Settings | None = None, journal: Journal | None = None) -> None:
        self.settings = settings or get_settings()
        self.journal = journal or get_journal()
        if self.name == "base":
            raise EngineError(f"{type(self).__name__} must set a `name` class attribute")
        self.config = self.settings.section(f"engines.{self.name}")

    # -- configuration ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def auto_trade(self) -> bool:
        """False means alert-only. Every engine starts here until PROMOTED (§4)."""
        return bool(self.config.get("auto_trade", False))

    @property
    def alert_only(self) -> bool:
        """§6.10 / §6.11 are structurally alert-only, not merely un-promoted."""
        return bool(self.config.get("alert_only", False))

    def universe(self) -> list[str]:
        """§6.0: the symbols this engine watches. Resolved from config."""
        name = self.config.get("universe", None)
        if name is None:
            return []
        return resolve_universe(str(name))

    # -- hooks ------------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """Timed evaluation. Default: nothing to do at this time."""
        return []

    def on_tick(self, tick: Tick, ctx: Context) -> list[Signal]:
        """Streaming evaluation. Default: this engine is not tick-driven."""
        return []

    def on_fill(self, fill: Fill, ctx: Context) -> None:
        """React to an execution. Default: no per-fill bookkeeping needed."""
        return None

    def manage(self, ctx: Context) -> list[Signal]:
        """Stops, trails and exits for this engine's open positions."""
        return []

    # -- helpers shared by concrete engines -------------------------------

    def signal(
        self,
        symbol: str,
        side: Any,
        *,
        stop: float | None,
        reference_price: float | None,
        targets: Sequence[float] = (),
        ttl: Any = None,
        entry_type: Any = None,
        reason: str = "",
        **meta: Any,
    ) -> Signal:
        """Build a Signal tagged with this engine's name and the current time."""
        from core.types import EntryType, TTL

        return Signal(
            symbol=symbol.upper(),
            side=side,
            entry_type=entry_type or EntryType.MARKET,
            stop=stop,
            targets=tuple(targets),
            ttl=ttl or TTL.INTRADAY,
            reason=reason,
            engine=self.name,
            meta=meta,
            reference_price=reference_price,
            created_at=clock.now_ist(),
        )

    def within_window(self, ctx: Context, window_key: str) -> bool:
        """True when ``ctx.now`` is inside a configured ``{start, end}`` window."""
        window = self.config.get(window_key, None)
        if not window:
            return True
        return clock.within(ctx.now, str(window["start"]), str(window["end"]))

    def trades_today(self, ctx: Context) -> int:
        return ctx.the_journal().new_entry_count_today(ctx.today.isoformat(), self.name)

    def at_daily_limit(self, ctx: Context, key: str = "max_trades_per_day") -> bool:
        """Engine-local trade cap. The §3 kernel enforces its own cap on top."""
        limit = self.config.get(key, None)
        if limit is None:
            return False
        return self.trades_today(ctx) >= int(limit)

    def concurrent_positions(self, ctx: Context) -> int:
        return len(ctx.positions_for_engine(self.name))

    def at_concurrency_limit(self, ctx: Context, key: str = "max_concurrent") -> bool:
        limit = self.config.get(key, None)
        if limit is None:
            return False
        return self.concurrent_positions(ctx) >= int(limit)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} enabled={self.enabled}>"


class AlertOnlyEngine(Engine):
    """Base for §6.10 and §6.11: engines that alert and never trade.

    ``on_schedule`` is final here and always returns ``[]``. Alert-only is a
    structural property, not a config flag someone can flip in a hurry -- the
    §3 kernel rejects their orders too, so this is defence in depth.
    """

    def alerts_for(self, ctx: Context) -> list[str]:
        """Return the alert lines this engine wants to send. Override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement alerts_for()"
        )

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """Alert-only engines never emit tradable Signals (§6.10, §6.11)."""
        return []


def load_engines(
    names: Iterable[str] | None = None,
    settings: Settings | None = None,
    journal: Journal | None = None,
) -> dict[str, Engine]:
    """Instantiate engines by name from the registry.

    Import is lazy and per-engine so one broken engine cannot stop the others
    from loading -- but the failure is logged and journalled, never hidden.
    """
    from engines import registry

    settings = settings or get_settings()
    journal = journal or get_journal()
    wanted = list(names) if names else list(registry.ENGINE_CLASSES)

    loaded: dict[str, Engine] = {}
    for name in wanted:
        cls = registry.ENGINE_CLASSES.get(name)
        if cls is None:
            raise EngineError(f"Unknown engine {name!r}. Known: {sorted(registry.ENGINE_CLASSES)}")
        try:
            loaded[name] = cls(settings=settings, journal=journal)
        except Exception as exc:
            log.error("Engine %s failed to load: %s", name, exc, exc_info=True)
            journal.record_error("engine_load", f"{name}: {exc}", severity="ERROR")
    return loaded
