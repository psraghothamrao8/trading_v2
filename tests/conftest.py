"""Shared pytest fixtures. Tests never touch the network or a real broker (§0.5)."""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import calendar as trading_calendar  # noqa: E402
from core import clock  # noqa: E402
from core.broker import PaperBroker  # noqa: E402
from core.config import reset_config_cache  # noqa: E402
from core.journal import Journal, set_journal  # noqa: E402
from core.risk import MarketState, RiskKernel, set_kernel  # noqa: E402
from core.types import EntryType, Order, Product, Segment, Side, Signal, TTL  # noqa: E402
from live.alerts import NullAlerts, set_alerts  # noqa: E402

IST = clock.IST


@pytest.fixture(autouse=True)
def _isolate_globals():
    """Reset every process-wide singleton between tests."""
    reset_config_cache()
    trading_calendar.reset_calendar_cache()
    set_journal(None)
    set_kernel(None)
    set_alerts(NullAlerts())
    clock.set_clock(None)
    yield
    clock.set_clock(None)
    set_journal(None)
    set_kernel(None)
    set_alerts(None)
    reset_config_cache()
    trading_calendar.reset_calendar_cache()


@pytest.fixture
def frozen_clock():
    """Freeze the clock. Usage: ``frozen_clock(2026, 7, 22, 10, 30)``."""

    def _freeze(*args: int) -> _dt.datetime:
        ts = _dt.datetime(*args, tzinfo=IST)
        clock.set_clock(lambda: ts)
        return ts

    return _freeze


@pytest.fixture
def journal() -> Journal:
    """In-memory journal, wired as the process singleton."""
    j = Journal(":memory:")
    set_journal(j)
    yield j
    j.close()


@pytest.fixture
def paper_broker() -> PaperBroker:
    return PaperBroker(price_source={}, starting_capital=800_000.0)


@pytest.fixture
def market() -> MarketState:
    return MarketState()


@pytest.fixture
def kernel(journal, market, paper_broker) -> RiskKernel:
    k = RiskKernel(journal=journal, market=market, positions=paper_broker)
    set_kernel(k)
    return k


@pytest.fixture
def make_signal():
    """Factory for Signals with sensible defaults."""

    def _make(
        symbol: str = "RELIANCE",
        side: Side = Side.BUY,
        stop: float | None = 2900.0,
        reference_price: float | None = 3000.0,
        engine: str = "filings",
        ttl: TTL = TTL.INTRADAY,
        entry_type: EntryType = EntryType.MARKET,
        targets: tuple[float, ...] = (3100.0,),
        **meta,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            side=side,
            entry_type=entry_type,
            stop=stop,
            targets=targets,
            ttl=ttl,
            reason="test signal",
            engine=engine,
            meta=meta,
            reference_price=reference_price,
        )

    return _make


@pytest.fixture
def make_order():
    """Factory for Orders with sensible defaults."""

    def _make(
        symbol: str = "RELIANCE",
        side: Side = Side.BUY,
        quantity: int = 10,
        engine: str = "filings",
        ttl: TTL = TTL.INTRADAY,
        segment: Segment = Segment.EQUITY_INTRADAY,
        product: Product = Product.MIS,
        entry_type: EntryType = EntryType.MARKET,
        stop: float | None = 2900.0,
        price: float | None = None,
        is_entry: bool = True,
        reference_price: float = 3000.0,
        **meta,
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_type=entry_type,
            product=product,
            engine=engine,
            signal_id="sig-test",
            segment=segment,
            price=price,
            stop=stop,
            ttl=ttl,
            reason="test order",
            meta={"reference_price": reference_price, **meta},
            created_at=clock.now_ist(),
            is_entry=is_entry,
        )

    return _make
