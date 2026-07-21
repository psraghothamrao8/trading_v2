"""Broker tests — §0.1 paper default and §8.6 product types."""

from __future__ import annotations

import io

import pytest

from core.broker import (
    BrokerError,
    PaperBroker,
    confirm_live,
    get_broker,
    product_for,
    resolve_mode,
)
from core.types import Position, Product, Segment, Side, TTL


class TestProductSelection:
    """§8.6: MIS intraday, CNC overnight equity, NRML overnight F&O."""

    def test_intraday_is_mis(self):
        assert product_for(TTL.INTRADAY, Segment.EQUITY_INTRADAY, Side.BUY) is Product.MIS
        assert product_for(TTL.INTRADAY, Segment.EQUITY_OPTIONS, Side.SELL) is Product.MIS

    def test_overnight_equity_long_is_cnc(self):
        assert product_for(TTL.OVERNIGHT, Segment.EQUITY_DELIVERY, Side.BUY) is Product.CNC

    def test_overnight_derivatives_are_nrml(self):
        assert product_for(TTL.OVERNIGHT, Segment.EQUITY_FUTURES, Side.SELL) is Product.NRML
        assert product_for(TTL.SWING, Segment.EQUITY_OPTIONS, Side.SELL) is Product.NRML

    def test_overnight_equity_short_is_impossible(self):
        with pytest.raises(BrokerError, match="cannot short equity overnight"):
            product_for(TTL.OVERNIGHT, Segment.EQUITY_DELIVERY, Side.SELL)


class TestPaperBroker:
    def test_never_needs_authentication(self, paper_broker):
        assert paper_broker.is_authenticated() is True
        assert paper_broker.mode == "paper"

    def test_fill_creates_a_position(self, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        broker = PaperBroker(price_source={"RELIANCE": 3000.0})
        broker.place_order(make_order(quantity=10))
        positions = broker.positions()
        assert len(positions) == 1
        assert positions[0].quantity == 10

    def test_slippage_hurts_in_both_directions(self, make_order, frozen_clock):
        """§4: slippage is a cost, never a gift."""
        frozen_clock(2026, 7, 22, 10, 0)
        broker = PaperBroker(price_source={"RELIANCE": 3000.0})
        broker.place_order(make_order(side=Side.BUY, quantity=1))
        buy_price = broker.fills[-1].price
        broker.place_order(make_order(side=Side.SELL, quantity=1, stop=3100.0))
        sell_price = broker.fills[-1].price
        assert buy_price > 3000.0
        assert sell_price < 3000.0

    def test_option_slippage_is_wider_than_equity(self, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        broker = PaperBroker(price_source={"X": 1000.0})
        broker.place_order(make_order(symbol="X", segment=Segment.EQUITY_INTRADAY, quantity=1))
        equity_price = broker.fills[-1].price
        broker.place_order(make_order(symbol="X", segment=Segment.EQUITY_OPTIONS, quantity=1))
        option_price = broker.fills[-1].price
        assert option_price > equity_price

    def test_closing_a_position_removes_it(self, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        broker = PaperBroker(price_source={"RELIANCE": 3000.0})
        broker.place_order(make_order(side=Side.BUY, quantity=10))
        broker.place_order(make_order(side=Side.SELL, quantity=10, stop=3100.0))
        assert broker.positions() == []

    def test_missing_price_raises_rather_than_inventing_one(self, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        broker = PaperBroker(price_source={})
        order = make_order()
        order.meta.pop("reference_price")
        with pytest.raises(BrokerError, match="needs a reference price"):
            broker.place_order(order)

    def test_seed_position_for_restart(self, paper_broker):
        paper_broker.seed_position(
            Position(symbol="A", quantity=5, average_price=100.0, engine="filings",
                     product=Product.MIS)
        )
        assert len(paper_broker.positions()) == 1

    def test_account_snapshot(self, paper_broker):
        snapshot = paper_broker.account()
        assert snapshot.mode == "paper"
        assert snapshot.authenticated is True
        assert snapshot.equity_available == 800_000.0


class TestModeResolution:
    def test_defaults_to_paper(self):
        """§0.1: paper is the default."""
        assert resolve_mode() == "paper"

    def test_explicit_override(self):
        assert resolve_mode("live") == "live"

    def test_get_broker_returns_paper_by_default(self):
        assert isinstance(get_broker(interactive=False), PaperBroker)

    def test_live_without_confirmation_falls_back_to_paper(self):
        """§0.1: live needs config AND an interactive y/N. Non-interactive => paper."""
        broker = get_broker("live", interactive=False)
        assert isinstance(broker, PaperBroker)

    def test_confirm_live_refuses_non_interactive(self):
        out = io.StringIO()
        assert confirm_live(interactive=False, stream=out) is False
        assert "non-interactive" in out.getvalue()

    def test_unknown_mode_raises(self):
        with pytest.raises(BrokerError, match="Unknown execution mode"):
            get_broker("backtest", interactive=False)
