"""Engine interface and registry tests — §6.0, §0.3."""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.types import Position, Product, Regime, Side, Signal, TTL
from engines.base import AlertOnlyEngine, Context, Engine, EngineError, load_engines
from engines.registry import ALERT_ONLY, ENGINE_CLASSES

IST = clock.IST


def bars(days: int = 30, start: _dt.date = _dt.date(2026, 6, 1), base: float = 100.0):
    index = pd.date_range(start, periods=days, freq="D", tz=IST)
    values = [base + i for i in range(days)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": [v + 0.5 for v in values],
            "volume": [100_000] * days,
        },
        index=index,
    )


class TestRegistry:
    def test_all_eleven_spec_engines_are_registered(self):
        """§6.1 - §6.11."""
        assert set(ENGINE_CLASSES) == {
            "filings", "sympathy", "pairs", "overnight", "preopen", "pead",
            "panic_reversion", "wheel", "flows", "surveillance", "special_situations",
        }

    def test_every_registered_engine_has_a_config_block(self):
        from core.config import get_settings

        configured = set(get_settings().get("engines", {}))
        assert set(ENGINE_CLASSES) == configured

    def test_every_registered_engine_has_a_capital_cap(self):
        from core.config import get_settings

        caps = set(get_settings().get("risk.per_engine_capital_cap_pct", {}))
        assert set(ENGINE_CLASSES) <= caps

    def test_alert_only_set_matches_the_risk_config(self):
        from core.config import get_settings

        assert ALERT_ONLY == set(get_settings().require("risk.alert_only_engines"))

    def test_all_engines_load(self, journal):
        engines = load_engines(journal=journal)
        assert len(engines) == 11

    def test_engine_names_match_their_keys(self, journal):
        for name, engine in load_engines(journal=journal).items():
            assert engine.name == name

    def test_unknown_engine_raises(self, journal):
        with pytest.raises(EngineError, match="Unknown engine"):
            load_engines(["nope"], journal=journal)


class TestEngineDefaults:
    def test_every_engine_ships_with_auto_trade_off(self, journal):
        """§4: an engine stays alert-only until it is PROMOTED."""
        for name, engine in load_engines(journal=journal).items():
            assert engine.auto_trade is False, f"{name} ships with auto_trade on"

    def test_alert_only_engines_are_flagged(self, journal):
        engines = load_engines(journal=journal)
        for name in ALERT_ONLY:
            assert engines[name].alert_only is True

    def test_base_engine_requires_a_name(self):
        class Nameless(Engine):
            pass

        with pytest.raises(EngineError, match="must set a `name`"):
            Nameless()

    def test_unimplemented_hooks_return_empty_not_raise(self, journal):
        """A scheduled engine needs no on_tick; that is 'not applicable', not a stub."""
        engine = load_engines(["overnight"], journal=journal)["overnight"]
        ctx = Context(now=clock.now_ist(), journal=journal)
        from core.types import Tick

        assert engine.on_tick(Tick("X", 1.0, clock.now_ist()), ctx) == []
        assert engine.on_fill(None, ctx) is None


class TestContext:
    def test_bars_for_and_require_bars(self, journal, frozen_clock):
        now = frozen_clock(2026, 6, 30, 10, 0)
        ctx = Context(now=now, bars={("X", "day"): bars()}, journal=journal)
        assert ctx.bars_for("x", "day") is not None
        with pytest.raises(EngineError, match="download_history"):
            ctx.require_bars("MISSING")

    def test_price_prefers_the_live_tick(self, journal, frozen_clock):
        now = frozen_clock(2026, 6, 30, 10, 0)
        ctx = Context(now=now, bars={("X", "day"): bars()}, prices={"X": 999.0}, journal=journal)
        assert ctx.price("X") == 999.0

    def test_price_falls_back_to_the_last_close(self, journal, frozen_clock):
        now = frozen_clock(2026, 6, 30, 10, 0)
        frame = bars()
        ctx = Context(now=now, bars={("X", "day"): frame}, journal=journal)
        assert ctx.price("X") == pytest.approx(float(frame["close"].iloc[-1]))

    def test_price_of_an_unknown_symbol_is_none(self, journal, frozen_clock):
        ctx = Context(now=frozen_clock(2026, 6, 30, 10, 0), journal=journal)
        assert ctx.price("NOPE") is None

    def test_positions_for_engine_filters(self, journal, frozen_clock):
        positions = [
            Position("A", 1, 100.0, "filings", Product.MIS),
            Position("B", 1, 100.0, "pairs", Product.MIS),
            Position("C", 0, 100.0, "filings", Product.MIS),
        ]
        ctx = Context(now=frozen_clock(2026, 6, 30, 10, 0), positions=positions, journal=journal)
        assert [p.symbol for p in ctx.positions_for_engine("filings")] == ["A"]


class TestAlertOnlyIsStructural:
    """§6.10 / §6.11: alert-only is a safety property, not a config preference."""

    def test_on_schedule_always_returns_nothing(self, journal, frozen_clock):
        ctx = Context(now=frozen_clock(2026, 6, 30, 10, 0), journal=journal)
        for name in ALERT_ONLY:
            engine = load_engines([name], journal=journal)[name]
            assert engine.on_schedule(ctx) == []

    def test_still_returns_nothing_with_auto_trade_forced_on(self, journal, frozen_clock):
        """Flipping the config flag must not make an alert-only engine trade."""
        engine = load_engines(["surveillance"], journal=journal)["surveillance"]
        engine.config._data["auto_trade"] = True    # type: ignore[attr-defined]
        ctx = Context(now=frozen_clock(2026, 6, 30, 10, 0), journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_the_kernel_rejects_their_orders_too(self, kernel, make_order, frozen_clock):
        """Defence in depth: even a hand-built order is refused (§3)."""
        from core.risk import Reason

        frozen_clock(2026, 7, 22, 10, 0)
        for name in ALERT_ONLY:
            decision = kernel.check(make_order(engine=name))
            assert decision.reason_code == Reason.ALERT_ONLY_ENGINE

    def test_alerts_for_must_be_implemented(self):
        class Incomplete(AlertOnlyEngine):
            name = "surveillance"

        with pytest.raises(NotImplementedError, match="alerts_for"):
            Incomplete().alerts_for(Context(now=clock.now_ist()))


class TestSignalConstruction:
    def test_signal_is_tagged_with_the_engine(self, journal, frozen_clock):
        frozen_clock(2026, 6, 30, 10, 0)
        engine = load_engines(["overnight"], journal=journal)["overnight"]
        signal = engine.signal("X", Side.BUY, stop=95.0, reference_price=100.0)
        assert isinstance(signal, Signal)
        assert signal.engine == "overnight"
        assert signal.created_at is not None

    def test_limit_signal_defaults_its_price_to_the_reference(self, journal):
        """The helper fills limit_price so engines cannot build an invalid Signal."""
        from core.types import EntryType

        engine = load_engines(["overnight"], journal=journal)["overnight"]
        signal = engine.signal("X", Side.BUY, stop=95.0, reference_price=100.0,
                               entry_type=EntryType.LIMIT)
        assert signal.limit_price == 100.0

    def test_raw_limit_signal_without_a_price_is_rejected(self):
        """The Signal dataclass itself still refuses an unpriced LIMIT order."""
        from core.types import EntryType, Signal as RawSignal

        with pytest.raises(ValueError, match="needs limit_price"):
            RawSignal(symbol="X", side=Side.BUY, entry_type=EntryType.LIMIT,
                      stop=95.0, targets=(), ttl=TTL.INTRADAY, reason="r",
                      engine="test")

    def test_risk_per_unit(self, journal):
        engine = load_engines(["overnight"], journal=journal)["overnight"]
        signal = engine.signal("X", Side.BUY, stop=95.0, reference_price=100.0)
        assert signal.risk_per_unit == pytest.approx(5.0)
