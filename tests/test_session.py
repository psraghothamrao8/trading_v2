"""Session runtime tests — §5.6, §7, and the §0.3 one-way path.

Includes the Phase 6 acceptance test: one complete simulated session
end-to-end with a journal and a 15:45 digest.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.broker import PaperBroker
from core.risk import Reason
from core.types import Position, Product, Regime, Segment, Side, Signal, TTL
from engines.base import Context, Engine
from live.alerts import Command, NullAlerts
from live.session import Session, _split_hhmmss

IST = clock.IST


class StubEngine(Engine):
    """A controllable engine: returns whatever it is told to."""

    name = "filings"

    def __init__(self, entries=None, exits=None, boom=False, **kwargs):
        super().__init__(**kwargs)
        self._entries = entries or []
        self._exits = exits or []
        self._boom = boom
        self.fills: list = []
        self.config._data["auto_trade"] = True      # type: ignore[attr-defined]

    def on_schedule(self, ctx):
        if self._boom:
            raise RuntimeError("engine exploded")
        return list(self._entries)

    def manage(self, ctx):
        return list(self._exits)

    def on_fill(self, fill, ctx):
        self.fills.append(fill)


def make_session(journal, engines=None, prices=None, alerts=None) -> Session:
    broker = PaperBroker(price_source=prices or {}, starting_capital=800_000.0)
    session = Session(
        broker=broker,
        journal=journal,
        alerts=alerts or NullAlerts(),
        feed=None,
        nse=object(),
        engines=engines if engines is not None else {},
        interactive=False,
    )
    session.update_prices(prices or {})
    return session


def entry_signal(engine="filings", symbol="RELIANCE", price=3000.0, stop=2900.0) -> Signal:
    from core.types import EntryType

    return Signal(
        symbol=symbol, side=Side.BUY, entry_type=EntryType.MARKET, stop=stop,
        targets=(3100.0,), ttl=TTL.INTRADAY, reason="test", engine=engine,
        meta={"segment": Segment.EQUITY_INTRADAY.value}, reference_price=price,
    )


# ===========================================================================
# The one-way path (§0.3)
# ===========================================================================


class TestRouting:
    def test_allowed_signal_reaches_the_broker(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        ctx = session.build_context()
        assert session.route(entry_signal(), ctx) is not None
        assert len(session.broker.positions()) == 1

    def test_signal_is_journalled_even_when_rejected(self, journal, frozen_clock):
        """§0.6: everything is journalled -- including what did not happen."""
        frozen_clock(2026, 7, 22, 10, 0)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        session.market.surveillance.add("RELIANCE")
        ctx = session.build_context()
        assert session.route(entry_signal(), ctx) is None
        assert len(journal.query("SELECT * FROM signals")) == 1
        assert len(journal.query("SELECT * FROM rejections")) == 1

    def test_rejected_signal_never_reaches_the_broker(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        session.market.surveillance.add("RELIANCE")
        session.route(entry_signal(), session.build_context())
        assert session.broker.positions() == []

    def test_rejection_is_alerted_with_its_code(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        session = make_session(journal, prices={"RELIANCE": 3000.0}, alerts=alerts)
        session.market.surveillance.add("RELIANCE")
        session.route(entry_signal(), session.build_context())
        assert any(Reason.SURVEILLANCE in m for m in alerts.sent_messages)

    def test_expiry_day_halves_the_size(self, journal, frozen_clock):
        """§7: expiry days halve all new sizes, applied in the session, not engines."""
        frozen_clock(2026, 7, 28, 10, 0)      # Tuesday == configured expiry weekday
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        session.job_regime()
        session.route(entry_signal(), session.build_context())
        row = journal.query("SELECT quantity FROM orders")[0]
        assert row["quantity"] == 20          # half of the normal 40

    def test_exit_signals_are_not_size_multiplied(self, journal, frozen_clock):
        frozen_clock(2026, 7, 28, 15, 0)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        session.job_regime()
        session.broker.seed_position(
            Position("RELIANCE", 40, 3000.0, "filings", Product.MIS,
                     last_price=3000.0, segment=Segment.EQUITY_INTRADAY)
        )
        exit_sig = entry_signal()
        exit_sig.meta.update({"exit": True, "quantity": 40})
        object.__setattr__(exit_sig, "side", Side.SELL)
        session.route(exit_sig, session.build_context())
        assert journal.query("SELECT quantity FROM orders")[0]["quantity"] == 40

    def test_routed_entry_alert_carries_stop_target_and_fill_price(self, journal, frozen_clock):
        """The alert a manual trader acts on must be a complete instruction."""
        frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        session = make_session(journal, prices={"RELIANCE": 3000.0}, alerts=alerts)
        session.route(entry_signal(), session.build_context())
        text = next(m for m in alerts.sent_messages if "ENTRY" in m)
        assert "2,900.00" in text        # stop
        assert "3,100.00" in text        # target
        assert "test" in text            # signal.reason

    def test_routed_exit_alert_omits_stop_and_target(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 15, 0)
        alerts = NullAlerts()
        session = make_session(journal, prices={"RELIANCE": 3000.0}, alerts=alerts)
        session.broker.seed_position(
            Position("RELIANCE", 40, 2950.0, "filings", Product.MIS,
                     last_price=3000.0, segment=Segment.EQUITY_INTRADAY)
        )
        exit_sig = entry_signal()
        exit_sig.meta.update({"exit": True, "quantity": 40})
        object.__setattr__(exit_sig, "side", Side.SELL)
        session.route(exit_sig, session.build_context())
        text = next(m for m in alerts.sent_messages if "EXIT" in m)
        assert "stop" not in text.lower()
        assert "target" not in text.lower()

    def test_missing_price_drops_the_signal_without_an_order(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        session = make_session(journal, prices={})
        signal = entry_signal()
        object.__setattr__(signal, "reference_price", None)
        assert session.route(signal, session.build_context()) is None


# ===========================================================================
# The engine loop
# ===========================================================================


class TestRunCycle:
    def test_entries_are_routed(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 30)
        engine = StubEngine(entries=[entry_signal()], journal=journal)
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.job_regime()
        session.state.decision.enabled_engines = ["filings"]
        assert session.run_cycle() == 1

    def test_auto_trade_off_blocks_new_entries_at_the_session_not_the_engine(
        self, journal, frozen_clock
    ):
        """The one authoritative auto_trade gate lives in run_cycle().

        Engines themselves always emit whatever their own criteria produce
        (see test_engines_news.py::test_signal_is_produced_even_with_auto_trade_off)
        so that a backtest can evaluate an unpromoted engine at all. This test
        is the other half: confirming the live loop still refuses to act on
        those signals for anything that has not been promoted.
        """
        frozen_clock(2026, 7, 22, 10, 30)
        engine = StubEngine(entries=[entry_signal()], journal=journal)
        engine.config._data["auto_trade"] = False       # type: ignore[attr-defined]
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.job_regime()
        session.state.decision.enabled_engines = ["filings"]
        assert session.run_cycle() == 0
        assert session.broker.positions() == []

    def test_auto_trade_off_still_manages_existing_positions(self, journal, frozen_clock):
        """A demoted engine's open position must still get exited."""
        frozen_clock(2026, 7, 22, 15, 0)
        exit_sig = entry_signal()
        exit_sig.meta.update({"exit": True, "quantity": 10})
        object.__setattr__(exit_sig, "side", Side.SELL)

        engine = StubEngine(exits=[exit_sig], journal=journal)
        engine.config._data["auto_trade"] = False       # type: ignore[attr-defined]
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.broker.seed_position(
            Position("RELIANCE", 10, 3000.0, "filings", Product.MIS,
                     last_price=3000.0, segment=Segment.EQUITY_INTRADAY)
        )
        assert session.run_cycle() == 1
        assert session.broker.positions() == []

    def test_management_runs_before_entries(self, journal, frozen_clock):
        """An exit that frees a slot must not queue behind signal generation."""
        frozen_clock(2026, 7, 22, 10, 30)
        order: list[str] = []

        class Ordered(StubEngine):
            def manage(self, ctx):
                order.append("manage")
                return []

            def on_schedule(self, ctx):
                order.append("schedule")
                return []

        session = make_session(journal, {"filings": Ordered(journal=journal)})
        session.job_regime()
        session.state.decision.enabled_engines = ["filings"]
        session.run_cycle()
        assert order == ["manage", "schedule"]

    def test_a_broken_engine_does_not_stop_the_others(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 30)
        broken = StubEngine(boom=True, journal=journal)
        working = StubEngine(entries=[entry_signal(engine="pairs", symbol="TCS")],
                             journal=journal)
        working.name = "pairs"
        session = make_session(journal, {"filings": broken, "pairs": working},
                               {"RELIANCE": 3000.0, "TCS": 3000.0})
        session.job_regime()
        session.state.decision.enabled_engines = ["filings", "pairs"]
        assert session.run_cycle() == 1
        errors = journal.query("SELECT * FROM errors WHERE source='filings'")
        assert len(errors) == 1

    def test_disabled_engines_are_skipped(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 30)
        engine = StubEngine(entries=[entry_signal()], journal=journal)
        engine.config._data["enabled"] = False      # type: ignore[attr-defined]
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.job_regime()
        session.state.decision.enabled_engines = ["filings"]
        assert session.run_cycle() == 0

    def test_regime_gate_blocks_new_entries(self, journal, frozen_clock):
        """§7: an engine not enabled in today's regime may not open."""
        frozen_clock(2026, 7, 22, 10, 30)
        engine = StubEngine(entries=[entry_signal()], journal=journal)
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.job_regime()
        session.state.decision.enabled_engines = ["pairs"]
        assert session.run_cycle() == 0

    def test_management_still_runs_after_the_entry_cutoff(self, journal, frozen_clock):
        """§3 blocks new entries at 14:45; exits must keep working."""
        frozen_clock(2026, 7, 22, 15, 0)
        exit_sig = entry_signal()
        exit_sig.meta.update({"exit": True, "quantity": 10})
        object.__setattr__(exit_sig, "side", Side.SELL)

        engine = StubEngine(entries=[entry_signal()], exits=[exit_sig], journal=journal)
        session = make_session(journal, {"filings": engine}, {"RELIANCE": 3000.0})
        session.broker.seed_position(
            Position("RELIANCE", 10, 3000.0, "filings", Product.MIS,
                     last_price=3000.0, segment=Segment.EQUITY_INTRADAY)
        )
        session.job_entry_cutoff()
        assert session.run_cycle() == 1
        assert session.broker.positions() == []


# ===========================================================================
# Scheduled jobs (§7)
# ===========================================================================


class TestOvernightCheckRespectsAutoTrade:
    """job_overnight_check() calls on_schedule() directly, bypassing
    run_cycle()'s entry loop -- it must repeat that loop's auto_trade check
    itself or an unpromoted overnight engine would place real paper orders
    every night it happens to pass its filters."""

    def _rising_bars(self, n=260, end=_dt.date(2026, 7, 22)):
        closes = [100.0 + i * 0.2 for i in range(n)]
        index = pd.date_range(end - _dt.timedelta(days=n - 1), periods=n, freq="D", tz=IST)
        return pd.DataFrame(
            {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
             "close": closes, "volume": [100_000.0] * n},
            index=index,
        )

    def test_unpromoted_overnight_never_routes_from_the_dedicated_job(
        self, journal, frozen_clock
    ):
        from engines.overnight import OvernightEngine

        frozen_clock(2026, 7, 22, 15, 20)
        engine = OvernightEngine(journal=journal)
        assert engine.auto_trade is False, "must be the shipped default for this test"
        session = make_session(journal, {"overnight": engine}, {"NIFTYBEES": 152.0})
        session.update_bars({("NIFTYBEES", "day"): self._rising_bars()})
        assert session.job_overnight_check() == 0
        assert session.broker.positions() == []

    def test_promoted_overnight_does_route(self, journal, frozen_clock):
        from engines.overnight import OvernightEngine

        frozen_clock(2026, 7, 22, 15, 20)
        engine = OvernightEngine(journal=journal)
        engine.config._data["auto_trade"] = True     # type: ignore[attr-defined]
        session = make_session(journal, {"overnight": engine}, {"NIFTYBEES": 152.0})
        session.update_bars({("NIFTYBEES", "day"): self._rising_bars()})
        assert session.job_overnight_check() == 1
        assert len(session.broker.positions()) == 1


class TestScheduledJobs:
    def test_every_spec_job_is_scheduled(self, journal, frozen_clock):
        """§7 names the job set explicitly."""
        frozen_clock(2026, 7, 22, 8, 0)
        session = make_session(journal)
        ids = set(session.scheduled_job_ids())
        assert {
            "auth_check", "preopen_context", "preopen_snapshot_1", "preopen_snapshot_2",
            "announcements_poll", "regime_classify", "entry_cutoff", "force_flat",
            "overnight_check", "digest", "nightly_downloads", "pairs_refresh_reminder",
            "engine_cycle",
        } <= ids
        session.stop()

    def test_auth_check_passes_in_paper_mode(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 8, 30)
        assert make_session(journal).job_auth_check() is True

    def test_entry_cutoff_latches(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 14, 45)
        session = make_session(journal)
        session.job_entry_cutoff()
        assert session.state.entry_cutoff_passed is True

    def test_force_flat_closes_intraday_only(self, journal, frozen_clock):
        """§3: 15:10, before the broker's ~15:20 auto square-off."""
        frozen_clock(2026, 7, 22, 15, 10)
        session = make_session(journal, prices={"A": 100.0, "B": 100.0})
        session.broker.seed_position(
            Position("A", 10, 100.0, "filings", Product.MIS, ttl=TTL.INTRADAY,
                     last_price=100.0)
        )
        session.broker.seed_position(
            Position("B", 10, 100.0, "pead", Product.CNC, ttl=TTL.SWING, last_price=100.0)
        )
        assert session.job_force_flat() == 1
        assert [p.symbol for p in session.broker.positions()] == ["B"]

    def test_regime_job_journals_and_alerts(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        session = make_session(journal, alerts=alerts)
        decision = session.job_regime()
        assert decision.regime in (Regime.CHOP, Regime.TREND, Regime.PANIC)
        assert journal.query("SELECT * FROM regime_log")
        assert any("REGIME" in m for m in alerts.sent_messages)

    def test_gift_nifty_degrades_gracefully(self, journal, frozen_clock, monkeypatch):
        """§8.3: if unavailable, degrade gracefully and log."""
        frozen_clock(2026, 7, 22, 8, 45)
        session = make_session(journal)

        import httpx

        def boom(*a, **k):
            raise httpx.ConnectError("no route")

        monkeypatch.setattr(httpx, "get", boom)
        assert session.gift_nifty_gap_pct() is None
        assert journal.query("SELECT * FROM errors WHERE source='gift_nifty'")

    def test_a_failing_job_never_kills_the_scheduler(self, journal):
        from live.session import _guard

        guarded = _guard(lambda: (_ for _ in ()).throw(RuntimeError("nope")), "boom", journal)
        assert guarded() is None
        assert journal.query("SELECT * FROM errors WHERE source='scheduler'")

    def test_split_hhmmss(self):
        assert _split_hhmmss("09:06:30") == (9, 6, 30)
        assert _split_hhmmss("15:10") == (15, 10, 0)


# ===========================================================================
# Telegram commands
# ===========================================================================


class TestCommands:
    def _cmd(self, name: str, *args: str) -> Command:
        return Command(name=name, args=list(args), chat_id="1", message_id=1, raw=name)

    def test_kill_flattens_and_arms_the_kernel(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 11, 0)
        session = make_session(journal, prices={"A": 100.0})
        session.broker.seed_position(
            Position("A", 10, 100.0, "filings", Product.MIS, last_price=100.0)
        )
        reply = session._cmd_kill(self._cmd("/kill"))
        assert "KILL" in reply
        assert session.broker.positions() == []
        assert session.kernel.is_killed is True
        assert journal.query("SELECT * FROM kill_events")

    def test_nothing_passes_the_kernel_after_kill(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 11, 0)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        session._cmd_kill(self._cmd("/kill"))
        assert session.route(entry_signal(), session.build_context()) is None

    def test_status_reports_positions(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 11, 0)
        session = make_session(journal, prices={"A": 100.0})
        session.broker.seed_position(
            Position("A", 10, 100.0, "filings", Product.MIS, last_price=100.0)
        )
        assert "A" in session._cmd_status(self._cmd("/status"))

    def test_confirm_requires_a_request_id(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 11, 0)
        session = make_session(journal)
        assert "Usage" in session._cmd_confirm(self._cmd("/confirm"))

    def test_confirm_unknown_id(self, journal, frozen_clock):
        from engines.wheel import WheelEngine

        frozen_clock(2026, 7, 22, 11, 0)
        session = make_session(journal, {"wheel": WheelEngine(alerts=NullAlerts(),
                                                             journal=journal)})
        assert "Unknown request id" in session._cmd_confirm(self._cmd("/confirm", "nope"))


# ===========================================================================
# Phase 6 acceptance: a complete simulated session
# ===========================================================================


class TestFullSessionAcceptance:
    """§5.6: one complete simulated session end-to-end with journal + digest."""

    def test_end_to_end_session(self, journal, frozen_clock):
        alerts = NullAlerts()
        prices = {"RELIANCE": 3000.0}

        # 08:30 — auth check
        frozen_clock(2026, 7, 22, 8, 30)
        engine = StubEngine(entries=[entry_signal()], journal=journal)
        session = make_session(journal, {"filings": engine}, prices, alerts)
        assert session.job_auth_check() is True

        # 10:00 — regime classification
        frozen_clock(2026, 7, 22, 10, 0)
        decision = session.job_regime()
        session.state.decision.enabled_engines = ["filings"]

        # 10:30 — the engine loop opens a position
        frozen_clock(2026, 7, 22, 10, 30)
        assert session.run_cycle() == 1
        assert len(session.broker.positions()) == 1

        # 14:45 — entry cutoff
        frozen_clock(2026, 7, 22, 14, 45)
        session.job_entry_cutoff()
        assert session.run_cycle() == 0, "no new entries after the cutoff"

        # 15:10 — force flat
        frozen_clock(2026, 7, 22, 15, 10)
        session.update_prices({"RELIANCE": 3050.0})
        assert session.job_force_flat() == 1
        assert session.broker.positions() == []

        # 15:45 — digest
        frozen_clock(2026, 7, 22, 15, 45)
        summary = session.job_digest()

        assert summary["trade_date"] == "2026-07-22"
        assert summary["mode"] == "paper"
        assert summary["regime"] == decision.regime.value
        assert summary["signals"] >= 1

        # The journal recorded the whole day.
        counts = journal.counts_for_date("2026-07-22")
        assert counts["signals"] >= 1
        assert counts["orders"] >= 2         # the entry plus the force-flat exit
        assert counts["fills"] >= 1

        stored = journal.query("SELECT * FROM daily_summary WHERE trade_date='2026-07-22'")
        assert len(stored) == 1
        assert stored[0]["mode"] == "paper"

        assert any("DIGEST" in m for m in alerts.sent_messages)

    def test_no_broker_orders_are_sent_in_paper_mode(self, journal, frozen_clock):
        """§0.1: paper mode simulates fills locally and never sends broker orders."""
        frozen_clock(2026, 7, 22, 10, 30)
        session = make_session(journal, prices={"RELIANCE": 3000.0})
        assert session.broker.mode == "paper"
        session.route(entry_signal(), session.build_context())
        fills = journal.query("SELECT * FROM fills")
        assert all(row["is_paper"] == 1 for row in fills)

    def test_digest_is_produced_even_on_a_day_with_no_trades(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 15, 45)
        session = make_session(journal)
        summary = session.job_digest()
        assert summary["trades"] == 0
        assert summary["net_pnl"] == 0.0
