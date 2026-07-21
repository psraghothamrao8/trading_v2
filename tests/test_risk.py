"""Risk kernel tests — §3. Heaviest coverage in the repo (§0.5).

Every veto listed in §3 has an explicit test here, plus a test that exits are
NOT blocked by the entry vetoes (which would trap us in a position).
"""

from __future__ import annotations

import datetime as _dt

import pytest

from core import clock
from core.config import get_settings
from core.risk import (
    BandInfo,
    Reason,
    RiskKernel,
    build_order,
    flatten_all,
    force_flat_intraday,
    size_quantity,
)
from core.types import (
    EntryType,
    Position,
    Product,
    Segment,
    Side,
    TTL,
    Verdict,
)

IST = clock.IST

# 2026-07-22 is a Wednesday, a normal trading day, not a blocked event day.
NORMAL_DAY = (2026, 7, 22)
# 2026-02-06 is an RBI MPC day in events.yaml.
BLOCKED_DAY = (2026, 2, 6)
# 2026-07-29 (Wed) -> next session 2026-07-30 (Thu) is a blocked US-Fed day.
DAY_BEFORE_BLOCKED = (2026, 7, 29)
# 2026-07-28 is a Tuesday == the configured weekly expiry weekday.
EXPIRY_DAY = (2026, 7, 28)


# ---------------------------------------------------------------------------
# Sizing (§3)
# ---------------------------------------------------------------------------


class TestSizing:
    def test_formula_matches_spec(self):
        """qty = floor((capital x 0.005) / |entry - stop|)."""
        # 800000 * 0.005 = 4000 budget; risk/unit = 100 -> 40 units
        assert size_quantity(3000.0, 2900.0, capital=800_000, risk_pct=0.5) == 40

    def test_floor_not_round(self):
        # budget 4000, risk/unit 300 -> 13.33 -> 13, never 14
        assert size_quantity(3000.0, 2700.0, capital=800_000, risk_pct=0.5) == 13

    def test_rounds_down_to_whole_lots(self):
        # 40 units with lot_size 15 -> 2 lots (30), never 3 lots
        assert size_quantity(3000.0, 2900.0, capital=800_000, risk_pct=0.5, lot_size=15) == 30

    def test_zero_when_the_stop_costs_more_than_the_budget(self):
        """Risk per unit above the whole budget must size to 0, not to 1."""
        # budget = 1000 * 0.5% = 5; risk/unit = 2900 -> 0 units
        assert size_quantity(3000.0, 100.0, capital=1_000, risk_pct=0.5) == 0

    def test_zero_risk_per_unit_raises(self):
        with pytest.raises(ValueError, match="risk per unit is zero"):
            size_quantity(3000.0, 3000.0)

    def test_defaults_come_from_settings_not_code(self):
        settings = get_settings()
        expected = size_quantity(
            3000.0, 2900.0,
            capital=settings.require("risk.capital"),
            risk_pct=settings.require("risk.risk_per_trade_pct"),
        )
        assert size_quantity(3000.0, 2900.0) == expected


# ---------------------------------------------------------------------------
# The India-specific vetoes (§3)
# ---------------------------------------------------------------------------


class TestBanListVeto:
    def test_derivatives_order_rejected_when_underlying_banned(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.fno_ban_list.add("RELIANCE")
        order = make_order(
            symbol="RELIANCE26JUL3000CE",
            segment=Segment.EQUITY_OPTIONS,
            underlying="RELIANCE",
        )
        decision = kernel.check(order)
        assert decision.verdict is Verdict.REJECT
        assert decision.reason_code == Reason.FNO_BAN_LIST

    def test_cash_equity_order_not_rejected_by_ban_list(
        self, kernel, market, make_order, frozen_clock
    ):
        """§3 says the ban list rejects DERIVATIVES orders. Cash is unaffected."""
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.fno_ban_list.add("RELIANCE")
        decision = kernel.check(make_order(symbol="RELIANCE"))
        assert decision.allowed


class TestSurveillanceVeto:
    def test_asm_gsm_blocks_every_engine(self, kernel, market, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.surveillance.add("YESBANK")
        for engine in ("filings", "pairs", "pead", "panic_reversion"):
            decision = kernel.check(make_order(symbol="YESBANK", engine=engine))
            assert decision.reason_code == Reason.SURVEILLANCE, engine

    def test_clean_symbol_passes(self, kernel, market, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.surveillance.add("YESBANK")
        assert kernel.check(make_order(symbol="RELIANCE")).allowed


class TestCircuitBandVeto:
    def test_within_one_percent_of_upper_band_rejected(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.bands["ADANIENT"] = BandInfo(last_price=100.0, upper=100.5, lower=80.0)
        decision = kernel.check(make_order(symbol="ADANIENT"))
        assert decision.reason_code == Reason.CIRCUIT_BAND
        assert decision.meta["distance_pct"] == pytest.approx(0.5)

    def test_within_one_percent_of_lower_band_rejected(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.bands["ADANIENT"] = BandInfo(last_price=100.0, upper=120.0, lower=99.5)
        assert kernel.check(make_order(symbol="ADANIENT")).reason_code == Reason.CIRCUIT_BAND

    def test_comfortably_inside_bands_allowed(self, kernel, market, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.bands["ADANIENT"] = BandInfo(last_price=100.0, upper=110.0, lower=90.0)
        assert kernel.check(make_order(symbol="ADANIENT")).allowed

    def test_threshold_is_config_driven(self, kernel, market, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        configured = get_settings().require("risk.vetoes.circuit_band_proximity_pct")
        assert configured == 1.0, "test assumes the shipped default"
        # Exactly at the threshold is a rejection (<=).
        market.bands["X"] = BandInfo(last_price=100.0, upper=101.0, lower=50.0)
        assert kernel.check(make_order(symbol="X")).reason_code == Reason.CIRCUIT_BAND


class TestEventDayVeto:
    def test_no_new_entries_on_blocked_day(self, kernel, make_order, frozen_clock):
        frozen_clock(*BLOCKED_DAY, 10, 0)
        decision = kernel.check(make_order())
        assert decision.reason_code == Reason.BLOCKED_EVENT_DAY
        assert "rbi_mpc" in decision.reason

    def test_no_overnight_hold_into_blocked_day(self, kernel, make_order, frozen_clock):
        frozen_clock(*DAY_BEFORE_BLOCKED, 15, 20)
        decision = kernel.check(make_order(ttl=TTL.OVERNIGHT, segment=Segment.EQUITY_DELIVERY))
        assert decision.reason_code == Reason.OVERNIGHT_INTO_EVENT

    def test_intraday_on_day_before_blocked_is_fine(self, kernel, make_order, frozen_clock):
        frozen_clock(*DAY_BEFORE_BLOCKED, 10, 0)
        assert kernel.check(make_order(ttl=TTL.INTRADAY)).allowed


class TestTimeVetoes:
    def test_no_new_intraday_entries_after_1445(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 14, 46)
        decision = kernel.check(make_order(ttl=TTL.INTRADAY))
        assert decision.reason_code == Reason.ENTRY_CUTOFF

    def test_entry_at_1444_still_allowed(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 14, 44)
        assert kernel.check(make_order(ttl=TTL.INTRADAY)).allowed

    def test_cutoff_time_comes_from_config(self):
        assert get_settings().require("risk.vetoes.no_new_intraday_entries_after") == "14:45"
        assert get_settings().require("risk.vetoes.mis_force_flat_at") == "15:10"


class TestPhysicalSettlementGuard:
    """§3: short stock options ITM/near-money near expiry are a delivery risk."""

    def _option(self, make_order, **kw):
        defaults = dict(
            symbol="RELIANCE26JUL3000PE",
            side=Side.SELL,
            segment=Segment.EQUITY_OPTIONS,
            ttl=TTL.SWING,
            product=Product.NRML,
            underlying="RELIANCE",
            underlying_type="stock",
            option_type="PE",
            strike=3000,
            expiry="2026-07-28",
        )
        defaults.update(kw)
        return make_order(**defaults)

    def test_short_itm_stock_option_within_two_sessions_rejected(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(2026, 7, 24, 10, 0)   # Fri; expiry Tue 28th -> 2 sessions
        market.spot["RELIANCE"] = 2950.0   # PE strike 3000 is ITM
        decision = kernel.check(self._option(make_order))
        assert decision.reason_code == Reason.PHYSICAL_SETTLEMENT

    def test_within_two_percent_of_spot_also_rejected(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(2026, 7, 24, 10, 0)
        market.spot["RELIANCE"] = 3050.0   # OTM but within 2% of the 3000 strike
        assert kernel.check(self._option(make_order)).reason_code == Reason.PHYSICAL_SETTLEMENT

    def test_far_otm_allowed(self, kernel, market, make_order, frozen_clock):
        frozen_clock(2026, 7, 24, 10, 0)
        market.spot["RELIANCE"] = 3400.0   # PE 3000 far OTM
        assert kernel.check(self._option(make_order)).allowed

    def test_allow_delivery_opt_out_respected(self, kernel, market, make_order, frozen_clock):
        """The owner can accept delivery per trade -- §3 says `unless allow_delivery`."""
        frozen_clock(2026, 7, 24, 10, 0)
        market.spot["RELIANCE"] = 2950.0
        assert kernel.check(self._option(make_order, allow_delivery=True)).allowed

    def test_missing_spot_fails_safe(self, kernel, make_order, frozen_clock):
        """No spot price -> assume near-money. A surprise delivery can exceed capital."""
        frozen_clock(2026, 7, 24, 10, 0)
        assert kernel.check(self._option(make_order)).reason_code == Reason.PHYSICAL_SETTLEMENT

    def test_index_options_are_cash_settled_and_exempt(
        self, kernel, market, make_order, frozen_clock
    ):
        frozen_clock(2026, 7, 24, 10, 0)
        market.spot["NIFTY"] = 24000.0
        order = self._option(
            make_order,
            symbol="NIFTY26JUL24000PE",
            underlying="NIFTY",
            underlying_type="index",
            strike=24000,
        )
        assert kernel.check(order).allowed

    def test_far_from_expiry_allowed(self, kernel, market, make_order, frozen_clock):
        frozen_clock(2026, 7, 15, 10, 0)   # ~9 sessions to expiry
        market.spot["RELIANCE"] = 2950.0
        assert kernel.check(self._option(make_order)).allowed


class TestSTTTrapGuard:
    def _long_option(self, make_order, ttl=TTL.SWING):
        return make_order(
            symbol="NIFTY26JUL24000CE",
            side=Side.BUY,
            segment=Segment.EQUITY_OPTIONS,
            ttl=ttl,
            option_type="CE",
            strike=24000,
            underlying="NIFTY",
            underlying_type="index",
            expiry="2026-07-28",
        )

    def test_no_long_option_entry_after_1500_on_expiry_day(
        self, kernel, make_order, frozen_clock
    ):
        frozen_clock(*EXPIRY_DAY, 15, 5)
        assert kernel.check(self._long_option(make_order)).reason_code == Reason.STT_TRAP

    def test_before_1500_on_expiry_day_is_allowed(self, kernel, make_order, frozen_clock):
        frozen_clock(*EXPIRY_DAY, 14, 30)
        assert kernel.check(self._long_option(make_order)).allowed

    def test_intraday_cutoff_fires_first_for_intraday_options(
        self, kernel, make_order, frozen_clock
    ):
        """The 14:45 cutoff is earlier than 15:00, so it wins for MIS orders.

        Both are rejections; this pins the ordering so the reason reported to
        Telegram is the one that actually applies.
        """
        frozen_clock(*EXPIRY_DAY, 15, 5)
        order = self._long_option(make_order, ttl=TTL.INTRADAY)
        assert kernel.check(order).reason_code == Reason.ENTRY_CUTOFF

    def test_long_itm_options_are_flagged_for_exit_by_1500(
        self, kernel, paper_broker, frozen_clock
    ):
        """§3: 'never carry long ITM options into expiry close'.

        Blocking new entries is not enough -- an option bought on Monday must
        be forced out on expiry day. The kernel exposes the due list; the
        orchestrator acts on it.
        """
        frozen_clock(*EXPIRY_DAY, 15, 1)
        kernel.market.spot["NIFTY"] = 24_500.0     # CE 24000 is ITM
        paper_broker.seed_position(
            Position(
                symbol="NIFTY26JUL24000CE", quantity=75, average_price=100.0,
                engine="filings", product=Product.NRML, segment=Segment.EQUITY_OPTIONS,
                last_price=520.0,
                meta={"option_type": "CE", "strike": 24000, "underlying": "NIFTY",
                      "expiry": "2026-07-28"},
            )
        )
        due = kernel.stt_trap_exits_due()
        assert [p.symbol for p in due] == ["NIFTY26JUL24000CE"]

    def test_otm_long_options_are_not_flagged(self, kernel, paper_broker, frozen_clock):
        frozen_clock(*EXPIRY_DAY, 15, 1)
        kernel.market.spot["NIFTY"] = 23_000.0     # CE 24000 is worthless
        paper_broker.seed_position(
            Position(
                symbol="NIFTY26JUL24000CE", quantity=75, average_price=100.0,
                engine="filings", product=Product.NRML, segment=Segment.EQUITY_OPTIONS,
                last_price=0.5,
                meta={"option_type": "CE", "strike": 24000, "underlying": "NIFTY",
                      "expiry": "2026-07-28"},
            )
        )
        assert kernel.stt_trap_exits_due() == []


class TestEquityShortOvernight:
    def test_rejected(self, kernel, make_order, frozen_clock):
        """§8.6: you cannot short equity overnight in the cash segment."""
        frozen_clock(*NORMAL_DAY, 10, 0)
        order = make_order(side=Side.SELL, ttl=TTL.OVERNIGHT, segment=Segment.EQUITY_DELIVERY)
        assert kernel.check(order).reason_code == Reason.EQUITY_SHORT_OVERNIGHT

    def test_intraday_short_is_fine(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        order = make_order(side=Side.SELL, ttl=TTL.INTRADAY, stop=3100.0)
        assert kernel.check(order).allowed


# ---------------------------------------------------------------------------
# Loss limits and the kill switch (§3)
# ---------------------------------------------------------------------------


class TestLossLimits:
    def test_daily_limit_blocks_new_orders(self, kernel, journal, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 11, 0)
        # 1.5% of 800000 = 12000
        journal.record_trade(
            trade_id="t1", engine="filings", symbol="X", side="BUY", quantity=1,
            entry_ts="x", entry_price=1.0, net_pnl=-12_500.0, mode="paper",
            trade_date=_dt.date(*NORMAL_DAY).isoformat(),
        )
        decision = kernel.check(make_order())
        assert decision.reason_code == Reason.DAILY_LOSS_LIMIT

    def test_just_inside_daily_limit_allowed(self, kernel, journal, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 11, 0)
        journal.record_trade(
            trade_id="t1", engine="filings", symbol="X", side="BUY", quantity=1,
            entry_ts="x", entry_price=1.0, net_pnl=-11_000.0, mode="paper",
            trade_date=_dt.date(*NORMAL_DAY).isoformat(),
        )
        assert kernel.check(make_order()).allowed

    def test_weekly_limit_blocks_new_orders(self, kernel, journal, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 11, 0)     # Wednesday
        monday = _dt.date(2026, 7, 20).isoformat()
        # 3% of 800000 = 24000; split across the week so no single day breaches.
        for i, day in enumerate([monday, "2026-07-21"]):
            journal.record_trade(
                trade_id=f"t{i}", engine="filings", symbol="X", side="BUY", quantity=1,
                entry_ts="x", entry_price=1.0, net_pnl=-11_000.0 - i, mode="paper",
                trade_date=day,
            )
        journal.record_trade(
            trade_id="t9", engine="filings", symbol="X", side="BUY", quantity=1,
            entry_ts="x", entry_price=1.0, net_pnl=-3_000.0, mode="paper",
            trade_date=_dt.date(*NORMAL_DAY).isoformat(),
        )
        decision = kernel.check(make_order())
        assert decision.reason_code == Reason.WEEKLY_LOSS_LIMIT

    def test_exits_are_never_blocked_by_loss_limits(
        self, kernel, journal, make_order, frozen_clock
    ):
        """The limits exist to get us OUT; blocking an exit would be perverse."""
        frozen_clock(*NORMAL_DAY, 11, 0)
        journal.record_trade(
            trade_id="t1", engine="filings", symbol="X", side="BUY", quantity=1,
            entry_ts="x", entry_price=1.0, net_pnl=-50_000.0, mode="paper",
            trade_date=_dt.date(*NORMAL_DAY).isoformat(),
        )
        assert kernel.check(make_order(is_entry=False)).allowed


class TestKillSwitch:
    def test_armed_kill_blocks_everything_including_exits(
        self, kernel, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        kernel.arm_kill("manual")
        assert kernel.check(make_order()).reason_code == Reason.KILL_SWITCH
        assert kernel.check(make_order(is_entry=False)).reason_code == Reason.KILL_SWITCH

    def test_clear_kill_restores_trading(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        kernel.arm_kill("manual")
        kernel.clear_kill()
        assert kernel.check(make_order()).allowed


# ---------------------------------------------------------------------------
# Exposure limits (§3)
# ---------------------------------------------------------------------------


class TestExposureLimits:
    def test_max_new_trades_per_day_per_engine(
        self, kernel, journal, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        limit = get_settings().require("risk.max_new_trades_per_day_per_engine")
        for i in range(limit):
            order = make_order(symbol=f"SYM{i}")
            journal.record_order(order, mode="paper")
        assert kernel.check(make_order()).reason_code == Reason.MAX_TRADES_PER_ENGINE

    def test_limit_is_per_engine_not_global(self, kernel, journal, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        limit = get_settings().require("risk.max_new_trades_per_day_per_engine")
        for i in range(limit):
            journal.record_order(make_order(symbol=f"SYM{i}", engine="filings"), mode="paper")
        assert kernel.check(make_order(engine="pairs")).allowed

    def test_max_concurrent_positions_total(
        self, kernel, paper_broker, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        limit = get_settings().require("risk.max_concurrent_positions_total")
        for i in range(limit):
            paper_broker.seed_position(
                Position(symbol=f"SYM{i}", quantity=1, average_price=100.0,
                         engine="filings", product=Product.MIS, last_price=100.0)
            )
        assert kernel.check(make_order(symbol="NEW")).reason_code == Reason.MAX_CONCURRENT_POSITIONS

    def test_adding_to_an_existing_symbol_is_not_a_new_position(
        self, kernel, paper_broker, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        limit = get_settings().require("risk.max_concurrent_positions_total")
        for i in range(limit):
            paper_broker.seed_position(
                Position(symbol=f"SYM{i}", quantity=1, average_price=100.0,
                         engine="filings", product=Product.MIS, last_price=100.0)
            )
        assert kernel.check(make_order(symbol="SYM0")).allowed

    def test_per_engine_capital_cap(self, kernel, paper_broker, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        # sympathy cap is 10% of 800000 = 80000
        paper_broker.seed_position(
            Position(symbol="OLD", quantity=100, average_price=750.0,
                     engine="sympathy", product=Product.MIS, last_price=750.0)
        )
        order = make_order(engine="sympathy", symbol="NEW", quantity=10, reference_price=3000.0)
        decision = kernel.check(order)
        assert decision.reason_code == Reason.ENGINE_CAPITAL_CAP
        assert decision.meta["cap_value"] == pytest.approx(80_000.0)


class TestStructuralChecks:
    def test_alert_only_engines_cannot_place_orders(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        for engine in get_settings().get("risk.alert_only_engines"):
            assert kernel.check(make_order(engine=engine)).reason_code == Reason.ALERT_ONLY_ENGINE

    def test_zero_quantity_rejected(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        assert kernel.check(make_order(quantity=0)).reason_code == Reason.ZERO_QUANTITY

    def test_entry_without_stop_rejected(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        assert kernel.check(make_order(stop=None)).reason_code == Reason.NO_STOP

    def test_exit_without_stop_allowed(self, kernel, make_order, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        assert kernel.check(make_order(stop=None, is_entry=False)).allowed


# ---------------------------------------------------------------------------
# Journalling of rejections (§3: "Every rejection journaled with reason")
# ---------------------------------------------------------------------------


class TestRejectionJournalling:
    def test_every_rejection_lands_in_the_journal(
        self, kernel, journal, market, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        market.surveillance.add("YESBANK")
        kernel.check(make_order(symbol="YESBANK"))

        rows = journal.query("SELECT * FROM rejections")
        assert len(rows) == 1
        assert rows[0]["reason_code"] == Reason.SURVEILLANCE
        assert rows[0]["symbol"] == "YESBANK"
        assert rows[0]["engine"] == "filings"
        assert rows[0]["reason"]

    def test_allowed_orders_are_not_journalled_as_rejections(
        self, kernel, journal, make_order, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 10, 0)
        kernel.check(make_order())
        assert journal.query("SELECT * FROM rejections") == []


# ---------------------------------------------------------------------------
# build_order (§6.0 -> §3)
# ---------------------------------------------------------------------------


class TestBuildOrder:
    def test_sizes_from_signal_stop(self, make_signal, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        order = build_order(make_signal(reference_price=3000.0, stop=2900.0))
        assert order.quantity == 40

    def test_expiry_day_halves_size(self, make_signal, frozen_clock):
        """§7: expiry days halve all new sizes -- applied here, not in engines."""
        frozen_clock(*NORMAL_DAY, 10, 0)
        order = build_order(make_signal(reference_price=3000.0, stop=2900.0), size_multiplier=0.5)
        assert order.quantity == 20

    def test_product_from_ttl(self, make_signal, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        assert build_order(make_signal(ttl=TTL.INTRADAY)).product is Product.MIS
        assert build_order(make_signal(ttl=TTL.OVERNIGHT)).product is Product.CNC

    def test_overnight_derivative_gets_nrml(self, make_signal, frozen_clock):
        frozen_clock(*NORMAL_DAY, 10, 0)
        signal = make_signal(ttl=TTL.OVERNIGHT, instrument_type="FUT")
        assert build_order(signal).product is Product.NRML

    def test_no_reference_price_raises(self, make_signal):
        with pytest.raises(ValueError, match="No reference price"):
            build_order(make_signal(reference_price=None))


# ---------------------------------------------------------------------------
# flatten_all / force_flat (§3 kill(), 15:10 rule)
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flatten_all_closes_every_position(self, paper_broker, journal, frozen_clock):
        frozen_clock(*NORMAL_DAY, 15, 10)
        paper_broker.seed_position(
            Position(symbol="A", quantity=10, average_price=100.0, engine="filings",
                     product=Product.MIS, last_price=105.0)
        )
        paper_broker.seed_position(
            Position(symbol="B", quantity=-5, average_price=200.0, engine="pairs",
                     product=Product.MIS, last_price=195.0)
        )
        assert flatten_all(paper_broker, journal) == 2
        assert paper_broker.positions() == []

    def test_force_flat_intraday_leaves_swing_positions_alone(
        self, paper_broker, journal, frozen_clock
    ):
        frozen_clock(*NORMAL_DAY, 15, 10)
        paper_broker.seed_position(
            Position(symbol="MIS1", quantity=10, average_price=100.0, engine="filings",
                     product=Product.MIS, ttl=TTL.INTRADAY, last_price=101.0)
        )
        paper_broker.seed_position(
            Position(symbol="SWING1", quantity=10, average_price=100.0, engine="pead",
                     product=Product.CNC, ttl=TTL.SWING, last_price=101.0)
        )
        assert force_flat_intraday(paper_broker, journal) == 1
        assert [p.symbol for p in paper_broker.positions()] == ["SWING1"]

    def test_flatten_orders_are_journalled_as_exits(self, paper_broker, journal, frozen_clock):
        frozen_clock(*NORMAL_DAY, 15, 10)
        paper_broker.seed_position(
            Position(symbol="A", quantity=10, average_price=100.0, engine="filings",
                     product=Product.MIS, last_price=105.0)
        )
        flatten_all(paper_broker, journal)
        rows = journal.query("SELECT * FROM orders")
        assert len(rows) == 1
        assert rows[0]["is_entry"] == 0
        assert rows[0]["side"] == "SELL"
