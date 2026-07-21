"""Journal tests — §0.6: everything is journalled, in IST."""

from __future__ import annotations

import datetime as _dt

from core import clock
from core.journal import Journal
from core.types import Fill, RiskDecision, Side


class TestSchema:
    def test_all_spec_tables_exist(self, journal):
        rows = journal.query("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in rows}
        # §0.6 names signals, orders, fills/trades, rejections; §7 adds the digest.
        assert {"signals", "orders", "fills", "trades", "rejections",
                "daily_summary", "errors", "announcements"} <= names


class TestSignalsAndOrders:
    def test_signal_roundtrip(self, journal, make_signal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        signal = make_signal()
        journal.record_signal(signal)
        rows = journal.query("SELECT * FROM signals")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "RELIANCE"
        assert rows[0]["engine"] == "filings"
        assert rows[0]["trade_date"] == "2026-07-22"

    def test_timestamps_are_ist(self, journal, make_signal, frozen_clock):
        """§0.6: timestamps in Asia/Kolkata."""
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_signal(make_signal())
        ts = journal.query("SELECT ts FROM signals")[0]["ts"]
        assert ts.endswith("+05:30")

    def test_order_roundtrip_and_status_update(self, journal, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        order = make_order()
        journal.record_order(order, mode="paper")
        journal.update_order_status(order.order_id, "FILLED", "BRK123")
        row = journal.query("SELECT * FROM orders")[0]
        assert row["status"] == "FILLED"
        assert row["broker_order_id"] == "BRK123"
        assert row["mode"] == "paper"

    def test_new_entry_count_per_engine(self, journal, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_order(make_order(engine="filings", symbol="A"), mode="paper")
        journal.record_order(make_order(engine="filings", symbol="B"), mode="paper")
        journal.record_order(make_order(engine="pairs", symbol="C"), mode="paper")
        assert journal.new_entry_count_today("2026-07-22", "filings") == 2
        assert journal.new_entry_count_today("2026-07-22", "pairs") == 1

    def test_exits_do_not_count_as_new_entries(self, journal, make_order, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_order(make_order(is_entry=False), mode="paper")
        assert journal.new_entry_count_today("2026-07-22", "filings") == 0


class TestFillsAndTrades:
    def test_fill_roundtrip(self, journal, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 0)
        journal.record_fill(
            Fill(order_id="o1", symbol="INFY", side=Side.BUY, quantity=10,
                 price=1500.0, timestamp=ts, engine="filings", costs=23.5)
        )
        row = journal.query("SELECT * FROM fills")[0]
        assert row["price"] == 1500.0
        assert row["costs"] == 23.5
        assert row["is_paper"] == 1

    def test_realised_pnl_between(self, journal):
        for day, pnl in [("2026-07-20", -5000.0), ("2026-07-21", 2000.0), ("2026-07-22", -1000.0)]:
            journal.record_trade(
                trade_id=f"t{day}", engine="filings", symbol="X", side="BUY", quantity=1,
                entry_ts="x", entry_price=1.0, net_pnl=pnl, mode="paper", trade_date=day,
            )
        assert journal.realised_pnl_between("2026-07-20", "2026-07-22") == -4000.0
        assert journal.realised_pnl_between("2026-07-22", "2026-07-22") == -1000.0

    def test_realised_pnl_with_no_trades_is_zero(self, journal):
        assert journal.realised_pnl_between("2026-01-01", "2026-01-31") == 0.0


class TestRejections:
    def test_rejection_records_the_code(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        decision = RiskDecision.reject("SOME_CODE", "because reasons", extra=1)
        journal.record_rejection(decision, engine="pairs", symbol="TCS", side="BUY", quantity=5)
        row = journal.query("SELECT * FROM rejections")[0]
        assert row["reason_code"] == "SOME_CODE"
        assert row["reason"] == "because reasons"
        assert '"extra":1' in row["meta"]


class TestAnnouncements:
    def test_dedupe_by_id_and_hash(self, journal, frozen_clock):
        """§6.1: dedupe by announcement ID + content hash."""
        frozen_clock(2026, 7, 22, 10, 0)
        payload = dict(announcement_id="A1", content_hash="h1", symbol="INFY",
                       headline="Order win", label="MATERIAL_POSITIVE", confidence=0.9)
        assert journal.record_announcement(**payload) is True
        assert journal.record_announcement(**payload) is False

    def test_same_id_different_content_is_a_new_row(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        assert journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="INFY")
        assert journal.record_announcement(announcement_id="A1", content_hash="h2", symbol="INFY")
        assert len(journal.query("SELECT * FROM announcements")) == 2

    def test_material_filing_symbols(self, journal, frozen_clock):
        """§6.3 asks this before entering a pair leg."""
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="INFY",
                                    label="MATERIAL_POSITIVE", trade_date="2026-07-22")
        journal.record_announcement(announcement_id="A2", content_hash="h2", symbol="TCS",
                                    label="NOISE", trade_date="2026-07-22")
        assert journal.material_filing_symbols("2026-07-22") == {"INFY"}

    def test_has_negative_filing(self, journal, frozen_clock):
        """§6.7 mandatory cross-check."""
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="YESBANK",
                                    label="MATERIAL_NEGATIVE", trade_date="2026-07-22")
        assert journal.has_negative_filing("YESBANK", "2026-07-22")
        assert not journal.has_negative_filing("INFY", "2026-07-22")


class TestSurveillanceAndFlows:
    def test_snapshot_and_lookup(self, journal):
        journal.record_surveillance_snapshot(
            "2026-07-22", "asm", [{"symbol": "ABC", "stage": "ST-I"}, {"symbol": "DEF", "stage": "ST-II"}]
        )
        journal.record_surveillance_snapshot("2026-07-22", "fno_ban", [{"symbol": "GHI"}])
        assert journal.surveillance_symbols("2026-07-22", ["asm", "gsm"]) == {"ABC", "DEF"}
        assert journal.surveillance_symbols("2026-07-22", ["fno_ban"]) == {"GHI"}
        assert journal.latest_surveillance_date("asm") == "2026-07-22"

    def test_flows_roundtrip(self, journal):
        journal.record_flows("2026-07-22", fii_cash_cr=-1200.5, dii_cash_cr=900.0,
                             long_ratio=0.31, ratio_percentile_3y=8.0)
        row = journal.query("SELECT * FROM flows")[0]
        assert row["ratio_percentile_3y"] == 8.0


class TestPairs:
    def test_save_pairs_deactivates_the_old_set(self, journal):
        journal.save_pairs(
            [{"sector": "it", "symbol_a": "TCS", "symbol_b": "INFY",
              "hedge_ratio": 1.2, "pvalue": 0.01, "lookback_days": 252}],
            refreshed_on="2026-06-01",
        )
        journal.save_pairs(
            [{"sector": "it", "symbol_a": "TCS", "symbol_b": "WIPRO",
              "hedge_ratio": 0.8, "pvalue": 0.02, "lookback_days": 252}],
            refreshed_on="2026-07-01",
        )
        active = journal.active_pairs()
        assert len(active) == 1
        assert active[0]["symbol_b"] == "WIPRO"


class TestBacktestRuns:
    def test_test_window_consumption_is_counted(self, journal):
        """§4: the untouched test window is consumed exactly once."""
        assert journal.test_window_runs("filings") == 0
        journal.record_backtest_run("filings", "test", "2025-01-01", None, "PROMOTED", {})
        assert journal.test_window_runs("filings") == 1


class TestDailySummary:
    def test_upsert_merges(self, journal):
        journal.upsert_daily_summary("2026-07-22", mode="paper", net_pnl=1500.0)
        journal.upsert_daily_summary("2026-07-22", trades=3, regime="TREND")
        row = journal.query("SELECT * FROM daily_summary")[0]
        assert row["net_pnl"] == 1500.0
        assert row["trades"] == 3
        assert row["regime"] == "TREND"

    def test_unknown_field_raises(self, journal):
        import pytest

        with pytest.raises(ValueError, match="Unknown daily_summary fields"):
            journal.upsert_daily_summary("2026-07-22", not_a_column=1)


class TestResilience:
    def test_journal_failure_never_raises_upward(self, journal, make_signal, frozen_clock):
        """Journalling must never be the reason a trade fails."""
        frozen_clock(2026, 7, 22, 10, 0)
        journal._conn.execute("DROP TABLE signals")
        journal.record_signal(make_signal())   # logged, not raised

    def test_counts_for_date(self, journal, make_signal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        journal.record_signal(make_signal())
        counts = journal.counts_for_date("2026-07-22")
        assert counts["signals"] == 1
        assert counts["orders"] == 0
