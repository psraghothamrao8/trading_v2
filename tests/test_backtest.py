"""Backtester and metrics tests — §4.

Covers the promotion gates (including that a FAILED verdict is produced and
never softened), the walk-forward windows, the test-window single-use guard,
and the lookahead-bias defences.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.costs import CostModel
from core.types import Segment, Side, TTL
from backtest.metrics import (
    BacktestTrade,
    GateResult,
    compute_metrics,
    engine_capital_for,
    equity_curve,
    evaluate_gates,
    max_drawdown,
    monthly_returns,
    monthly_returns_table,
    write_equity_curve_csv,
)
from backtest.runner import (
    BacktestBook,
    BacktestError,
    Backtester,
    BuyAndHold,
    Window,
    load_windows,
)

IST = clock.IST


def trade(net: float, *, engine="filings", day=1, month=1, year=2024,
          entry=100.0, stop=95.0, quantity=10, costs=50.0) -> BacktestTrade:
    """A trade whose NET pnl is exactly `net` (gross is derived)."""
    ts = _dt.datetime(year, month, day, 10, 0, tzinfo=IST)
    return BacktestTrade(
        engine=engine, symbol="X", side="BUY", quantity=quantity,
        entry_ts=ts, entry_price=entry,
        exit_ts=ts + _dt.timedelta(hours=2), exit_price=entry + net / quantity,
        gross_pnl=net + costs, costs=costs, stop=stop,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_no_trades_produces_empty_metrics(self):
        metrics = compute_metrics([], "filings", "validate", 200_000)
        assert metrics.trades == 0
        assert metrics.profit_factor == 0.0

    def test_basic_counts(self):
        trades = [trade(100), trade(-50), trade(200)]
        metrics = compute_metrics(trades, "filings", "validate", 200_000)
        assert metrics.trades == 3
        assert metrics.wins == 2
        assert metrics.losses == 1
        assert metrics.win_rate == pytest.approx(66.67, abs=0.01)

    def test_expectancy_is_net_per_trade(self):
        metrics = compute_metrics([trade(100), trade(-50)], "filings", "validate", 200_000)
        assert metrics.expectancy == pytest.approx(25.0)

    def test_profit_factor(self):
        metrics = compute_metrics(
            [trade(300), trade(-100), trade(-50)], "filings", "validate", 200_000
        )
        assert metrics.profit_factor == pytest.approx(2.0)

    def test_profit_factor_is_inf_with_no_losers(self):
        """Honest, and the gate handles it explicitly rather than hiding it."""
        import math

        metrics = compute_metrics([trade(100), trade(50)], "filings", "validate", 200_000)
        assert math.isinf(metrics.profit_factor)

    def test_costs_are_carried_through(self):
        metrics = compute_metrics([trade(100, costs=75)], "filings", "validate", 200_000)
        assert metrics.total_costs == 75.0
        assert metrics.gross_pnl == 175.0
        assert metrics.net_pnl == 100.0

    def test_r_multiple(self):
        # entry 100, stop 95 -> 5 risk/unit x 10 units = 50 risk; net 100 -> 2R
        metrics = compute_metrics([trade(100)], "filings", "validate", 200_000)
        assert metrics.avg_r == pytest.approx(2.0)


class TestDrawdown:
    def test_equity_curve_accumulates(self):
        curve = equity_curve([trade(100, day=1), trade(-30, day=2)], 1000.0)
        assert [e for _, e in curve] == [1100.0, 1070.0]

    def test_max_drawdown_measures_peak_to_trough(self):
        curve = [
            (_dt.datetime(2024, 1, d, tzinfo=IST), v)
            for d, v in enumerate([1000, 1200, 900, 1100], start=1)
        ]
        absolute, percent = max_drawdown(curve)
        assert absolute == 300.0
        assert percent == pytest.approx(25.0)

    def test_no_drawdown_on_a_monotonic_curve(self):
        curve = [(_dt.datetime(2024, 1, d, tzinfo=IST), 1000 + d) for d in range(1, 5)]
        assert max_drawdown(curve) == (0.0, 0.0)

    def test_empty_curve(self):
        assert max_drawdown([]) == (0.0, 0.0)


class TestMonthlyReturns:
    def test_grouped_by_exit_month(self):
        returns = monthly_returns([
            trade(100, month=1), trade(50, month=1), trade(-30, month=3)
        ])
        assert returns == {"2024-01": 150.0, "2024-03": -30.0}

    def test_table_renders(self):
        metrics = compute_metrics([trade(100, month=1)], "filings", "validate", 200_000)
        table = monthly_returns_table(metrics)
        assert "2024" in table
        assert "Jan" in table

    def test_table_with_no_trades(self):
        metrics = compute_metrics([], "filings", "validate", 200_000)
        assert monthly_returns_table(metrics) == "(no trades)"

    def test_equity_csv_written(self, tmp_path):
        metrics = compute_metrics([trade(100), trade(-20)], "filings", "validate", 200_000)
        path = write_equity_curve_csv(metrics, tmp_path / "eq.csv")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "timestamp,equity"
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Promotion gates (§4)
# ---------------------------------------------------------------------------


class TestPromotionGates:
    # `pairs` is NOT in backtest.promotion_gates.event_engines, so it carries
    # the 150-trade bar. `pead` and `filings` are, so they carry 40.
    def _passing_metrics(self, engine="pairs", n=200):
        # 70% winners of +300, 30% losers of -100 -> PF = 7.0
        trades = [trade(300, engine=engine, day=(i % 28) + 1, month=(i % 12) + 1)
                  for i in range(int(n * 0.7))]
        trades += [trade(-100, engine=engine, day=(i % 28) + 1, month=(i % 12) + 1)
                   for i in range(int(n * 0.3))]
        return compute_metrics(trades, engine, "validate", 200_000)

    def test_a_good_engine_is_promoted(self):
        verdict = evaluate_gates(self._passing_metrics(), 200_000)
        assert verdict.promoted
        assert verdict.verdict == "PROMOTED"

    def test_too_few_trades_fails(self):
        """§4: >= 150 trades for a normal engine."""
        verdict = evaluate_gates(self._passing_metrics(engine="pairs", n=100), 200_000)
        assert not verdict.promoted
        assert "trades" in [g.name for g in verdict.failures]

    def test_event_engines_get_the_lower_trade_bar(self):
        """§4: >= 40 for event/quarterly engines. 60 trades passes for pead..."""
        verdict = evaluate_gates(self._passing_metrics(engine="pead", n=60), 200_000)
        assert "trades" not in [g.name for g in verdict.failures]

    def test_the_same_trade_count_fails_a_non_event_engine(self):
        """...and the identical record fails for pairs. The distinction is real."""
        verdict = evaluate_gates(self._passing_metrics(engine="pairs", n=60), 200_000)
        assert "trades" in [g.name for g in verdict.failures]

    def test_low_profit_factor_fails(self):
        """§4: profit factor >= 1.3. 10,000 / 9,000 = 1.11 must FAIL."""
        trades = [trade(100, engine="pairs") for _ in range(100)]
        trades += [trade(-100, engine="pairs") for _ in range(90)]
        metrics = compute_metrics(trades, "pairs", "validate", 200_000)
        assert metrics.trades >= 150, "the trade-count gate must not be what fails here"
        verdict = evaluate_gates(metrics, 200_000)
        assert not verdict.promoted
        assert "profit_factor" in [g.name for g in verdict.failures]

    def test_drawdown_over_twelve_percent_fails(self):
        """§4: max DD <= 12% of engine capital (12% of 200,000 = 24,000)."""
        trades = [trade(50_000, engine="pairs", day=1)]
        trades += [trade(-40_000, engine="pairs", day=2)]
        trades += [trade(1000, engine="pairs", day=(i % 28) + 1) for i in range(200)]
        metrics = compute_metrics(trades, "pairs", "validate", 200_000)
        verdict = evaluate_gates(metrics, 200_000)
        dd_gate = next(g for g in verdict.gates if g.name == "max_drawdown")
        assert not dd_gate.passed
        assert metrics.max_drawdown > 24_000

    def test_negative_expectancy_fails(self):
        trades = [trade(100, engine="pairs") for _ in range(100)]
        trades += [trade(-200, engine="pairs") for _ in range(100)]
        metrics = compute_metrics(trades, "pairs", "validate", 200_000)
        verdict = evaluate_gates(metrics, 200_000)
        assert "net_expectancy" in [g.name for g in verdict.failures]

    def test_verdict_render_shows_every_number(self):
        verdict = evaluate_gates(self._passing_metrics(engine="pairs", n=50), 200_000)
        text = verdict.render()
        assert "FAILED" in text
        assert "profit_factor" in text
        assert "trades" in text
        assert "actual=" in text and "required=" in text

    def test_failed_verdict_says_it_is_not_a_bug(self):
        """§4: 'a FAILED verdict is valuable output, not a bug'."""
        verdict = evaluate_gates(self._passing_metrics(engine="pairs", n=10), 200_000)
        assert "not a bug" in verdict.render()
        assert "alert-only" in verdict.render()

    def test_thresholds_come_from_config(self):
        from core.config import get_settings

        gates = get_settings().section("backtest.promotion_gates")
        verdict = evaluate_gates(self._passing_metrics(n=50), 200_000)
        pf_gate = next(g for g in verdict.gates if g.name == "profit_factor")
        assert str(gates.require("profit_factor_min")) in str(pf_gate.required)

    def test_engine_capital_from_per_engine_cap(self):
        # filings cap is 25% of 800,000
        assert engine_capital_for("filings") == pytest.approx(200_000.0)
        assert engine_capital_for("sympathy") == pytest.approx(80_000.0)

    def test_explicit_zero_cap_is_honoured(self):
        """Alert-only engines really do get no capital."""
        assert engine_capital_for("surveillance") == pytest.approx(0.0)

    def test_unconfigured_engine_falls_back_to_full_capital(self):
        """A 0 base would make the drawdown gate unfailable, so it must not be 0."""
        assert engine_capital_for("not_a_configured_engine") == pytest.approx(800_000.0)

    def test_the_drawdown_gate_can_actually_fail_for_an_unconfigured_engine(self):
        trades = [trade(200_000, engine="adhoc", day=1), trade(-150_000, engine="adhoc", day=2)]
        metrics = compute_metrics(trades, "adhoc", "validate", 800_000)
        verdict = evaluate_gates(metrics, engine_capital_for("adhoc"))
        dd_gate = next(g for g in verdict.gates if g.name == "max_drawdown")
        assert not dd_gate.passed


# ---------------------------------------------------------------------------
# Walk-forward windows (§4)
# ---------------------------------------------------------------------------


class TestWindows:
    def test_spec_windows_are_configured(self):
        windows = load_windows()
        assert windows["tune"].start == _dt.date(2019, 1, 1)
        assert windows["tune"].end == _dt.date(2022, 12, 31)
        assert windows["validate"].start == _dt.date(2023, 1, 1)
        assert windows["validate"].end == _dt.date(2024, 12, 31)
        assert windows["test"].start == _dt.date(2025, 1, 1)
        assert windows["test"].end is None      # present

    def test_windows_do_not_overlap(self):
        windows = load_windows()
        assert windows["tune"].end < windows["validate"].start
        assert windows["validate"].end < windows["test"].start

    def test_contains(self):
        window = Window("validate", _dt.date(2023, 1, 1), _dt.date(2024, 12, 31))
        assert window.contains(_dt.date(2023, 6, 1))
        assert not window.contains(_dt.date(2022, 12, 31))
        assert not window.contains(_dt.date(2025, 1, 1))

    def test_open_ended_window_contains_the_future(self):
        window = Window("test", _dt.date(2025, 1, 1), None)
        assert window.contains(_dt.date(2030, 1, 1))


class TestTestWindowGuard:
    def test_first_run_is_silent(self, journal, capsys):
        backtester = Backtester(journal=journal)
        backtester._guard_test_window("filings", Window("test", _dt.date(2025, 1, 1), None))
        assert "WARNING" not in capsys.readouterr().out

    def test_second_run_warns_loudly(self, journal, capsys):
        """§4: the test window is consumed exactly once."""
        journal.record_backtest_run("filings", "test", "2025-01-01", None, "FAILED", {})
        backtester = Backtester(journal=journal)
        backtester._guard_test_window("filings", Window("test", _dt.date(2025, 1, 1), None))
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "EXACTLY ONCE" in out

    def test_other_windows_are_unrestricted(self, journal, capsys):
        journal.record_backtest_run("filings", "tune", "2019-01-01", "2022-12-31", "FAILED", {})
        backtester = Backtester(journal=journal)
        backtester._guard_test_window("filings", Window("tune", _dt.date(2019, 1, 1), None))
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# The simulated book
# ---------------------------------------------------------------------------


class TestBacktestBook:
    def _order(self, make_order, **kw):
        return make_order(**kw)

    def test_open_then_close_records_a_trade(self, make_order, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel(), starting_equity=200_000)
        book.open_position(make_order(quantity=10), 1000.0, ts)
        assert len(book.positions()) == 1
        result = book.close_position("filings", "RELIANCE", 1100.0, ts, "target")
        assert result is not None
        assert book.positions() == []
        assert len(book.trades) == 1

    def test_costs_are_applied_to_both_legs(self, make_order, frozen_clock):
        """§4: every backtest fill runs through the cost model."""
        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel(), starting_equity=200_000)
        book.open_position(make_order(quantity=10), 1000.0, ts)
        book.close_position("filings", "RELIANCE", 1100.0, ts, "target")
        assert book.trades[0].costs > 0
        assert book.trades[0].net_pnl < book.trades[0].gross_pnl

    def test_slippage_moves_the_fill_against_us(self, make_order, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel(), starting_equity=200_000)
        fill = book.open_position(make_order(side=Side.BUY, quantity=10), 1000.0, ts)
        assert fill.price > 1000.0

    def test_partial_close(self, make_order, frozen_clock):
        """§6.1 books 50% at +1R and trails the rest."""
        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel(), starting_equity=200_000)
        book.open_position(make_order(quantity=10), 1000.0, ts)
        book.close_position("filings", "RELIANCE", 1050.0, ts, "+1R", quantity=5)
        assert abs(book.positions()[0].quantity) == 5
        assert len(book.trades) == 1

    def test_closing_a_missing_position_is_a_no_op(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel())
        assert book.close_position("filings", "NOPE", 100.0, ts, "x") is None

    def test_book_satisfies_the_kernel_position_source(self, make_order, frozen_clock):
        """The kernel's concurrency veto must work unchanged in a backtest."""
        from core.risk import RiskKernel

        ts = frozen_clock(2026, 7, 22, 10, 0)
        book = BacktestBook(CostModel(), starting_equity=200_000)
        book.open_position(make_order(quantity=10), 1000.0, ts)
        kernel = RiskKernel(positions=book)
        assert len(kernel.current_positions()) == 1


# ---------------------------------------------------------------------------
# End-to-end: the §5.3 buy-and-hold sanity backtest
# ---------------------------------------------------------------------------


def write_daily_bars(feed, symbol: str, start: _dt.date, days: int, drift: float = 0.5):
    """Store a synthetic daily series so the runner has something to walk."""
    index = pd.date_range(start, periods=days, freq="D", tz=IST)
    base = [100.0 + i * drift for i in range(days)]
    frame = pd.DataFrame(
        {
            "open": base,
            "high": [b + 1 for b in base],
            "low": [b - 1 for b in base],
            "close": [b + 0.5 for b in base],
            "volume": [100_000] * days,
        },
        index=index,
    )
    frame.index.name = "date"
    feed.save(symbol, "day", frame)
    return frame


class TestSanityBacktest:
    @pytest.fixture
    def feed(self, journal, tmp_path, monkeypatch):
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        return DataFeed(kite=None, journal=journal)

    def test_buy_and_hold_produces_believable_net_numbers(
        self, feed, journal, frozen_clock
    ):
        """§5.3 acceptance: the sanity backtest must produce believable net numbers."""
        frozen_clock(2024, 12, 31, 16, 0)
        write_daily_bars(feed, "NIFTYBEES", _dt.date(2023, 1, 2), 500, drift=0.2)

        backtester = Backtester(feed=feed, journal=journal)
        window = Window("validate", _dt.date(2023, 1, 2), _dt.date(2024, 5, 1))
        result = backtester.run(BuyAndHold("NIFTYBEES"), window)

        assert result.metrics.trades == 1
        assert result.metrics.net_pnl > 0, "a rising series must produce a profit"
        assert result.metrics.total_costs > 0, "§4: costs are never zero"
        assert result.metrics.net_pnl < result.metrics.gross_pnl

    def test_a_falling_market_produces_a_loss(self, feed, journal, frozen_clock):
        frozen_clock(2024, 12, 31, 16, 0)
        write_daily_bars(feed, "NIFTYBEES", _dt.date(2023, 1, 2), 400, drift=-0.1)

        backtester = Backtester(feed=feed, journal=journal)
        window = Window("validate", _dt.date(2023, 1, 2), _dt.date(2024, 1, 1))
        result = backtester.run(BuyAndHold("NIFTYBEES"), window)
        assert result.metrics.net_pnl < 0

    def test_run_is_journalled_with_a_verdict(self, feed, journal, frozen_clock):
        frozen_clock(2024, 12, 31, 16, 0)
        write_daily_bars(feed, "NIFTYBEES", _dt.date(2023, 1, 2), 400)
        backtester = Backtester(feed=feed, journal=journal)
        window = Window("validate", _dt.date(2023, 1, 2), _dt.date(2024, 1, 1))
        backtester.run(BuyAndHold("NIFTYBEES"), window)

        rows = journal.query("SELECT * FROM backtest_runs")
        assert len(rows) == 1
        assert rows[0]["verdict"] in ("PROMOTED", "FAILED")

    def test_one_trade_cannot_pass_the_trade_count_gate(self, feed, journal, frozen_clock):
        """Buy-and-hold is a plumbing check, not a promotable strategy."""
        frozen_clock(2024, 12, 31, 16, 0)
        write_daily_bars(feed, "NIFTYBEES", _dt.date(2023, 1, 2), 400)
        backtester = Backtester(feed=feed, journal=journal)
        window = Window("validate", _dt.date(2023, 1, 2), _dt.date(2024, 1, 1))
        result = backtester.run(BuyAndHold("NIFTYBEES"), window)
        assert not result.verdict.promoted

    def test_missing_data_raises_rather_than_reporting_zero_trades(
        self, feed, journal, frozen_clock
    ):
        """Zero trades from missing data reads as 'no edge'. Say what is wrong."""
        frozen_clock(2024, 12, 31, 16, 0)
        backtester = Backtester(feed=feed, journal=journal)
        window = Window("validate", _dt.date(2023, 1, 2), _dt.date(2024, 1, 1))
        with pytest.raises(BacktestError, match="download_history"):
            backtester.run(BuyAndHold("NOTDOWNLOADED"), window)


class TestLookaheadDefence:
    def test_context_frames_are_truncated_to_now(self, journal, tmp_path, monkeypatch):
        """An engine physically cannot see a future bar."""
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        feed = DataFeed(kite=None, journal=journal)
        frame = write_daily_bars(feed, "X", _dt.date(2024, 1, 1), 10)

        backtester = Backtester(feed=feed, journal=journal)
        bars = {("X", "day"): frame}
        indices = {("X", "day"): frame.index}
        sliced = backtester._slice(bars, indices, frame.index[4])
        assert len(sliced[("X", "day")]) == 5
        assert sliced[("X", "day")].index[-1] == frame.index[4]
