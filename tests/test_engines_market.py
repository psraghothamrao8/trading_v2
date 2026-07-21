"""Engine tests: §6.3 pairs, §6.4 overnight, §6.5 preopen, §6.7 panic,
§6.8 wheel, §6.9 flows, §6.10 surveillance.

The market-data engines. No network, no broker, no LLM.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from core import clock
from core.types import Position, Product, Regime, Segment, Side, TTL
from engines.base import Context
from engines.flows import FlowsEngine
from engines.overnight import OvernightEngine
from engines.pairs import PairsEngine, Pair, spread_series, zscore
from engines.panic_reversion import PanicReversionEngine
from engines.preopen import PreopenEngine, imbalance_ratio
from engines.surveillance import SurveillanceEngine
from engines.wheel import WheelEngine
from live.alerts import NullAlerts

IST = clock.IST


def five_min(values: list[float], day: _dt.date = _dt.date(2026, 7, 22)) -> pd.DataFrame:
    index = pd.date_range(
        _dt.datetime.combine(day, _dt.time(9, 15), tzinfo=IST), periods=len(values), freq="5min"
    )
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 0.5 for v in values],
            "low": [v - 0.5 for v in values],
            "close": values,
            "volume": [1000.0] * len(values),
        },
        index=index,
    )


def daily(values: list[float], end: _dt.date = _dt.date(2026, 7, 22)) -> pd.DataFrame:
    index = pd.date_range(end - _dt.timedelta(days=len(values) - 1), periods=len(values),
                          freq="D", tz=IST)
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": values,
            "volume": [100_000.0] * len(values),
        },
        index=index,
    )


# ===========================================================================
# §6.3 pairs
# ===========================================================================


class TestPairsMath:
    def test_spread_uses_the_hedge_ratio(self):
        pair = Pair("it", "TCS", "INFY", hedge_ratio=2.0, pvalue=0.01)
        assert pair.spread(300.0, 100.0) == pytest.approx(100.0)

    def test_zscore(self):
        series = pd.Series([0.0] * 19 + [10.0])
        assert zscore(series, 20) > 3

    def test_zscore_needs_enough_history(self):
        assert zscore(pd.Series([1.0, 2.0]), 20) is None

    def test_zero_sigma_returns_none_not_infinity(self):
        """A flat spread must not trigger an entry."""
        assert zscore(pd.Series([5.0] * 30), 20) is None


class TestPairsEngine:
    def _engine(self, journal, auto_trade=True):
        engine = PairsEngine(journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def _wide_context(self, journal, now, z_high=True):
        """Two series whose spread sits far from its mean."""
        base = list(np.linspace(100, 100, 300))
        a = base[:-1] + [130.0 if z_high else 70.0]
        b = list(np.linspace(100, 100, 300))
        return Context(
            now=now,
            bars={("TCS", "5minute"): five_min(a), ("INFY", "5minute"): five_min(b)},
            prices={"TCS": a[-1], "INFY": b[-1]},
            journal=journal,
        )

    def _store_pair(self, journal):
        journal.save_pairs(
            [{"sector": "it", "symbol_a": "TCS", "symbol_b": "INFY",
              "hedge_ratio": 1.0, "pvalue": 0.01, "lookback_days": 252}],
            refreshed_on="2026-07-01",
        )

    def test_no_pairs_means_no_signals(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 11, 0)
        engine = self._engine(journal)
        assert engine.on_schedule(self._wide_context(journal, now)) == []

    def test_wide_spread_produces_two_rupee_neutral_legs(self, journal, frozen_clock,
                                                          monkeypatch):
        now = frozen_clock(2026, 7, 22, 11, 0)
        self._store_pair(journal)
        engine = self._engine(journal)
        monkeypatch.setattr(engine, "_rolling_window", lambda ctx, interval: 20)

        signals = engine.on_schedule(self._wide_context(journal, now))
        assert len(signals) == 2
        assert {s.side for s in signals} == {Side.BUY, Side.SELL}
        # TCS is rich (z > 0) so it is the short leg.
        short = next(s for s in signals if s.side is Side.SELL)
        assert short.symbol == "TCS"

    def test_material_filing_on_a_leg_blocks_the_pair(self, journal, frozen_clock,
                                                      monkeypatch):
        """§6.3 MANDATORY: a real event breaks mean reversion."""
        now = frozen_clock(2026, 7, 22, 11, 0)
        self._store_pair(journal)
        journal.record_announcement(
            announcement_id="A1", content_hash="h1", symbol="TCS",
            label="MATERIAL_NEGATIVE", trade_date=now.date().isoformat(),
        )
        engine = self._engine(journal)
        monkeypatch.setattr(engine, "_rolling_window", lambda ctx, interval: 20)
        assert engine.on_schedule(self._wide_context(journal, now)) == []

    def test_narrow_spread_produces_nothing(self, journal, frozen_clock, monkeypatch):
        now = frozen_clock(2026, 7, 22, 11, 0)
        self._store_pair(journal)
        engine = self._engine(journal)
        monkeypatch.setattr(engine, "_rolling_window", lambda ctx, interval: 20)
        flat = list(np.linspace(100, 100, 300))
        ctx = Context(
            now=now,
            bars={("TCS", "5minute"): five_min(flat), ("INFY", "5minute"): five_min(flat)},
            prices={"TCS": 100.0, "INFY": 100.0}, journal=journal,
        )
        assert engine.on_schedule(ctx) == []

    def test_converged_pair_is_closed(self, journal, frozen_clock, monkeypatch):
        """§6.3: exit |z| <= 0.25."""
        now = frozen_clock(2026, 7, 22, 14, 0)
        self._store_pair(journal)
        engine = self._engine(journal)
        monkeypatch.setattr(engine, "_rolling_window", lambda ctx, interval: 20)

        # A spread with real variance whose LATEST value sits on its mean:
        # z ~= 0, which is the convergence the exit rule is looking for. A
        # perfectly flat spread has zero sigma and correctly yields no z-score.
        noisy = [100.0 + (2.0 if i % 2 else -2.0) for i in range(299)] + [100.0]
        flat = [100.0] * 300
        positions = [
            Position("TCS", -10, 130.0, "pairs", Product.MIS, meta={"pair_key": "TCS/INFY"}),
            Position("INFY", 13, 100.0, "pairs", Product.MIS, meta={"pair_key": "TCS/INFY"}),
        ]
        ctx = Context(
            now=now, positions=positions,
            bars={("TCS", "5minute"): five_min(noisy), ("INFY", "5minute"): five_min(flat)},
            prices={"TCS": 100.0, "INFY": 100.0}, journal=journal,
        )
        signals = engine.manage(ctx)
        assert len(signals) == 2, "both legs must close together"
        assert all(s.meta.get("exit") for s in signals)

    def test_dropped_pair_is_closed(self, journal, frozen_clock):
        """A pair that failed the last refresh has no hedge ratio we believe in."""
        now = frozen_clock(2026, 7, 22, 14, 0)
        engine = self._engine(journal)
        positions = [
            Position("TCS", -10, 130.0, "pairs", Product.MIS, meta={"pair_key": "TCS/INFY"}),
            Position("INFY", 13, 100.0, "pairs", Product.MIS, meta={"pair_key": "TCS/INFY"}),
        ]
        ctx = Context(now=now, positions=positions,
                      prices={"TCS": 100.0, "INFY": 100.0}, journal=journal)
        signals = engine.manage(ctx)
        assert len(signals) == 2
        assert all("no longer cointegrated" in s.reason for s in signals)

    def test_rolling_window_is_derived_not_hardcoded(self, journal):
        engine = self._engine(journal)
        ctx = Context(now=clock.now_ist(), journal=journal)
        # 20 days x 75 five-minute bars in a 09:15-15:30 session
        assert engine._rolling_window(ctx, "5minute") == 20 * 75


# ===========================================================================
# §6.4 overnight
# ===========================================================================


class TestOvernight:
    def _engine(self, journal, auto_trade=True):
        engine = OvernightEngine(journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def _ctx(self, journal, now, closes, extras=None):
        symbol = "NIFTYBEES"
        return Context(now=now, bars={(symbol, "day"): daily(closes, end=now.date())},
                       prices={symbol: closes[-1]}, journal=journal, extras=extras or {})

    def _rising(self, n=260):
        return [100.0 + i * 0.2 for i in range(n)]

    def test_enters_when_all_filters_pass(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 20)
        engine = self._engine(journal)
        signals = engine.on_schedule(self._ctx(journal, now, self._rising()))
        assert len(signals) == 1
        assert signals[0].ttl is TTL.OVERNIGHT

    def test_below_the_200_dma_blocks(self, journal, frozen_clock):
        """§6.4: index close > 200-DMA."""
        now = frozen_clock(2026, 7, 22, 15, 20)
        engine = self._engine(journal)
        falling = [200.0 - i * 0.2 for i in range(260)]
        reasons = engine.blocking_reasons(self._ctx(journal, now, falling))
        assert any("200-DMA" in r for r in reasons)

    def test_blocked_event_tomorrow_blocks(self, journal, frozen_clock):
        """§6.4: next session not in events.yaml."""
        now = frozen_clock(2026, 7, 29, 15, 20)     # 2026-07-30 is a US-Fed block
        engine = self._engine(journal)
        reasons = engine.blocking_reasons(self._ctx(journal, now, self._rising()))
        assert any("blocked" in r for r in reasons)
        assert engine.on_schedule(self._ctx(journal, now, self._rising())) == []

    def test_ugly_day_blocks(self, journal, frozen_clock):
        """§6.4: today >= -1.5%."""
        now = frozen_clock(2026, 7, 22, 15, 20)
        engine = self._engine(journal)
        closes = self._rising()
        closes[-1] = closes[-2] * 0.97          # -3% day
        reasons = engine.blocking_reasons(self._ctx(journal, now, closes))
        assert any("today is" in r for r in reasons)

    def test_flows_veto_blocks(self, journal, frozen_clock):
        """§6.9: percentile >= 90 vetoes new overnight longs."""
        now = frozen_clock(2026, 7, 22, 15, 20)
        engine = self._engine(journal)
        ctx = self._ctx(journal, now, self._rising(),
                        extras={"flows_veto_overnight_longs": True})
        assert any("6.9" in r for r in engine.blocking_reasons(ctx))

    def test_stop_is_one_adverse_gap_away(self, journal):
        """§6.4: size so a 2% adverse gap ~= the daily loss limit."""
        engine = self._engine(journal)
        assert engine._stop_for(100.0) == pytest.approx(98.0)

    def test_gift_nifty_triggers_a_preopen_exit(self, journal, frozen_clock):
        """§6.4: GIFT Nifty <= -1% -> exit in pre-open, not at 09:16."""
        now = frozen_clock(2026, 7, 23, 9, 5)
        engine = self._engine(journal)
        position = Position("NIFTYBEES", 40, 100.0, "overnight", Product.CNC,
                            ttl=TTL.OVERNIGHT, last_price=99.0,
                            segment=Segment.EQUITY_DELIVERY)
        ctx = Context(now=now, positions=[position], prices={"NIFTYBEES": 99.0},
                      journal=journal, extras={"gift_nifty_gap_pct": -1.5})
        signals = engine.manage(ctx)
        assert len(signals) == 1
        assert "GIFT Nifty" in signals[0].reason

    def test_scheduled_exit_at_0916(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 23, 9, 16)
        engine = self._engine(journal)
        position = Position("NIFTYBEES", 40, 100.0, "overnight", Product.CNC,
                            ttl=TTL.OVERNIGHT, last_price=101.0,
                            segment=Segment.EQUITY_DELIVERY)
        ctx = Context(now=now, positions=[position], prices={"NIFTYBEES": 101.0},
                      journal=journal)
        assert any("09:16" in s.reason for s in engine.manage(ctx))


# ===========================================================================
# §6.5 preopen
# ===========================================================================


class FakePreopenNSE:
    def __init__(self, rows):
        self.rows = rows

    def preopen_snapshot(self, key="NIFTY"):
        return self.rows


class TestPreopen:
    def _rows(self, symbol="TCS", indicative=101.5, prev=100.0, buy=30000, sell=5000):
        return [{
            "symbol": symbol, "indicative_price": indicative, "prev_close": prev,
            "total_buy_quantity": buy, "total_sell_quantity": sell,
        }]

    def _engine(self, journal, rows, auto_trade=True):
        engine = PreopenEngine(nse=FakePreopenNSE(rows), journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def test_imbalance_ratio(self):
        assert imbalance_ratio(30000, 10000) == pytest.approx(3.0)
        assert imbalance_ratio(0, 0) is None
        assert imbalance_ratio(100, 0) == float("inf")

    def test_snapshot_screens_gap_and_imbalance(self, journal, frozen_clock):
        """§6.5: |gap| >= 1% AND ratio >= 3 for longs."""
        now = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows())
        candidates = engine.take_snapshot(Context(now=now, journal=journal), "s1")
        assert len(candidates) == 1
        assert candidates[0].direction is Side.BUY

    def test_small_gap_is_screened_out(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows(indicative=100.3))
        assert engine.take_snapshot(Context(now=now, journal=journal), "s1") == []

    def test_weak_imbalance_is_screened_out(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows(buy=11000, sell=10000))
        assert engine.take_snapshot(Context(now=now, journal=journal), "s1") == []

    def test_persistence_requires_both_snapshots(self, journal, frozen_clock):
        """A single reading can be one large order that gets pulled."""
        now = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows())
        ctx = Context(now=now, journal=journal)
        engine.take_snapshot(ctx, "s1")
        engine._nse = FakePreopenNSE(self._rows(symbol="INFY"))
        engine.take_snapshot(Context(now=now, journal=journal), "s2")
        assert engine.persistent_candidates(ctx) == []

    def test_continuation_required_for_entry(self, journal, frozen_clock):
        """§6.5: enter on continuation only -- price beyond the indicative price."""
        snap = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows())
        ctx_snap = Context(now=snap, journal=journal)
        engine.take_snapshot(ctx_snap, "s1")
        engine.take_snapshot(ctx_snap, "s2")

        entry_now = frozen_clock(2026, 7, 22, 9, 17)
        bars = five_min([101.0] * 20)
        # Price BELOW the 101.5 indicative -> no continuation, no entry.
        ctx = Context(now=entry_now, bars={("TCS", "5minute"): bars},
                      prices={"TCS": 101.0}, journal=journal)
        assert engine.on_schedule(ctx) == []

        # Price ABOVE it -> entry.
        ctx = Context(now=entry_now, bars={("TCS", "5minute"): bars},
                      prices={"TCS": 102.0}, journal=journal)
        signals = engine.on_schedule(ctx)
        assert len(signals) == 1
        assert signals[0].side is Side.BUY

    def test_disagreeing_overnight_filing_vetoes(self, journal, frozen_clock):
        """§6.5: if a filing exists and disagrees, veto."""
        snap = frozen_clock(2026, 7, 22, 9, 6, 30)
        engine = self._engine(journal, self._rows())
        ctx_snap = Context(now=snap, journal=journal)
        engine.take_snapshot(ctx_snap, "s1")
        engine.take_snapshot(ctx_snap, "s2")

        journal.record_announcement(
            announcement_id="A1", content_hash="h1", symbol="TCS",
            label="MATERIAL_NEGATIVE", trade_date="2026-07-22",
        )
        entry_now = frozen_clock(2026, 7, 22, 9, 17)
        ctx = Context(now=entry_now, bars={("TCS", "5minute"): five_min([101.0] * 20)},
                      prices={"TCS": 102.0}, journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_flat_by_1030(self, journal, frozen_clock):
        """§6.5: flat by 10:30 -- this edge is minutes-scale."""
        now = frozen_clock(2026, 7, 22, 10, 35)
        engine = self._engine(journal, [])
        position = Position("TCS", 10, 102.0, "preopen", Product.MIS,
                            stop=100.0, last_price=103.0, segment=Segment.EQUITY_INTRADAY)
        ctx = Context(now=now, positions=[position], prices={"TCS": 103.0}, journal=journal)
        signals = engine.manage(ctx)
        assert any("10:30" in s.reason for s in signals)


# ===========================================================================
# §6.7 panic reversion
# ===========================================================================


class TestPanicReversion:
    def _engine(self, journal, auto_trade=True):
        engine = PanicReversionEngine(journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def test_stock_crash_is_detected(self, journal, frozen_clock):
        """§6.7 trigger B: a NIFTY-100 stock <= -6%."""
        now = frozen_clock(2026, 7, 22, 15, 30)
        engine = self._engine(journal)
        crashed = daily([100.0] * 20 + [92.0], end=now.date())
        ctx = Context(now=now, bars={("HDFCBANK", "day"): crashed}, journal=journal)
        found = [c for c in engine.detect(ctx) if c.symbol == "HDFCBANK"]
        assert len(found) == 1
        assert found[0].trigger == "stock"

    def test_negative_filing_blocks_the_trigger(self, journal, frozen_clock):
        """§6.7 MANDATORY cross-check: this is a repricing, not a sentiment crash."""
        now = frozen_clock(2026, 7, 22, 15, 30)
        engine = self._engine(journal)
        journal.record_announcement(
            announcement_id="A1", content_hash="h1", symbol="HDFCBANK",
            label="MATERIAL_NEGATIVE", trade_date=now.date().isoformat(),
        )
        crashed = daily([100.0] * 20 + [92.0], end=now.date())
        ctx = Context(now=now, bars={("HDFCBANK", "day"): crashed}, journal=journal)
        assert [c for c in engine.detect(ctx) if c.symbol == "HDFCBANK"] == []

    def test_small_drop_is_not_a_trigger(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 30)
        engine = self._engine(journal)
        ctx = Context(now=now,
                      bars={("HDFCBANK", "day"): daily([100.0] * 20 + [98.0], end=now.date())},
                      journal=journal)
        assert engine.detect(ctx) == []

    def test_entry_requires_reclaiming_the_first_15_min_high(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 23, 10, 0)
        engine = self._engine(journal)
        from engines.panic_reversion import PanicCandidate

        engine.watchlist["HDFCBANK"] = PanicCandidate(
            symbol="HDFCBANK", trigger="stock", crash_date=_dt.date(2026, 7, 22),
            move_pct=-8.0, session_low=90.0,
        )
        # opening range high is 92.5 (92 + 0.5)
        bars = five_min([92.0, 92.0, 92.0] + [93.0] * 6, day=_dt.date(2026, 7, 23))
        ctx = Context(now=now, bars={("HDFCBANK", "5minute"): bars},
                      prices={"HDFCBANK": 93.0}, regime=Regime.PANIC, journal=journal)
        signals = engine.on_schedule(ctx)
        assert len(signals) == 1
        assert signals[0].side is Side.BUY
        assert signals[0].stop < 93.0

    def test_no_entry_without_a_reclaim(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 23, 10, 0)
        engine = self._engine(journal)
        from engines.panic_reversion import PanicCandidate

        engine.watchlist["HDFCBANK"] = PanicCandidate(
            symbol="HDFCBANK", trigger="stock", crash_date=_dt.date(2026, 7, 22),
            move_pct=-8.0, session_low=90.0,
        )
        bars = five_min([95.0, 95.0, 95.0] + [91.0] * 6, day=_dt.date(2026, 7, 23))
        ctx = Context(now=now, bars={("HDFCBANK", "5minute"): bars},
                      prices={"HDFCBANK": 91.0}, regime=Regime.PANIC, journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_only_runs_in_the_panic_regime(self, journal, frozen_clock):
        """§6.7: enabled only in PANIC regime."""
        now = frozen_clock(2026, 7, 23, 10, 0)
        engine = self._engine(journal)
        ctx = Context(now=now, regime=Regime.CHOP, journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_entry_window_is_respected(self, journal, frozen_clock):
        """§6.7: entry next session 09:30-10:30."""
        now = frozen_clock(2026, 7, 23, 11, 30)
        engine = self._engine(journal)
        from engines.panic_reversion import PanicCandidate

        engine.watchlist["HDFCBANK"] = PanicCandidate(
            symbol="HDFCBANK", trigger="stock", crash_date=_dt.date(2026, 7, 22),
            move_pct=-8.0, session_low=90.0,
        )
        ctx = Context(now=now, regime=Regime.PANIC, journal=journal)
        assert engine.on_schedule(ctx) == []


# ===========================================================================
# §6.8 wheel
# ===========================================================================


class TestWheel:
    def _engine(self, journal, alerts=None):
        return WheelEngine(alerts=alerts or NullAlerts(), journal=journal)

    def test_universe_is_the_approved_list_only(self, journal):
        assert self._engine(journal).universe() == ["RELIANCE", "INFY", "ITC"]

    def test_on_schedule_never_returns_a_tradable_signal(self, journal, frozen_clock):
        """§6.8: every wheel order requires Telegram confirmation, even in paper."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        assert engine.on_schedule(Context(now=now, journal=journal)) == []

    def test_iv_gate_closed_without_enough_vix_history(self, journal, frozen_clock):
        """A percentile from 12 observations is not a percentile."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        ctx = Context(now=now, india_vix=18.0, journal=journal,
                      extras={"india_vix_history": [12.0] * 5})
        assert engine.iv_gate_open(ctx) is False

    def test_iv_gate_closed_at_low_percentile(self, journal, frozen_clock):
        """§6.8: run only when India VIX 1-year percentile >= 50."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        ctx = Context(now=now, india_vix=10.0, journal=journal,
                      extras={"india_vix_history": list(np.linspace(12, 30, 250))})
        assert engine.iv_gate_open(ctx) is False

    def test_iv_gate_open_at_high_percentile(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        ctx = Context(now=now, india_vix=28.0, journal=journal,
                      extras={"india_vix_history": list(np.linspace(12, 30, 250))})
        assert engine.iv_gate_open(ctx) is True

    def test_proposal_is_sent_for_confirmation(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        engine = self._engine(journal, alerts)
        chain = [{
            "tradingsymbol": "RELIANCE26AUG2800PE", "instrument_type": "PE",
            "strike": 2800.0, "expiry": "2026-08-25", "delta": -0.25, "last_price": 45.0,
        }]
        ctx = Context(now=now, india_vix=28.0, journal=journal, extras={
            "india_vix_history": list(np.linspace(12, 30, 250)),
            "option_chains": {"RELIANCE": chain},
        })
        proposals = engine.propose_new_puts(ctx)
        assert len(proposals) == 1
        assert any("CONFIRMATION REQUIRED" in m for m in alerts.sent_messages)
        assert any("physical settlement" in m.lower() for m in alerts.sent_messages)

    def test_unconfirmed_proposal_yields_no_signal(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        chain = [{
            "tradingsymbol": "RELIANCE26AUG2800PE", "instrument_type": "PE",
            "strike": 2800.0, "expiry": "2026-08-25", "delta": -0.25, "last_price": 45.0,
        }]
        ctx = Context(now=now, india_vix=28.0, journal=journal, extras={
            "india_vix_history": list(np.linspace(12, 30, 250)),
            "option_chains": {"RELIANCE": chain},
        })
        proposal = engine.propose_new_puts(ctx)[0]
        assert engine.signal_for_confirmed(proposal.request_id, ctx) is None

    def test_confirmed_proposal_yields_a_signal(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        chain = [{
            "tradingsymbol": "RELIANCE26AUG2800PE", "instrument_type": "PE",
            "strike": 2800.0, "expiry": "2026-08-25", "delta": -0.25, "last_price": 45.0,
        }]
        ctx = Context(now=now, india_vix=28.0, journal=journal, extras={
            "india_vix_history": list(np.linspace(12, 30, 250)),
            "option_chains": {"RELIANCE": chain},
        })
        proposal = engine.propose_new_puts(ctx)[0]
        engine.confirm(proposal.request_id, True)
        signal = engine.signal_for_confirmed(proposal.request_id, ctx)
        assert signal is not None
        assert signal.side is Side.SELL
        assert signal.meta["underlying_type"] == "stock"
        assert signal.meta["allow_delivery"] is False   # §3 guard stays armed

    def test_rejected_proposal_yields_no_signal(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        chain = [{
            "tradingsymbol": "RELIANCE26AUG2800PE", "instrument_type": "PE",
            "strike": 2800.0, "expiry": "2026-08-25", "delta": -0.25, "last_price": 45.0,
        }]
        ctx = Context(now=now, india_vix=28.0, journal=journal, extras={
            "india_vix_history": list(np.linspace(12, 30, 250)),
            "option_chains": {"RELIANCE": chain},
        })
        proposal = engine.propose_new_puts(ctx)[0]
        engine.confirm(proposal.request_id, False)
        assert engine.signal_for_confirmed(proposal.request_id, ctx) is None

    def test_flows_veto_stops_premium_selling(self, journal, frozen_clock):
        """§6.9: halve premium selling at an extreme percentile; 1 lot cannot halve."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        engine = self._engine(journal)
        ctx = Context(now=now, india_vix=28.0, journal=journal, extras={
            "india_vix_history": list(np.linspace(12, 30, 250)),
            "flows_halve_premium_selling": True,
        })
        assert engine.propose_new_puts(ctx) == []


# ===========================================================================
# §6.9 flows
# ===========================================================================


class TestFlows:
    def _engine(self, journal, auto_trade=True):
        engine = FlowsEngine(journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def test_no_data_means_no_signal(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 20)
        assert self._engine(journal).on_schedule(Context(now=now, journal=journal)) == []

    def test_low_percentile_produces_a_long(self, journal, frozen_clock):
        """§6.9: percentile <= 10 -> swing long the index."""
        now = frozen_clock(2026, 7, 22, 9, 20)
        journal.record_flows("2026-07-21", long_ratio=0.28, ratio_percentile_3y=6.0)
        engine = self._engine(journal)
        ctx = Context(now=now, prices={"NIFTYBEES": 250.0}, journal=journal)
        signals = engine.on_schedule(ctx)
        assert len(signals) == 1
        assert signals[0].side is Side.BUY
        assert signals[0].stop == pytest.approx(245.0)      # -2%

    def test_mid_percentile_produces_nothing(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 20)
        journal.record_flows("2026-07-21", long_ratio=0.5, ratio_percentile_3y=55.0)
        engine = self._engine(journal)
        ctx = Context(now=now, prices={"NIFTYBEES": 250.0}, journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_high_percentile_never_shorts(self, journal, frozen_clock):
        """§6.9: percentile >= 90 -> NO shorting. It is a veto, not a signal."""
        now = frozen_clock(2026, 7, 22, 9, 20)
        journal.record_flows("2026-07-21", long_ratio=0.9, ratio_percentile_3y=95.0)
        engine = self._engine(journal)
        ctx = Context(now=now, prices={"NIFTYBEES": 250.0}, journal=journal)
        assert engine.on_schedule(ctx) == []

    def test_high_percentile_emits_the_two_vetoes(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 20)
        journal.record_flows("2026-07-21", long_ratio=0.9, ratio_percentile_3y=95.0)
        context = self._engine(journal).regime_context(Context(now=now, journal=journal))
        assert context["flows_veto_overnight_longs"] is True
        assert context["flows_halve_premium_selling"] is True

    def test_null_percentile_is_not_usable(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 9, 20)
        journal.record_flows("2026-07-21", long_ratio=0.28, ratio_percentile_3y=None)
        context = self._engine(journal).regime_context(Context(now=now, journal=journal))
        assert context["flows_available"] is False

    def test_profit_target_exit(self, journal, frozen_clock):
        """§6.9 exit: +4%."""
        now = frozen_clock(2026, 8, 5, 10, 0)
        journal.record_flows("2026-08-04", long_ratio=0.3, ratio_percentile_3y=15.0)
        engine = self._engine(journal)
        position = Position("NIFTYBEES", 100, 250.0, "flows", Product.CNC, ttl=TTL.SWING,
                            opened_at=_dt.datetime(2026, 7, 22, 9, 20, tzinfo=IST),
                            last_price=261.0, segment=Segment.EQUITY_DELIVERY)
        ctx = Context(now=now, positions=[position], prices={"NIFTYBEES": 261.0},
                      journal=journal)
        assert any("target" in s.reason for s in engine.manage(ctx))

    def test_percentile_normalisation_exit(self, journal, frozen_clock):
        """§6.9 exit: percentile > 40."""
        now = frozen_clock(2026, 8, 5, 10, 0)
        journal.record_flows("2026-08-04", long_ratio=0.6, ratio_percentile_3y=55.0)
        engine = self._engine(journal)
        position = Position("NIFTYBEES", 100, 250.0, "flows", Product.CNC, ttl=TTL.SWING,
                            opened_at=_dt.datetime(2026, 7, 22, 9, 20, tzinfo=IST),
                            last_price=252.0, segment=Segment.EQUITY_DELIVERY)
        ctx = Context(now=now, positions=[position], prices={"NIFTYBEES": 252.0},
                      journal=journal)
        assert any("percentile back to" in s.reason for s in engine.manage(ctx))


# ===========================================================================
# §6.10 surveillance
# ===========================================================================


class TestSurveillance:
    def test_diff_reports_entries_and_exits(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 20, 30)
        journal.record_surveillance_snapshot("2026-07-21", "asm",
                                             [{"symbol": "ABC", "stage": "ST-I"}])
        journal.record_surveillance_snapshot("2026-07-22", "asm",
                                             [{"symbol": "XYZ", "stage": "ST-II"}])
        engine = SurveillanceEngine(alerts=NullAlerts(), journal=journal)
        change = engine.diff(Context(now=now, journal=journal))["asm"]
        assert [e["symbol"] for e in change.added] == ["XYZ"]
        assert change.removed == ["ABC"]

    def test_veto_set_feeds_the_kernel(self, journal, frozen_clock):
        """§6.10: additions feed the §3 kernel veto -- no engine touches them."""
        now = frozen_clock(2026, 7, 22, 20, 30)
        journal.record_surveillance_snapshot("2026-07-22", "asm",
                                             [{"symbol": "ABC", "stage": "ST-I"}])
        journal.record_surveillance_snapshot("2026-07-22", "gsm",
                                             [{"symbol": "DEF", "stage": "ST-1"}])
        engine = SurveillanceEngine(alerts=NullAlerts(), journal=journal)
        assert engine.veto_symbols(Context(now=now, journal=journal)) == {"ABC", "DEF"}

    def test_ban_list_is_separate_from_asm_gsm(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 20, 30)
        journal.record_surveillance_snapshot("2026-07-22", "fno_ban", [{"symbol": "IDEA"}])
        engine = SurveillanceEngine(alerts=NullAlerts(), journal=journal)
        ctx = Context(now=now, journal=journal)
        assert engine.ban_list(ctx) == {"IDEA"}
        assert engine.veto_symbols(ctx) == set()

    def test_stale_lists_are_alerted(self, journal, frozen_clock):
        """§8.2: never silently degrade. A stale veto set is worse than none."""
        now = frozen_clock(2026, 7, 22, 20, 30)
        journal.record_surveillance_snapshot("2026-07-01", "asm", [{"symbol": "ABC"}])
        alerts = NullAlerts()
        engine = SurveillanceEngine(alerts=alerts, journal=journal)
        engine.send_digest(Context(now=now, journal=journal))
        assert any("days old" in m for m in alerts.sent_messages)

    def test_digest_is_not_sent_when_nothing_changed(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 20, 30)
        journal.record_surveillance_snapshot("2026-07-22", "asm", [{"symbol": "ABC"}])
        engine = SurveillanceEngine(alerts=NullAlerts(), journal=journal)
        # Only one snapshot date exists, so everything reads as "added".
        assert engine.send_digest(Context(now=now, journal=journal)) is True

        journal.record_surveillance_snapshot("2026-07-23", "asm", [{"symbol": "ABC"}])
        assert engine.send_digest(Context(now=frozen_clock(2026, 7, 23, 20, 30),
                                          journal=journal)) is False
