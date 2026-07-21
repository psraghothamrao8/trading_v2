"""Regime router tests — §7."""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.types import Regime
from live.regime import RegimeInputs, RegimeRouter, vwap_side_fraction

IST = clock.IST


@pytest.fixture
def router() -> RegimeRouter:
    return RegimeRouter()


def inputs(**kw) -> RegimeInputs:
    defaults = dict(index_gap_pct=0.1, index_intraday_pct=0.1, vix_intraday_pct=1.0,
                    vwap_side_fraction=0.5, advance_decline_ratio=1.0)
    defaults.update(kw)
    return RegimeInputs(**defaults)


# 2026-07-22 is a Wednesday, not an expiry day.
NORMAL_DAY = _dt.date(2026, 7, 22)
# 2026-07-28 is a Tuesday == the configured expiry weekday.
EXPIRY_DAY = _dt.date(2026, 7, 28)


class TestPanic:
    def test_vix_spike(self, router):
        """§7: India VIX +8% intraday."""
        decision = router.classify(inputs(vix_intraday_pct=9.0), NORMAL_DAY)
        assert decision.regime is Regime.PANIC
        assert "VIX" in decision.reason

    def test_index_drop(self, router):
        """§7: index <= -1.5% intraday."""
        decision = router.classify(inputs(index_intraday_pct=-2.0), NORMAL_DAY)
        assert decision.regime is Regime.PANIC

    def test_just_under_the_thresholds_is_not_panic(self, router):
        assert router.classify(
            inputs(vix_intraday_pct=7.9, index_intraday_pct=-1.4), NORMAL_DAY
        ).regime is not Regime.PANIC

    def test_panic_wins_over_a_down_trend(self, router):
        """A panic day can also satisfy the down-trend test; PANIC is checked first."""
        decision = router.classify(
            inputs(index_intraday_pct=-3.0, index_gap_pct=-1.0,
                   vwap_side_fraction=0.95, advance_decline_ratio=0.1),
            NORMAL_DAY,
        )
        assert decision.regime is Regime.PANIC

    def test_panic_enables_only_panic_reversion(self, router):
        decision = router.classify(inputs(index_intraday_pct=-2.0), NORMAL_DAY)
        assert decision.enabled_engines == ["panic_reversion"]

    def test_panic_disables_premium_selling(self, router):
        """§7: PANIC -> all new premium selling disabled."""
        decision = router.classify(inputs(index_intraday_pct=-2.0), NORMAL_DAY)
        assert decision.premium_selling_disabled is True


class TestTrend:
    def test_up_trend(self, router):
        decision = router.classify(
            inputs(index_gap_pct=0.8, vwap_side_fraction=0.9, advance_decline_ratio=3.0),
            NORMAL_DAY,
        )
        assert decision.regime is Regime.TREND
        assert "up-trend" in decision.reason

    def test_down_trend(self, router):
        """§7: A/D <= 1:2 for a down-trend."""
        decision = router.classify(
            inputs(index_gap_pct=-0.8, vwap_side_fraction=0.9, advance_decline_ratio=0.4),
            NORMAL_DAY,
        )
        assert decision.regime is Regime.TREND
        assert "down-trend" in decision.reason

    def test_small_gap_is_chop(self, router):
        assert router.classify(
            inputs(index_gap_pct=0.2, vwap_side_fraction=0.9, advance_decline_ratio=3.0),
            NORMAL_DAY,
        ).regime is Regime.CHOP

    def test_choppy_vwap_is_chop(self, router):
        assert router.classify(
            inputs(index_gap_pct=0.8, vwap_side_fraction=0.6, advance_decline_ratio=3.0),
            NORMAL_DAY,
        ).regime is Regime.CHOP

    def test_flat_breadth_is_chop(self, router):
        assert router.classify(
            inputs(index_gap_pct=0.8, vwap_side_fraction=0.9, advance_decline_ratio=1.2),
            NORMAL_DAY,
        ).regime is Regime.CHOP

    def test_trend_enables_the_directional_engines(self, router):
        """§7: TREND -> filings, sympathy, preopen, overnight."""
        decision = router.classify(
            inputs(index_gap_pct=0.8, vwap_side_fraction=0.9, advance_decline_ratio=3.0),
            NORMAL_DAY,
        )
        assert decision.enabled_engines == ["filings", "sympathy", "preopen", "overnight"]


class TestChop:
    def test_default_is_chop(self, router):
        assert router.classify(inputs(), NORMAL_DAY).regime is Regime.CHOP

    def test_chop_enables_pairs_and_wheel_management(self, router):
        """§7: CHOP -> pairs, wheel management."""
        assert router.classify(inputs(), NORMAL_DAY).enabled_engines == [
            "pairs", "wheel_management"
        ]

    def test_missing_inputs_fall_back_to_chop_and_say_so(self, router):
        """Partial evidence must never be inflated into a trend."""
        decision = router.classify(
            RegimeInputs(index_gap_pct=0.9, vwap_side_fraction=None,
                         advance_decline_ratio=None),
            NORMAL_DAY,
        )
        assert decision.regime is Regime.CHOP
        assert "unavailable" in decision.reason


class TestAlwaysOnAlerts:
    def test_filings_and_surveillance_alert_in_every_regime(self, router):
        """§7: filings/surveillance ALERTS run in every regime."""
        for measured in (inputs(index_intraday_pct=-2.0), inputs(),
                         inputs(index_gap_pct=0.8, vwap_side_fraction=0.9,
                                advance_decline_ratio=3.0)):
            decision = router.classify(measured, NORMAL_DAY)
            assert decision.may_alert("filings")
            assert decision.may_alert("surveillance")

    def test_may_open_is_stricter_than_may_alert(self, router):
        decision = router.classify(inputs(), NORMAL_DAY)     # CHOP
        assert decision.may_alert("filings") is True
        assert decision.may_open("filings") is False


class TestExpiryDay:
    def test_sizes_are_halved(self, router):
        """§7: expiry days halve all new sizes."""
        decision = router.classify(inputs(), EXPIRY_DAY)
        assert decision.size_multiplier == 0.5
        assert "expiry day" in decision.reason

    def test_normal_day_is_full_size(self, router):
        assert router.classify(inputs(), NORMAL_DAY).size_multiplier == 1.0


class TestMeasurement:
    def _bars(self, closes, day=_dt.date(2026, 7, 22), opens=None):
        index = pd.date_range(
            _dt.datetime.combine(day, _dt.time(9, 15), tzinfo=IST),
            periods=len(closes), freq="5min",
        )
        opens = opens or closes
        return pd.DataFrame(
            {
                "open": opens,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000.0] * len(closes),
            },
            index=index,
        )

    def test_gap_and_intraday_move(self, router, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        bars = self._bars([101.0] * 10, opens=[101.0] * 10)
        measured = router.measure(bars, prev_close=100.0, now=now)
        assert measured.index_gap_pct == pytest.approx(1.0)
        assert measured.index_intraday_pct == pytest.approx(1.0)

    def test_vix_intraday_percent(self, router, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        measured = router.measure(None, prev_close=None, vix_open=12.0, vix_now=13.2, now=now)
        assert measured.vix_intraday_pct == pytest.approx(10.0)

    def test_advance_decline_ratio(self, router, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        measured = router.measure(None, None, advances=1200, declines=400, now=now)
        assert measured.advance_decline_ratio == pytest.approx(3.0)

    def test_missing_data_leaves_inputs_none(self, router, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        measured = router.measure(None, None, now=now)
        assert measured.index_gap_pct is None
        assert measured.vwap_side_fraction is None

    def test_preopen_context_is_carried_into_the_record(self, router, frozen_clock):
        """§7: log GIFT Nifty, prior US session and FII percentile with the day."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        measured = router.measure(None, None, now=now, extras={
            "gift_nifty_gap_pct": -0.6, "prior_us_session_pct": 1.2,
            "fii_ratio_percentile_3y": 8.0,
        })
        recorded = measured.as_dict()
        assert recorded["gift_nifty_gap_pct"] == -0.6
        assert recorded["prior_us_session_pct"] == 1.2
        assert recorded["fii_ratio_percentile_3y"] == 8.0

    def test_vwap_side_fraction_is_the_majority_side(self):
        index = pd.date_range("2026-07-22 09:15", periods=10, freq="5min", tz=IST)
        rising = pd.DataFrame(
            {"open": range(100, 110), "high": range(101, 111), "low": range(99, 109),
             "close": range(100, 110), "volume": [1000.0] * 10},
            index=index,
        )
        fraction = vwap_side_fraction(rising, index[0], index[-1])
        assert fraction is not None and fraction > 0.5

    def test_too_few_bars_returns_none(self):
        index = pd.date_range("2026-07-22 09:15", periods=1, freq="5min", tz=IST)
        frame = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0],
             "volume": [1000.0]},
            index=index,
        )
        assert vwap_side_fraction(frame, index[0], index[-1]) is None


class TestNADecision:
    def test_nothing_opens_before_ten(self, router):
        decision = router.na_decision()
        assert decision.regime is Regime.NA
        assert decision.enabled_engines == []
        assert decision.may_alert("filings") is True
