"""Cost model tests — §4. Heaviest coverage alongside the risk kernel (§0.5).

The three sample trades at the bottom are the §4 brokerage-calculator
comparison. Their expected totals are computed by hand from the rate table in
``config/settings.yaml`` (``costs.as_of``), with the arithmetic written out so
a human can check each line against Zerodha's calculator without re-deriving
anything.
"""

from __future__ import annotations

import pytest

from core.config import ConfigError, get_settings
from core.costs import Charges, CostModel, RoundTrip
from core.types import Segment, Side


@pytest.fixture
def model() -> CostModel:
    return CostModel()


class TestBrokerage:
    def test_equity_delivery_is_free(self, model):
        assert model.brokerage(Segment.EQUITY_DELIVERY, 100_000) == 0.0

    def test_intraday_is_capped_at_twenty(self, model):
        """min(0.03%, ₹20) -- the cap binds on anything above ₹66,667 turnover."""
        assert model.brokerage(Segment.EQUITY_INTRADAY, 1_000_000) == 20.0

    def test_intraday_percentage_applies_on_small_turnover(self, model):
        # 0.03% of 10,000 = 3, which is below the ₹20 cap
        assert model.brokerage(Segment.EQUITY_INTRADAY, 10_000) == pytest.approx(3.0)

    def test_options_are_flat_twenty(self, model):
        assert model.brokerage(Segment.EQUITY_OPTIONS, 5_000) == 20.0
        assert model.brokerage(Segment.EQUITY_OPTIONS, 5_000_000) == 20.0


class TestSTT:
    def test_delivery_charges_both_sides(self, model):
        assert model.stt(Segment.EQUITY_DELIVERY, Side.BUY, 100_000) > 0
        assert model.stt(Segment.EQUITY_DELIVERY, Side.SELL, 100_000) > 0

    def test_intraday_charges_sell_only(self, model):
        assert model.stt(Segment.EQUITY_INTRADAY, Side.BUY, 100_000) == 0.0
        assert model.stt(Segment.EQUITY_INTRADAY, Side.SELL, 100_000) > 0

    def test_futures_charges_sell_only(self, model):
        assert model.stt(Segment.EQUITY_FUTURES, Side.BUY, 100_000) == 0.0
        assert model.stt(Segment.EQUITY_FUTURES, Side.SELL, 100_000) > 0

    def test_options_charge_on_premium_not_notional(self, model):
        """Charging options STT on notional inflates costs by ~100x."""
        premium_based = model.stt(Segment.EQUITY_OPTIONS, Side.SELL, 7_500, premium=7_500)
        notional_based = model.stt(Segment.EQUITY_OPTIONS, Side.SELL, 1_800_000, premium=7_500)
        assert premium_based == notional_based

    def test_exercised_option_uses_intrinsic_value(self, model):
        """§3 STT trap: a long ITM option held to expiry pays on intrinsic."""
        normal_sell = model.stt(Segment.EQUITY_OPTIONS, Side.SELL, 7_500, premium=7_500)
        exercised = model.stt(
            Segment.EQUITY_OPTIONS, Side.BUY, 7_500, intrinsic=750_000, exercised=True
        )
        assert exercised > normal_sell * 20, (
            "the STT trap must be visibly, painfully larger than a normal exit"
        )

    def test_unknown_segment_raises(self, model):
        class FakeSegment:
            value = "crypto_perp"

        with pytest.raises(ConfigError, match="No STT rule"):
            model.stt(FakeSegment(), Side.BUY, 1000)


class TestOtherCharges:
    def test_stamp_duty_is_buy_side_only(self, model):
        assert model.stamp_duty(Segment.EQUITY_DELIVERY, Side.SELL, 100_000) == 0.0
        assert model.stamp_duty(Segment.EQUITY_DELIVERY, Side.BUY, 100_000) > 0

    def test_gst_applies_to_the_right_base(self, model):
        """18% of (brokerage + exchange + SEBI). Not on STT, not on stamp duty."""
        assert model.gst(100.0, 10.0, 1.0) == pytest.approx(0.18 * 111.0)

    def test_dp_charges_only_on_delivery_sells(self, model):
        assert model.dp_charges(Segment.EQUITY_DELIVERY, Side.SELL) == 15.34
        assert model.dp_charges(Segment.EQUITY_DELIVERY, Side.BUY) == 0.0
        assert model.dp_charges(Segment.EQUITY_INTRADAY, Side.SELL) == 0.0

    def test_options_exchange_charges_use_premium(self, model):
        on_premium = model.exchange_charges(Segment.EQUITY_OPTIONS, 1_800_000, premium=7_500)
        assert on_premium == pytest.approx(
            7_500 * get_settings().require("costs.exchange_transaction_charges.equity_options")
        )


class TestSlippage:
    def test_equity_slippage_is_three_bps_per_side(self, model):
        """§4: 0.03%/side liquid equity."""
        assert model.slippage_per_unit(Segment.EQUITY_INTRADAY, 1000.0) == pytest.approx(0.30)

    def test_option_slippage_is_five_bps_per_side(self, model):
        assert model.slippage_per_unit(Segment.EQUITY_OPTIONS, 1000.0) == pytest.approx(0.50)

    def test_one_tick_minimum(self, model):
        """§4: 1 tick minimum. 0.03% of ₹10 is ₹0.003, below the ₹0.05 tick."""
        assert model.slippage_per_unit(Segment.EQUITY_INTRADAY, 10.0) == pytest.approx(0.05)

    def test_slippage_always_costs_money(self, model):
        assert model.apply_slippage(Segment.EQUITY_INTRADAY, Side.BUY, 1000.0) > 1000.0
        assert model.apply_slippage(Segment.EQUITY_INTRADAY, Side.SELL, 1000.0) < 1000.0

    def test_slippage_never_produces_a_negative_price(self, model):
        assert model.apply_slippage(Segment.EQUITY_INTRADAY, Side.SELL, 0.05) > 0


class TestCharges:
    def test_total_sums_every_component(self):
        charges = Charges(brokerage=20, stt=25, exchange=6, sebi=0.2, stamp_duty=3, gst=8, dp=0)
        assert charges.total == pytest.approx(62.2)

    def test_breakdown_keys(self, model):
        breakdown = model.charges(
            Segment.EQUITY_INTRADAY, Side.BUY, 1000.0, 100
        ).breakdown()
        assert set(breakdown) == {
            "brokerage", "stt", "exchange", "sebi", "stamp_duty", "gst", "dp", "total"
        }

    def test_charges_are_additive(self, model):
        entry = model.charges(Segment.EQUITY_INTRADAY, Side.BUY, 1000.0, 100)
        exit_ = model.charges(Segment.EQUITY_INTRADAY, Side.SELL, 1010.0, 100)
        assert (entry + exit_).total == pytest.approx(entry.total + exit_.total)


class TestRoundTrip:
    def test_net_is_gross_minus_costs(self, model):
        trip = model.round_trip(Segment.EQUITY_INTRADAY, Side.BUY, 1000.0, 1010.0, 100)
        assert trip.gross_pnl == pytest.approx(1000.0)
        assert trip.net_pnl == pytest.approx(1000.0 - trip.costs)

    def test_a_small_winner_can_be_a_net_loser(self, model):
        """§4: gross backtests are lies. This is the whole point of the module."""
        trip = model.round_trip(Segment.EQUITY_INTRADAY, Side.BUY, 1000.0, 1000.30, 100)
        assert trip.gross_pnl > 0
        assert trip.net_pnl < 0

    def test_short_round_trip_pnl_sign(self, model):
        trip = model.round_trip(Segment.EQUITY_INTRADAY, Side.SELL, 1010.0, 1000.0, 100)
        assert trip.gross_pnl == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# §4 BROKERAGE-CALCULATOR COMPARISON
#
# Three sample trades, with the arithmetic written out line by line against
# the rate table in config/settings.yaml (costs.as_of). To validate against
# Zerodha's public calculator at https://zerodha.com/brokerage-calculator/,
# enter the same buy/sell/quantity and compare each line below.
#
# `costs.verified_against_calculator` in settings.yaml stays FALSE until a
# human has done that comparison. These tests pin the model's internal
# arithmetic; they cannot by themselves prove the RATES are current.
# ---------------------------------------------------------------------------


class TestBrokerageCalculatorSamples:
    """§4: 'validate the model against Zerodha's public brokerage calculator
    for 3 sample trades (document the comparison in tests)'."""

    def test_sample_1_equity_intraday(self, model):
        """SAMPLE 1 — Equity intraday: buy 100 @ 1000, sell 100 @ 1010.

        buy turnover   = 100 x 1000    = 100,000.00
        sell turnover  = 100 x 1010    = 101,000.00
        total turnover                 = 201,000.00

        brokerage   min(0.03% x 100000, 20) + min(0.03% x 101000, 20)
                    = 20.00 + 20.00                       =  40.0000
        STT         0.025% x 101,000 (sell only)          =  25.2500
        exchange    0.00307% x 201,000                    =   6.1707
        SEBI        0.0001% x 201,000                     =   0.2010
        stamp duty  0.003% x 100,000 (buy only)           =   3.0000
        GST         18% x (40.0000 + 6.1707 + 0.2010)     =   8.3469
        DP          n/a (intraday)                        =   0.0000
                                                            ---------
        TOTAL                                             =  82.9686
        """
        entry = model.charges(Segment.EQUITY_INTRADAY, Side.BUY, 1000.0, 100)
        exit_ = model.charges(Segment.EQUITY_INTRADAY, Side.SELL, 1010.0, 100)
        total = entry + exit_

        assert total.brokerage == pytest.approx(40.0)
        assert total.stt == pytest.approx(25.25)
        assert total.exchange == pytest.approx(6.1707)
        assert total.sebi == pytest.approx(0.201)
        assert total.stamp_duty == pytest.approx(3.0)
        assert total.gst == pytest.approx(8.346906)
        assert total.dp == pytest.approx(0.0)
        assert total.total == pytest.approx(82.97, abs=0.01)

    def test_sample_2_equity_delivery(self, model):
        """SAMPLE 2 — Equity delivery: buy 50 @ 2000, sell 50 @ 2100.

        buy turnover   = 50 x 2000     = 100,000.00
        sell turnover  = 50 x 2100     = 105,000.00
        total turnover                 = 205,000.00

        brokerage   zero on delivery                      =   0.0000
        STT         0.1% x (100,000 + 105,000) both sides = 205.0000
        exchange    0.00307% x 205,000                    =   6.2935
        SEBI        0.0001% x 205,000                     =   0.2050
        stamp duty  0.015% x 100,000 (buy only)           =  15.0000
        GST         18% x (0 + 6.2935 + 0.2050)           =   1.1697
        DP          flat, delivery SELL only              =  15.3400
                                                            ---------
        TOTAL                                             = 243.0082
        """
        entry = model.charges(Segment.EQUITY_DELIVERY, Side.BUY, 2000.0, 50)
        exit_ = model.charges(Segment.EQUITY_DELIVERY, Side.SELL, 2100.0, 50)
        total = entry + exit_

        assert total.brokerage == pytest.approx(0.0)
        assert total.stt == pytest.approx(205.0)
        assert total.exchange == pytest.approx(6.2935)
        assert total.sebi == pytest.approx(0.205)
        assert total.stamp_duty == pytest.approx(15.0)
        assert total.gst == pytest.approx(1.16973)
        assert total.dp == pytest.approx(15.34)
        assert total.total == pytest.approx(243.01, abs=0.01)

    def test_sample_3_equity_options(self, model):
        """SAMPLE 3 — Options: buy 75 (1 NIFTY lot) @ 100, sell 75 @ 150.

        buy premium    = 75 x 100      =   7,500.00
        sell premium   = 75 x 150      =  11,250.00
        total premium                  =  18,750.00

        brokerage   flat 20 x 2 legs                      =  40.0000
        STT         0.15% x 11,250 (sell, on premium)     =  16.8750
        exchange    0.03553% x 18,750 (on premium)        =   6.6619
        SEBI        0.0001% x 18,750                      =   0.0188
        stamp duty  0.003% x 7,500 (buy only)             =   0.2250
        GST         18% x (40 + 6.6619 + 0.0188)          =   8.4025
        DP          n/a                                   =   0.0000
                                                            ---------
        TOTAL                                             =  72.1831
        """
        entry = model.charges(Segment.EQUITY_OPTIONS, Side.BUY, 100.0, 75)
        exit_ = model.charges(Segment.EQUITY_OPTIONS, Side.SELL, 150.0, 75)
        total = entry + exit_

        assert total.brokerage == pytest.approx(40.0)
        assert total.stt == pytest.approx(16.875)
        assert total.exchange == pytest.approx(6.661875)
        assert total.sebi == pytest.approx(0.01875)
        assert total.stamp_duty == pytest.approx(0.225)
        assert total.gst == pytest.approx(8.4025125)
        assert total.total == pytest.approx(72.18, abs=0.01)

    def test_rates_are_dated_and_flagged_unverified(self):
        """The model must never claim to be calculator-verified without a human."""
        settings = get_settings()
        assert settings.require("costs.as_of")
        assert settings.get("costs.verified_against_calculator") is False, (
            "Flip this to true only after running the three samples above through "
            "https://zerodha.com/brokerage-calculator/ by hand."
        )
