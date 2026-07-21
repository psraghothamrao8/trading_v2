"""Calendar tests — §8.3 session structure, expiries, holidays, event blocks."""

from __future__ import annotations

import datetime as _dt

import pytest

from core import calendar as cal
from core.config import ConfigError


class TestTradingDays:
    def test_weekend_is_not_a_trading_day(self):
        assert not cal.is_trading_day(_dt.date(2026, 7, 25))   # Saturday
        assert not cal.is_trading_day(_dt.date(2026, 7, 26))   # Sunday

    def test_normal_weekday_is_a_trading_day(self):
        assert cal.is_trading_day(_dt.date(2026, 7, 22))       # Wednesday

    def test_configured_holiday_is_not_a_trading_day(self):
        assert cal.is_holiday(_dt.date(2026, 8, 15))           # Independence Day
        assert not cal.is_trading_day(_dt.date(2026, 8, 15))

    def test_next_trading_day_skips_the_weekend(self):
        assert cal.next_trading_day(_dt.date(2026, 7, 24)) == _dt.date(2026, 7, 27)

    def test_previous_trading_day_skips_the_weekend(self):
        assert cal.previous_trading_day(_dt.date(2026, 7, 27)) == _dt.date(2026, 7, 24)

    def test_trading_days_between_excludes_weekends(self):
        days = cal.trading_days_between(_dt.date(2026, 7, 20), _dt.date(2026, 7, 26))
        assert days == [
            _dt.date(2026, 7, d) for d in (20, 21, 22, 23, 24)
        ]

    def test_sessions_until_counts_sessions_not_days(self):
        # Fri 24th -> Tue 28th is 2 sessions (Mon 27, Tue 28), not 4 days.
        assert cal.sessions_until(_dt.date(2026, 7, 28), _dt.date(2026, 7, 24)) == 2


class TestMuhurat:
    def test_muhurat_day_is_recognised(self):
        assert cal.is_muhurat(_dt.date(2025, 10, 21))

    def test_muhurat_day_is_not_a_normal_trading_day(self):
        """It is a listed holiday with a special evening window."""
        assert not cal.is_trading_day(_dt.date(2025, 10, 21))

    def test_engines_disabled_for_muhurat_by_default(self):
        assert cal.muhurat_engines_enabled() is False


class TestSessionWindows:
    def test_continuous_session(self):
        start, end = cal.session_window(_dt.date(2026, 7, 22), "continuous")
        assert (start.hour, start.minute) == (9, 15)
        assert (end.hour, end.minute) == (15, 30)

    def test_preopen_session(self):
        start, end = cal.session_window(_dt.date(2026, 7, 22), "preopen")
        assert (start.hour, start.minute) == (9, 0)
        assert (end.hour, end.minute) == (9, 12)

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError, match="Unknown session window"):
            cal.session_window(_dt.date(2026, 7, 22), "lunch")

    def test_is_market_open(self, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 30)
        assert cal.is_market_open()
        frozen_clock(2026, 7, 22, 16, 30)
        assert not cal.is_market_open()
        frozen_clock(2026, 7, 25, 10, 30)   # Saturday
        assert not cal.is_market_open()


class TestExpiry:
    def test_weekly_expiry_uses_the_configured_weekday(self):
        """§8.3: never hardcode the expiry weekday."""
        expiry = cal.weekly_expiry(_dt.date(2026, 7, 22), "NIFTY")
        assert expiry == _dt.date(2026, 7, 28)
        assert expiry.strftime("%A").lower() == "tuesday"

    def test_weekly_expiry_on_the_day_itself(self):
        assert cal.weekly_expiry(_dt.date(2026, 7, 28)) == _dt.date(2026, 7, 28)

    def test_monthly_expiry_is_the_last_configured_weekday(self):
        expiry = cal.monthly_expiry(2026, 7, "index")
        assert expiry == _dt.date(2026, 7, 28)

    def test_expiry_rolls_back_over_a_holiday(self, monkeypatch):
        """If the expiry weekday is a holiday, expiry moves to the prior session."""
        original = cal.holidays(None)
        patched = frozenset(original | {_dt.date(2026, 7, 28)})
        monkeypatch.setattr(cal, "holidays", lambda config_dir=None: patched)
        assert cal.weekly_expiry(_dt.date(2026, 7, 22)) == _dt.date(2026, 7, 27)

    def test_is_expiry_day(self):
        assert cal.is_expiry_day(_dt.date(2026, 7, 28))
        assert not cal.is_expiry_day(_dt.date(2026, 7, 22))

    def test_expiry_config_ships_unverified(self):
        """§8.3: must be verified against NSE/Kite in Phase 2 before F&O go-live."""
        assert cal.expiry_config_verified() is False


class TestLotSize:
    def test_known_lot_size(self):
        assert cal.lot_size("NIFTY") == 75

    def test_unknown_lot_size_raises_rather_than_guessing(self):
        with pytest.raises(ConfigError, match="never be guessed"):
            cal.lot_size("SOMERANDOMSTOCK")


class TestEventBlocks:
    def test_rbi_mpc_day_is_blocked(self):
        assert cal.is_blocked_event_day(_dt.date(2026, 2, 6))
        assert "rbi_mpc" in cal.event_note(_dt.date(2026, 2, 6))

    def test_budget_day_is_blocked(self):
        assert cal.is_blocked_event_day(_dt.date(2026, 2, 1))

    def test_us_fed_follow_on_session_is_blocked(self):
        assert cal.is_blocked_event_day(_dt.date(2026, 7, 30))

    def test_normal_day_is_not_blocked(self):
        assert not cal.is_blocked_event_day(_dt.date(2026, 7, 22))

    def test_next_session_is_blocked_detects_the_eve(self):
        """§6.4: no overnight entry when tomorrow is blocked."""
        assert cal.next_session_is_blocked(_dt.date(2026, 7, 29))
        assert not cal.next_session_is_blocked(_dt.date(2026, 7, 22))


class TestDescribe:
    def test_describe_covers_the_status_fields(self, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        info = cal.describe()
        assert info["date"] == "2026-07-22"
        assert info["trading_day"] is True
        assert info["blocked_event"] is False
        assert "next_weekly_expiry" in info
