"""Clock tests — §0.6: every timestamp is Asia/Kolkata."""

from __future__ import annotations

import datetime as _dt

import pytest

from core import clock


class TestNow:
    def test_now_is_tz_aware_ist(self):
        now = clock.now_ist()
        assert now.tzinfo is not None
        assert now.utcoffset() == _dt.timedelta(hours=5, minutes=30)

    def test_clock_override_is_respected(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 30)
        assert clock.now_ist() == ts
        assert clock.today_ist() == _dt.date(2026, 7, 22)

    def test_naive_override_is_assumed_ist(self):
        clock.set_clock(lambda: _dt.datetime(2026, 7, 22, 10, 30))
        assert clock.now_ist().utcoffset() == _dt.timedelta(hours=5, minutes=30)

    def test_utc_input_is_converted(self):
        utc = _dt.datetime(2026, 7, 22, 5, 0, tzinfo=_dt.timezone.utc)
        assert clock.to_ist(utc).hour == 10
        assert clock.to_ist(utc).minute == 30


class TestParsing:
    def test_hhmm(self):
        assert clock.parse_hhmm("14:45") == _dt.time(14, 45)

    def test_hhmmss(self):
        """Config uses 09:06:30 for the pre-open snapshots (§6.5)."""
        assert clock.parse_hhmm("09:06:30") == _dt.time(9, 6, 30)

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="expected HH:MM"):
            clock.parse_hhmm("9am")


class TestComparisons:
    def test_is_after_cutoff(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 14, 46)
        assert clock.is_after(ts, "14:45")
        assert not clock.is_before(ts, "14:45")

    def test_within_window(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 9, 30)
        assert clock.within(ts, "09:20", "14:30")
        assert not clock.within(ts, "09:00", "09:15")

    def test_boundaries_are_inclusive(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 9, 20)
        assert clock.within(ts, "09:20", "14:30")


class TestFormatting:
    def test_isoformat_carries_the_ist_offset(self, frozen_clock):
        ts = frozen_clock(2026, 7, 22, 10, 30)
        assert clock.isoformat(ts) == "2026-07-22T10:30:00+05:30"

    def test_at_combines_date_and_config_time(self):
        ts = clock.at(_dt.date(2026, 7, 22), "15:10")
        assert (ts.hour, ts.minute) == (15, 10)
        assert ts.utcoffset() == _dt.timedelta(hours=5, minutes=30)
