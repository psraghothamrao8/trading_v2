"""Datafeed tests — chunking, storage, indicators, websocket gap handling."""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.config import ConfigError
from core.datafeed import (
    DataError,
    DataFeed,
    TickStream,
    atr,
    ema,
    first_n_minutes,
    percentile_rank,
    plan_chunks,
    sma,
    vwap,
)


class FakeKite:
    """Minimal Kite double: instruments, historical_data, ltp, quote."""

    def __init__(self, candles: list[dict] | None = None, fail_on: set[int] | None = None) -> None:
        self.candles = candles or []
        self.fail_on = fail_on or set()
        self.calls: list[dict] = []

    def instruments(self, exchange="NSE"):
        return [
            {"instrument_token": 111, "tradingsymbol": "RELIANCE", "segment": "NSE",
             "instrument_type": "EQ", "name": "RELIANCE"},
            {"instrument_token": 222, "tradingsymbol": "INFY", "segment": "NSE",
             "instrument_type": "EQ", "name": "INFY"},
            {"instrument_token": 333, "tradingsymbol": "NIFTYBEES", "segment": "NSE",
             "instrument_type": "EQ", "name": "NIFTYBEES"},
        ]

    def historical_data(self, instrument_token, from_date, to_date, interval):
        index = len(self.calls)
        self.calls.append({"from": from_date, "to": to_date, "interval": interval})
        if index in self.fail_on:
            raise RuntimeError("simulated Kite failure")
        return [
            row for row in self.candles
            if from_date <= row["date"].date() <= to_date
        ]

    def ltp(self, keys):
        return {k: {"last_price": 100.0} for k in keys}

    def quote(self, keys):
        return {
            k: {"last_price": 100.0, "upper_circuit_limit": 110.0, "lower_circuit_limit": 90.0}
            for k in keys
        }


def make_candles(days: int, start: _dt.date = _dt.date(2026, 1, 1)) -> list[dict]:
    out = []
    for i in range(days):
        day = start + _dt.timedelta(days=i)
        out.append({
            "date": _dt.datetime.combine(day, _dt.time(9, 15), tzinfo=clock.IST),
            "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i,
            "close": 101.0 + i, "volume": 1000 + i,
        })
    return out


class TestChunking:
    def test_single_chunk_when_range_fits(self):
        chunks = plan_chunks(_dt.date(2026, 1, 1), _dt.date(2026, 1, 10), 60)
        assert len(chunks) == 1

    def test_splits_at_the_cap(self):
        """Kite rejects an over-long range; chunking is correctness, not polish."""
        chunks = plan_chunks(_dt.date(2026, 1, 1), _dt.date(2026, 12, 31), 100)
        assert len(chunks) == 4
        assert chunks[0].start == _dt.date(2026, 1, 1)
        assert chunks[-1].end == _dt.date(2026, 12, 31)

    def test_chunks_are_contiguous_and_non_overlapping(self):
        chunks = plan_chunks(_dt.date(2026, 1, 1), _dt.date(2026, 6, 30), 45)
        for a, b in zip(chunks, chunks[1:]):
            assert b.start == a.end + _dt.timedelta(days=1)

    def test_bad_range_raises(self):
        with pytest.raises(ValueError, match="before start"):
            plan_chunks(_dt.date(2026, 2, 1), _dt.date(2026, 1, 1), 60)

    def test_zero_cap_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            plan_chunks(_dt.date(2026, 1, 1), _dt.date(2026, 1, 2), 0)


class TestHistorical:
    def test_downloads_and_normalises(self, journal, tmp_path, monkeypatch):
        kite = FakeKite(make_candles(30))
        feed = DataFeed(kite=kite, journal=journal, sleeper=lambda s: None)
        frame = feed.historical("RELIANCE", "day", _dt.date(2026, 1, 1), _dt.date(2026, 1, 30))
        assert len(frame) == 30
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert str(frame.index.tz) == "Asia/Kolkata"

    def test_rate_limit_sleep_between_chunks(self, journal):
        slept: list[float] = []
        kite = FakeKite(make_candles(400))
        feed = DataFeed(kite=kite, journal=journal, sleeper=slept.append)
        feed.historical("RELIANCE", "5minute", _dt.date(2026, 1, 1), _dt.date(2026, 12, 31))
        assert len(kite.calls) == 4          # 100-day cap
        assert len(slept) == 3               # no sleep after the last chunk

    def test_one_failed_chunk_does_not_lose_the_rest(self, journal):
        kite = FakeKite(make_candles(400), fail_on={1})
        feed = DataFeed(kite=kite, journal=journal, sleeper=lambda s: None)
        frame = feed.historical("RELIANCE", "5minute", _dt.date(2026, 1, 1), _dt.date(2026, 12, 31))
        assert not frame.empty
        errors = journal.query("SELECT * FROM errors WHERE source='datafeed'")
        assert len(errors) == 1, "the gap must be journalled, not swallowed"

    def test_no_session_raises_rather_than_returning_empty(self, journal):
        feed = DataFeed(kite=None, journal=journal)
        with pytest.raises(DataError, match="morning_auth"):
            feed.historical("RELIANCE", "day", _dt.date(2026, 1, 1), _dt.date(2026, 1, 5))

    def test_unknown_interval_names_the_config(self, journal):
        feed = DataFeed(kite=FakeKite(), journal=journal)
        with pytest.raises(ConfigError, match="max_days_per_request"):
            feed.max_days_for("7minute")

    def test_unknown_symbol_raises(self, journal):
        feed = DataFeed(kite=FakeKite(), journal=journal)
        with pytest.raises(DataError, match="not found"):
            feed.instrument_token("NOTREAL")


class TestStorage:
    def test_save_and_load_roundtrip(self, journal, monkeypatch, tmp_path):
        monkeypatch.setattr("core.datafeed.data_path",
                            lambda *p: _mkpath(tmp_path, *p))
        kite = FakeKite(make_candles(10))
        feed = DataFeed(kite=kite, journal=journal, sleeper=lambda s: None)
        frame = feed.historical("RELIANCE", "day", _dt.date(2026, 1, 1), _dt.date(2026, 1, 10))
        feed.save("RELIANCE", "day", frame)
        loaded = feed.load("RELIANCE", "day")
        assert len(loaded) == 10

    def test_save_merges_without_duplicating(self, journal, monkeypatch, tmp_path):
        monkeypatch.setattr("core.datafeed.data_path", lambda *p: _mkpath(tmp_path, *p))
        kite = FakeKite(make_candles(20))
        feed = DataFeed(kite=kite, journal=journal, sleeper=lambda s: None)
        first = feed.historical("RELIANCE", "day", _dt.date(2026, 1, 1), _dt.date(2026, 1, 10))
        second = feed.historical("RELIANCE", "day", _dt.date(2026, 1, 5), _dt.date(2026, 1, 20))
        feed.save("RELIANCE", "day", first)
        feed.save("RELIANCE", "day", second)
        assert len(feed.load("RELIANCE", "day")) == 20

    def test_missing_data_raises_with_the_fix_command(self, journal, monkeypatch, tmp_path):
        """An empty frame would read as 'no trades found'. Say what is wrong."""
        monkeypatch.setattr("core.datafeed.data_path", lambda *p: _mkpath(tmp_path, *p))
        feed = DataFeed(kite=None, journal=journal)
        with pytest.raises(DataError, match="download_history.py"):
            feed.load("RELIANCE", "day")


class TestBands:
    def test_bands_feed_the_kernel_veto(self, journal):
        feed = DataFeed(kite=FakeKite(), journal=journal)
        bands = feed.bands(["RELIANCE"])
        assert bands["RELIANCE"].upper == 110.0
        assert bands["RELIANCE"].distance_to_band_pct() == pytest.approx(10.0)

    def test_band_failure_degrades_to_empty_and_journals(self, journal):
        class Boom(FakeKite):
            def quote(self, keys):
                raise RuntimeError("quote down")

        feed = DataFeed(kite=Boom(), journal=journal)
        assert feed.bands(["RELIANCE"]) == {}
        assert journal.query("SELECT * FROM errors WHERE source='datafeed'")


class TestIndicators:
    def _frame(self) -> pd.DataFrame:
        index = pd.date_range("2026-01-01 09:15", periods=60, freq="5min", tz=clock.IST)
        return pd.DataFrame(
            {
                "open": range(100, 160),
                "high": [x + 2 for x in range(100, 160)],
                "low": [x - 2 for x in range(100, 160)],
                "close": [x + 1 for x in range(100, 160)],
                "volume": [1000] * 60,
            },
            index=index,
        )

    def test_atr_is_positive_after_the_warmup(self):
        values = atr(self._frame(), 14)
        assert values.iloc[:13].isna().all()
        assert values.iloc[-1] > 0

    def test_vwap_resets_each_session(self):
        """A VWAP carried across days is meaningless."""
        day1 = self._frame()
        day2 = day1.copy()
        day2.index = day2.index + pd.Timedelta(days=1)
        day2["close"] = day2["close"] + 1000
        combined = pd.concat([day1, day2])
        values = vwap(combined)
        # Day 2's first VWAP equals its own first typical price, not a carryover.
        first_day2 = values.loc[day2.index[0]]
        typical = (day2["high"].iloc[0] + day2["low"].iloc[0] + day2["close"].iloc[0]) / 3
        assert first_day2 == pytest.approx(typical)

    def test_ema_and_sma_warmup(self):
        series = self._frame()["close"]
        assert ema(series, 20).iloc[:19].isna().all()
        assert sma(series, 20).iloc[:19].isna().all()
        assert sma(series, 20).iloc[-1] == pytest.approx(series.iloc[-20:].mean())

    def test_percentile_rank(self):
        series = pd.Series(range(100))
        assert percentile_rank(series, 49) == pytest.approx(50.0)
        assert percentile_rank(series, 0) == pytest.approx(1.0)

    def test_percentile_of_empty_raises(self):
        with pytest.raises(ValueError, match="empty series"):
            percentile_rank(pd.Series(dtype=float), 1.0)

    def test_first_n_minutes_window(self):
        frame = self._frame()
        window = first_n_minutes(frame, _dt.date(2026, 1, 1), 15)
        assert len(window) == 3          # 3 x 5-minute bars
        assert window.index[0].hour == 9


class TestTickStream:
    def test_resubscribes_on_reconnect(self, journal):
        class FakeTicker:
            MODE_FULL = "full"

            def __init__(self):
                self.subscribed: list[list[int]] = []

            def subscribe(self, tokens):
                self.subscribed.append(list(tokens))

            def set_mode(self, mode, tokens):
                pass

        ticker = FakeTicker()
        stream = TickStream(kite_ticker=ticker, journal=journal)
        stream.subscribe([111, 222], {111: "RELIANCE", 222: "INFY"})
        stream.handle_connect()
        assert sorted(ticker.subscribed[-1]) == [111, 222]

    def test_gap_is_journalled_on_reconnect(self, journal, frozen_clock):
        """§9.5: a silent gap looks exactly like a quiet market."""
        frozen_clock(2026, 7, 22, 10, 0)
        stream = TickStream(kite_ticker=None, journal=journal)
        stream._connected = True
        stream.handle_close(1006, "abnormal")
        frozen_clock(2026, 7, 22, 10, 2)
        stream.handle_connect()
        errors = journal.query("SELECT * FROM errors WHERE source='websocket'")
        assert len(errors) == 1
        assert "120s data gap" in errors[0]["message"]

    def test_no_gap_recorded_on_the_first_connect(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        stream = TickStream(kite_ticker=None, journal=journal)
        stream.handle_connect()
        assert journal.query("SELECT * FROM errors WHERE source='websocket'") == []

    def test_backoff_grows_and_caps(self, journal):
        stream = TickStream(kite_ticker=None, journal=journal, max_backoff_seconds=30.0)
        delays = [stream.backoff_delay() for _ in range(10)]
        assert delays[0] < delays[1] < delays[2]
        assert max(delays) == 30.0

    def test_handler_exception_does_not_stop_the_feed(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 10, 0)
        seen: list[str] = []
        stream = TickStream(kite_ticker=None, journal=journal)
        stream.subscribe([111], {111: "RELIANCE"})
        stream.on_tick(lambda t: (_ for _ in ()).throw(RuntimeError("bad handler")))
        stream.on_tick(lambda t: seen.append(t.symbol))
        stream.handle_ticks([{"instrument_token": 111, "last_price": 100.0}])
        assert seen == ["RELIANCE"]


def _mkpath(root, *parts):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
