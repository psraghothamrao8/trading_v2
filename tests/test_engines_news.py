"""Engine tests: §6.1 filings, §6.2 sympathy, §6.6 pead, §6.11 special situations.

The LLM-driven engines. Every LLM call is faked; every spec gate is asserted.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd
import pytest

from core import clock
from core.types import Position, Product, Segment, Side, TTL
from engines.base import Context
from engines.filings import ClassifiedFiling, FilingsEngine, content_hash
from engines.pead import PeadEngine
from engines.special_situations import SpecialSituation, SpecialSituationsEngine
from engines.sympathy import SympathyEngine
from live.alerts import NullAlerts

IST = clock.IST


class FakeLLM:
    """Returns a queued response per call; records payloads."""

    def __init__(self, responses: list[dict] | dict) -> None:
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls: list[tuple[str, dict]] = []

    def classify(self, task: str, payload: dict, schema: dict | None = None):
        self.calls.append((task, payload))
        data = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

        class _Result(dict):
            latency_sec = 1.0

        return _Result(data)


class FakeNSE:
    def __init__(self, announcements: list[dict] | None = None) -> None:
        self._announcements = announcements or []

    def announcements(self, since=None):
        return self._announcements


def intraday_bars(
    base: float = 100.0,
    day: _dt.date = _dt.date(2026, 7, 22),
    bars: int = 60,
    drift: float = 0.0,
) -> pd.DataFrame:
    index = pd.date_range(
        _dt.datetime.combine(day, _dt.time(9, 15), tzinfo=IST), periods=bars, freq="5min"
    )
    values = [base + i * drift for i in range(bars)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 0.6 for v in values],
            "low": [v - 0.6 for v in values],
            "close": values,
            "volume": [10_000] * bars,
        },
        index=index,
    )


def daily_bars(days: int = 60, base: float = 100.0, end: _dt.date = _dt.date(2026, 7, 22),
               volume: int = 100_000) -> pd.DataFrame:
    index = pd.date_range(end - _dt.timedelta(days=days - 1), periods=days, freq="D", tz=IST)
    values = [base + i * 0.1 for i in range(days)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": values,
            "volume": [float(volume)] * days,
        },
        index=index,
    )


def make_filing(**kw) -> ClassifiedFiling:
    defaults = dict(
        symbol="RELIANCE", announcement_id="A1", content_hash="h1",
        headline="Order win", body="big order",
        timestamp=_dt.datetime(2026, 7, 22, 10, 0, tzinfo=IST),
        label="MATERIAL_POSITIVE", confidence=0.9, reason="large order",
        est_revenue_impact_pct=10.0, latency_sec=1.0,
    )
    defaults.update(kw)
    return ClassifiedFiling(**defaults)


# ===========================================================================
# §6.1 filings
# ===========================================================================


class TestFilingsClassification:
    def _engine(self, journal, announcements, llm_response, auto_trade=True):
        engine = FilingsEngine(
            nse=FakeNSE(announcements),
            llm=FakeLLM(llm_response),
            alerts=NullAlerts(),
            journal=journal,
        )
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def test_content_hash_is_stable_and_content_sensitive(self):
        assert content_hash("A", "h", "b") == content_hash("A", "h", "b")
        assert content_hash("A", "h", "b") != content_hash("A", "h", "b2")

    def test_material_filing_is_journalled_and_alerted(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        engine = FilingsEngine(
            nse=FakeNSE([{"symbol": "INFY", "headline": "Order win", "body": "big",
                          "announcement_id": "A1", "timestamp": now}]),
            llm=FakeLLM({"label": "MATERIAL_POSITIVE", "confidence": 0.92,
                         "reason": "big order"}),
            alerts=alerts, journal=journal,
        )
        ctx = Context(now=now, journal=journal)
        material = engine.poll(ctx)

        assert len(material) == 1
        assert journal.query("SELECT * FROM announcements")[0]["label"] == "MATERIAL_POSITIVE"
        assert any("MATERIAL_POSITIVE" in m for m in alerts.sent_messages)

    def test_noise_is_journalled_but_not_alerted(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        engine = FilingsEngine(
            nse=FakeNSE([{"symbol": "TCS", "headline": "Board meeting", "body": "notice",
                          "announcement_id": "A2", "timestamp": now}]),
            llm=FakeLLM({"label": "NOISE", "confidence": 0.95, "reason": "routine"}),
            alerts=alerts, journal=journal,
        )
        assert engine.poll(Context(now=now, journal=journal)) == []
        assert len(journal.query("SELECT * FROM announcements")) == 1
        assert alerts.sent_messages == []

    def test_duplicate_announcement_is_not_alerted_twice(self, journal, frozen_clock):
        """§6.1: dedupe by announcement ID + content hash."""
        now = frozen_clock(2026, 7, 22, 10, 0)
        alerts = NullAlerts()
        row = {"symbol": "INFY", "headline": "Order win", "body": "big",
               "announcement_id": "A1", "timestamp": now}
        engine = FilingsEngine(
            nse=FakeNSE([row]),
            llm=FakeLLM({"label": "MATERIAL_POSITIVE", "confidence": 0.9, "reason": "r"}),
            alerts=alerts, journal=journal,
        )
        ctx = Context(now=now, journal=journal)
        engine.poll(ctx)
        engine.poll(ctx)
        assert len(alerts.sent_messages) == 1

    def test_poll_outside_the_window_does_nothing(self, journal, frozen_clock):
        """§6.1: poll 08:00-15:35 IST."""
        now = frozen_clock(2026, 7, 22, 16, 30)
        engine = FilingsEngine(
            nse=FakeNSE([{"symbol": "INFY", "headline": "x", "announcement_id": "A1"}]),
            llm=FakeLLM({"label": "MATERIAL_POSITIVE", "confidence": 0.9, "reason": "r"}),
            alerts=NullAlerts(), journal=journal,
        )
        assert engine.poll(Context(now=now, journal=journal)) == []


class TestFilingsTradeGate:
    """§6.1: NIFTY-200, confidence >= 0.8, 09:20-14:30, gap <= 5%, age <= 10 min."""

    def _engine(self, journal, auto_trade=True):
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        return engine

    def _ctx(self, journal, now, symbol="RELIANCE", price=100.0):
        return Context(
            now=now,
            bars={(symbol, "5minute"): intraday_bars()},
            prices={symbol: price},
            journal=journal,
        )

    def test_signal_produced_when_every_gate_passes(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        signals = [engine.build_signal(make_filing(timestamp=now), self._ctx(journal, now))]
        assert signals[0] is not None
        assert signals[0].side is Side.BUY
        assert signals[0].stop < 100.0

    def test_negative_filing_becomes_a_short(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        signal = engine.build_signal(
            make_filing(label="MATERIAL_NEGATIVE", timestamp=now), self._ctx(journal, now)
        )
        assert signal.side is Side.SELL
        assert signal.stop > 100.0
        assert signal.ttl is TTL.INTRADAY      # §6.1: NEGATIVE => MIS short

    def test_auto_trade_off_means_alert_only(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal, auto_trade=False)
        assert engine.build_signal(make_filing(timestamp=now), self._ctx(journal, now)) is None

    def test_low_confidence_is_rejected(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        assert engine.build_signal(
            make_filing(confidence=0.75, timestamp=now), self._ctx(journal, now)
        ) is None

    def test_symbol_outside_nifty200_is_alert_only(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        ctx = self._ctx(journal, now, symbol="SOMESMALLCAP")
        assert engine.build_signal(
            make_filing(symbol="SOMESMALLCAP", timestamp=now), ctx
        ) is None

    def test_outside_the_trade_window(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 14, 45)
        engine = self._engine(journal)
        assert engine.build_signal(make_filing(timestamp=now), self._ctx(journal, now)) is None

    def test_filing_older_than_ten_minutes_is_skipped(self, journal, frozen_clock):
        """§6.1 NOTE: never trade a filing older than 10 minutes -- the edge is gone."""
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal)
        stale = make_filing(timestamp=now - _dt.timedelta(minutes=15))
        assert engine.build_signal(stale, self._ctx(journal, now)) is None

    def test_slow_classification_is_alerted_but_not_traded(self, journal, frozen_clock):
        """§6.1 NOTE: latency > 20s -> alert anyway, skip auto-trade."""
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        slow = make_filing(timestamp=now, latency_sec=25.0)
        assert engine.build_signal(slow, self._ctx(journal, now)) is None

    def test_already_gapped_more_than_five_percent_is_skipped(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        # bars sit at 100; a price of 110 is a 10% move since the filing.
        ctx = self._ctx(journal, now, price=110.0)
        assert engine.build_signal(make_filing(timestamp=now), ctx) is None

    def test_swing_hold_moves_the_stop_to_entry(self, journal, frozen_clock):
        """§6.1: POSITIVE trades may carry swing_hold -> hold <=5 sessions, stop at entry."""
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        engine.config._data["swing_hold"] = True        # type: ignore[attr-defined]
        signal = engine.build_signal(make_filing(timestamp=now), self._ctx(journal, now))
        assert signal.ttl is TTL.SWING
        assert signal.stop == pytest.approx(100.0)

    def test_swing_hold_never_applies_to_a_negative_filing(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 5)
        engine = self._engine(journal)
        engine.config._data["swing_hold"] = True        # type: ignore[attr-defined]
        signal = engine.build_signal(
            make_filing(label="MATERIAL_NEGATIVE", timestamp=now), self._ctx(journal, now)
        )
        assert signal.ttl is TTL.INTRADAY


class TestFilingsStop:
    """§6.1: stop = entry -/+ 1.5xATR(5m,14), TIGHTENED to VWAP if VWAP is nearer."""

    def _stop_for(self, journal, now, frame, price, vwap_override=None, monkeypatch=None):
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        engine.config._data["auto_trade"] = True         # type: ignore[attr-defined]
        if vwap_override is not None:
            monkeypatch.setattr(
                "engines.filings.vwap",
                lambda f: pd.Series([vwap_override] * len(f), index=f.index),
            )
        ctx = Context(now=now, bars={("RELIANCE", "5minute"): frame},
                      prices={"RELIANCE": price}, journal=journal)
        return engine.build_signal(make_filing(timestamp=now), ctx)

    def test_vwap_replaces_the_atr_stop_when_it_is_nearer(self, journal, frozen_clock,
                                                          monkeypatch):
        """§6.1: 'tightened to VWAP if VWAP is nearer'."""
        from core.datafeed import atr

        now = frozen_clock(2026, 7, 22, 12, 0)
        frame = intraday_bars(base=100.0, drift=0.1)
        price = float(frame["close"].iloc[-1])
        atr_stop = price - 1.5 * float(atr(frame, 14).iloc[-1])
        nearer = (atr_stop + price) / 2.0        # between the ATR stop and price

        signal = self._stop_for(journal, now, frame, price, nearer, monkeypatch)
        assert signal.stop == pytest.approx(round(nearer, 2))

    def test_vwap_is_ignored_when_it_would_loosen_the_stop(self, journal, frozen_clock,
                                                           monkeypatch):
        """A looser stop would silently increase size via the §3 sizing formula."""
        from core.datafeed import atr

        now = frozen_clock(2026, 7, 22, 12, 0)
        frame = intraday_bars(base=100.0, drift=0.1)
        price = float(frame["close"].iloc[-1])
        atr_stop = price - 1.5 * float(atr(frame, 14).iloc[-1])
        looser = atr_stop - 5.0                  # further from price than the ATR stop

        signal = self._stop_for(journal, now, frame, price, looser, monkeypatch)
        assert signal.stop == pytest.approx(round(atr_stop, 2))

    def test_short_stop_is_mirrored_above_price(self, journal, frozen_clock, monkeypatch):
        """§6.1: NEGATIVE => MIS short, mirrored stop above VWAP."""
        now = frozen_clock(2026, 7, 22, 12, 0)
        frame = intraday_bars(base=100.0, drift=0.1)
        price = float(frame["close"].iloc[-1])
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        engine.config._data["auto_trade"] = True         # type: ignore[attr-defined]
        ctx = Context(now=now, bars={("RELIANCE", "5minute"): frame},
                      prices={"RELIANCE": price}, journal=journal)
        signal = engine.build_signal(
            make_filing(label="MATERIAL_NEGATIVE", timestamp=now), ctx
        )
        assert signal.stop > price


class TestFilingsManage:
    def test_books_half_at_one_r(self, journal, frozen_clock):
        """§6.1: book 50% at +1R."""
        now = frozen_clock(2026, 7, 22, 11, 0)
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        position = Position("RELIANCE", 40, 100.0, "filings", Product.MIS,
                            stop=95.0, last_price=105.0,
                            segment=Segment.EQUITY_INTRADAY)
        ctx = Context(now=now, positions=[position], prices={"RELIANCE": 105.0},
                      bars={("RELIANCE", "5minute"): intraday_bars()}, journal=journal)
        signals = engine.manage(ctx)
        scale_out = [s for s in signals if "+1R" in s.reason]
        assert len(scale_out) == 1
        assert scale_out[0].meta["quantity"] == 20

    def test_scale_out_happens_only_once(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 11, 0)
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        position = Position("RELIANCE", 40, 100.0, "filings", Product.MIS,
                            stop=95.0, last_price=105.0, segment=Segment.EQUITY_INTRADAY)
        ctx = Context(now=now, positions=[position], prices={"RELIANCE": 105.0},
                      bars={("RELIANCE", "5minute"): intraday_bars()}, journal=journal)
        engine.manage(ctx)
        second = [s for s in engine.manage(ctx) if "+1R" in s.reason]
        assert second == []

    def test_vwap_break_exits_the_rest(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 11, 0)
        engine = FilingsEngine(nse=FakeNSE(), llm=FakeLLM({}), alerts=NullAlerts(),
                               journal=journal)
        frame = intraday_bars(base=100.0, drift=0.5)
        position = Position("RELIANCE", 40, 100.0, "filings", Product.MIS,
                            stop=95.0, last_price=50.0, segment=Segment.EQUITY_INTRADAY)
        ctx = Context(now=now, positions=[position], prices={"RELIANCE": 50.0},
                      bars={("RELIANCE", "5minute"): frame}, journal=journal)
        assert any("VWAP trail broken" in s.reason for s in engine.manage(ctx))


# ===========================================================================
# §6.2 sympathy
# ===========================================================================


class TestSympathy:
    def _engine(self, journal, response, auto_trade=True):
        engine = SympathyEngine(llm=FakeLLM(response), journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        engine.universe = lambda: ["INFY", "TCS", "MOTHERSON"]   # type: ignore[method-assign]
        return engine

    def _ctx(self, journal, now, prices, primary_move=5.0):
        bars_map = {}
        for symbol in prices:
            bars_map[(symbol, "5minute")] = intraday_bars(base=100.0)
            bars_map[(symbol, "day")] = daily_bars(volume=5_000_000)
        return Context(now=now, bars=bars_map, prices=prices, journal=journal)

    def test_only_triggers_from_a_material_filing(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "supplies"}]})
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 100.5})
        assert engine.on_filing(make_filing(label="NOISE"), ctx) == []

    def test_signal_when_every_gate_passes(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "supplies"}]})
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 100.5})
        signals = engine.on_filing(make_filing(), ctx)
        assert len(signals) == 1
        assert signals[0].symbol == "INFY"
        assert signals[0].side is Side.BUY

    def test_low_confidence_is_rejected(self, journal, frozen_clock):
        """§6.2: confidence >= 0.7."""
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.5, "reason": "supplies"}]})
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 100.5})
        assert engine.on_filing(make_filing(), ctx) == []

    def test_name_that_already_moved_is_rejected(self, journal, frozen_clock):
        """§6.2: its move must be < 1/3 of the primary's -- else the crowd got there."""
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "supplies"}]})
        # primary +5%, INFY +4% -> 4 >= 5 * 1/3
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 104.0})
        assert engine.on_filing(make_filing(), ctx) == []

    def test_illiquid_name_is_rejected(self, journal, frozen_clock):
        """§6.2: avg daily turnover > Rs 25 cr."""
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "supplies"}]})
        ctx = Context(
            now=now,
            bars={("INFY", "5minute"): intraday_bars(),
                  ("INFY", "day"): daily_bars(volume=100),      # ~tiny turnover
                  ("RELIANCE", "5minute"): intraday_bars()},
            prices={"RELIANCE": 105.0, "INFY": 100.5},
            journal=journal,
        )
        assert engine.on_filing(make_filing(), ctx) == []

    def test_max_one_trade_per_filing(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "a"},
            {"symbol": "TCS", "relation": "peer", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "b"}]})
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 100.5, "TCS": 100.5})
        assert len(engine.on_filing(make_filing(), ctx)) == 1

    def test_max_two_trades_per_day(self, journal, frozen_clock):
        """§6.2: 2 per day."""
        now = frozen_clock(2026, 7, 22, 10, 30)
        engine = self._engine(journal, {"related": [
            {"symbol": "INFY", "relation": "supplier", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "a"}]})
        ctx = self._ctx(journal, now, {"RELIANCE": 105.0, "INFY": 100.5, "TCS": 100.5})
        engine._day = ctx.today                 # already "today", so no reset
        engine._traded_today = {"A", "B"}
        assert engine.on_filing(make_filing(), ctx) == []

    def test_the_subject_itself_is_never_returned(self, journal, frozen_clock):
        engine = self._engine(journal, {"related": [
            {"symbol": "RELIANCE", "relation": "peer", "direction": "POSITIVE",
             "confidence": 0.9, "reason": "self"}]})
        assert engine.find_related(make_filing()) == []


# ===========================================================================
# §6.6 pead
# ===========================================================================


class TestPead:
    def _engine(self, journal, tone_response, auto_trade=True):
        engine = PeadEngine(llm=FakeLLM(tone_response), journal=journal)
        engine.config._data["auto_trade"] = auto_trade      # type: ignore[attr-defined]
        engine.universe = lambda: ["INFY"]                  # type: ignore[method-assign]
        return engine

    def _gap_bars(self, gap_pct: float = 5.0, volume_multiple: float = 3.0):
        frame = daily_bars(days=40, base=100.0)
        previous_close = float(frame["close"].iloc[-2])
        frame.iloc[-1, frame.columns.get_loc("open")] = previous_close * (1 + gap_pct / 100)
        frame.iloc[-1, frame.columns.get_loc("close")] = previous_close * (1 + gap_pct / 100)
        frame.iloc[-1, frame.columns.get_loc("high")] = previous_close * (1 + gap_pct / 100) + 1
        frame.iloc[-1, frame.columns.get_loc("volume")] = 100_000 * volume_multiple
        return frame

    def test_detects_a_gap_with_volume(self, journal, frozen_clock):
        """§6.6: gap >= +3% with volume >= 2x the 20-day average."""
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 8, "confidence": 0.9, "reason": "r"})
        ctx = Context(now=now, bars={("INFY", "day"): self._gap_bars()}, journal=journal)
        found = engine.detect(ctx)
        assert len(found) == 1
        assert found[0].gap_pct == pytest.approx(5.0, abs=0.2)

    def test_small_gap_is_ignored(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 8, "confidence": 0.9, "reason": "r"})
        ctx = Context(now=now, bars={("INFY", "day"): self._gap_bars(gap_pct=1.0)},
                      journal=journal)
        assert engine.detect(ctx) == []

    def test_low_volume_gap_is_ignored(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 8, "confidence": 0.9, "reason": "r"})
        ctx = Context(now=now,
                      bars={("INFY", "day"): self._gap_bars(volume_multiple=1.1)},
                      journal=journal)
        assert engine.detect(ctx) == []

    def test_low_tone_keeps_it_off_the_watchlist(self, journal, frozen_clock):
        """§6.6: require tone >= 7. A beat management won't back does not drift."""
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 4, "confidence": 0.9, "reason": "hedged"})
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="INFY",
                                    headline="Results", body="commentary",
                                    trade_date=now.date().isoformat())
        ctx = Context(now=now, bars={("INFY", "day"): self._gap_bars()}, journal=journal)
        engine.on_schedule(ctx)
        assert engine.watchlist == {}

    def test_no_transcript_means_no_score_and_no_trade(self, journal, frozen_clock):
        """Refusing to score is a valid outcome; a guessed tone is not."""
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 9, "confidence": 0.9, "reason": "r"})
        ctx = Context(now=now, bars={("INFY", "day"): self._gap_bars()}, journal=journal)
        engine.on_schedule(ctx)
        assert engine.watchlist == {}

    def test_high_tone_goes_on_the_watchlist(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 9, "confidence": 0.9, "reason": "raised"})
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="INFY",
                                    headline="Results", body="Guidance raised to 20%",
                                    trade_date=now.date().isoformat())
        ctx = Context(now=now, bars={("INFY", "day"): self._gap_bars()}, journal=journal)
        engine.on_schedule(ctx)
        assert "INFY" in engine.watchlist
        assert engine.watchlist["INFY"].tone == 9

    def test_quarter_label_uses_the_indian_fiscal_year(self):
        assert PeadEngine._quarter_label(_dt.date(2026, 5, 15)) == "Q1FY27"
        assert PeadEngine._quarter_label(_dt.date(2026, 1, 15)) == "Q4FY26"

    def test_time_exit_after_thirty_sessions(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 15, 0)
        engine = self._engine(journal, {"tone": 9, "confidence": 0.9, "reason": "r"})
        position = Position("INFY", 10, 100.0, "pead", Product.CNC, ttl=TTL.SWING,
                            opened_at=_dt.datetime(2026, 3, 2, 10, 0, tzinfo=IST),
                            last_price=105.0, segment=Segment.EQUITY_DELIVERY)
        ctx = Context(now=now, positions=[position], prices={"INFY": 105.0},
                      bars={("INFY", "day"): daily_bars()}, journal=journal)
        assert any("time exit" in s.reason for s in engine.manage(ctx))


# ===========================================================================
# §6.11 special situations
# ===========================================================================


class TestSpecialSituations:
    def test_economics_are_computed_not_guessed(self):
        situation = SpecialSituation(
            symbol="X", event_type="buyback", offer_price=1850.0, market_price=1600.0,
            record_date="2026-06-12", offer_size_shares=12_000_000,
            total_shares=500_000_000, reserved_retail_pct=15.0,
            summary="s", confidence=0.9,
        )
        assert situation.premium_pct == pytest.approx(15.625, abs=0.01)
        assert situation.retail_entitlement_shares == 125       # 200000 // 1600
        assert situation.acceptance_ratio == pytest.approx(0.036, abs=0.001)
        assert situation.expected_value_inr is not None

    def test_missing_inputs_yield_none_not_a_made_up_number(self):
        situation = SpecialSituation(
            symbol="X", event_type="buyback", offer_price=None, market_price=None,
            record_date=None, offer_size_shares=None, total_shares=None,
            reserved_retail_pct=None, summary="s", confidence=0.5,
        )
        assert situation.premium_pct is None
        assert situation.acceptance_ratio is None
        assert situation.expected_value_inr is None
        assert situation.retail_entitlement_shares is None

    def test_alert_says_not_determinable_rather_than_inventing(self, journal):
        engine = SpecialSituationsEngine(llm=FakeLLM({}), alerts=NullAlerts(), journal=journal)
        text = engine.format_alert(SpecialSituation(
            symbol="X", event_type="buyback", offer_price=100.0, market_price=None,
            record_date=None, offer_size_shares=None, total_shares=None,
            reserved_retail_pct=None, summary="s", confidence=0.7,
        ))
        assert "not determinable" in text
        assert "ALERT ONLY" in text

    def test_keyword_filter_runs_before_the_llm(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 20, 30)
        llm = FakeLLM({"event_type": "buyback", "confidence": 0.9, "offer_price": 100.0,
                       "summary": "s"})
        engine = SpecialSituationsEngine(llm=llm, alerts=NullAlerts(), journal=journal)
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="X",
                                    headline="Buyback approved", body="tender offer",
                                    trade_date=now.date().isoformat())
        journal.record_announcement(announcement_id="A2", content_hash="h2", symbol="Y",
                                    headline="Analyst meet", body="routine",
                                    trade_date=now.date().isoformat())
        ctx = Context(now=now, journal=journal, prices={"X": 90.0})
        engine.scan(ctx)
        assert len(llm.calls) == 1, "the routine filing must never reach the model"

    def test_same_announcement_is_only_scanned_once(self, journal, frozen_clock):
        now = frozen_clock(2026, 7, 22, 20, 30)
        llm = FakeLLM({"event_type": "buyback", "confidence": 0.9, "summary": "s"})
        engine = SpecialSituationsEngine(llm=llm, alerts=NullAlerts(), journal=journal)
        journal.record_announcement(announcement_id="A1", content_hash="h1", symbol="X",
                                    headline="Buyback approved", body="tender",
                                    trade_date=now.date().isoformat())
        ctx = Context(now=now, journal=journal, prices={"X": 90.0})
        engine.scan(ctx)
        engine.scan(ctx)
        assert len(llm.calls) == 1
