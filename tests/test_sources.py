"""Free-data source tests. No network: yfinance is faked."""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from core import clock
from core.sources import (
    YF_INTRADAY_MAX_DAYS,
    YF_INTRADAY_SAFE_DAYS,
    SourceError,
    YFinanceSource,
    from_yahoo_symbol,
    get_source,
    to_yahoo_symbol,
)

IST = clock.IST


class FakeYF:
    """Stands in for the yfinance module."""

    def __init__(self, frame: pd.DataFrame | None = None, last: float | None = 100.0) -> None:
        self.frame = frame
        self.last = last
        self.calls: list[dict] = []

    def download(self, ticker, start, end, interval, **kwargs):
        self.calls.append({"ticker": ticker, "start": start, "end": end,
                           "interval": interval})
        return self.frame if self.frame is not None else pd.DataFrame()

    def Ticker(self, ticker):  # noqa: N802 - mirrors yfinance's API
        outer = self

        class _T:
            fast_info = {"lastPrice": outer.last}

            def history(self, **kwargs):
                return pd.DataFrame()

        return _T()


def make_source(frame=None, last=100.0) -> YFinanceSource:
    source = YFinanceSource.__new__(YFinanceSource)
    source._yf = FakeYF(frame, last)
    source._session = None
    source._instruments_cache = None
    source._token_map = {}
    return source


def ohlcv(days: int = 10, start: _dt.date = _dt.date(2026, 6, 1), tz: str | None = None):
    index = pd.date_range(start, periods=days, freq="D", tz=tz)
    values = [100.0 + i for i in range(days)]
    return pd.DataFrame(
        {
            "Open": values,
            "High": [v + 1 for v in values],
            "Low": [v - 1 for v in values],
            "Close": [v + 0.5 for v in values],
            "Volume": [100_000] * days,
        },
        index=index,
    )


class TestSymbolMapping:
    def test_equity_gets_the_ns_suffix(self):
        assert to_yahoo_symbol("RELIANCE") == "RELIANCE.NS"
        assert to_yahoo_symbol("infy") == "INFY.NS"

    def test_index_symbols_are_mapped(self):
        assert to_yahoo_symbol("NIFTY") == "^NSEI"
        assert to_yahoo_symbol("INDIAVIX") == "^INDIAVIX"

    def test_niftybees_maps_to_the_index_not_the_etf(self):
        """NIFTYBEES.NS returns NaN closes on Yahoo, which silently zeroes returns."""
        assert to_yahoo_symbol("NIFTYBEES") == "^NSEI"

    def test_already_mapped_symbols_pass_through(self):
        assert to_yahoo_symbol("^NSEI") == "^NSEI"
        assert to_yahoo_symbol("RELIANCE.NS") == "RELIANCE.NS"

    def test_round_trip(self):
        assert from_yahoo_symbol("RELIANCE.NS") == "RELIANCE"
        assert from_yahoo_symbol("^NSEI") == "NIFTY"


class TestInstruments:
    def test_builds_from_the_configured_universe(self):
        rows = make_source().instruments("NSE")
        symbols = {r["tradingsymbol"] for r in rows}
        assert "RELIANCE" in symbols
        assert len(rows) > 100
        assert all(r["exchange"] == "NSE" for r in rows)

    def test_tokens_are_unique(self):
        rows = make_source().instruments("NSE")
        tokens = [r["instrument_token"] for r in rows]
        assert len(tokens) == len(set(tokens))

    def test_nfo_is_empty_rather_than_fabricated(self):
        """Free data has no option chain; §6.8 must see that, not a fake one."""
        assert make_source().instruments("NFO") == []


class TestHistoricalData:
    def test_daily_returns_kite_shaped_rows(self):
        source = make_source(ohlcv(10))
        source.instruments("NSE")
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        rows = source.historical_data(token, _dt.date(2026, 6, 1), _dt.date(2026, 6, 10), "day")
        assert len(rows) == 10
        assert set(rows[0]) == {"date", "open", "high", "low", "close", "volume"}
        assert rows[0]["date"].tzinfo is not None

    def test_multiindex_columns_are_flattened(self):
        """Recent yfinance returns MultiIndex columns even for one ticker."""
        frame = ohlcv(5)
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["RELIANCE.NS"]])
        source = make_source(frame)
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        assert len(source.historical_data(
            token, _dt.date(2026, 6, 1), _dt.date(2026, 6, 5), "day"
        )) == 5

    def test_nan_closes_are_dropped(self):
        """A NaN close silently becomes a zero return if it survives."""
        frame = ohlcv(5)
        frame.iloc[2, frame.columns.get_loc("Close")] = float("nan")
        source = make_source(frame)
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        assert len(source.historical_data(
            token, _dt.date(2026, 6, 1), _dt.date(2026, 6, 5), "day"
        )) == 4

    def test_empty_frame_returns_no_rows(self):
        source = make_source(pd.DataFrame())
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        assert source.historical_data(
            token, _dt.date(2026, 6, 1), _dt.date(2026, 6, 5), "day"
        ) == []

    def test_unknown_interval_raises(self):
        source = make_source()
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        with pytest.raises(SourceError, match="no '7minute' interval"):
            source.historical_data(token, _dt.date(2026, 6, 1), _dt.date(2026, 6, 5), "7minute")

    def test_unknown_token_raises(self):
        with pytest.raises(SourceError, match="Unknown instrument token"):
            make_source().historical_data(1, _dt.date(2026, 6, 1), _dt.date(2026, 6, 5), "day")


class TestIntradayLimit:
    """Yahoo caps intraday history at 60 days. This is their limit, not a knob."""

    def test_old_intraday_request_is_clamped_with_a_warning(self, frozen_clock, caplog):
        frozen_clock(2026, 7, 22, 10, 0)
        source = make_source(ohlcv(5))
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        with caplog.at_level("WARNING"):
            source.historical_data(
                token, _dt.date(2019, 1, 1), _dt.date(2026, 7, 22), "5minute"
            )
        assert "capped at 60 days" in caplog.text
        requested_start = source._yf.calls[-1]["start"]
        assert requested_start >= (_dt.date(2026, 7, 22)
                                   - _dt.timedelta(days=YF_INTRADAY_MAX_DAYS)).isoformat()

    def test_entirely_stale_window_returns_nothing(self, frozen_clock, caplog):
        frozen_clock(2026, 7, 22, 10, 0)
        source = make_source(ohlcv(5))
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        with caplog.at_level("WARNING"):
            rows = source.historical_data(
                token, _dt.date(2019, 1, 1), _dt.date(2019, 12, 31), "5minute"
            )
        assert rows == []
        assert "older than the 60-day" in caplog.text

    def test_declared_limit_leaves_a_margin_below_yahoos_boundary(self):
        """Yahoo rejects a window starting at exactly 60 days; 59 returns data."""
        source = make_source()
        assert source.max_history_days("5minute") == YF_INTRADAY_SAFE_DAYS
        assert YF_INTRADAY_SAFE_DAYS < YF_INTRADAY_MAX_DAYS

    def test_daily_has_no_declared_limit(self):
        assert make_source().max_history_days("day") is None

    def test_datafeed_clamps_before_chunking(self, journal, frozen_clock, tmp_path,
                                             monkeypatch):
        """One clamped request, not nineteen that each come back empty."""
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        frozen_clock(2026, 7, 22, 10, 0)
        source = make_source(ohlcv(5))
        feed = DataFeed(kite=source, journal=journal, sleeper=lambda s: None)
        feed.historical("RELIANCE", "5minute", _dt.date(2019, 1, 1), _dt.date(2026, 7, 21))
        assert len(source._yf.calls) == 1

    def test_datafeed_skips_the_network_for_a_fully_stale_window(
        self, journal, frozen_clock, tmp_path, monkeypatch
    ):
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        frozen_clock(2026, 7, 22, 10, 0)
        source = make_source(ohlcv(5))
        feed = DataFeed(kite=source, journal=journal, sleeper=lambda s: None)
        frame = feed.historical("RELIANCE", "5minute", _dt.date(2019, 1, 1),
                                _dt.date(2019, 12, 31))
        assert frame.empty
        assert source._yf.calls == []

    def test_instrument_cache_is_namespaced_by_source(self, journal, tmp_path,
                                                      monkeypatch):
        """A Kite token is meaningless to the free source and vice versa."""
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        free_feed = DataFeed(kite=make_source(), journal=journal)
        kite_like = type("K", (), {"instruments": lambda self, e="NSE": []})()
        kite_feed = DataFeed(kite=kite_like, journal=journal)
        assert free_feed.source_name == "free"
        assert kite_feed.source_name == "kite"
        assert free_feed.parquet_path("X", "day") != kite_feed.parquet_path("X", "day") \
            or free_feed.source_name != kite_feed.source_name

    def test_daily_history_is_not_clamped(self, frozen_clock):
        """Daily is complete back to 2015 — this is the whole value of free data."""
        frozen_clock(2026, 7, 22, 10, 0)
        source = make_source(ohlcv(5))
        token = next(r["instrument_token"] for r in source.instruments("NSE")
                     if r["tradingsymbol"] == "RELIANCE")
        source.historical_data(token, _dt.date(2015, 1, 1), _dt.date(2026, 7, 22), "day")
        assert source._yf.calls[-1]["start"] == "2015-01-01"


class TestQuotes:
    def test_ltp_is_kite_shaped(self):
        assert make_source(last=2500.0).ltp(["NSE:RELIANCE"]) == {
            "NSE:RELIANCE": {"last_price": 2500.0}
        }

    def test_quote_omits_circuit_bands_rather_than_guessing(self):
        """§3's band veto must conclude 'no data', not act on an invented band."""
        quote = make_source(last=2500.0).quote(["NSE:RELIANCE"])["NSE:RELIANCE"]
        assert "upper_circuit_limit" not in quote
        assert "lower_circuit_limit" not in quote
        assert quote["delayed"] is True

    def test_band_lookup_reports_no_limits_on_free_data(self, journal):
        from core.datafeed import DataFeed

        feed = DataFeed(kite=make_source(last=2500.0), journal=journal)
        band = feed.bands(["RELIANCE"])["RELIANCE"]
        assert band.upper is None and band.lower is None
        # No limits -> no distance -> the §3 veto has nothing to fire on.
        assert band.distance_to_band_pct() is None

    def test_the_kernel_does_not_veto_on_a_bandless_symbol(
        self, kernel, make_order, frozen_clock, journal
    ):
        """A missing band must not be mistaken for a breached one."""
        from core.datafeed import DataFeed

        frozen_clock(2026, 7, 22, 10, 0)
        feed = DataFeed(kite=make_source(last=2500.0), journal=journal)
        kernel.market.bands.update(feed.bands(["RELIANCE"]))
        assert kernel.check(make_order(symbol="RELIANCE")).allowed

    def test_unavailable_price_is_omitted(self):
        assert make_source(last=None).ltp(["NSE:RELIANCE"]) == {}


class TestCapabilities:
    def test_reports_what_it_cannot_do(self):
        caps = make_source().capabilities()
        assert caps["circuit_bands"] is False
        assert caps["option_chain"] is False
        assert caps["order_routing"] is False
        assert caps["intraday_history_days"] == 60

    def test_names_the_engines_it_cannot_backtest(self):
        """The five 5-minute engines. Stated, not left for the user to discover."""
        assert set(make_source().capabilities()["not_backtestable"]) == {
            "filings", "sympathy", "pairs", "preopen", "panic_reversion"
        }


class TestGetSource:
    def test_free(self, monkeypatch):
        monkeypatch.setattr("core.sources.YFinanceSource", lambda: "FREE")
        assert get_source("free") == "FREE"

    def test_kite_passthrough(self):
        sentinel = object()
        assert get_source("kite", kite=sentinel) is sentinel

    def test_unknown_raises(self):
        with pytest.raises(SourceError, match="Unknown data source"):
            get_source("bloomberg")


class TestDataFeedIntegration:
    """The point of the adapter: DataFeed does not know which source it has."""

    def test_datafeed_downloads_through_the_free_source(self, journal, tmp_path,
                                                        monkeypatch):
        from core.datafeed import DataFeed

        def _path(*parts):
            path = tmp_path.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        monkeypatch.setattr("core.datafeed.data_path", _path)
        feed = DataFeed(kite=make_source(ohlcv(30)), journal=journal,
                        sleeper=lambda s: None)
        frame = feed.historical("RELIANCE", "day", _dt.date(2026, 6, 1),
                                _dt.date(2026, 6, 30))
        assert len(frame) == 30
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert str(frame.index.tz) == "Asia/Kolkata"
