"""NSE client tests — §8.2 scraping etiquette. No network anywhere here."""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from core.nse import NSEClient, NSEError, RateLimiter, _parse_nse_datetime


class FakeResponse:
    def __init__(self, payload: Any = None, status_code: int = 200, text: str = "") -> None:
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return self._payload


class FakeHTTP:
    """Scriptable HTTP double. ``script`` maps a URL fragment to responses."""

    def __init__(self, script: dict[str, list[FakeResponse]] | None = None,
                 default: FakeResponse | None = None) -> None:
        self.script = script or {}
        self.default = default or FakeResponse({})
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> FakeResponse:
        self.calls.append((url, params))
        for fragment, responses in self.script.items():
            if fragment in url:
                if responses:
                    return responses.pop(0)
                return self.default
        return self.default

    def close(self) -> None:
        pass


class RecordingSleeper:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def make_client(script=None, default=None, journal=None, alerts=None):
    from live.alerts import NullAlerts

    http = FakeHTTP(script, default)
    sleeper = RecordingSleeper()
    client = NSEClient(
        http=http, journal=journal, alerts=alerts or NullAlerts(), sleeper=sleeper
    )
    return client, http, sleeper


class TestRateLimiter:
    def test_first_call_does_not_wait(self):
        sleeper = RecordingSleeper()
        limiter = RateLimiter(2.0, sleeper=sleeper)
        assert limiter.wait("x") == 0.0
        assert sleeper.slept == []

    def test_second_call_waits(self):
        sleeper = RecordingSleeper()
        limiter = RateLimiter(2.0, sleeper=sleeper)
        limiter.wait("x")
        delay = limiter.wait("x")
        assert delay > 0
        assert sleeper.slept

    def test_per_endpoint_budgets_are_independent(self):
        sleeper = RecordingSleeper()
        limiter = RateLimiter(2.0, sleeper=sleeper)
        limiter.wait("a")
        assert limiter.wait("b") == 0.0

    def test_override_for_the_announcements_poll(self):
        """§8.2: 30s poll is the one exception to the 2s floor."""
        limiter = RateLimiter(2.0, sleeper=lambda s: None)
        limiter.set_interval("announcements", 30.0)
        limiter.wait("announcements")
        assert limiter.wait("announcements") == pytest.approx(30.0, abs=0.5)


class TestEtiquette:
    def test_warms_up_before_the_first_api_call(self, journal):
        client, http, _ = make_client(journal=journal)
        client.fetch("ban_list")
        # First call is the homepage warm-up, second is the endpoint.
        assert len(http.calls) == 2
        assert http.calls[0][0].rstrip("/").endswith("nseindia.com")

    def test_cookies_are_reused_within_the_ttl(self, journal):
        client, http, _ = make_client(journal=journal)
        client.fetch("ban_list")
        client.fetch("asm_list")
        warmups = [c for c in http.calls if c[0].rstrip("/").endswith("nseindia.com")]
        assert len(warmups) == 1

    def test_401_triggers_a_rewarm_then_succeeds(self, journal):
        script = {"fno-ban-list": [FakeResponse(status_code=401), FakeResponse({"data": []})]}
        client, http, _ = make_client(script=script, journal=journal)
        client.fetch("ban_list")
        warmups = [c for c in http.calls if c[0].rstrip("/").endswith("nseindia.com")]
        assert len(warmups) == 2, "a 401 must force a cookie re-warm (§8.2)"

    def test_403_also_rewarms(self, journal):
        script = {"fno-ban-list": [FakeResponse(status_code=403), FakeResponse({"data": []})]}
        client, http, _ = make_client(script=script, journal=journal)
        client.fetch("ban_list")
        assert client.consecutive_failures == 0

    def test_backoff_is_exponential(self, journal):
        script = {"fno-ban-list": [FakeResponse(status_code=500) for _ in range(4)]}
        client, _, sleeper = make_client(script=script, journal=journal)
        with pytest.raises(NSEError):
            client.fetch("ban_list")
        # Backoff sleeps only; rate-limiter sleeps are zero on a first call.
        backoffs = [s for s in sleeper.slept if s > 0]
        assert len(backoffs) >= 3
        assert backoffs[1] > backoffs[0]

    def test_headers_come_from_config(self, journal):
        client, _, _ = make_client(journal=journal)
        headers = client.settings.get("headers")
        assert "Mozilla" in headers["User-Agent"]
        assert headers["Referer"].startswith("https://www.nseindia.com")


class TestLoudFailure:
    def test_failures_are_journalled(self, journal):
        script = {"fno-ban-list": [FakeResponse(status_code=500) for _ in range(8)]}
        client, _, _ = make_client(script=script, journal=journal)
        with pytest.raises(NSEError):
            client.fetch("ban_list")
        rows = journal.query("SELECT * FROM errors WHERE source='nse'")
        assert len(rows) == 1

    def test_alerts_after_five_consecutive_failures(self, journal):
        """§8.2: fail LOUDLY after 5 consecutive failures. Never degrade quietly."""
        from live.alerts import NullAlerts

        alerts = NullAlerts()
        client, _, _ = make_client(
            default=FakeResponse(status_code=500), journal=journal, alerts=alerts
        )
        for _ in range(5):
            with pytest.raises(NSEError):
                client.fetch("ban_list")
        assert any("consecutive NSE failures" in m for m in alerts.sent_messages)

    def test_no_alert_before_the_threshold(self, journal):
        from live.alerts import NullAlerts

        alerts = NullAlerts()
        client, _, _ = make_client(
            default=FakeResponse(status_code=500), journal=journal, alerts=alerts
        )
        for _ in range(4):
            with pytest.raises(NSEError):
                client.fetch("ban_list")
        assert not any("consecutive NSE failures" in m for m in alerts.sent_messages)

    def test_recovery_is_announced(self, journal):
        from live.alerts import NullAlerts

        alerts = NullAlerts()
        script = {"fno-ban-list": [FakeResponse(status_code=500)] * 20}
        client, http, _ = make_client(script=script, journal=journal, alerts=alerts)
        for _ in range(5):
            with pytest.raises(NSEError):
                client.fetch("ban_list")
        http.script = {}
        http.default = FakeResponse({"data": []})
        client.fetch("ban_list")
        assert any("recovered" in m for m in alerts.sent_messages)


class TestEndpointConfig:
    def test_unknown_endpoint_names_the_config_file(self, journal):
        client, _, _ = make_client(journal=journal)
        with pytest.raises(NSEError, match="nse.endpoints"):
            client.fetch("does_not_exist")

    def test_path_templates_are_filled(self, journal):
        client, http, _ = make_client(journal=journal)
        client.fetch("equity_quote", symbol="RELIANCE")
        assert any("symbol=RELIANCE" in url for url, _ in http.calls)


class TestParsers:
    def test_announcements_normalised(self, journal):
        payload = [{
            "symbol": "infy", "desc": "Order win", "seqId": "123",
            "an_dt": "22-Jul-2026 10:15:00", "attchmntFile": "http://x/y.pdf",
        }]
        client, _, _ = make_client(default=FakeResponse(payload), journal=journal)
        rows = client.announcements()
        assert rows[0]["symbol"] == "INFY"
        assert rows[0]["announcement_id"] == "123"
        assert rows[0]["timestamp"].hour == 10

    def test_announcements_since_filter(self, journal):
        from core import clock

        payload = [
            {"symbol": "A", "seqId": "1", "an_dt": "22-Jul-2026 09:00:00"},
            {"symbol": "B", "seqId": "2", "an_dt": "22-Jul-2026 11:00:00"},
        ]
        client, _, _ = make_client(default=FakeResponse(payload), journal=journal)
        since = _dt.datetime(2026, 7, 22, 10, 0, tzinfo=clock.IST)
        assert [r["symbol"] for r in client.announcements(since=since)] == ["B"]

    def test_ban_list_handles_strings_and_dicts(self, journal):
        client, _, _ = make_client(
            default=FakeResponse({"data": ["idea", {"symbol": "pnb"}]}), journal=journal
        )
        assert client.fno_ban_list() == {"IDEA", "PNB"}

    def test_preopen_snapshot_extracts_imbalance_inputs(self, journal):
        """§6.5 needs indicative price plus buy/sell quantities."""
        payload = {"data": [{
            "metadata": {"symbol": "TCS", "lastPrice": 3900.0, "previousClose": 3850.0,
                         "pChange": 1.3},
            "detail": {"preOpenMarket": {"IEP": 3900.0, "totalBuyQuantity": 30000,
                                         "totalSellQuantity": 8000, "finalQuantity": 5000}},
        }]}
        client, _, _ = make_client(default=FakeResponse(payload), journal=journal)
        row = client.preopen_snapshot()[0]
        assert row["symbol"] == "TCS"
        assert row["total_buy_quantity"] == 30000
        assert row["total_sell_quantity"] == 8000

    def test_equity_quote_exposes_circuit_bands(self, journal):
        """§8.4 -> the §3 kernel veto."""
        payload = {"priceInfo": {"lastPrice": 100.0, "previousClose": 99.0,
                                 "pPriceBand": {"upper": 110.0, "lower": 90.0}}}
        client, _, _ = make_client(default=FakeResponse(payload), journal=journal)
        quote = client.equity_quote("ADANIENT")
        assert quote["upper_band"] == 110.0
        assert quote["lower_band"] == 90.0

    def test_india_vix_extracted_from_all_indices(self, journal):
        payload = {"data": [{"index": "NIFTY 50", "last": 24000.0},
                            {"index": "INDIA VIX", "last": 13.4}]}
        client, _, _ = make_client(default=FakeResponse(payload), journal=journal)
        assert client.india_vix() == 13.4

    def test_surveillance_lists_survive_one_endpoint_failing(self, journal):
        script = {"reportASM": [FakeResponse({"data": [{"symbol": "abc", "stage": "ST-I"}]})],
                  "reportGSM": [FakeResponse(status_code=500)] * 8}
        client, _, _ = make_client(script=script, journal=journal)
        lists = client.surveillance_lists()
        assert lists["asm"] == [{"symbol": "ABC", "stage": "ST-I"}]
        assert lists["gsm"] == []


class TestDatetimeParsing:
    @pytest.mark.parametrize("text,expected_hour", [
        ("22-Jul-2026 10:15:00", 10),
        ("22-Jul-2026 10:15", 10),
        ("2026-07-22 10:15:00", 10),
        ("2026-07-22T10:15:00", 10),
    ])
    def test_known_formats(self, text, expected_hour):
        assert _parse_nse_datetime(text).hour == expected_hour

    def test_unknown_format_returns_none(self):
        assert _parse_nse_datetime("sometime yesterday") is None

    def test_result_is_ist(self):
        assert _parse_nse_datetime("22-Jul-2026 10:15:00").utcoffset() == _dt.timedelta(
            hours=5, minutes=30
        )
