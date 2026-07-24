"""Telegram alert tests. No network: the transport is faked."""

from __future__ import annotations

from typing import Any

from live.alerts import Alerts, Command, NullAlerts


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeTransport:
    """Records outbound calls; replays a queue of getUpdates payloads."""

    def __init__(self, updates: list[dict[str, Any]] | None = None) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.updates = updates or []

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.posts.append((url, json))
        return FakeResponse({"ok": True})

    def get(self, url: str, params: dict[str, Any]) -> FakeResponse:
        payload = {"result": self.updates}
        self.updates = []
        return FakeResponse(payload)


def make_alerts(updates=None) -> tuple[Alerts, FakeTransport]:
    transport = FakeTransport(updates)
    alerts = Alerts(bot_token="TOKEN", chat_id="12345", transport=transport, enabled=True)
    return alerts, transport


class TestSending:
    def test_send_posts_to_telegram(self):
        alerts, transport = make_alerts()
        assert alerts.send("hello") is True
        url, payload = transport.posts[0]
        assert url.endswith("/sendMessage")
        assert payload["chat_id"] == "12345"
        assert payload["text"] == "hello"

    def test_unconfigured_alerts_log_instead_of_raising(self):
        alerts = Alerts(bot_token=None, chat_id=None, transport=None, enabled=True)
        assert alerts.send("hello") is False
        assert alerts.sent_messages == ["hello"]

    def test_transport_failure_is_swallowed(self):
        class Boom:
            def post(self, *a, **k):
                raise RuntimeError("network down")

        alerts = Alerts(bot_token="T", chat_id="1", transport=Boom(), enabled=True)
        assert alerts.send("hello") is False   # never raises: trading continues

    def test_rate_limit_drops_excess(self):
        alerts, transport = make_alerts()
        alerts.rate_limit_per_minute = 3
        results = [alerts.send(f"m{i}") for i in range(5)]
        assert results.count(True) == 3
        assert len(transport.posts) == 3

    def test_html_is_escaped(self):
        alerts, transport = make_alerts()
        alerts.material_filing("A<B>", "MATERIAL_POSITIVE", 0.9, "reason & more")
        text = transport.posts[0][1]["text"]
        assert "A&lt;B&gt;" in text
        assert "&amp;" in text


class TestFormattedAlerts:
    def test_material_filing_flags_late_classification(self):
        """§6.1 NOTE: latency > 20s -> alert anyway, skip auto-trade."""
        alerts, transport = make_alerts()
        alerts.material_filing("INFY", "MATERIAL_POSITIVE", 0.91, "big order", latency_sec=25.0)
        text = transport.posts[0][1]["text"]
        assert "over budget" in text

    def test_order_placed_entry_includes_stop_and_target(self):
        """A manual trader needs the full instruction, not just symbol/side/qty."""
        alerts, transport = make_alerts()
        alerts.order_placed("filings", "RELIANCE", "BUY", 40, 3005.50, "paper",
                            stop=2900.0, targets=(3110.0,), reason="MATERIAL_POSITIVE (0.94)")
        text = transport.posts[0][1]["text"]
        assert "ENTRY" in text
        assert "2,900.00" in text
        assert "3,110.00" in text
        assert "MATERIAL_POSITIVE" in text

    def test_order_placed_exit_omits_stop_and_target(self):
        """An exit alert should not show a stop/target for a position already closing."""
        alerts, transport = make_alerts()
        alerts.order_placed("filings", "RELIANCE", "SELL", 20, 3110.0, "paper",
                            stop=2900.0, targets=(3110.0,), reason="+1R scale-out (50%)",
                            is_entry=False)
        text = transport.posts[0][1]["text"]
        assert "EXIT" in text
        assert "stop" not in text.lower()
        assert "target" not in text.lower()
        assert "+1R scale-out" in text

    def test_order_placed_without_stop_or_targets_omits_those_lines(self):
        alerts, transport = make_alerts()
        alerts.order_placed("wheel", "RELIANCE26AUG2800PE", "SELL", 500, 45.0, "paper")
        text = transport.posts[0][1]["text"]
        assert "stop" not in text.lower()
        assert "target" not in text.lower()

    def test_order_placed_market_order_shows_mkt(self):
        alerts, transport = make_alerts()
        alerts.order_placed("overnight", "NIFTYBEES", "BUY", 100, None, "paper")
        assert "MKT" in transport.posts[0][1]["text"]

    def test_rejection_alert_carries_the_code(self):
        alerts, transport = make_alerts()
        alerts.rejection("filings", "YESBANK", "ASM_GSM_SURVEILLANCE", "under surveillance")
        text = transport.posts[0][1]["text"]
        assert "ASM_GSM_SURVEILLANCE" in text

    def test_digest_includes_pnl_and_counts(self):
        alerts, transport = make_alerts()
        alerts.digest({
            "trade_date": "2026-07-22", "regime": "TREND", "mode": "paper",
            "net_pnl": 4200.0, "gross_pnl": 4700.0, "costs": 500.0,
            "trades": 3, "wins": 2, "losses": 1, "signals": 9, "rejections": 4,
        })
        text = transport.posts[0][1]["text"]
        assert "DIGEST 2026-07-22" in text
        assert "TREND" in text
        assert "trades 3" in text


class TestCommands:
    def _update(self, text: str, chat_id: str = "12345", update_id: int = 1) -> dict:
        return {
            "update_id": update_id,
            "message": {"message_id": 10, "chat": {"id": chat_id}, "text": text},
        }

    def test_kill_command_dispatches(self):
        fired: list[Command] = []
        alerts, _ = make_alerts([self._update("/kill")])
        alerts.register("/kill", lambda cmd: fired.append(cmd) or "killed")
        handled = alerts.poll_once()
        assert len(handled) == 1
        assert fired[0].name == "/kill"

    def test_command_from_another_chat_is_ignored(self):
        """Only the configured chat may command this system."""
        fired: list[Command] = []
        alerts, _ = make_alerts([self._update("/kill", chat_id="99999")])
        alerts.register("/kill", lambda cmd: fired.append(cmd) or "killed")
        assert alerts.poll_once() == []
        assert fired == []

    def test_disallowed_command_is_rejected(self):
        alerts, transport = make_alerts([self._update("/selleverything")])
        alerts.poll_once()
        assert "Unknown command" in transport.posts[-1][1]["text"]

    def test_handler_exception_does_not_kill_the_poller(self):
        def boom(cmd):
            raise RuntimeError("handler bug")

        alerts, transport = make_alerts([self._update("/kill")])
        alerts.register("/kill", boom)
        handled = alerts.poll_once()
        assert len(handled) == 1
        assert "Command failed" in transport.posts[-1][1]["text"]

    def test_offset_advances_so_updates_are_not_replayed(self):
        alerts, transport = make_alerts([self._update("/status", update_id=7)])
        alerts.register("/status", lambda cmd: "ok")
        alerts.poll_once()
        assert alerts._offset == 8


class TestNullAlerts:
    def test_records_without_sending(self):
        alerts = NullAlerts()
        alerts.send("x")
        assert alerts.sent_messages == ["x"]
        assert alerts.configured is False
