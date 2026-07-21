"""Phase 1 acceptance (§5.1):

    python -m live.orchestrator --status

prints account, config, and regime=NA.
"""

from __future__ import annotations

import io
import json

from live.orchestrator import build_parser, build_status, kill, main, print_status
from core.types import Position, Product


class TestStatus:
    def test_status_reports_regime_na(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        assert status["regime"] == "NA"

    def test_status_reports_the_account(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        assert status["account"]["user_id"] == "PAPER"
        assert status["account"]["equity_available"] == 800_000.0

    def test_status_reports_paper_mode(self, journal, paper_broker, frozen_clock):
        """§0.1: paper is the default and --status says so."""
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        assert status["mode"]["configured"] == "paper"
        assert status["mode"]["effective"] == "paper"

    def test_status_reports_config(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        assert status["risk"]["capital"] == 800000
        assert status["risk"]["entry_cutoff"] == "14:45"
        assert len(status["engines"]) == 11          # §6.1 - §6.11
        assert status["calendar"]["date"] == "2026-07-22"

    def test_status_marks_alert_only_engines(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        assert status["engines"]["surveillance"]["alert_only"] is True
        assert status["engines"]["special_situations"]["alert_only"] is True

    def test_status_never_leaks_secret_values(self, journal, paper_broker, frozen_clock):
        """§0.4: credentials are reported as booleans, never as values."""
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        for value in status["credentials"].values():
            assert isinstance(value, bool)

    def test_printed_status_is_readable(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        out = io.StringIO()
        print_status(build_status(broker=paper_broker, journal=journal), stream=out)
        text = out.getvalue()
        assert "MODE            : PAPER" in text
        assert "REGIME          : NA" in text
        assert "RISK KERNEL (§3)" in text
        assert "ENGINES" in text
        assert "CALENDAR" in text

    def test_json_output_is_serialisable(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        status = build_status(broker=paper_broker, journal=journal)
        json.loads(json.dumps(status, default=str))


class TestCLI:
    def test_parser_accepts_the_acceptance_command(self):
        args = build_parser().parse_args(["--status"])
        assert args.status is True

    def test_main_status_exits_zero(self, journal, capsys, frozen_clock):
        frozen_clock(2026, 7, 22, 9, 0)
        assert main(["--status"]) == 0
        assert "REGIME" in capsys.readouterr().out

    def test_main_with_no_args_prints_help_and_fails(self, capsys):
        assert main([]) == 1
        assert "usage" in capsys.readouterr().out.lower()


class TestKillSwitch:
    def test_kill_flattens_positions_and_journals(self, journal, paper_broker, frozen_clock):
        frozen_clock(2026, 7, 22, 11, 0)
        paper_broker.seed_position(
            Position(symbol="A", quantity=10, average_price=100.0, engine="filings",
                     product=Product.MIS, last_price=101.0)
        )
        result = kill(broker=paper_broker, journal=journal, reason="test kill")
        assert result["positions_flattened"] == 1
        assert paper_broker.positions() == []

        rows = journal.query("SELECT * FROM kill_events")
        assert len(rows) == 1
        assert rows[0]["source"] == "cli"
        assert rows[0]["reason"] == "test kill"

    def test_kill_alerts(self, journal, paper_broker, frozen_clock):
        from live.alerts import NullAlerts

        frozen_clock(2026, 7, 22, 11, 0)
        alerts = NullAlerts()
        kill(broker=paper_broker, journal=journal, alerts=alerts, reason="boom")
        assert any("KILL SWITCH FIRED" in m for m in alerts.sent_messages)
