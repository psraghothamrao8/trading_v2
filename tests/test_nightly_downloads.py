"""Nightly download tests — §6.9 flows, §6.10 surveillance diff, §7 20:30 job."""

from __future__ import annotations

import pytest

from live.alerts import NullAlerts
from scripts.nightly_downloads import _diff, run_nightly


class FakeNSE:
    """Stands in for NSEClient. ``fail`` names methods that should blow up."""

    def __init__(self, fail: set[str] | None = None, **overrides) -> None:
        self.fail = fail or set()
        self.overrides = overrides

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} endpoint down")

    def surveillance_lists(self):
        self._maybe_fail("surveillance_lists")
        return self.overrides.get("surveillance", {
            "asm": [{"symbol": "ABC", "stage": "ST-I"}],
            "gsm": [],
        })

    def fno_ban_list(self):
        self._maybe_fail("fno_ban_list")
        return self.overrides.get("ban", {"IDEA"})

    def fii_dii_cash(self):
        self._maybe_fail("fii_dii_cash")
        return self.overrides.get("cash", [
            {"category": "FII/FPI", "net_value_cr": -1200.0},
            {"category": "DII", "net_value_cr": 900.0},
        ])

    def fii_derivatives(self):
        self._maybe_fail("fii_derivatives")
        return self.overrides.get("derivatives", [
            {"client_type": "FII", "future_index_long": 30000.0, "future_index_short": 70000.0},
        ])

    def bulk_deals(self):
        self._maybe_fail("bulk_deals")
        return []

    def block_deals(self):
        self._maybe_fail("block_deals")
        return []

    def india_vix(self):
        self._maybe_fail("india_vix")
        return self.overrides.get("vix", 13.4)


class TestDiff:
    def test_additions_and_removals(self):
        added, removed = _diff({"A", "B"}, {"B", "C"})
        assert added == ["C"]
        assert removed == ["A"]

    def test_no_change(self):
        assert _diff({"A"}, {"A"}) == ([], [])


class TestNightlyRun:
    def test_all_fetchers_run(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        result = run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        assert result.all_ok
        assert set(result.ok) == {
            "surveillance", "fno_ban", "flows", "bulk_deals", "block_deals", "india_vix"
        }

    def test_one_failing_endpoint_does_not_stop_the_rest(self, journal, frozen_clock):
        """§8.2: never silently degrade -- but never abort the whole job either."""
        frozen_clock(2026, 7, 22, 20, 30)
        result = run_nightly(
            client=FakeNSE(fail={"fii_derivatives"}), journal=journal, alerts=NullAlerts()
        )
        assert "flows" in result.failed
        assert "surveillance" in result.ok
        assert not result.all_ok

    def test_failures_are_alerted(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        alerts = NullAlerts()
        run_nightly(client=FakeNSE(fail={"bulk_deals"}), journal=journal, alerts=alerts)
        assert any("Nightly download failures" in m for m in alerts.sent_messages)

    def test_surveillance_snapshot_is_stored(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        assert journal.surveillance_symbols("2026-07-22", ["asm"]) == {"ABC"}
        assert journal.surveillance_symbols("2026-07-22", ["fno_ban"]) == {"IDEA"}

    def test_surveillance_diff_is_alerted(self, journal, frozen_clock):
        """§6.10 is ALERT-ONLY: the diff is the whole product."""
        frozen_clock(2026, 7, 21, 20, 30)
        run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())

        frozen_clock(2026, 7, 22, 20, 30)
        alerts = NullAlerts()
        client = FakeNSE(surveillance={"asm": [{"symbol": "XYZ", "stage": "ST-II"}], "gsm": []})
        result = run_nightly(client=client, journal=journal, alerts=alerts)

        assert result.surveillance_added["asm"] == ["XYZ"]
        assert result.surveillance_removed["asm"] == ["ABC"]
        assert any("SURVEILLANCE DIFF" in m for m in alerts.sent_messages)

    def test_no_diff_no_alert(self, journal, frozen_clock):
        frozen_clock(2026, 7, 21, 20, 30)
        run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        frozen_clock(2026, 7, 22, 20, 30)
        alerts = NullAlerts()
        run_nightly(client=FakeNSE(), journal=journal, alerts=alerts)
        assert not any("SURVEILLANCE DIFF" in m for m in alerts.sent_messages)

    def test_dry_run_writes_nothing(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts(), dry_run=True)
        assert journal.surveillance_symbols("2026-07-22", ["asm"]) == set()
        assert journal.query("SELECT * FROM flows") == []


class TestFlows:
    def test_long_ratio_computed(self, journal, frozen_clock):
        """§6.9: FII index-futures long share."""
        frozen_clock(2026, 7, 22, 20, 30)
        result = run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        assert result.flows["long_ratio"] == pytest.approx(0.30)

    def test_percentile_needs_history_before_it_means_anything(self, journal, frozen_clock):
        """With <30 stored readings the percentile is None, so §6.9 stays flat."""
        frozen_clock(2026, 7, 22, 20, 30)
        result = run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        assert result.flows["ratio_percentile_3y"] is None

    def test_percentile_computed_once_history_exists(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        for i in range(40):
            journal.record_flows(f"2026-05-{(i % 28) + 1:02d}-{i}", long_ratio=0.40 + i * 0.005)
        result = run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        # 0.30 is below every stored reading -> bottom of the range.
        assert result.flows["ratio_percentile_3y"] == pytest.approx(0.0)

    def test_flows_row_persisted(self, journal, frozen_clock):
        frozen_clock(2026, 7, 22, 20, 30)
        run_nightly(client=FakeNSE(), journal=journal, alerts=NullAlerts())
        row = journal.query("SELECT * FROM flows WHERE trade_date='2026-07-22'")[0]
        assert row["fii_cash_cr"] == -1200.0
        assert row["dii_cash_cr"] == 900.0
