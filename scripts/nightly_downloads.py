"""Nightly NSE downloads. Implements the 20:30 scheduler job (§7).

    python scripts/nightly_downloads.py
    python scripts/nightly_downloads.py --dry-run

Pulls, journals and diffs:
  * ASM / GSM surveillance lists and the F&O ban list  (§6.10, feeds the §3 veto)
  * FII/DII cash and FII index-futures positioning     (§6.9)
  * bulk and block deals                                (§5.2)
  * India VIX                                           (§6.8, §7)

Each fetcher is independent: one failing endpoint does not stop the others, and
every failure is journalled and surfaced (§8.2 -- never silently degrade).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` as well as `python -m scripts.<name>`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from core import clock
from core.datafeed import percentile_rank
from core.journal import Journal, get_journal
from core.logging_config import setup_logging
from core.nse import NSEClient, NSEError

log = logging.getLogger(__name__)


@dataclass
class NightlyResult:
    """What ran, what worked, and what changed -- the digest input."""

    trade_date: str
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    surveillance_added: dict[str, list[str]] = field(default_factory=dict)
    surveillance_removed: dict[str, list[str]] = field(default_factory=dict)
    ban_added: list[str] = field(default_factory=list)
    ban_removed: list[str] = field(default_factory=list)
    flows: dict[str, Any] = field(default_factory=dict)
    india_vix: float | None = None

    @property
    def all_ok(self) -> bool:
        return not self.failed


def _diff(previous: set[str], current: set[str]) -> tuple[list[str], list[str]]:
    return sorted(current - previous), sorted(previous - current)


def run_nightly(
    client: NSEClient | None = None,
    journal: Journal | None = None,
    alerts: Any = None,
    dry_run: bool = False,
) -> NightlyResult:
    """Run every nightly fetcher. Returns a structured result for the digest."""
    journal = journal or get_journal()
    client = client or NSEClient(journal=journal, alerts=alerts)
    if alerts is None:
        from live.alerts import get_alerts

        alerts = get_alerts()

    today = clock.today_ist().isoformat()
    result = NightlyResult(trade_date=today)

    def step(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            result.ok.append(name)
        except (NSEError, Exception) as exc:  # noqa: B902 - isolate each fetcher
            result.failed[name] = str(exc)
            log.error("nightly %s failed: %s", name, exc)
            journal.record_error("nightly", f"{name}: {exc}", severity="ERROR")

    # --- surveillance lists (§6.10) -> feeds the §3 veto ------------------
    def surveillance() -> None:
        lists = client.surveillance_lists()
        for list_name, rows in lists.items():
            previous_date = journal.latest_surveillance_date(list_name)
            previous = (
                journal.surveillance_symbols(previous_date, [list_name]) if previous_date else set()
            )
            current = {r["symbol"] for r in rows}
            added, removed = _diff(previous, current)
            result.surveillance_added[list_name] = added
            result.surveillance_removed[list_name] = removed
            if not dry_run:
                journal.record_surveillance_snapshot(today, list_name, rows)

    step("surveillance", surveillance)

    # --- F&O ban list -> §3 derivatives veto -----------------------------
    def ban_list() -> None:
        previous_date = journal.latest_surveillance_date("fno_ban")
        previous = (
            journal.surveillance_symbols(previous_date, ["fno_ban"]) if previous_date else set()
        )
        current = client.fno_ban_list()
        result.ban_added, result.ban_removed = _diff(previous, current)
        if not dry_run:
            journal.record_surveillance_snapshot(
                today, "fno_ban", [{"symbol": s} for s in sorted(current)]
            )

    step("fno_ban", ban_list)

    # --- FII/DII flows (§6.9) --------------------------------------------
    def flows() -> None:
        cash = client.fii_dii_cash()
        fii_cash = next((r["net_value_cr"] for r in cash if r["category"].upper().startswith("FII")), None)
        dii_cash = next((r["net_value_cr"] for r in cash if r["category"].upper().startswith("DII")), None)

        derivatives = client.fii_derivatives()
        fii_row = next(
            (r for r in derivatives if r["client_type"].upper() in ("FII", "FPI")), None
        )
        long_oi = fii_row["future_index_long"] if fii_row else None
        short_oi = fii_row["future_index_short"] if fii_row else None
        ratio = None
        if long_oi is not None and short_oi is not None and (long_oi + short_oi):
            ratio = long_oi / (long_oi + short_oi)

        percentile = None
        if ratio is not None:
            history = journal.query(
                "SELECT long_ratio FROM flows WHERE long_ratio IS NOT NULL "
                "ORDER BY trade_date DESC LIMIT 750"      # ~3 years of sessions
            )
            values = [r["long_ratio"] for r in history]
            if len(values) >= 30:
                import pandas as pd

                percentile = percentile_rank(pd.Series(values), ratio)
            else:
                log.warning(
                    "Only %d historical FII ratios stored; the §6.9 3-year percentile "
                    "is not meaningful yet and the engine will stay flat.", len(values)
                )

        result.flows = {
            "fii_cash_cr": fii_cash, "dii_cash_cr": dii_cash,
            "fii_idx_fut_long": long_oi, "fii_idx_fut_short": short_oi,
            "long_ratio": ratio, "ratio_percentile_3y": percentile,
        }
        if not dry_run:
            journal.record_flows(today, **result.flows)

    step("flows", flows)

    # --- deals and VIX ----------------------------------------------------
    step("bulk_deals", lambda: client.bulk_deals())
    step("block_deals", lambda: client.block_deals())

    def vix() -> None:
        result.india_vix = client.india_vix()

    step("india_vix", vix)

    # --- alerting ---------------------------------------------------------
    if result.failed:
        alerts.error(
            "nightly",
            "Nightly download failures: "
            + "; ".join(f"{k} ({v[:80]})" for k, v in result.failed.items()),
            severity="ERROR",
        )
    _alert_surveillance_diff(result, alerts)
    return result


def _alert_surveillance_diff(result: NightlyResult, alerts: Any) -> None:
    """§6.10: ALERT-ONLY digest of surveillance entries and exits."""
    lines: list[str] = []
    for list_name in sorted(set(result.surveillance_added) | set(result.surveillance_removed)):
        added = result.surveillance_added.get(list_name, [])
        removed = result.surveillance_removed.get(list_name, [])
        if added:
            lines.append(f"{list_name.upper()} IN : {', '.join(added[:20])}")
        if removed:
            lines.append(f"{list_name.upper()} OUT: {', '.join(removed[:20])}")
    if result.ban_added:
        lines.append(f"F&O BAN IN : {', '.join(result.ban_added[:20])}")
    if result.ban_removed:
        lines.append(f"F&O BAN OUT: {', '.join(result.ban_removed[:20])}")

    if not lines:
        return
    alerts.send(
        "🛡️ <b>SURVEILLANCE DIFF</b> "
        f"{result.trade_date}\n<code>" + "\n".join(lines) + "</code>\n"
        "<i>Additions feed the §3 kernel veto. Exits are a watchlist note only.</i>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly NSE downloads (§7 20:30 job)")
    parser.add_argument("--dry-run", action="store_true", help="fetch but do not write")
    args = parser.parse_args(argv)

    setup_logging()
    result = run_nightly(dry_run=args.dry_run)

    print(f"Nightly downloads for {result.trade_date}")
    print(f"  ok     : {', '.join(result.ok) or '-'}")
    print(f"  failed : {', '.join(result.failed) or '-'}")
    for list_name, added in result.surveillance_added.items():
        if added:
            print(f"  {list_name} added  : {', '.join(added[:15])}")
    if result.ban_added:
        print(f"  ban added   : {', '.join(result.ban_added[:15])}")
    if result.india_vix is not None:
        print(f"  india vix   : {result.india_vix}")
    if result.flows.get("ratio_percentile_3y") is not None:
        print(f"  FII long-ratio percentile (3y): {result.flows['ratio_percentile_3y']:.1f}")
    return 0 if result.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
