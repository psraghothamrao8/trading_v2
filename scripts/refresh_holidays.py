"""Diff ``config/holidays.yaml`` against NSE's live holiday master.

    python scripts/refresh_holidays.py --check

Prints a diff and exits non-zero when they disagree. It deliberately does
**not** rewrite the config: holidays.yaml is owner-maintained (§8.3 principle),
and a scraper silently editing the trading calendar is exactly the kind of
change that corrupts a backtest without anyone noticing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys

from core import calendar as trading_calendar
from core.logging_config import setup_logging
from core.nse import NSEClient, NSEError, _parse_nse_datetime


def fetch_nse_holidays(client: NSEClient, year: int) -> set[_dt.date]:
    """Holiday dates for ``year`` from NSE's holiday-master endpoint."""
    rows = client.trading_holidays()
    out: set[_dt.date] = set()
    for row in rows:
        raw = row.get("tradingDate") or row.get("date")
        parsed = _parse_nse_datetime(raw)
        if parsed is None:
            try:
                parsed_date = _dt.datetime.strptime(str(raw).strip(), "%d-%b-%Y").date()
            except (ValueError, TypeError):
                continue
        else:
            parsed_date = parsed.date()
        if parsed_date.year == year:
            out.add(parsed_date)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff holidays.yaml against NSE")
    parser.add_argument("--check", action="store_true", help="exit non-zero on any difference")
    parser.add_argument("--year", type=int, default=_dt.date.today().year)
    args = parser.parse_args(argv)

    setup_logging()

    try:
        with NSEClient() as client:
            live = fetch_nse_holidays(client, args.year)
    except NSEError as exc:
        print(f"ERROR: could not reach NSE: {exc}", file=sys.stderr)
        print("       See docs/RUNBOOK.md -> 'scraper breaks'.", file=sys.stderr)
        return 2

    configured = {d for d in trading_calendar.holidays() if d.year == args.year}

    missing = sorted(live - configured)      # NSE says holiday, we do not
    extra = sorted(configured - live)        # we say holiday, NSE does not

    print(f"Holiday diff for {args.year}")
    print(f"  configured : {len(configured)}")
    print(f"  live (NSE) : {len(live)}")
    if missing:
        print("\n  MISSING from config/holidays.yaml (NSE has them):")
        for day in missing:
            print(f"    - {{date: \"{day.isoformat()}\", name: \"?\"}}   # {day:%A}")
    if extra:
        print("\n  EXTRA in config/holidays.yaml (NSE does not list them):")
        for day in extra:
            print(f"    - {day.isoformat()}  ({day:%A})")

    if not missing and not extra:
        print("\n  MATCH. Set `meta.verified_against_nse: true` in config/holidays.yaml.")
        return 0

    print("\n  Edit config/holidays.yaml by hand, then re-run. This script never")
    print("  rewrites the calendar -- a silent edit here corrupts every backtest.")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
