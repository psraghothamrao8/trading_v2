"""Orchestrator CLI and runtime entry point.

Phase 1 scope: ``--status`` (account, config, calendar, regime=NA) and
``--kill``. The regime router and APScheduler loop arrive in Phase 5 (§7);
until then ``--run`` raises loudly rather than pretending to trade (§0.8).

Usage
-----
    python -m live.orchestrator --status
    python -m live.orchestrator --kill --reason "manual"
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from core import calendar as trading_calendar
from core import clock
from core.broker import Broker, PaperBroker, get_broker, resolve_mode
from core.config import get_secrets, get_settings, get_universe
from core.journal import Journal, get_journal
from core.logging_config import log_path, setup_logging
from core.types import Regime
from live.alerts import Alerts, get_alerts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def build_status(
    broker: Broker | None = None,
    journal: Journal | None = None,
    alerts: Alerts | None = None,
) -> dict[str, Any]:
    """Collect everything ``--status`` prints, as data (so tests can assert)."""
    settings = get_settings()
    secrets = get_secrets()
    universe = get_universe()

    mode = resolve_mode()
    # --status must never trip the §0.1 live confirmation prompt; it is a
    # read-only command, so it always inspects through a paper broker unless
    # one was injected.
    broker = broker or PaperBroker()
    journal = journal or get_journal()
    alerts = alerts or get_alerts()

    account = broker.account()
    today = clock.today_ist()

    return {
        "timestamp": clock.isoformat(clock.now_ist()),
        "spec_version": settings.get("meta.spec_version", "?"),
        "config_as_of": settings.get("meta.config_as_of", "?"),
        "mode": {
            "configured": mode,
            "effective": broker.mode,
            "live_requires_confirmation": True,
        },
        "account": {
            "user_id": account.user_id,
            "user_name": account.user_name,
            "authenticated": account.authenticated,
            "equity_available": account.equity_available,
            "equity_used": account.equity_used,
            "net": account.net,
            "error": account.error,
        },
        # §7 classifies at 10:00 and journals the decision. Before that (or on
        # a day the router has not run) the honest answer is NA.
        "regime": _todays_regime(journal, today),
        "risk": {
            "capital": settings.require("risk.capital"),
            "risk_per_trade_pct": settings.require("risk.risk_per_trade_pct"),
            "daily_loss_limit_pct": settings.require("risk.daily_loss_limit_pct"),
            "weekly_loss_limit_pct": settings.require("risk.weekly_loss_limit_pct"),
            "max_concurrent_positions_total": settings.require("risk.max_concurrent_positions_total"),
            "max_new_trades_per_day_per_engine": settings.require(
                "risk.max_new_trades_per_day_per_engine"
            ),
            "entry_cutoff": settings.require("risk.vetoes.no_new_intraday_entries_after"),
            "force_flat": settings.require("risk.vetoes.mis_force_flat_at"),
        },
        "engines": {
            name: {
                "enabled": bool(cfg.get("enabled", False)),
                "auto_trade": bool(cfg.get("auto_trade", False)),
                "cap_pct": settings.get(f"risk.per_engine_capital_cap_pct.{name}", 0),
                "alert_only": bool(cfg.get("alert_only", False)),
            }
            for name, cfg in (settings.get("engines", {}) or {}).items()
        },
        "calendar": trading_calendar.describe(today),
        "universe": {
            "as_of": universe.get("meta.as_of", "?"),
            "nifty50": len(universe.get("nifty50", [])),
            "nifty_next_50": len(universe.get("nifty_next_50", [])),
            "nifty_midcap_100": len(universe.get("nifty_midcap_100", [])),
            "pair_sectors": len(universe.get("pair_sectors", {}) or {}),
            "wheel_approved": [w["symbol"] for w in universe.get("wheel_approved", [])],
        },
        "credentials": {
            "kite_api_key": bool(secrets.kite_api_key),
            "kite_access_token": bool(secrets.kite_access_token),
            "anthropic_api_key": bool(secrets.anthropic_api_key),
            "telegram": alerts.configured,
        },
        "storage": {
            "journal_db": str(journal.db_path),
            "log_file": str(log_path()),
        },
        "journal_today": journal.counts_for_date(today.isoformat()),
    }


def _todays_regime(journal: Journal, today: Any) -> str:
    """The regime the §7 router recorded today, or NA if it has not run."""
    rows = journal.query(
        "SELECT regime FROM regime_log WHERE trade_date=? ORDER BY ts DESC LIMIT 1",
        (today.isoformat(),),
    )
    return rows[0]["regime"] if rows else Regime.NA.value


def _yn(value: Any) -> str:
    return "yes" if value else "NO"


def print_status(status: dict[str, Any], stream: Any = None) -> None:
    """Render the status dict. This is the §5 phase-1 acceptance output."""
    out = stream or sys.stdout
    w = lambda line="": print(line, file=out)  # noqa: E731 - local shorthand

    w("=" * 72)
    w(f"  INDIAN MARKETS MULTI-ENGINE TRADING SYSTEM  (spec {status['spec_version']})")
    w(f"  {status['timestamp']}")
    w("=" * 72)

    mode = status["mode"]
    w("")
    w(f"  MODE            : {mode['effective'].upper()}  (configured: {mode['configured']})")
    if mode["effective"] == "paper":
        w("                    no broker orders will be sent (§0.1)")

    acct = status["account"]
    w("")
    w("  ACCOUNT")
    w(f"    user          : {acct['user_id'] or '-'}  {acct['user_name'] or ''}")
    w(f"    authenticated : {_yn(acct['authenticated'])}")
    if acct["equity_available"] is not None:
        w(f"    available     : INR {acct['equity_available']:,.2f}")
    if acct["equity_used"] is not None:
        w(f"    used          : INR {acct['equity_used']:,.2f}")
    if acct["error"]:
        w(f"    ERROR         : {acct['error']}")

    w("")
    w(f"  REGIME          : {status['regime']}")
    if status["regime"] == "NA":
        w("                    (classified at 10:00 IST by the §7 router)")

    risk = status["risk"]
    w("")
    w("  RISK KERNEL (§3)")
    w(f"    capital       : INR {risk['capital']:,.0f}")
    w(f"    per trade     : {risk['risk_per_trade_pct']}%   "
      f"daily {risk['daily_loss_limit_pct']}%   weekly {risk['weekly_loss_limit_pct']}%")
    w(f"    max positions : {risk['max_concurrent_positions_total']}  "
      f"(max {risk['max_new_trades_per_day_per_engine']} new/day/engine)")
    w(f"    entry cutoff  : {risk['entry_cutoff']}   force-flat {risk['force_flat']}")

    w("")
    w("  ENGINES")
    w(f"    {'engine':<20} {'enabled':<9} {'auto-trade':<12} {'cap %':<7}")
    for name, cfg in status["engines"].items():
        auto = "ALERT-ONLY" if cfg["alert_only"] else _yn(cfg["auto_trade"])
        w(f"    {name:<20} {_yn(cfg['enabled']):<9} {auto:<12} {cfg['cap_pct']:<7}")

    cal = status["calendar"]
    w("")
    w("  CALENDAR")
    w(f"    {cal['date']} ({cal['weekday']})  trading_day={_yn(cal['trading_day'])}"
      f"  holiday={_yn(cal['holiday'])}")
    w(f"    blocked event : {_yn(cal['blocked_event'])}  {cal.get('event_note', '')}")
    if "is_expiry_day" in cal:
        w(f"    expiry today  : {_yn(cal['is_expiry_day'])}   "
          f"next weekly {cal.get('next_weekly_expiry')}   "
          f"next monthly {cal.get('next_monthly_expiry')}")
    if not cal.get("expiry_config_verified", False):
        w("    ⚠ expiry/lot-size config is UNVERIFIED (§8.3) -- verify before F&O go-live")
    if cal.get("expiry_error"):
        w(f"    expiry ERROR  : {cal['expiry_error']}")

    uni = status["universe"]
    w("")
    w("  UNIVERSE")
    w(f"    as_of {uni['as_of']}   nifty50={uni['nifty50']}  next50={uni['nifty_next_50']}"
      f"  midcap100={uni['nifty_midcap_100']}  pair_sectors={uni['pair_sectors']}")
    w(f"    wheel approved: {', '.join(uni['wheel_approved']) or '-'}")

    creds = status["credentials"]
    w("")
    w("  CREDENTIALS (.env)")
    w(f"    kite api key  : {_yn(creds['kite_api_key'])}")
    w(f"    kite token    : {_yn(creds['kite_access_token'])}"
      f"   (§8.1 expires daily ~07:30 IST; run scripts/morning_auth.py)")
    w(f"    anthropic key : {_yn(creds['anthropic_api_key'])}")
    w(f"    telegram      : {_yn(creds['telegram'])}")

    store = status["storage"]
    counts = status["journal_today"]
    w("")
    w("  STORAGE")
    w(f"    journal       : {store['journal_db']}")
    w(f"    log           : {store['log_file']}")
    w(f"    today         : " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    w("")
    w("=" * 72)


# ---------------------------------------------------------------------------
# Kill switch (§3)
# ---------------------------------------------------------------------------


def kill(
    broker: Broker | None = None,
    journal: Journal | None = None,
    alerts: Alerts | None = None,
    *,
    source: str = "cli",
    reason: str = "manual kill",
) -> dict[str, int]:
    """§3 ``kill()``: cancel all open orders and flatten everything.

    Callable from the CLI and from the Telegram ``/kill`` handler. Failures on
    individual instruments are logged and counted, never allowed to abort the
    rest of the sweep -- a partial flatten beats an exception halfway through.
    """
    from core.risk import flatten_all  # local import: risk imports broker types

    broker = broker or get_broker(interactive=False)
    journal = journal or get_journal()
    alerts = alerts or get_alerts()

    cancelled = 0
    for order in broker.open_orders():
        try:
            broker.cancel_order(order.get("order_id") or order.get("broker_order_id"))
            cancelled += 1
        except Exception as exc:
            log.error("kill: cancel failed for %r: %s", order, exc)
            journal.record_error("kill", f"cancel failed: {exc}", severity="ERROR")

    flattened = flatten_all(broker, journal, reason=f"KILL: {reason}")

    journal.record_kill(source, reason, cancelled, flattened)
    alerts.killed(source, cancelled, flattened, reason)
    log.critical("KILL SWITCH: cancelled=%d flattened=%d source=%s reason=%s",
                 cancelled, flattened, source, reason)
    return {"orders_cancelled": cancelled, "positions_flattened": flattened}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m live.orchestrator",
        description="Indian markets multi-engine trading system (paper by default).",
    )
    parser.add_argument("--status", action="store_true", help="print account, config and regime")
    parser.add_argument("--run", action="store_true", help="start the live session loop (§7)")
    parser.add_argument("--kill", action="store_true", help="§3 kill switch: cancel + flatten all")
    parser.add_argument("--reason", default="manual kill", help="reason recorded with --kill")
    parser.add_argument("--mode", choices=["paper", "live"], help="override execution mode")
    parser.add_argument("--json", action="store_true", help="emit --status as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.status:
        status = build_status()
        if args.json:
            import json

            print(json.dumps(status, indent=2, default=str))
        else:
            print_status(status)
        return 0

    if args.kill:
        result = kill(reason=args.reason)
        print(f"KILL: cancelled {result['orders_cancelled']} orders, "
              f"flattened {result['positions_flattened']} positions")
        return 0

    if args.run:
        from live.session import run_session

        return run_session(mode=args.mode)

    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
