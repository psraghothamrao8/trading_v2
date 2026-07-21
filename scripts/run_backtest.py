"""Backtest CLI — §4.

    python scripts/run_backtest.py --engine filings --window validate
    python scripts/run_backtest.py --sanity                     # buy-and-hold NIFTY
    python scripts/run_backtest.py --engine pairs --window tune --interval 5minute
    python scripts/run_backtest.py --all --window validate

Prints the §4 metric set, the monthly returns table, and a PROMOTED / FAILED
verdict with every gate's number. Writes the equity curve to CSV.

The `test` window is consumed exactly once (§4); a re-run prints a loud
warning and the result should be treated as tainted.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` as well as `python -m scripts.<name>`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
import sys
from pathlib import Path

from core.config import REPO_ROOT, get_settings
from core.logging_config import setup_logging
from backtest.metrics import monthly_returns_table, write_equity_curve_csv
from backtest.runner import Backtester, BacktestError, BuyAndHold, load_windows

log = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "backtest_output"


def report(result, output_dir: Path) -> None:
    """Print the §4 metric set and write the equity-curve CSV."""
    metrics = result.metrics
    print()
    print(result.verdict.render())
    print()
    print("  MONTHLY RETURNS (net of costs)")
    print(monthly_returns_table(metrics))
    print()
    print(f"  bars processed   : {result.bars_processed:,}")
    print(f"  signals emitted  : {result.signals_emitted:,}")
    print(f"  signals rejected : {result.signals_rejected:,}")
    for code, count in sorted(result.rejection_reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {code:<34} {count:,}")

    if metrics.equity_curve:
        path = write_equity_curve_csv(
            metrics, output_dir / f"{result.engine}_{result.window.name}_equity.csv"
        )
        print(f"\n  equity curve     : {path}")
    print()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    windows = load_windows(settings)

    parser = argparse.ArgumentParser(description="Run a backtest (§4)")
    parser.add_argument("--engine", help="engine name from config/settings.yaml")
    parser.add_argument("--all", action="store_true", help="every enabled engine")
    parser.add_argument("--sanity", action="store_true",
                        help="buy-and-hold NIFTY sanity check (§5.3 acceptance)")
    parser.add_argument("--symbol", default="NIFTYBEES", help="symbol for --sanity")
    parser.add_argument("--window", default="validate", choices=sorted(windows),
                        help="walk-forward window (§4)")
    parser.add_argument("--interval", default="day", help="bar interval")
    parser.add_argument("--symbols", nargs="+", help="override the engine universe")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="where to write CSVs")
    args = parser.parse_args(argv)

    setup_logging()
    window = windows[args.window]
    output_dir = Path(args.output)
    backtester = Backtester()

    if args.sanity:
        print(f"\nSanity backtest: buy and hold {args.symbol} over {window}")
        print("If this does not produce a believable net number, no engine result can be.")
        engine = BuyAndHold(symbol=args.symbol)
        try:
            result = backtester.run(engine, window, intervals=(args.interval,))
        except BacktestError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 2
        report(result, output_dir)
        return 0

    if args.all:
        from engines.base import load_engines

        engines = load_engines()
        names = [n for n, e in engines.items() if e.enabled and not e.alert_only]
    elif args.engine:
        names = [args.engine]
    else:
        parser.print_help()
        return 1

    from engines.base import load_engines

    engines = load_engines(names)
    exit_code = 0
    promoted, failed = [], []

    for name in names:
        engine = engines.get(name)
        if engine is None:
            print(f"ERROR: unknown engine {name!r}", file=sys.stderr)
            exit_code = 2
            continue
        try:
            result = backtester.run(
                engine, window, symbols=args.symbols, intervals=(args.interval,)
            )
        except BacktestError as exc:
            print(f"\n{name}: ERROR {exc}", file=sys.stderr)
            exit_code = 2
            continue
        report(result, output_dir)
        (promoted if result.verdict.promoted else failed).append(name)

    if len(names) > 1:
        print("=" * 72)
        print(f"  PROMOTED: {', '.join(promoted) or '-'}")
        print(f"  FAILED  : {', '.join(failed) or '-'}")
        print("  A FAILED engine stays alert-only. Do not soften a gate (§4).")
        print("=" * 72)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
