"""Monthly cointegration refresh for §6.3.

    python scripts/refresh_pairs.py
    python scripts/refresh_pairs.py --dry-run

Runs an Engle-Granger two-step test on 1 year of daily closes for every
same-sector combination in ``universe.yaml -> pair_sectors``, keeps pairs with
p < 0.05, and stores the hedge ratio in the journal.

Only same-sector pairs are tested. Cointegration found across unrelated sectors
on a 1-year window is usually an artefact: with enough pairs tested, some will
pass p < 0.05 by chance alone, and those are exactly the ones that blow up.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` as well as `python -m scripts.<name>`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import datetime as _dt
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core import clock
from core.config import get_settings, get_universe
from core.datafeed import DataFeed, DataError
from core.journal import Journal, get_journal
from core.logging_config import setup_logging

log = logging.getLogger(__name__)


@dataclass
class PairResult:
    """One tested pair."""

    sector: str
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    pvalue: float
    observations: int

    @property
    def passed(self) -> bool:
        return self.pvalue < 0.05


def engle_granger(
    series_a: pd.Series, series_b: pd.Series
) -> tuple[Optional[float], Optional[float]]:
    """Engle-Granger two-step: returns ``(hedge_ratio, pvalue)``.

    Step 1 regresses A on B (OLS) to get the hedge ratio; step 2 runs an ADF
    test on the residual. ``statsmodels.coint`` does both, but returning the
    hedge ratio matters -- §6.3 trades a rupee-neutral spread and needs beta.
    """
    joined = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if len(joined) < 60:
        return None, None

    try:
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log.error("statsmodels is not installed; pip install -r requirements.txt")
        return None, None

    design = sm.add_constant(joined["b"].values)
    model = sm.OLS(joined["a"].values, design).fit()
    hedge_ratio = float(model.params[1])
    residual = joined["a"].values - (model.params[0] + hedge_ratio * joined["b"].values)

    if not np.isfinite(residual).all() or np.std(residual) == 0:
        return None, None

    try:
        pvalue = float(adfuller(residual, autolag="AIC")[1])
    except Exception as exc:
        log.debug("ADF failed: %s", exc)
        return None, None
    return hedge_ratio, pvalue


def test_sector(
    sector: str,
    symbols: Sequence[str],
    feed: DataFeed,
    lookback_days: int,
    end: _dt.date,
) -> list[PairResult]:
    """Test every combination within one sector."""
    start = end - _dt.timedelta(days=int(lookback_days * 1.6))   # calendar pad for sessions

    frames: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            frame = feed.load(symbol, "day", start=start, end=end)
        except DataError as exc:
            log.warning("%s: no daily data (%s)", symbol, exc)
            continue
        if len(frame) >= 60:
            frames[symbol] = frame["close"].iloc[-lookback_days:]

    results: list[PairResult] = []
    for symbol_a, symbol_b in itertools.combinations(sorted(frames), 2):
        hedge_ratio, pvalue = engle_granger(frames[symbol_a], frames[symbol_b])
        if hedge_ratio is None or pvalue is None:
            continue
        results.append(
            PairResult(
                sector=sector, symbol_a=symbol_a, symbol_b=symbol_b,
                hedge_ratio=round(hedge_ratio, 6), pvalue=round(pvalue, 6),
                observations=min(len(frames[symbol_a]), len(frames[symbol_b])),
            )
        )
    return results


def refresh(
    feed: DataFeed | None = None,
    journal: Journal | None = None,
    end: _dt.date | None = None,
    dry_run: bool = False,
) -> list[PairResult]:
    """Run the full refresh. Returns every tested pair, passing or not."""
    settings = get_settings()
    journal = journal or get_journal()
    feed = feed or DataFeed(journal=journal)
    end = end or clock.today_ist()

    lookback = int(settings.require("engines.pairs.cointegration.lookback_days"))
    pvalue_max = float(settings.require("engines.pairs.cointegration.pvalue_max"))
    sectors = get_universe().get("pair_sectors", {}) or {}

    all_results: list[PairResult] = []
    for sector, symbols in sectors.items():
        results = test_sector(sector, symbols, feed, lookback, end)
        all_results.extend(results)
        passing = [r for r in results if r.pvalue < pvalue_max]
        log.info("%s: %d pairs tested, %d passed p<%.2f",
                 sector, len(results), len(passing), pvalue_max)

    passing = [r for r in all_results if r.pvalue < pvalue_max]
    if not dry_run:
        journal.save_pairs(
            [
                {
                    "sector": r.sector, "symbol_a": r.symbol_a, "symbol_b": r.symbol_b,
                    "hedge_ratio": r.hedge_ratio, "pvalue": r.pvalue,
                    "lookback_days": lookback,
                }
                for r in passing
            ],
            refreshed_on=end.isoformat(),
        )
    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly Engle-Granger pair refresh (§6.3)")
    parser.add_argument("--dry-run", action="store_true", help="test but do not store")
    parser.add_argument("--end", help="YYYY-MM-DD, default today")
    args = parser.parse_args(argv)

    setup_logging()
    end = _dt.date.fromisoformat(args.end) if args.end else clock.today_ist()
    pvalue_max = float(get_settings().require("engines.pairs.cointegration.pvalue_max"))

    results = refresh(end=end, dry_run=args.dry_run)
    passing = sorted((r for r in results if r.pvalue < pvalue_max), key=lambda r: r.pvalue)

    print(f"\nEngle-Granger refresh as of {end}: {len(results)} pairs tested, "
          f"{len(passing)} passed p < {pvalue_max}\n")
    print(f"  {'sector':<22} {'pair':<24} {'beta':>10} {'p':>10}")
    for result in passing:
        print(f"  {result.sector:<22} {result.symbol_a + '/' + result.symbol_b:<24} "
              f"{result.hedge_ratio:>10.4f} {result.pvalue:>10.4f}")

    if not passing:
        print("\n  No pairs passed. §6.3 will not trade until some do — that is the")
        print("  gate working, not a bug. Check that daily data is downloaded.")
    if args.dry_run:
        print("\n  (dry run — nothing stored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
