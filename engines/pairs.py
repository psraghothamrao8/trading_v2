"""pairs.py — Intraday pair mean reversion. Implements §6.3.

WHY (from the spec): same-sector large caps are chained together by flows;
intraday divergences snap back. Market-neutral, so it earns on days the
directional engines sit out -- its job in the ensemble is smoothing.

The mandatory cross-check with §6.1's database is the load-bearing rule here:
a real event breaks mean reversion, and a pair that has diverged *because one
leg had material news* is not a divergence, it is a repricing.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.types import EntryType, Segment, Side, Signal, TTL
from engines.base import Context, Engine

log = logging.getLogger(__name__)


@dataclass
class Pair:
    """A cointegrated pair with its hedge ratio, from the monthly refresh."""

    sector: str
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    pvalue: float

    @property
    def key(self) -> str:
        return f"{self.symbol_a}/{self.symbol_b}"

    def spread(self, price_a: float, price_b: float) -> float:
        """``a - beta*b``. The residual that is supposed to be stationary."""
        return price_a - self.hedge_ratio * price_b


def spread_series(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame, hedge_ratio: float
) -> pd.Series:
    """Aligned spread series for two price frames."""
    joined = pd.concat(
        [frame_a["close"].rename("a"), frame_b["close"].rename("b")], axis=1
    ).dropna()
    return joined["a"] - hedge_ratio * joined["b"]


def zscore(spread: pd.Series, window: int) -> Optional[float]:
    """Latest z-score of ``spread`` against a rolling mean/sigma.

    Returns None when there is not enough history, or when sigma is zero --
    dividing by a zero sigma yields inf and would trigger an entry on a pair
    that simply has not moved.
    """
    if len(spread) < window:
        return None
    recent = spread.iloc[-window:]
    sigma = float(recent.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return float((spread.iloc[-1] - recent.mean()) / sigma)


class PairsEngine(Engine):
    """§6.3."""

    name = "pairs"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pairs_cache: Optional[list[Pair]] = None

    # -- pair set ---------------------------------------------------------

    def pairs(self) -> list[Pair]:
        """The active cointegrated pairs, from the monthly refresh (§6.3)."""
        if self._pairs_cache is not None:
            return self._pairs_cache
        rows = self.journal.active_pairs()
        self._pairs_cache = [
            Pair(
                sector=row["sector"],
                symbol_a=row["symbol_a"],
                symbol_b=row["symbol_b"],
                hedge_ratio=float(row["hedge_ratio"]),
                pvalue=float(row["pvalue"]),
            )
            for row in rows
        ]
        if not self._pairs_cache:
            log.warning(
                "No active pairs. Run `python scripts/refresh_pairs.py` -- §6.3 needs "
                "an Engle-Granger refresh before it can trade."
            )
        return self._pairs_cache

    def universe(self) -> list[str]:
        symbols: set[str] = set()
        for pair in self.pairs():
            symbols.add(pair.symbol_a)
            symbols.add(pair.symbol_b)
        return sorted(symbols)

    # -- entries ----------------------------------------------------------

    def on_schedule(self, ctx: Context) -> list[Signal]:
        """§6.3: enter |z| >= 2, rupee-neutral, max 2 concurrent pairs.

        Always evaluated regardless of ``auto_trade`` -- see filings.py for
        why; the one authoritative gate lives in Session.run_cycle.
        """
        zconfig = self.config.section("zscore")
        entry_threshold = float(zconfig.require("entry_abs"))
        interval = str(zconfig.get("timeframe", "5minute"))
        window = self._rolling_window(ctx, interval)

        open_pairs = self._open_pair_keys(ctx)
        max_pairs = int(self.config.require("max_concurrent_pairs"))
        if len(open_pairs) >= max_pairs:
            return []

        # §6.3 MANDATORY: ask 6.1's DB. A real event breaks mean reversion.
        material_today: set[str] = set()
        if self.config.get("skip_if_material_filing_today", True):
            material_today = ctx.the_journal().material_filing_symbols(ctx.today.isoformat())

        signals: list[Signal] = []
        for pair in self.pairs():
            if pair.key in open_pairs:
                continue
            if len(open_pairs) + len(signals) // 2 >= max_pairs:
                break

            if pair.symbol_a in material_today or pair.symbol_b in material_today:
                log.info(
                    "skipping %s: a leg has a MATERIAL filing today (§6.3) -- an "
                    "event is a repricing, not a divergence", pair.key,
                )
                continue

            frame_a = ctx.bars_for(pair.symbol_a, interval)
            frame_b = ctx.bars_for(pair.symbol_b, interval)
            if frame_a is None or frame_b is None or frame_a.empty or frame_b.empty:
                continue

            spread = spread_series(frame_a, frame_b, pair.hedge_ratio)
            z = zscore(spread, window)
            if z is None or abs(z) < entry_threshold:
                continue

            price_a = float(frame_a["close"].iloc[-1])
            price_b = float(frame_b["close"].iloc[-1])
            stop_z = float(zconfig.require("stop_abs"))

            # z > 0 means A is rich relative to B: short A, long B.
            rich_is_a = z > 0
            signals.extend(
                self._pair_signals(pair, z, price_a, price_b, stop_z, spread, window, rich_is_a)
            )
        return signals

    def _pair_signals(
        self,
        pair: Pair,
        z: float,
        price_a: float,
        price_b: float,
        stop_z: float,
        spread: pd.Series,
        window: int,
        rich_is_a: bool,
    ) -> list[Signal]:
        """Both legs of one pair, sized rupee-neutral (§6.3)."""
        recent = spread.iloc[-window:]
        sigma = float(recent.std(ddof=1))
        mean = float(recent.mean())

        # The stop is |z| >= 3.5 on the SPREAD. Translate that into a per-leg
        # price stop so the §3 sizing formula has something to work with.
        stop_spread = mean + stop_z * sigma * (1 if z > 0 else -1)
        spread_move = abs(stop_spread - spread.iloc[-1])

        # Rupee neutrality: the two legs carry equal notional. Attributing the
        # whole spread move to each leg would double-count the risk, so each
        # leg is stopped at half of it.
        leg_move = spread_move / 2.0

        short_symbol, short_price = (pair.symbol_a, price_a) if rich_is_a else (pair.symbol_b, price_b)
        long_symbol, long_price = (pair.symbol_b, price_b) if rich_is_a else (pair.symbol_a, price_a)

        common = dict(
            ttl=TTL.INTRADAY,
            entry_type=EntryType.MARKET,
            segment=Segment.EQUITY_INTRADAY.value,
            pair_key=pair.key,
            hedge_ratio=pair.hedge_ratio,
            entry_z=round(z, 3),
            exit_abs=float(self.config.require("zscore.exit_abs")),
            stop_abs=stop_z,
        )
        reason = f"pair {pair.key} z={z:+.2f} ({pair.sector})"

        return [
            self.signal(
                short_symbol, Side.SELL,
                stop=round(short_price + leg_move, 2),
                reference_price=short_price,
                reason=f"{reason} short rich leg",
                leg="short", **common,
            ),
            self.signal(
                long_symbol, Side.BUY,
                stop=round(long_price - leg_move, 2),
                reference_price=long_price,
                reason=f"{reason} long cheap leg",
                leg="long", **common,
            ),
        ]

    # -- exits ------------------------------------------------------------

    def manage(self, ctx: Context) -> list[Signal]:
        """§6.3: exit |z| <= 0.25, stop |z| >= 3.5, force-flat EOD."""
        zconfig = self.config.section("zscore")
        exit_threshold = float(zconfig.require("exit_abs"))
        stop_threshold = float(zconfig.require("stop_abs"))
        interval = str(zconfig.get("timeframe", "5minute"))
        window = self._rolling_window(ctx, interval)

        by_pair: dict[str, list[Any]] = {}
        for position in ctx.positions_for_engine(self.name):
            key = position.meta.get("pair_key")
            if key:
                by_pair.setdefault(key, []).append(position)

        signals: list[Signal] = []
        for key, positions in by_pair.items():
            pair = next((p for p in self.pairs() if p.key == key), None)
            if pair is None:
                # The monthly refresh dropped this pair. Close it: we no longer
                # have a hedge ratio we believe in.
                signals.extend(self._close_all(positions, ctx, "pair no longer cointegrated"))
                continue

            frame_a = ctx.bars_for(pair.symbol_a, interval)
            frame_b = ctx.bars_for(pair.symbol_b, interval)
            if frame_a is None or frame_b is None or frame_a.empty or frame_b.empty:
                continue

            z = zscore(spread_series(frame_a, frame_b, pair.hedge_ratio), window)
            if z is None:
                continue

            if abs(z) <= exit_threshold:
                signals.extend(self._close_all(positions, ctx, f"converged z={z:+.2f}"))
            elif abs(z) >= stop_threshold:
                signals.extend(self._close_all(positions, ctx, f"spread stop z={z:+.2f}"))
        return signals

    def _close_all(self, positions: list[Any], ctx: Context, reason: str) -> list[Signal]:
        """Close every leg together. A half-closed pair is a naked directional bet."""
        out: list[Signal] = []
        for position in positions:
            price = ctx.price(position.symbol)
            if price is None:
                continue
            out.append(
                self.signal(
                    position.symbol,
                    Side.SELL if position.is_long else Side.BUY,
                    stop=None,
                    reference_price=price,
                    ttl=TTL.INTRADAY,
                    reason=f"{reason} [{position.meta.get('pair_key', '?')}]",
                    segment=Segment.EQUITY_INTRADAY.value,
                    exit=True,
                    quantity=abs(position.quantity),
                    pair_key=position.meta.get("pair_key"),
                )
            )
        return out

    # -- helpers ----------------------------------------------------------

    def _open_pair_keys(self, ctx: Context) -> set[str]:
        return {
            position.meta.get("pair_key")
            for position in ctx.positions_for_engine(self.name)
            if position.meta.get("pair_key")
        }

    def _rolling_window(self, ctx: Context, interval: str) -> int:
        """§6.3: 'rolling 20-day mean/sigma' on 5-minute bars.

        20 days of 5-minute bars is 20 x 75 = 1500 observations for a full NSE
        session (09:15-15:30). Deriving it from config rather than hardcoding
        1500 keeps the rule intact if the timeframe changes.
        """
        days = int(self.config.require("zscore.rolling_days"))
        if interval == "day":
            return days
        minutes_per_bar = _interval_minutes(interval)
        session_minutes = self._session_minutes()
        bars_per_day = max(int(session_minutes // minutes_per_bar), 1)
        return days * bars_per_day

    def _session_minutes(self) -> int:
        from core import clock

        start = clock.parse_hhmm(str(self.settings.require("market.session.continuous_start")))
        end = clock.parse_hhmm(str(self.settings.require("market.session.continuous_end")))
        return (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)


def _interval_minutes(interval: str) -> int:
    mapping = {
        "minute": 1, "3minute": 3, "5minute": 5, "10minute": 10,
        "15minute": 15, "30minute": 30, "60minute": 60,
    }
    if interval not in mapping:
        raise ValueError(f"Unknown interval {interval!r} for the pairs z-score window")
    return mapping[interval]
