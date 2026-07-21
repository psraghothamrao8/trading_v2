"""Backtest metrics and promotion gates — §4.

Metrics: trades, win rate, avg win/loss, expectancy, profit factor, max
drawdown, a monthly returns table, and an equity-curve CSV.

Promotion gates (validation window, per engine): profit factor >= 1.3,
trades >= 150 (>= 40 for event/quarterly engines), max drawdown <= 12% of
engine capital, net-of-costs expectancy > 0.

**A FAILED verdict is valuable output, not a bug.** Nothing in this module
softens a gate, and :func:`evaluate_gates` takes its thresholds from
``settings.yaml`` so there is no code path where a number gets "adjusted" to
make an engine pass.
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from core.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """One completed round trip in a backtest. All P&L is net of costs (§4)."""

    engine: str
    symbol: str
    side: str
    quantity: int
    entry_ts: _dt.datetime
    entry_price: float
    exit_ts: _dt.datetime
    exit_price: float
    gross_pnl: float
    costs: float
    exit_reason: str = ""
    stop: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        return round(self.gross_pnl - self.costs, 2)

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def r_multiple(self) -> Optional[float]:
        """Net P&L in units of initial risk. None when no stop was set."""
        if self.stop is None or self.quantity == 0:
            return None
        risk_per_unit = abs(self.entry_price - self.stop)
        if risk_per_unit <= 0:
            return None
        return round(self.net_pnl / (risk_per_unit * self.quantity), 3)

    @property
    def holding_days(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds() / 86400.0


@dataclass
class Metrics:
    """§4 metric set. Every P&L figure here is net of costs."""

    engine: str
    window: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_r: Optional[float] = None
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_days: float = 0.0
    start: Optional[_dt.date] = None
    end: Optional[_dt.date] = None
    monthly_returns: dict[str, float] = field(default_factory=dict)
    equity_curve: list[tuple[_dt.datetime, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for journalling; the equity curve is dropped (too large)."""
        return {
            "engine": self.engine, "window": self.window, "trades": self.trades,
            "wins": self.wins, "losses": self.losses, "win_rate": self.win_rate,
            "gross_pnl": self.gross_pnl, "total_costs": self.total_costs,
            "net_pnl": self.net_pnl, "avg_win": self.avg_win, "avg_loss": self.avg_loss,
            "expectancy": self.expectancy, "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown, "max_drawdown_pct": self.max_drawdown_pct,
            "avg_r": self.avg_r, "best_trade": self.best_trade,
            "worst_trade": self.worst_trade, "avg_holding_days": self.avg_holding_days,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "monthly_returns": self.monthly_returns,
        }


def compute_metrics(
    trades: Sequence[BacktestTrade],
    engine: str,
    window: str,
    starting_equity: float,
) -> Metrics:
    """Compute the §4 metric set from a list of completed trades."""
    metrics = Metrics(engine=engine, window=window)
    if not trades:
        log.info("No trades for %s in the %s window", engine, window)
        return metrics

    ordered = sorted(trades, key=lambda t: t.exit_ts)
    nets = [t.net_pnl for t in ordered]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]

    metrics.trades = len(ordered)
    metrics.wins = len(wins)
    metrics.losses = len(losses)
    metrics.win_rate = round(len(wins) / len(ordered) * 100.0, 2)
    metrics.gross_pnl = round(sum(t.gross_pnl for t in ordered), 2)
    metrics.total_costs = round(sum(t.costs for t in ordered), 2)
    metrics.net_pnl = round(sum(nets), 2)
    metrics.avg_win = round(sum(wins) / len(wins), 2) if wins else 0.0
    metrics.avg_loss = round(sum(losses) / len(losses), 2) if losses else 0.0
    metrics.expectancy = round(metrics.net_pnl / len(ordered), 2)
    metrics.best_trade = round(max(nets), 2)
    metrics.worst_trade = round(min(nets), 2)
    metrics.avg_holding_days = round(
        sum(t.holding_days for t in ordered) / len(ordered), 2
    )

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        metrics.profit_factor = round(gross_profit / gross_loss, 3)
    elif gross_profit > 0:
        # No losing trades at all. `inf` is honest but useless in a comparison,
        # so report it as inf and let the gate handle it explicitly.
        metrics.profit_factor = math.inf

    r_values = [t.r_multiple for t in ordered if t.r_multiple is not None]
    if r_values:
        metrics.avg_r = round(sum(r_values) / len(r_values), 3)

    metrics.start = ordered[0].entry_ts.date()
    metrics.end = ordered[-1].exit_ts.date()

    curve = equity_curve(ordered, starting_equity)
    metrics.equity_curve = curve
    metrics.max_drawdown, metrics.max_drawdown_pct = max_drawdown(curve)
    metrics.monthly_returns = monthly_returns(ordered)
    return metrics


def equity_curve(
    trades: Sequence[BacktestTrade], starting_equity: float
) -> list[tuple[_dt.datetime, float]]:
    """Equity after each trade's exit, starting from ``starting_equity``."""
    equity = starting_equity
    curve: list[tuple[_dt.datetime, float]] = []
    for trade in sorted(trades, key=lambda t: t.exit_ts):
        equity += trade.net_pnl
        curve.append((trade.exit_ts, round(equity, 2)))
    return curve


def max_drawdown(curve: Sequence[tuple[_dt.datetime, float]]) -> tuple[float, float]:
    """Largest peak-to-trough decline, as ``(absolute, percent_of_peak)``."""
    if not curve:
        return 0.0, 0.0
    peak = curve[0][1]
    worst_abs = 0.0
    worst_pct = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        decline = peak - equity
        if decline > worst_abs:
            worst_abs = decline
            worst_pct = (decline / peak * 100.0) if peak else 0.0
    return round(worst_abs, 2), round(worst_pct, 2)


def monthly_returns(trades: Sequence[BacktestTrade]) -> dict[str, float]:
    """``{"2024-03": net_pnl}`` keyed by exit month."""
    out: dict[str, float] = {}
    for trade in trades:
        key = f"{trade.exit_ts.year:04d}-{trade.exit_ts.month:02d}"
        out[key] = round(out.get(key, 0.0) + trade.net_pnl, 2)
    return dict(sorted(out.items()))


def monthly_returns_table(metrics: Metrics) -> str:
    """The §4 monthly returns table, as text."""
    if not metrics.monthly_returns:
        return "(no trades)"
    by_year: dict[int, dict[int, float]] = {}
    for key, value in metrics.monthly_returns.items():
        year, month = key.split("-")
        by_year.setdefault(int(year), {})[int(month)] = value

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lines = ["      " + "".join(f"{m:>10}" for m in months) + f"{'YEAR':>12}"]
    for year in sorted(by_year):
        row = by_year[year]
        cells = "".join(
            f"{row[m]:>10,.0f}" if m in row else f"{'-':>10}" for m in range(1, 13)
        )
        lines.append(f"{year}  {cells}{sum(row.values()):>12,.0f}")
    return "\n".join(lines)


def write_equity_curve_csv(metrics: Metrics, path: str | Path) -> Path:
    """§4 deliverable: the equity curve as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "equity"])
        for timestamp, equity in metrics.equity_curve:
            writer.writerow([timestamp.isoformat(), equity])
    return path


# ---------------------------------------------------------------------------
# Promotion gates (§4)
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """One gate's outcome, with the number that produced it."""

    name: str
    passed: bool
    actual: Any
    required: Any
    detail: str = ""


@dataclass
class PromotionVerdict:
    """PROMOTED or FAILED, with every gate's number (§4)."""

    engine: str
    window: str
    promoted: bool
    gates: list[GateResult] = field(default_factory=list)
    metrics: Optional[Metrics] = None

    @property
    def verdict(self) -> str:
        return "PROMOTED" if self.promoted else "FAILED"

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def render(self) -> str:
        """The §4 printed verdict: PROMOTED / FAILED with the numbers."""
        icon = "PROMOTED" if self.promoted else "FAILED"
        lines = [
            "=" * 72,
            f"  {self.engine.upper()}  [{self.window}]  ->  {icon}",
            "=" * 72,
        ]
        for gate in self.gates:
            mark = "PASS" if gate.passed else "FAIL"
            lines.append(
                f"  [{mark}] {gate.name:<28} actual={gate.actual!s:<14} "
                f"required={gate.required!s}"
            )
            if gate.detail:
                lines.append(f"         {gate.detail}")
        if self.metrics:
            m = self.metrics
            lines += [
                "",
                f"  trades {m.trades}   win rate {m.win_rate}%   "
                f"expectancy {m.expectancy:,.2f}",
                f"  net P&L {m.net_pnl:,.2f}  (gross {m.gross_pnl:,.2f}, "
                f"costs {m.total_costs:,.2f})",
                f"  avg win {m.avg_win:,.2f}   avg loss {m.avg_loss:,.2f}   "
                f"avg R {m.avg_r}",
                f"  max drawdown {m.max_drawdown:,.2f} ({m.max_drawdown_pct}%)",
            ]
        if not self.promoted:
            lines += [
                "",
                "  FAILED is a verdict, not a bug. This engine stays alert-only",
                "  (auto_trade: false). Do not soften a gate to make it pass (§4).",
            ]
        lines.append("=" * 72)
        return "\n".join(lines)


def evaluate_gates(
    metrics: Metrics,
    engine_capital: float,
    settings: Settings | None = None,
) -> PromotionVerdict:
    """Run the §4 promotion gates. Thresholds come from config, never from code."""
    settings = settings or get_settings()
    gates_config = settings.section("backtest.promotion_gates")

    event_engines = set(gates_config.get("event_engines", []) or [])
    min_trades = int(
        gates_config.require("min_trades_event_engines")
        if metrics.engine in event_engines
        else gates_config.require("min_trades_default")
    )
    pf_min = float(gates_config.require("profit_factor_min"))
    dd_max_pct = float(gates_config.require("max_drawdown_pct_of_engine_capital"))
    require_expectancy = bool(gates_config.get("require_net_expectancy_positive", True))

    dd_limit = engine_capital * dd_max_pct / 100.0

    gates = [
        GateResult(
            name="profit_factor",
            passed=metrics.profit_factor >= pf_min,
            actual=metrics.profit_factor,
            required=f">= {pf_min}",
        ),
        GateResult(
            name="trades",
            passed=metrics.trades >= min_trades,
            actual=metrics.trades,
            required=f">= {min_trades}",
            detail=(
                "event/quarterly engine threshold"
                if metrics.engine in event_engines else ""
            ),
        ),
        GateResult(
            name="max_drawdown",
            passed=metrics.max_drawdown <= dd_limit,
            actual=round(metrics.max_drawdown, 2),
            required=f"<= {dd_limit:,.2f} ({dd_max_pct}% of {engine_capital:,.0f})",
        ),
    ]
    if require_expectancy:
        gates.append(
            GateResult(
                name="net_expectancy",
                passed=metrics.expectancy > 0,
                actual=metrics.expectancy,
                required="> 0",
                detail="net of costs (§4)",
            )
        )

    return PromotionVerdict(
        engine=metrics.engine,
        window=metrics.window,
        promoted=all(g.passed for g in gates),
        gates=gates,
        metrics=metrics,
    )


def engine_capital_for(engine: str, settings: Settings | None = None) -> float:
    """Capital allocated to an engine, from its §3 per-engine cap.

    A cap explicitly configured as 0 (``surveillance``, ``special_situations``)
    means exactly that: no capital. A *missing* cap is different -- it is an
    engine the risk config has never heard of, and returning 0 there would make
    the drawdown gate read "<= 0.00 (12% of 0)", which passes for any engine
    that happens not to draw down. That is a gate that cannot fail, so it falls
    back to the full capital and says so.
    """
    settings = settings or get_settings()
    capital = float(settings.require("risk.capital"))
    caps = settings.get("risk.per_engine_capital_cap_pct", {}) or {}
    if engine not in caps:
        log.warning(
            "No per-engine capital cap configured for %r; using the full capital "
            "(%.0f) as the drawdown base. Add it to settings.yaml "
            "`risk.per_engine_capital_cap_pct` so the §4 drawdown gate is meaningful.",
            engine, capital,
        )
        return capital
    return capital * float(caps[engine]) / 100.0
