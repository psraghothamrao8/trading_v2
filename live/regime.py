"""Regime router — §7.

WHY (from the spec): no engine wins every day; each wins on its *kind* of day.
The router's job is deciding which engines are allowed to play today, so each
engine avoids the days it loses on. This meta-layer is worth more than any
single signal.

Classification at 10:00 IST:
  PANIC  India VIX +8% intraday OR index <= -1.5% intraday
  TREND  |gap| >= 0.4% AND index one side of VWAP >= 80% of 09:15-10:00
         AND advance/decline >= 2:1 (or <= 1:2 for a down-trend)
  else   CHOP

The order matters: PANIC is checked first because a panic day can technically
satisfy the down-trend conditions too, and the panic enablement map is the
more conservative one.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.config import Settings, get_settings
from core.datafeed import vwap
from core.types import Regime

log = logging.getLogger(__name__)


@dataclass
class RegimeInputs:
    """Everything the 10:00 classification reads, kept for the day's record."""

    index_gap_pct: Optional[float] = None
    index_intraday_pct: Optional[float] = None
    vix_intraday_pct: Optional[float] = None
    vwap_side_fraction: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    # §7 pre-open context, logged with the day's record.
    gift_nifty_gap_pct: Optional[float] = None
    prior_us_session_pct: Optional[float] = None
    fii_ratio_percentile_3y: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RegimeDecision:
    """The classification plus the enabled-engine set it implies."""

    regime: Regime
    inputs: RegimeInputs
    enabled_engines: list[str] = field(default_factory=list)
    alerts_always: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0
    premium_selling_disabled: bool = False
    reason: str = ""

    def may_open(self, engine: str) -> bool:
        """True if ``engine`` may open a NEW position in this regime."""
        return engine in self.enabled_engines

    def may_alert(self, engine: str) -> bool:
        """§7: filings and surveillance ALERTS run in every regime."""
        return engine in self.enabled_engines or engine in self.alerts_always


def vwap_side_fraction(bars: pd.DataFrame, start: _dt.datetime, end: _dt.datetime) -> Optional[float]:
    """Fraction of bars in ``[start, end]`` on the majority side of VWAP.

    §7's TREND test is "index one side of VWAP >= 80% of 09:15-10:00". This
    returns the *majority* side's fraction, so 0.85 means 85% of the window sat
    on one side -- direction is read separately from the gap.
    """
    window = bars[(bars.index >= start) & (bars.index <= end)]
    if len(window) < 2:
        return None
    vwap_series = vwap(window)
    aligned = pd.concat([window["close"], vwap_series.rename("vwap")], axis=1).dropna()
    if aligned.empty:
        return None
    above = (aligned["close"] > aligned["vwap"]).sum()
    return max(above, len(aligned) - above) / len(aligned)


class RegimeRouter:
    """§7 classification and the engine-enablement map."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.section("regime")

    # -- classification ---------------------------------------------------

    def classify(self, inputs: RegimeInputs, today: _dt.date | None = None) -> RegimeDecision:
        """Apply the §7 rules to a set of measured inputs."""
        today = today or clock.today_ist()
        regime, reason = self._classify_regime(inputs)

        enablement = self.config.get("enablement", {}) or {}
        enabled = list(enablement.get(regime.value, []) or [])
        alerts_always = list(self.config.get("alerts_always", []) or [])

        # §7: expiry days halve all new sizes.
        size_multiplier = 1.0
        if trading_calendar.is_expiry_day(today):
            size_multiplier = float(self.config.get("expiry_day_size_multiplier", 0.5))
            reason += f"; expiry day -> sizes x{size_multiplier}"

        # §7: PANIC disables all new premium selling.
        premium_disabled = (
            regime is Regime.PANIC
            and bool(self.config.get("panic_disables_premium_selling", True))
        )

        return RegimeDecision(
            regime=regime,
            inputs=inputs,
            enabled_engines=enabled,
            alerts_always=alerts_always,
            size_multiplier=size_multiplier,
            premium_selling_disabled=premium_disabled,
            reason=reason,
        )

    def _classify_regime(self, inputs: RegimeInputs) -> tuple[Regime, str]:
        panic = self.config.section("panic")
        trend = self.config.section("trend")

        # PANIC first: a panic day can also look like a down-trend, and the
        # panic enablement map is the more conservative of the two.
        vix_threshold = float(panic.require("vix_intraday_rise_pct"))
        index_threshold = float(panic.require("index_intraday_drop_pct"))

        if inputs.vix_intraday_pct is not None and inputs.vix_intraday_pct >= vix_threshold:
            return Regime.PANIC, f"India VIX +{inputs.vix_intraday_pct:.1f}% (>= {vix_threshold}%)"
        if inputs.index_intraday_pct is not None and inputs.index_intraday_pct <= index_threshold:
            return Regime.PANIC, f"index {inputs.index_intraday_pct:.2f}% (<= {index_threshold}%)"

        min_gap = float(trend.require("min_abs_gap_pct"))
        min_side = float(trend.require("vwap_side_fraction_min"))
        min_ad = float(trend.require("advance_decline_ratio_min"))

        have_all = all(
            value is not None
            for value in (inputs.index_gap_pct, inputs.vwap_side_fraction,
                          inputs.advance_decline_ratio)
        )
        if have_all:
            gap_ok = abs(inputs.index_gap_pct) >= min_gap
            side_ok = inputs.vwap_side_fraction >= min_side
            up_trend = inputs.advance_decline_ratio >= min_ad
            down_trend = inputs.advance_decline_ratio <= 1.0 / min_ad
            if gap_ok and side_ok and (up_trend or down_trend):
                direction = "up" if up_trend else "down"
                return Regime.TREND, (
                    f"{direction}-trend: gap {inputs.index_gap_pct:+.2f}%, "
                    f"{inputs.vwap_side_fraction:.0%} one side of VWAP, "
                    f"A/D {inputs.advance_decline_ratio:.2f}"
                )

        missing = [
            name for name, value in (
                ("gap", inputs.index_gap_pct),
                ("vwap_side", inputs.vwap_side_fraction),
                ("advance_decline", inputs.advance_decline_ratio),
            ) if value is None
        ]
        if missing:
            return Regime.CHOP, f"CHOP (trend inputs unavailable: {', '.join(missing)})"
        return Regime.CHOP, "no panic trigger and trend conditions not met"

    # -- measurement ------------------------------------------------------

    def measure(
        self,
        index_bars: Optional[pd.DataFrame],
        prev_close: Optional[float],
        vix_open: Optional[float] = None,
        vix_now: Optional[float] = None,
        advances: Optional[int] = None,
        declines: Optional[int] = None,
        now: Optional[_dt.datetime] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> RegimeInputs:
        """Build :class:`RegimeInputs` from live data.

        Any input that cannot be measured stays None; the classifier then falls
        back to CHOP and *says so*, rather than inventing a trend from partial
        evidence.
        """
        now = now or clock.now_ist()
        extras = extras or {}
        inputs = RegimeInputs(
            gift_nifty_gap_pct=extras.get("gift_nifty_gap_pct"),
            prior_us_session_pct=extras.get("prior_us_session_pct"),
            fii_ratio_percentile_3y=extras.get("fii_ratio_percentile_3y"),
        )

        if vix_open and vix_now and vix_open > 0:
            inputs.vix_intraday_pct = (vix_now - vix_open) / vix_open * 100.0

        if advances is not None and declines is not None and declines > 0:
            inputs.advance_decline_ratio = advances / declines

        if index_bars is None or index_bars.empty or not prev_close:
            return inputs

        today_bars = index_bars[index_bars.index.date == now.date()]
        if today_bars.empty:
            return inputs

        session_open = float(today_bars["open"].iloc[0])
        latest = float(today_bars["close"].iloc[-1])
        inputs.index_gap_pct = (session_open - prev_close) / prev_close * 100.0
        inputs.index_intraday_pct = (latest - prev_close) / prev_close * 100.0

        start, _ = trading_calendar.session_window(now.date(), "continuous")
        end = clock.at(now.date(), str(self.config.get("classify_at", "10:00")))
        inputs.vwap_side_fraction = vwap_side_fraction(today_bars, start, end)
        return inputs

    # -- enablement -------------------------------------------------------

    def enabled_for(self, regime: Regime) -> list[str]:
        return list((self.config.get("enablement", {}) or {}).get(regime.value, []) or [])

    def na_decision(self) -> RegimeDecision:
        """The pre-10:00 state: nothing is enabled to open new positions."""
        return RegimeDecision(
            regime=Regime.NA,
            inputs=RegimeInputs(),
            enabled_engines=self.enabled_for(Regime.NA),
            alerts_always=list(self.config.get("alerts_always", []) or []),
            reason="not yet classified (§7 classifies at 10:00 IST)",
        )
