"""Live/paper session runtime — §5.6, §7.

Owns the whole trading day: the APScheduler job set, the engine loop, the
Signal -> Order -> ``risk.check()`` -> broker path, and the 15:45 digest.

The one-way rule (§0.3) is implemented in exactly one place, :meth:`Session.route`.
Engines return Signals; nothing else in this file lets a Signal reach the
broker without a kernel ALLOW.

Restart safety (§9.5): every scheduler job is idempotent and every piece of
per-day state is keyed by date, so restarting mid-session re-reads the journal
and continues rather than double-trading.
"""

from __future__ import annotations

import datetime as _dt
import logging
import signal as _signal
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd

from core import calendar as trading_calendar
from core import clock
from core.broker import Broker, PaperBroker, get_broker, resolve_mode
from core.config import Settings, get_settings, get_universe
from core.costs import get_cost_model
from core.datafeed import DataFeed
from core.journal import Journal, get_journal
from core.risk import MarketState, RiskKernel, build_order, flatten_all, force_flat_intraday
from core.types import Fill, Regime, Signal, Verdict
from engines.base import Context, Engine, load_engines
from engines.registry import ALERT_ONLY
from live.alerts import Alerts, Command, get_alerts
from live.regime import RegimeDecision, RegimeRouter

log = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-day mutable state. Rebuilt from the journal on a mid-day restart."""

    trade_date: _dt.date
    decision: Optional[RegimeDecision] = None
    entry_cutoff_passed: bool = False
    forced_flat: bool = False
    digest_sent: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def regime(self) -> Regime:
        return self.decision.regime if self.decision else Regime.NA


class Session:
    """The live/paper runtime."""

    def __init__(
        self,
        broker: Broker | None = None,
        journal: Journal | None = None,
        alerts: Alerts | None = None,
        settings: Settings | None = None,
        feed: DataFeed | None = None,
        nse: Any = None,
        engines: dict[str, Engine] | None = None,
        interactive: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.journal = journal or get_journal()
        self.alerts = alerts or get_alerts()
        self.cost_model = get_cost_model()

        self.market = MarketState()
        self.feed = feed if feed is not None else DataFeed(journal=self.journal)
        self._nse = nse

        self.broker = broker or get_broker(
            interactive=interactive, price_source=self, cost_model=self.cost_model
        )
        self.kernel = RiskKernel(
            journal=self.journal, settings=self.settings,
            market=self.market, positions=self.broker,
        )
        self.router = RegimeRouter(self.settings)
        self.engines = engines if engines is not None else load_engines(journal=self.journal)

        self.state = SessionState(trade_date=clock.today_ist())
        self._scheduler: Any = None
        self._stop = threading.Event()
        self._prices: dict[str, float] = {}
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}

        self._register_commands()

    # -- collaborators ----------------------------------------------------

    @property
    def nse(self) -> Any:
        if self._nse is None:
            from core.nse import NSEClient

            self._nse = NSEClient(journal=self.journal, alerts=self.alerts)
        return self._nse

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        """Price source for the paper broker."""
        return {s: self._prices[s] for s in symbols if s in self._prices}

    # -- context ----------------------------------------------------------

    def build_context(self, now: _dt.datetime | None = None) -> Context:
        """The read-only world engines see this cycle."""
        now = now or clock.now_ist()
        return Context(
            now=now,
            regime=self.state.regime,
            bars=dict(self._bars),
            prices=dict(self._prices),
            positions=self.broker.positions(),
            journal=self.journal,
            india_vix=self.state.extras.get("india_vix"),
            index_level=self.state.extras.get("index_level"),
            extras=dict(self.state.extras),
            is_backtest=False,
        )

    # -- the one-way path (§0.3) ------------------------------------------

    def route(self, signal: Signal, ctx: Context) -> Optional[str]:
        """Signal -> Order -> risk.check() -> broker. The ONLY path to an order.

        Returns the broker order id on a fill, or None when the kernel rejected
        it or the broker refused it.
        """
        self.journal.record_signal(signal)

        is_exit = bool(signal.meta.get("exit"))
        reference = signal.reference_price or ctx.price(signal.symbol)
        if reference is None:
            log.warning("no reference price for %s; dropping signal", signal.symbol)
            return None

        lot_size = 1
        if signal.is_derivative:
            try:
                lot_size = trading_calendar.lot_size(
                    str(signal.meta.get("underlying") or signal.symbol)
                )
            except Exception as exc:
                log.error("cannot size %s: %s", signal.symbol, exc)
                self.journal.record_error("session", f"lot size: {exc}", severity="ERROR")
                return None

        multiplier = self.state.decision.size_multiplier if self.state.decision else 1.0
        try:
            order = build_order(
                signal, reference, kernel=self.kernel, lot_size=lot_size,
                size_multiplier=1.0 if is_exit else multiplier, is_entry=not is_exit,
            )
        except Exception as exc:
            log.error("could not build an order for %s: %s", signal.symbol, exc)
            self.journal.record_error("session", f"build_order {signal.symbol}: {exc}",
                                      severity="ERROR")
            return None

        # An explicit quantity in meta wins: exits and option orders size
        # themselves rather than going through the stop-distance formula.
        explicit = signal.meta.get("quantity")
        if explicit:
            order.quantity = int(explicit)

        decision = self.kernel.check(order, now=ctx.now)
        if decision.verdict is Verdict.REJECT:
            self.alerts.rejection(order.engine, order.symbol,
                                  decision.reason_code, decision.reason)
            return None

        try:
            broker_order_id = self.broker.place_order(order)
        except Exception as exc:
            log.error("broker rejected %s: %s", order.symbol, exc)
            self.journal.record_order(order, mode=self.broker.mode, status="FAILED")
            self.journal.record_error("broker", f"{order.symbol}: {exc}", severity="ERROR")
            self.alerts.error("broker", f"{order.symbol}: {exc}")
            return None

        order.broker_order_id = broker_order_id
        self.journal.record_order(order, mode=self.broker.mode, status="FILLED")
        self.alerts.order_placed(
            order.engine, order.symbol, order.side.value, order.quantity,
            order.price, self.broker.mode,
        )

        for fill in getattr(self.broker, "fills", [])[-1:]:
            if fill.order_id == order.order_id:
                self.journal.record_fill(fill)
                self._notify_fill(fill, ctx)
        return broker_order_id

    def _notify_fill(self, fill: Fill, ctx: Context) -> None:
        engine = self.engines.get(fill.engine)
        if engine is not None:
            try:
                engine.on_fill(fill, ctx)
            except Exception as exc:
                log.error("on_fill failed for %s: %s", fill.engine, exc, exc_info=True)

    # -- the engine loop --------------------------------------------------

    def run_cycle(self, now: _dt.datetime | None = None) -> int:
        """One pass: manage open positions, then look for new entries.

        Management runs first, always. An exit that frees a concurrency slot
        should be able to do so before a new entry competes for it, and a stop
        that needs hitting must not wait behind signal generation.
        """
        ctx = self.build_context(now)
        routed = 0

        for name, engine in self.engines.items():
            if name in ALERT_ONLY or not engine.enabled:
                continue
            for signal in self._safe(engine.manage, ctx, name, "manage"):
                if self.route(signal, ctx) is not None:
                    routed += 1

        if self.state.entry_cutoff_passed:
            return routed

        ctx = self.build_context(now)      # refresh: management may have closed
        for name, engine in self.engines.items():
            if name in ALERT_ONLY or not engine.enabled:
                continue
            if self.state.decision and not self.state.decision.may_open(name):
                continue
            for signal in self._safe(engine.on_schedule, ctx, name, "on_schedule"):
                if self.route(signal, ctx) is not None:
                    routed += 1
        return routed

    def _safe(self, fn: Any, ctx: Context, engine: str, hook: str) -> list[Signal]:
        """Run an engine hook; one engine's exception never stops the others."""
        try:
            return list(fn(ctx) or [])
        except Exception as exc:
            log.error("%s.%s failed: %s", engine, hook, exc, exc_info=True)
            self.journal.record_error(engine, f"{hook}: {exc}", severity="ERROR")
            return []

    # -- scheduled jobs (§7) ----------------------------------------------

    def job_auth_check(self) -> bool:
        """08:30 — §8.1: verify the Kite token and alert if unauthenticated."""
        if self.broker.mode == "paper":
            log.info("auth check: paper mode, no broker session needed")
            return True
        if self.broker.is_authenticated():
            log.info("auth check: authenticated")
            return True
        self.alerts.error(
            "auth",
            "NOT AUTHENTICATED with Kite. Access tokens expire daily ~07:30 IST (§8.1).\n"
            "Run: python scripts/morning_auth.py",
            severity="CRITICAL",
        )
        self.journal.record_error("auth", "Kite token invalid at 08:30", severity="CRITICAL")
        return False

    def job_preopen_context(self) -> dict[str, Any]:
        """08:45 — §7 pre-open context, logged with the day's record."""
        extras: dict[str, Any] = {}

        gap = self.gift_nifty_gap_pct()
        if gap is not None:
            extras["gift_nifty_gap_pct"] = gap

        flows_engine = self.engines.get("flows")
        if flows_engine is not None:
            extras.update(self._safe_call(flows_engine.regime_context, self.build_context(),
                                          default={}))

        try:
            vix = self.nse.india_vix()
            if vix is not None:
                extras["india_vix"] = vix
        except Exception as exc:
            log.warning("India VIX unavailable: %s", exc)

        self.state.extras.update(extras)
        log.info("pre-open context: %s", extras)
        return extras

    def gift_nifty_gap_pct(self) -> Optional[float]:
        """§8.3: GIFT Nifty vs prior close as the overnight gap proxy.

        Degrades gracefully and logs when unavailable, because the spec says so
        and because §6.4's early exit must not fire on a missing number.
        """
        if not self.settings.get("market.gift_nifty.enabled", True):
            return None
        try:
            import httpx

            url = str(self.settings.require("market.gift_nifty.url"))
            response = httpx.get(url, timeout=10.0)
            if getattr(response, "status_code", 200) >= 400:
                raise RuntimeError(f"status {response.status_code}")
            payload = response.json()
            last = payload.get("last") or payload.get("lastPrice")
            previous = payload.get("previousClose") or payload.get("prevClose")
            if last and previous:
                return (float(last) - float(previous)) / float(previous) * 100.0
            raise RuntimeError("no last/previousClose in the payload")
        except Exception as exc:
            log.warning("GIFT Nifty unavailable (§8.3, degrading gracefully): %s", exc)
            self.journal.record_error("gift_nifty", str(exc), severity="WARNING")
            return None

    def job_preopen_snapshot(self, label: str) -> int:
        """09:06:30 / 09:07:45 — §6.5 pre-open auction snapshots."""
        engine = self.engines.get("preopen")
        if engine is None or not engine.enabled:
            return 0
        candidates = self._safe_call(
            lambda: engine.take_snapshot(self.build_context(), label), default=[]
        )
        return len(candidates)

    def job_announcements_poll(self) -> int:
        """Every 30s, 08:00-15:35 — §6.1, and it feeds §6.2."""
        engine = self.engines.get("filings")
        if engine is None or not engine.enabled:
            return 0
        ctx = self.build_context()
        material = self._safe_call(lambda: engine.poll(ctx), default=[])

        # §6.2 is triggered by §6.1's output; hand it over explicitly rather
        # than letting sympathy re-poll and re-classify.
        if material:
            self.state.extras["material_filings"] = material
        return len(material)

    def job_regime(self) -> RegimeDecision:
        """10:00 — §7 classification."""
        index_symbol = str(get_universe().get("index_proxies.NIFTY.etf_proxy", "NIFTYBEES"))
        index_bars = self._bars.get((index_symbol, "5minute"))
        daily_bars = self._bars.get((index_symbol, "day"))
        prev_close = (
            float(daily_bars["close"].iloc[-2])
            if daily_bars is not None and len(daily_bars) >= 2 else None
        )

        inputs = self.router.measure(
            index_bars=index_bars,
            prev_close=prev_close,
            vix_open=self.state.extras.get("india_vix_open"),
            vix_now=self.state.extras.get("india_vix"),
            advances=self.state.extras.get("advances"),
            declines=self.state.extras.get("declines"),
            extras=self.state.extras,
        )
        decision = self.router.classify(inputs, self.state.trade_date)
        self.state.decision = decision

        if decision.premium_selling_disabled:
            self.state.extras["premium_selling_disabled"] = True

        self.journal.record_regime(
            decision.regime.value, inputs.as_dict(), decision.enabled_engines
        )
        self.alerts.regime_change(
            decision.regime.value, inputs.as_dict(), decision.enabled_engines
        )
        log.info("REGIME %s: %s | enabled=%s",
                 decision.regime.value, decision.reason, decision.enabled_engines)
        return decision

    def job_entry_cutoff(self) -> None:
        """14:45 — §3: no new intraday entries after this."""
        self.state.entry_cutoff_passed = True
        log.info("entry cutoff reached (14:45); management only from here")
        self.alerts.send("⏱ <b>14:45</b> entry cutoff — management only (§3)", silent=True)

    def job_force_flat(self) -> int:
        """15:10 — §3: MIS force-flat, before the broker's ~15:20 square-off.

        Also runs the two §3 expiry-day sweeps, which are position-side rules
        the entry vetoes cannot express.
        """
        ctx = self.build_context()

        for position in self.kernel.stt_trap_exits_due(ctx.now):
            log.warning("STT trap: forcing an exit on long ITM %s", position.symbol)
        for position in self.kernel.physical_settlement_rolls_due(ctx.now):
            log.warning("physical settlement: %s must be closed or rolled", position.symbol)
            self.alerts.error(
                "physical_settlement",
                f"{position.symbol} is a short stock option near expiry and ITM/near-money. "
                f"Closing it now (§3). Set allow_delivery on the trade to accept delivery.",
                severity="WARNING",
            )

        flattened = force_flat_intraday(self.broker, self.journal)
        self.state.forced_flat = True
        if flattened:
            self.alerts.send(f"🔻 <b>15:10 force-flat</b>: closed {flattened} intraday position(s)")
        return flattened

    def job_overnight_check(self) -> int:
        """15:20 — §6.4's entry window."""
        engine = self.engines.get("overnight")
        if engine is None or not engine.enabled:
            return 0
        if self.state.decision and not self.state.decision.may_open("overnight"):
            log.info("overnight: not enabled in the %s regime", self.state.regime.value)
            return 0

        ctx = self.build_context()
        routed = 0
        for signal in self._safe(engine.on_schedule, ctx, "overnight", "on_schedule"):
            if self.route(signal, ctx) is not None:
                routed += 1
        if not routed:
            reasons = self._safe_call(lambda: engine.blocking_reasons(ctx), default=[])
            if reasons:
                self.alerts.send(
                    "🌙 <b>overnight</b>: no position tonight — " + "; ".join(reasons),
                    silent=True,
                )
        return routed

    def job_digest(self) -> dict[str, Any]:
        """15:45 — the daily digest (§7)."""
        today = self.state.trade_date.isoformat()
        counts = self.journal.counts_for_date(today)

        trades = self.journal.query(
            "SELECT net_pnl, gross_pnl, costs FROM trades WHERE trade_date=?", (today,)
        )
        net = sum(float(r["net_pnl"] or 0) for r in trades)
        gross = sum(float(r["gross_pnl"] or 0) for r in trades)
        costs = sum(float(r["costs"] or 0) for r in trades)
        wins = sum(1 for r in trades if (r["net_pnl"] or 0) > 0)

        rejections = self.journal.query(
            "SELECT reason_code, COUNT(*) AS n FROM rejections WHERE trade_date=? "
            "GROUP BY reason_code ORDER BY n DESC", (today,)
        )
        material = self.journal.query(
            "SELECT symbol, label, confidence FROM announcements "
            "WHERE trade_date=? AND label != 'NOISE'", (today,)
        )
        errors = self.journal.query(
            "SELECT source, message FROM errors WHERE trade_date=? AND severity != 'WARNING'",
            (today,),
        )

        summary = {
            "trade_date": today,
            "mode": self.broker.mode,
            "regime": self.state.regime.value,
            "net_pnl": round(net, 2),
            "gross_pnl": round(gross, 2),
            "costs": round(costs, 2),
            "trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "signals": counts["signals"],
            "rejections": counts["rejections"],
            "sections": {
                "material filings": [
                    f"{r['symbol']} {r['label']} ({r['confidence']:.2f})" for r in material
                ],
                "rejections by reason": [f"{r['reason_code']} x{r['n']}" for r in rejections],
                "errors": [f"{r['source']}: {r['message'][:100]}" for r in errors],
            },
        }

        self.journal.upsert_daily_summary(
            today, mode=self.broker.mode, regime=self.state.regime.value,
            net_pnl=summary["net_pnl"], gross_pnl=summary["gross_pnl"],
            costs=summary["costs"], trades=summary["trades"], wins=wins,
            losses=summary["losses"], signals=counts["signals"],
            rejections=counts["rejections"],
        )
        self.alerts.digest(summary)
        self.state.digest_sent = True
        return summary

    def job_nightly_downloads(self) -> Any:
        """20:30 — surveillance/ban diff, flows, deals, VIX (§6.9, §6.10)."""
        from scripts.nightly_downloads import run_nightly

        result = run_nightly(client=self.nse, journal=self.journal, alerts=self.alerts)
        self.refresh_market_state()

        engine = self.engines.get("special_situations")
        if engine is not None and engine.enabled:
            self._safe_call(lambda: engine.send_alerts(self.build_context()), default=0)
        return result

    def job_pairs_refresh_reminder(self) -> None:
        """Sunday — §6.3 wants a monthly Engle-Granger refresh."""
        rows = self.journal.query("SELECT MAX(refreshed_on) AS d FROM pairs")
        last = rows[0]["d"] if rows and rows[0]["d"] else "never"
        self.alerts.send(
            f"📅 <b>Pairs refresh</b> — last run: {last}\n"
            f"Run <code>python scripts/refresh_pairs.py</code> monthly (§6.3)."
        )

    def _safe_call(self, fn: Any, default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:
            log.error("scheduled call failed: %s", exc, exc_info=True)
            self.journal.record_error("session", str(exc), severity="ERROR")
            return default

    # -- market state -----------------------------------------------------

    def refresh_market_state(self) -> None:
        """Reload the §3 veto inputs: ban list, ASM/GSM, and price bands."""
        surveillance = self.engines.get("surveillance")
        if surveillance is not None:
            ctx = self.build_context()
            self.market.surveillance = self._safe_call(
                lambda: surveillance.veto_symbols(ctx), default=set()
            ) or set()
            self.market.fno_ban_list = self._safe_call(
                lambda: surveillance.ban_list(ctx), default=set()
            ) or set()

        symbols = sorted({p.symbol for p in self.broker.positions()} | set(self._prices))
        if symbols:
            bands = self._safe_call(lambda: self.feed.bands(symbols[:200]), default={})
            if bands:
                self.market.bands.update(bands)
        self.market.spot.update(self._prices)
        self.market.lists_as_of = self.state.trade_date

    def update_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)
        self.market.spot.update(prices)

    def update_bars(self, bars: dict[tuple[str, str], pd.DataFrame]) -> None:
        self._bars.update(bars)

    # -- Telegram commands ------------------------------------------------

    def _register_commands(self) -> None:
        self.alerts.register("/kill", self._cmd_kill)
        self.alerts.register("/status", self._cmd_status)
        self.alerts.register("/confirm", self._cmd_confirm)
        self.alerts.register("/reject", self._cmd_reject)

    def _cmd_kill(self, command: Command) -> str:
        """§3 kill(): cancel all open orders and flatten everything."""
        self.kernel.arm_kill("Telegram /kill")
        cancelled = 0
        for order in self._safe_call(self.broker.open_orders, default=[]) or []:
            try:
                self.broker.cancel_order(order.get("order_id") or order.get("broker_order_id"))
                cancelled += 1
            except Exception as exc:
                log.error("kill: cancel failed: %s", exc)
        flattened = flatten_all(self.broker, self.journal, reason="Telegram /kill")
        self.journal.record_kill("telegram", "user /kill", cancelled, flattened)
        return (f"🚨 KILL: cancelled {cancelled} orders, flattened {flattened} positions. "
                f"No new orders will pass the kernel until restart.")

    def _cmd_status(self, command: Command) -> str:
        account = self.broker.account()
        positions = self.broker.positions()
        lines = [
            f"mode {self.broker.mode} | regime {self.state.regime.value}",
            f"positions {len(positions)} | killed {self.kernel.is_killed}",
        ]
        for position in positions[:10]:
            lines.append(f"  {position.symbol} {position.quantity:+d} @ {position.average_price:,.2f}"
                         f" [{position.engine}]")
        if account.equity_available is not None:
            lines.append(f"available ₹{account.equity_available:,.0f}")
        return "\n".join(lines)

    def _cmd_confirm(self, command: Command) -> str:
        """§6.8: the owner confirming a wheel proposal."""
        return self._resolve_proposal(command, approved=True)

    def _cmd_reject(self, command: Command) -> str:
        return self._resolve_proposal(command, approved=False)

    def _resolve_proposal(self, command: Command, approved: bool) -> str:
        if not command.args:
            return "Usage: /confirm <request_id>"
        request_id = command.args[0]
        wheel = self.engines.get("wheel")
        if wheel is None:
            return "The wheel engine is not loaded."

        proposal = wheel.confirm(request_id, approved)
        if proposal is None:
            return f"Unknown request id {request_id}"
        if not approved:
            return f"Rejected {request_id}."

        ctx = self.build_context()
        signal = wheel.signal_for_confirmed(request_id, ctx)
        if signal is None:
            return f"Confirmed {request_id}, but no signal could be built."
        order_id = self.route(signal, ctx)
        return (f"Confirmed {request_id} -> order {order_id}"
                if order_id else f"Confirmed {request_id}, but the risk kernel rejected it.")

    # -- scheduler --------------------------------------------------------

    def build_scheduler(self) -> Any:
        """Wire the §7 job set onto APScheduler. Jobs are idempotent."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("APScheduler is not installed; pip install -r requirements.txt") from exc

        config = self.settings.section("scheduler")
        timezone = str(config.get("timezone", "Asia/Kolkata"))
        scheduler = BackgroundScheduler(timezone=timezone)
        jobs = config.get("jobs", {}) or {}

        def cron(key: str, fn: Any, **extra: Any) -> None:
            spec = jobs.get(key)
            if not spec or "at" not in spec:
                return
            hour, minute, second = _split_hhmmss(str(spec["at"]))
            scheduler.add_job(
                _guard(fn, key, self.journal),
                CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute,
                            second=second, timezone=timezone),
                id=key, replace_existing=True,
                misfire_grace_time=int(config.get("misfire_grace_time_sec", 300)),
                coalesce=bool(config.get("coalesce", True)),
                kwargs=extra,
            )

        cron("auth_check", self.job_auth_check)
        cron("preopen_context", self.job_preopen_context)
        cron("preopen_snapshot_1", lambda: self.job_preopen_snapshot("s1"))
        cron("preopen_snapshot_2", lambda: self.job_preopen_snapshot("s2"))
        cron("regime_classify", self.job_regime)
        cron("entry_cutoff", self.job_entry_cutoff)
        cron("force_flat", self.job_force_flat)
        cron("overnight_check", self.job_overnight_check)
        cron("digest", self.job_digest)
        cron("nightly_downloads", self.job_nightly_downloads)

        poll = jobs.get("announcements_poll", {})
        if poll:
            start_h, start_m, _ = _split_hhmmss(str(poll.get("start", "08:00")))
            end_h, end_m, _ = _split_hhmmss(str(poll.get("end", "15:35")))
            scheduler.add_job(
                _guard(self.job_announcements_poll, "announcements_poll", self.journal),
                CronTrigger(day_of_week="mon-fri",
                            hour=f"{start_h}-{end_h}", minute="*",
                            second=f"*/{int(poll.get('every_seconds', 30))}",
                            timezone=timezone),
                id="announcements_poll", replace_existing=True,
                max_instances=1, coalesce=True,
                misfire_grace_time=int(config.get("misfire_grace_time_sec", 300)),
            )

        reminder = jobs.get("pairs_refresh_reminder", {})
        if reminder:
            hour, minute, _ = _split_hhmmss(str(reminder.get("at", "10:00")))
            scheduler.add_job(
                _guard(self.job_pairs_refresh_reminder, "pairs_refresh_reminder", self.journal),
                CronTrigger(day_of_week=str(reminder.get("day_of_week", "sun")),
                            hour=hour, minute=minute, timezone=timezone),
                id="pairs_refresh_reminder", replace_existing=True,
            )

        # The engine loop itself, between the open and the close.
        scheduler.add_job(
            _guard(self.run_cycle, "engine_cycle", self.journal),
            IntervalTrigger(seconds=int(self.settings.get("scheduler.cycle_seconds", 60)),
                            timezone=timezone),
            id="engine_cycle", replace_existing=True, max_instances=1, coalesce=True,
        )

        self._scheduler = scheduler
        return scheduler

    def start(self) -> None:
        scheduler = self._scheduler or self.build_scheduler()
        scheduler.start()
        self.alerts.start_polling()
        log.info("session started in %s mode; %d jobs scheduled",
                 self.broker.mode, len(scheduler.get_jobs()))

    def stop(self) -> None:
        """Shut down cleanly. Safe to call on a session that never started."""
        self._stop.set()
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as exc:
                # A scheduler that was built but never started raises here.
                # Shutdown must be idempotent so the signal handler and the
                # `finally` in run_session cannot fight each other.
                log.debug("scheduler shutdown: %s", exc)
        self.alerts.stop_polling()
        log.info("session stopped")

    def scheduled_job_ids(self) -> list[str]:
        scheduler = self._scheduler or self.build_scheduler()
        return sorted(job.id for job in scheduler.get_jobs())


def _split_hhmmss(value: str) -> tuple[int, int, int]:
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _guard(fn: Any, name: str, journal: Journal) -> Any:
    """Wrap a job so a failure is journalled and never kills the scheduler."""

    def wrapped(**kwargs: Any) -> Any:
        try:
            return fn(**kwargs) if kwargs else fn()
        except Exception as exc:
            log.error("scheduled job %s failed: %s", name, exc, exc_info=True)
            journal.record_error("scheduler", f"{name}: {exc}", severity="ERROR")
            return None

    wrapped.__name__ = f"job_{name}"
    return wrapped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_session(mode: str | None = None, interactive: bool = True) -> int:
    """Start the session loop and block until interrupted (§5.6)."""
    settings = get_settings()
    resolved = resolve_mode(mode)

    session = Session(interactive=interactive)
    if resolved == "live" and session.broker.mode != "live":
        print("Live mode was requested but not confirmed; running in PAPER (§0.1).")

    today = clock.today_ist()
    if not trading_calendar.is_trading_day(today):
        print(f"{today} is not a trading day "
              f"({'holiday' if trading_calendar.is_holiday(today) else 'weekend'}).")
        print("Scheduler jobs will still run so nightly downloads happen.")

    session.refresh_market_state()
    session.start()

    stop_event = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        print("\nShutting down...")
        stop_event.set()

    _signal.signal(_signal.SIGINT, handle_signal)
    try:
        _signal.signal(_signal.SIGTERM, handle_signal)
    except (AttributeError, ValueError):  # pragma: no cover - platform dependent
        pass

    print(f"Session running in {session.broker.mode.upper()} mode. Ctrl-C to stop.")
    print(f"Jobs: {', '.join(session.scheduled_job_ids())}")
    try:
        stop_event.wait()
    finally:
        session.stop()
    return 0
