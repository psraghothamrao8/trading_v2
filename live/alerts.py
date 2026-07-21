"""Telegram alerts: real-time notifications, the 15:45 digest, and ``/kill``.

Implements the alerting half of §5.1 and §7, and the Telegram entry point for
the §3 kill switch.

Why raw HTTP instead of ``python-telegram-bot``: this system needs exactly two
verbs -- send a message, and long-poll for a handful of commands -- from
inside a synchronous APScheduler process. A full async bot framework would
force an event loop into the scheduler thread for no benefit, and it would be
much harder to test. ``httpx`` against the Bot API is ~80 lines and is
trivially faked in tests.

Security: ``/kill`` is honoured **only** from the chat id in ``.env``. A
message from any other chat is logged and dropped.
"""

from __future__ import annotations

import html
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from core import clock
from core.config import get_secrets, get_settings

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class AlertError(RuntimeError):
    """Telegram transport failure. Never allowed to break the trading loop."""


@dataclass
class Command:
    """A parsed inbound Telegram command."""

    name: str
    args: list[str]
    chat_id: str
    message_id: int
    raw: str


class Alerts:
    """Outbound alerts and inbound command polling.

    ``transport`` is injectable: production uses ``httpx``, tests pass a fake
    with ``post(url, json)`` and ``get(url, params)``.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        transport: Any = None,
        enabled: bool | None = None,
    ) -> None:
        secrets = get_secrets()
        settings = get_settings()
        self.bot_token = bot_token if bot_token is not None else secrets.telegram_bot_token
        self.chat_id = chat_id if chat_id is not None else secrets.telegram_chat_id
        self.enabled = (
            bool(settings.get("alerts.telegram.enabled", True)) if enabled is None else enabled
        )
        self.allowed_commands = set(
            settings.get("alerts.telegram.allowed_commands", ["/kill", "/status"])
        )
        self.rate_limit_per_minute = int(settings.get("alerts.telegram.rate_limit_per_minute", 20))
        self._transport = transport
        self._sent_timestamps: list[float] = []
        self._lock = threading.Lock()
        self._offset: Optional[int] = None
        self._handlers: dict[str, Callable[[Command], str]] = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.sent_messages: list[str] = []   # kept for the digest and for tests

    # -- configuration ----------------------------------------------------

    @property
    def configured(self) -> bool:
        """True when a token and chat id are both present."""
        return bool(self.enabled and self.bot_token and self.chat_id)

    def _client(self) -> Any:
        if self._transport is not None:
            return self._transport
        import httpx

        self._transport = httpx.Client(timeout=15.0)
        return self._transport

    def _url(self, method: str) -> str:
        return f"{TELEGRAM_API}/bot{self.bot_token}/{method}"

    # -- outbound ---------------------------------------------------------

    def send(self, text: str, *, silent: bool = False) -> bool:
        """Send a message. Returns False on failure -- never raises upward.

        A Telegram outage must not stop trading, so every failure here is
        logged and swallowed. The journal remains the source of truth.
        """
        self.sent_messages.append(text)
        if not self.configured:
            log.info("[telegram not configured] %s", text.replace("\n", " | ")[:400])
            return False
        if not self._allow_rate():
            log.warning("Telegram rate limit hit; dropping message: %s", text[:80])
            return False
        try:
            response = self._client().post(
                self._url("sendMessage"),
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                    "disable_web_page_preview": True,
                },
            )
            ok = getattr(response, "status_code", 200) < 300
            if not ok:
                log.error("Telegram send failed: %s %s", response.status_code, getattr(response, "text", ""))
            return ok
        except Exception as exc:  # transport-agnostic on purpose
            log.error("Telegram send raised: %s", exc)
            return False

    def _allow_rate(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._sent_timestamps = [t for t in self._sent_timestamps if now - t < 60.0]
            if len(self._sent_timestamps) >= self.rate_limit_per_minute:
                return False
            self._sent_timestamps.append(now)
            return True

    # -- formatted alerts -------------------------------------------------

    def material_filing(
        self,
        symbol: str,
        label: str,
        confidence: float,
        reason: str,
        headline: str = "",
        latency_sec: float | None = None,
    ) -> None:
        """§6.1: every MATERIAL item triggers an instant alert."""
        icon = "🟢" if label == "MATERIAL_POSITIVE" else "🔴"
        lines = [
            f"{icon} <b>{_e(label)}</b>  <code>{_e(symbol)}</code>",
            f"confidence {confidence:.2f}",
        ]
        if headline:
            lines.append(f"<i>{_e(headline[:200])}</i>")
        lines.append(_e(reason[:400]))
        if latency_sec is not None:
            budget = float(get_settings().get("llm.classify_latency_budget_seconds", 20))
            flag = "  ⚠️ over budget, auto-trade skipped" if latency_sec > budget else ""
            lines.append(f"<code>classify {latency_sec:.1f}s{flag}</code>")
        lines.append(f"<code>{clock.now_ist():%H:%M:%S}</code>")
        self.send("\n".join(lines))

    def order_placed(
        self, engine: str, symbol: str, side: str, quantity: int, price: float | None, mode: str
    ) -> None:
        px = f"@ {price:,.2f}" if price is not None else "@ MKT"
        tag = "PAPER" if mode == "paper" else "LIVE"
        self.send(
            f"📤 <b>{_e(tag)} ORDER</b> [{_e(engine)}]\n"
            f"{_e(side)} <code>{_e(symbol)}</code> x{quantity} {px}"
        )

    def fill(self, engine: str, symbol: str, side: str, quantity: int, price: float, costs: float) -> None:
        self.send(
            f"✅ <b>FILL</b> [{_e(engine)}]\n"
            f"{_e(side)} <code>{_e(symbol)}</code> x{quantity} @ {price:,.2f}\n"
            f"costs ₹{costs:,.2f}"
        )

    def rejection(self, engine: str, symbol: str, reason_code: str, reason: str) -> None:
        """§3: rejections are journalled *and* surfaced -- a silent veto teaches
        nothing about why an engine is not trading."""
        self.send(
            f"⛔ <b>REJECTED</b> [{_e(engine)}] <code>{_e(symbol)}</code>\n"
            f"<code>{_e(reason_code)}</code>: {_e(reason)}",
            silent=True,
        )

    def error(self, source: str, message: str, severity: str = "ERROR") -> None:
        """§8.2: fail LOUDLY -- never silently degrade."""
        icon = {"WARNING": "⚠️", "ERROR": "❗", "CRITICAL": "🚨"}.get(severity, "❗")
        self.send(f"{icon} <b>{_e(severity)}</b> [{_e(source)}]\n{_e(message[:800])}")

    def regime_change(self, regime: str, inputs: dict[str, Any], enabled: Sequence[str]) -> None:
        detail = "\n".join(f"  {_e(k)}: {_e(str(v))}" for k, v in inputs.items())
        self.send(
            f"🧭 <b>REGIME: {_e(regime)}</b>\n{detail}\n"
            f"enabled: <code>{_e(', '.join(enabled) or 'none')}</code>"
        )

    def confirmation_request(self, request_id: str, description: str) -> None:
        """§6.8: wheel orders require a typed confirmation, even in paper."""
        self.send(
            f"❓ <b>CONFIRMATION REQUIRED</b>\n{_e(description)}\n\n"
            f"Reply <code>/confirm {_e(request_id)}</code> or "
            f"<code>/reject {_e(request_id)}</code>"
        )

    def killed(self, source: str, orders: int, positions: int, reason: str) -> None:
        self.send(
            f"🚨 <b>KILL SWITCH FIRED</b> ({_e(source)})\n"
            f"cancelled {orders} orders, flattened {positions} positions\n{_e(reason)}"
        )

    def digest(self, summary: dict[str, Any]) -> None:
        """The 15:45 nightly digest (§7)."""
        lines = [f"📊 <b>DIGEST {_e(str(summary.get('trade_date', '')))}</b>"]
        if summary.get("regime"):
            lines.append(f"regime: <b>{_e(str(summary['regime']))}</b>   mode: {_e(str(summary.get('mode', '')))}")
        net = summary.get("net_pnl")
        if net is not None:
            icon = "🟢" if net >= 0 else "🔴"
            lines.append(
                f"{icon} net ₹{net:,.0f}  (gross ₹{summary.get('gross_pnl', 0):,.0f}, "
                f"costs ₹{summary.get('costs', 0):,.0f})"
            )
        lines.append(
            f"trades {summary.get('trades', 0)}  "
            f"W/L {summary.get('wins', 0)}/{summary.get('losses', 0)}  "
            f"signals {summary.get('signals', 0)}  "
            f"rejections {summary.get('rejections', 0)}"
        )
        for section, items in (summary.get("sections") or {}).items():
            if items:
                lines.append(f"\n<b>{_e(section)}</b>")
                lines.extend(f"  • {_e(str(i))}" for i in items[:15])
        if summary.get("notes"):
            lines.append(f"\n<i>{_e(str(summary['notes']))}</i>")
        self.send("\n".join(lines))

    # -- inbound commands -------------------------------------------------

    def register(self, command: str, handler: Callable[[Command], str]) -> None:
        """Bind a handler. Handlers return the reply text."""
        if not command.startswith("/"):
            command = "/" + command
        self._handlers[command] = handler

    def poll_once(self, timeout: int = 0) -> list[Command]:
        """Fetch and dispatch pending commands. Returns what was handled."""
        if not self.configured:
            return []
        try:
            params: dict[str, Any] = {"timeout": timeout}
            if self._offset is not None:
                params["offset"] = self._offset
            response = self._client().get(self._url("getUpdates"), params=params)
            payload = response.json()
        except Exception as exc:
            log.warning("Telegram getUpdates failed: %s", exc)
            return []

        handled: list[Command] = []
        for update in payload.get("result", []) or []:
            self._offset = int(update["update_id"]) + 1
            message = update.get("message") or update.get("edited_message") or {}
            text = (message.get("text") or "").strip()
            chat_id = str((message.get("chat") or {}).get("id", ""))
            if not text.startswith("/"):
                continue

            # Allowlist: only the configured chat may command this system.
            if chat_id != str(self.chat_id):
                log.warning("Ignoring command %r from unauthorised chat %s", text[:32], chat_id)
                continue

            parts = text.split()
            name = parts[0].split("@")[0].lower()
            command = Command(
                name=name,
                args=parts[1:],
                chat_id=chat_id,
                message_id=int(message.get("message_id", 0)),
                raw=text,
            )
            if name not in self.allowed_commands:
                self.send(f"Unknown command <code>{_e(name)}</code>")
                continue
            handler = self._handlers.get(name)
            if handler is None:
                self.send(f"<code>{_e(name)}</code> is allowed but has no handler bound yet.")
                continue
            try:
                reply = handler(command)
            except Exception as exc:
                log.error("Command %s failed: %s", name, exc, exc_info=True)
                reply = f"Command failed: {exc}"
            if reply:
                self.send(reply)
            handled.append(command)
        return handled

    def start_polling(self, interval_seconds: float = 3.0) -> None:
        """Long-poll for commands on a daemon thread.

        ``/kill`` must be reachable while the scheduler is busy, so this cannot
        live on the scheduler's own thread.
        """
        if not self.configured:
            log.info("Telegram not configured; command polling disabled")
            return
        if self._poll_thread and self._poll_thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.poll_once(timeout=int(interval_seconds))
                except Exception as exc:  # pragma: no cover - defensive
                    log.error("Telegram poll loop error: %s", exc)
                self._stop.wait(interval_seconds)

        self._stop.clear()
        self._poll_thread = threading.Thread(target=loop, name="telegram-poll", daemon=True)
        self._poll_thread.start()
        log.info("Telegram command polling started (commands: %s)", sorted(self.allowed_commands))

    def stop_polling(self) -> None:
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None


def _e(text: Any) -> str:
    """Escape for Telegram HTML parse mode."""
    return html.escape(str(text), quote=False)


class NullAlerts(Alerts):
    """Alerts sink that records but never sends. Used in tests and backtests."""

    def __init__(self) -> None:
        super().__init__(bot_token=None, chat_id=None, transport=None, enabled=False)

    def send(self, text: str, *, silent: bool = False) -> bool:
        self.sent_messages.append(text)
        return True


_default: Optional[Alerts] = None


def get_alerts() -> Alerts:
    """Process-wide alerts singleton."""
    global _default
    if _default is None:
        _default = Alerts()
    return _default


def set_alerts(alerts: Optional[Alerts]) -> None:
    global _default
    _default = alerts
